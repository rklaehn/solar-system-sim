//! Trajectory optimization from the lunar north pole.
//!
//! Finds launch parameters (angle, speed) from the lunar north pole such that
//! a single payload arrives at EML3 with minimal velocity.
//!
//! Separate from the equatorial optimizer — reuses shared propagation and
//! rendering utilities but has its own panel, state, and launch geometry.

use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, Ordering},
    mpsc,
};
use std::thread::JoinHandle;

use anise::prelude::Almanac;
use bevy::prelude::*;
use bevy_egui::egui;
use hifitime::{Duration, Epoch};
use nyx_space::{
    cosmic::{Mass, Spacecraft},
    dynamics::SpacecraftDynamics,
    propagators::Propagator,
};
use rayon::prelude::*;

use crate::{
    bodies,
    constants::*,
    lagrange,
    optimizer::{
        BevyStrip, closest_approach_to_lagrange, end_condition_ui, moon_frame,
        precompute_strip, propagate_to_trail, range_control,
    },
    orbit,
    visualization::{
        SimulationState, SpaceResources, TrailFrame,
        UserBodies, UserBody, UserBodyMarker,
    },
};

// === Data Structures ===

/// User-editable pole optimizer configuration.
#[derive(Clone, Debug)]
pub struct PoleOptimizerConfig {
    pub speed_min: f64,
    pub speed_max: f64,
    pub angle_min: f64,
    pub angle_max: f64,
    pub max_prop_days: f64,
    pub distance_weight: f64,
    pub velocity_weight: f64,
    pub grid_n_speed: usize,
    pub grid_n_angle: usize,
    pub nm_max_iter: usize,
    pub target: lagrange::LagrangeId,
}

impl Default for PoleOptimizerConfig {
    fn default() -> Self {
        Self {
            speed_min: 2.2,
            speed_max: 2.5,
            angle_min: -180.0,
            angle_max: 180.0,
            max_prop_days: 20.0,
            distance_weight: 1.0,
            velocity_weight: 0.0,
            grid_n_speed: 10,
            grid_n_angle: 36,
            nm_max_iter: 100,
            target: lagrange::LagrangeId::L3,
        }
    }
}

/// Result of evaluating a single (angle, speed) parameter pair from the pole.
#[derive(Clone, Debug)]
pub struct PoleEvalResult {
    pub angle_deg: f64,
    pub speed_km_s: f64,
    pub cost: f64,
    pub trail: Vec<(Epoch, [f64; 3], [f64; 3])>,
    pub closest_km: f64,
    pub arrival_v: f64,
}

/// Messages from pole optimizer thread to main thread.
pub enum PoleOptimizerMsg {
    Current(PoleEvalResult),
    NewBest { rank: usize, result: PoleEvalResult },
    Progress { phase: String, done: usize, total: usize },
    Finished { best: PoleEvalResult },
}

/// Colors for pole optimizer trails (distinct from equatorial optimizer).
const POLE_COLORS: [Color; 3] = [
    Color::srgb(1.0, 0.5, 0.0),   // orange
    Color::srgb(0.5, 1.0, 0.0),   // lime
    Color::srgb(0.3, 0.7, 1.0),   // sky blue
];

/// Bevy resource holding pole optimizer state.
#[derive(Resource)]
pub struct PoleOptimizerState {
    pub config: PoleOptimizerConfig,
    pub running: bool,
    pub cancel: Arc<AtomicBool>,
    receiver: Option<Mutex<mpsc::Receiver<PoleOptimizerMsg>>>,
    thread_handle: Option<Mutex<JoinHandle<()>>>,
    pub best_results: Vec<PoleEvalResult>,
    pub display_strips: Vec<(Color, BevyStrip)>,
    pub phase: String,
    pub progress_done: usize,
    pub progress_total: usize,
    pub final_result: Option<PoleEvalResult>,
    pub current_strip: Option<BevyStrip>,
    pub opt_epoch: Option<Epoch>,
    trail_frame_cached: TrailFrame,
    pub adopt_requests: Vec<usize>,
}

impl Default for PoleOptimizerState {
    fn default() -> Self {
        Self {
            config: PoleOptimizerConfig::default(),
            running: false,
            cancel: Arc::new(AtomicBool::new(false)),
            receiver: None,
            thread_handle: None,
            best_results: Vec::new(),
            display_strips: Vec::new(),
            phase: String::new(),
            progress_done: 0,
            progress_total: 0,
            final_result: None,
            current_strip: None,
            opt_epoch: None,
            trail_frame_cached: TrailFrame::default(),
            adopt_requests: Vec::new(),
        }
    }
}

// === Core Functions ===

/// Create a spacecraft launched from the lunar north pole at `angle_deg`
/// with velocity magnitude `speed_km_s`.
///
/// At the north pole, the surface normal is the orbital angular momentum direction.
/// The tangent plane is spanned by `away` (Earth-Moon radial) and `prograde`.
/// `angle_deg` rotates the velocity direction in this tangent plane (0 = away from Earth).
fn create_pole_spacecraft(
    angle_deg: f64,
    speed_km_s: f64,
    epoch: Epoch,
    almanac: &Almanac,
) -> Option<Spacecraft> {
    let moon_pos = bodies::moon_position(almanac, epoch).ok()?;
    let moon_vel = bodies::moon_velocity(almanac, epoch).ok()?;
    let (away, north, prograde) = moon_frame(&moon_pos, &moon_vel);

    // Position on the north pole of the Moon
    let pos = [
        moon_pos.x + north[0] * MOON_RADIUS_KM,
        moon_pos.y + north[1] * MOON_RADIUS_KM,
        moon_pos.z + north[2] * MOON_RADIUS_KM,
    ];

    // Velocity direction in the tangent plane at the pole
    // 0 deg = away from Earth, 90 deg = prograde
    let az = angle_deg.to_radians();
    let launch_dir = [
        az.cos() * away[0] + az.sin() * prograde[0],
        az.cos() * away[1] + az.sin() * prograde[1],
        az.cos() * away[2] + az.sin() * prograde[2],
    ];

    let vel = [
        moon_vel.x + speed_km_s * launch_dir[0],
        moon_vel.y + speed_km_s * launch_dir[1],
        moon_vel.z + speed_km_s * launch_dir[2],
    ];

    let orb = orbit::create_orbit_cartesian(
        pos, vel, epoch, almanac, orbit::ReferenceFrame::EarthJ2000,
    ).ok()?;
    Some(Spacecraft::builder()
        .orbit(orb)
        .mass(Mass::from_dry_mass(1.0))
        .build())
}

/// Evaluate a single (angle, speed) parameter pair from the north pole.
fn evaluate_pole(
    angle_deg: f64,
    speed_km_s: f64,
    epoch: Epoch,
    almanac: &Arc<Almanac>,
    propagator: &Propagator<SpacecraftDynamics>,
    config: &PoleOptimizerConfig,
) -> Option<PoleEvalResult> {
    let sc = create_pole_spacecraft(angle_deg, speed_km_s, epoch, almanac)?;

    let duration = Duration::from_seconds(config.max_prop_days * 86400.0);
    let sample_step = 120.0;

    let trail = propagate_to_trail(sc, propagator, almanac.clone(), duration, sample_step)?;

    let (closest_km, arrival_v) = closest_approach_to_lagrange(&trail, almanac, config.target)?;

    let cost = config.distance_weight * closest_km.powi(2)
        + config.velocity_weight * arrival_v.powi(2);

    Some(PoleEvalResult {
        angle_deg,
        speed_km_s,
        cost,
        trail,
        closest_km,
        arrival_v,
    })
}

// === Optimization Algorithms ===

/// Uniform grid scan over the parameter space.
fn grid_scan(
    config: &PoleOptimizerConfig,
    epoch: Epoch,
    almanac: &Arc<Almanac>,
    propagator: &Propagator<SpacecraftDynamics>,
    sender: &mpsc::Sender<PoleOptimizerMsg>,
    cancel: &AtomicBool,
) -> Vec<PoleEvalResult> {
    let total = config.grid_n_speed * config.grid_n_angle;

    let params: Vec<(f64, f64)> = (0..config.grid_n_speed)
        .flat_map(|i_s| {
            let speed = if config.grid_n_speed <= 1 {
                config.speed_min
            } else {
                config.speed_min
                    + (config.speed_max - config.speed_min)
                        * i_s as f64
                        / (config.grid_n_speed - 1) as f64
            };
            (0..config.grid_n_angle).map(move |i_a| {
                let angle = if config.grid_n_angle <= 1 {
                    config.angle_min
                } else {
                    config.angle_min
                        + (config.angle_max - config.angle_min)
                            * i_a as f64
                            / (config.grid_n_angle - 1) as f64
                };
                (angle, speed)
            })
        })
        .collect();

    let done_count = std::sync::atomic::AtomicUsize::new(0);
    let results: Vec<PoleEvalResult> = params
        .par_iter()
        .filter_map(|&(angle, speed)| {
            if cancel.load(Ordering::Relaxed) {
                return None;
            }
            let result = evaluate_pole(angle, speed, epoch, almanac, propagator, config);
            let done = done_count.fetch_add(1, Ordering::Relaxed) + 1;
            if let Some(ref r) = result {
                let _ = sender.send(PoleOptimizerMsg::Current(r.clone()));
            }
            let _ = sender.send(PoleOptimizerMsg::Progress {
                phase: "Grid scan".into(),
                done,
                total,
            });
            result
        })
        .collect();

    let mut results = results;
    results.sort_by(|a, b| a.cost.partial_cmp(&b.cost).unwrap_or(std::cmp::Ordering::Equal));

    for (rank, r) in results.iter().take(3).enumerate() {
        let _ = sender.send(PoleOptimizerMsg::NewBest {
            rank,
            result: r.clone(),
        });
    }

    results
}

/// 2D Nelder-Mead optimizer for [angle_deg, speed_km_s].
fn nelder_mead_2d(
    initial_simplex: [[f64; 2]; 3],
    config: &PoleOptimizerConfig,
    epoch: Epoch,
    almanac: &Arc<Almanac>,
    propagator: &Propagator<SpacecraftDynamics>,
    sender: &mpsc::Sender<PoleOptimizerMsg>,
    cancel: &AtomicBool,
) -> Option<PoleEvalResult> {
    let clamp = |p: [f64; 2]| -> [f64; 2] {
        [
            p[0].clamp(config.angle_min, config.angle_max),
            p[1].clamp(config.speed_min, config.speed_max),
        ]
    };

    let eval_point = |p: [f64; 2]| -> (f64, Option<PoleEvalResult>) {
        let p = clamp(p);
        let result = evaluate_pole(p[0], p[1], epoch, almanac, propagator, config);
        if let Some(ref r) = result {
            let _ = sender.send(PoleOptimizerMsg::Current(r.clone()));
        }
        let cost = result.as_ref().map(|r| r.cost).unwrap_or(f64::INFINITY);
        (cost, result)
    };

    let mut simplex: Vec<([f64; 2], f64, Option<PoleEvalResult>)> = initial_simplex
        .iter()
        .map(|&p| {
            let (c, r) = eval_point(p);
            (p, c, r)
        })
        .collect();

    let alpha = 1.0;
    let gamma = 2.0;
    let rho = 0.5;
    let sigma = 0.5;

    for iter in 0..config.nm_max_iter {
        if cancel.load(Ordering::Relaxed) {
            break;
        }

        simplex.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        let _ = sender.send(PoleOptimizerMsg::Progress {
            phase: "Nelder-Mead".into(),
            done: iter,
            total: config.nm_max_iter,
        });

        if let Some(ref r) = simplex[0].2 {
            let _ = sender.send(PoleOptimizerMsg::NewBest {
                rank: 0,
                result: r.clone(),
            });
        }

        let n = simplex.len() - 1;

        let centroid = [
            simplex[..n].iter().map(|s| s.0[0]).sum::<f64>() / n as f64,
            simplex[..n].iter().map(|s| s.0[1]).sum::<f64>() / n as f64,
        ];

        let worst = simplex[n].0;

        let reflected = [
            centroid[0] + alpha * (centroid[0] - worst[0]),
            centroid[1] + alpha * (centroid[1] - worst[1]),
        ];
        let (fr, rr) = eval_point(reflected);

        if fr < simplex[0].1 {
            let expanded = [
                centroid[0] + gamma * (reflected[0] - centroid[0]),
                centroid[1] + gamma * (reflected[1] - centroid[1]),
            ];
            let (fe, re) = eval_point(expanded);
            if fe < fr {
                simplex[n] = (expanded, fe, re);
            } else {
                simplex[n] = (reflected, fr, rr);
            }
        } else if fr < simplex[n - 1].1 {
            simplex[n] = (reflected, fr, rr);
        } else {
            let contracted = [
                centroid[0] + rho * (worst[0] - centroid[0]),
                centroid[1] + rho * (worst[1] - centroid[1]),
            ];
            let (fc, rc) = eval_point(contracted);
            if fc < simplex[n].1 {
                simplex[n] = (contracted, fc, rc);
            } else {
                let best = simplex[0].0;
                for i in 1..simplex.len() {
                    let p = [
                        best[0] + sigma * (simplex[i].0[0] - best[0]),
                        best[1] + sigma * (simplex[i].0[1] - best[1]),
                    ];
                    let (c, r) = eval_point(p);
                    simplex[i] = (p, c, r);
                }
            }
        }

        let dx = (simplex[0].0[0] - simplex[n].0[0]).abs();
        let ds = (simplex[0].0[1] - simplex[n].0[1]).abs();
        if dx < 0.1 && ds < 0.001 {
            break;
        }
    }

    simplex.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    simplex[0].2.clone()
}

// === Threading ===

fn launch_pole_optimizer(
    config: PoleOptimizerConfig,
    epoch: Epoch,
    almanac: Arc<Almanac>,
) -> (mpsc::Receiver<PoleOptimizerMsg>, JoinHandle<()>, Arc<AtomicBool>) {
    let (sender, receiver) = mpsc::channel();
    let cancel = Arc::new(AtomicBool::new(false));
    let cancel_clone = cancel.clone();

    let handle = std::thread::spawn(move || {
        let propagator = match orbit::setup_propagator(&almanac) {
            Ok(p) => p,
            Err(_) => return,
        };

        let grid_results = grid_scan(&config, epoch, &almanac, &propagator, &sender, &cancel_clone);
        if cancel_clone.load(Ordering::Relaxed) || grid_results.is_empty() {
            return;
        }

        if grid_results.len() >= 3 {
            let initial = [
                [grid_results[0].angle_deg, grid_results[0].speed_km_s],
                [grid_results[1].angle_deg, grid_results[1].speed_km_s],
                [grid_results[2].angle_deg, grid_results[2].speed_km_s],
            ];
            if let Some(best) = nelder_mead_2d(
                initial, &config, epoch, &almanac, &propagator, &sender, &cancel_clone,
            ) {
                let _ = sender.send(PoleOptimizerMsg::Finished { best });
                return;
            }
        }

        if let Some(best) = grid_results.into_iter().next() {
            let _ = sender.send(PoleOptimizerMsg::Finished { best });
        }
    });

    (receiver, handle, cancel)
}

// === Bevy Systems ===

fn recompute_all_strips(state: &mut PoleOptimizerState, almanac: &Almanac, moon_now: [f64; 3]) {
    let frame = state.trail_frame_cached;
    state.display_strips.clear();
    for (i, result) in state.best_results.iter().enumerate() {
        let color = POLE_COLORS[i % POLE_COLORS.len()];
        let strip = precompute_strip(&result.trail, frame, almanac, moon_now);
        state.display_strips.push((color, strip));
    }
}

/// Poll the pole optimizer channel each frame and update state.
pub fn poll_pole_optimizer(
    mut opt_state: ResMut<PoleOptimizerState>,
    sim_state: Res<SimulationState>,
    space: Res<SpaceResources>,
) {
    let frame = sim_state.trail_frame;
    let moon_now = bodies::moon_position(&space.almanac, sim_state.epoch)
        .map(|m| [m.x, m.y, m.z])
        .unwrap_or([0.0; 3]);

    if frame != opt_state.trail_frame_cached && !opt_state.best_results.is_empty() {
        opt_state.trail_frame_cached = frame;
        recompute_all_strips(&mut opt_state, &space.almanac, moon_now);
    }

    let messages: Vec<PoleOptimizerMsg> = {
        let Some(ref receiver) = opt_state.receiver else {
            return;
        };
        let Ok(receiver) = receiver.lock() else {
            return;
        };
        let mut msgs = Vec::new();
        while let Ok(msg) = receiver.try_recv() {
            msgs.push(msg);
        }
        msgs
    };

    if messages.is_empty() {
        return;
    }

    let mut finished = false;
    for msg in messages {
        match msg {
            PoleOptimizerMsg::Current(result) => {
                let strip = precompute_strip(&result.trail, frame, &space.almanac, moon_now);
                opt_state.current_strip = Some(strip);
            }
            PoleOptimizerMsg::NewBest { rank, result } => {
                let color = POLE_COLORS[rank % POLE_COLORS.len()];
                let strip = precompute_strip(&result.trail, frame, &space.almanac, moon_now);

                if rank < opt_state.best_results.len() {
                    opt_state.best_results[rank] = result;
                    opt_state.display_strips[rank] = (color, strip);
                } else {
                    opt_state.best_results.push(result);
                    opt_state.display_strips.push((color, strip));
                }
                opt_state.best_results.truncate(3);
                opt_state.display_strips.truncate(3);
            }
            PoleOptimizerMsg::Progress { phase, done, total } => {
                opt_state.phase = phase;
                opt_state.progress_done = done;
                opt_state.progress_total = total;
            }
            PoleOptimizerMsg::Finished { best } => {
                opt_state.final_result = Some(best);
                finished = true;
            }
        }
    }

    opt_state.trail_frame_cached = frame;

    if finished {
        opt_state.running = false;
        opt_state.receiver = None;
        opt_state.thread_handle = None;
        opt_state.current_strip = None;
    }
}

/// Draw pole optimizer trails with gizmos.
pub fn draw_pole_optimizer_trails(
    opt_state: Res<PoleOptimizerState>,
    mut gizmos: Gizmos,
) {
    let has_current = opt_state.current_strip.is_some();
    let has_best = !opt_state.display_strips.is_empty();
    if !has_current && !has_best {
        return;
    }

    let draw_strip = |gizmos: &mut Gizmos, strip: &[Vec3], color: Color| {
        if strip.len() < 2 {
            return;
        }
        gizmos.linestrip(strip.iter().copied(), color);
    };

    if let Some(ref strip) = opt_state.current_strip {
        let dim = Color::srgba(0.4, 0.4, 0.4, 0.6);
        draw_strip(&mut gizmos, strip, dim);
    }

    for (color, strip) in &opt_state.display_strips {
        draw_strip(&mut gizmos, strip, *color);
    }
}

/// Draw pole optimizer UI content into the given `Ui`.
pub(crate) fn pole_optimizer_ui_content(
    ui: &mut egui::Ui,
    opt_state: &mut PoleOptimizerState,
    sim_state: &mut SimulationState,
    space: &SpaceResources,
) {
    let config = &mut opt_state.config;

    egui::CollapsingHeader::new("Parameters")
        .default_open(true)
        .show(ui, |ui| {
            range_control(ui, "Launch speed:", &mut config.speed_min, &mut config.speed_max, 0.01, " km/s");
            range_control(ui, "Launch angle:", &mut config.angle_min, &mut config.angle_max, 1.0, " deg");
            ui.label("(0\u{00b0} = away from Earth, 90\u{00b0} = prograde)");
            ui.horizontal(|ui| {
                ui.label("Max propagation:");
                ui.add(
                    egui::DragValue::new(&mut config.max_prop_days)
                        .range(1.0..=60.0)
                        .speed(0.5)
                        .suffix(" days"),
                );
            });
        });

    egui::CollapsingHeader::new("Cost Function")
        .default_open(true)
        .show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.label("Target:");
                egui::ComboBox::from_id_salt("pole_target")
                    .selected_text(config.target.label())
                    .show_ui(ui, |ui| {
                        for lp in lagrange::LagrangeId::ALL {
                            ui.selectable_value(&mut config.target, lp, lp.label());
                        }
                    });
            });
            end_condition_ui(ui, &mut config.distance_weight, &mut config.velocity_weight);
        });

    egui::CollapsingHeader::new("Algorithm")
        .default_open(false)
        .show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.label("Grid speed steps:");
                ui.add(egui::DragValue::new(&mut config.grid_n_speed).range(2..=50));
            });
            ui.horizontal(|ui| {
                ui.label("Grid angle steps:");
                ui.add(egui::DragValue::new(&mut config.grid_n_angle).range(4..=360));
            });
            ui.horizontal(|ui| {
                ui.label("NM max iterations:");
                ui.add(egui::DragValue::new(&mut config.nm_max_iter).range(10..=1000));
            });
        });

    ui.separator();

    if let Some(epoch) = opt_state.opt_epoch {
        ui.label(format!("Epoch: {epoch}"));
    } else {
        ui.label(format!("Epoch: {} (current)", sim_state.epoch));
    }

    if opt_state.running {
        let frac = if opt_state.progress_total > 0 {
            opt_state.progress_done as f32 / opt_state.progress_total as f32
        } else {
            0.0
        };
        ui.add(
            egui::ProgressBar::new(frac).text(format!(
                "{}: {}/{}",
                opt_state.phase, opt_state.progress_done, opt_state.progress_total
            )),
        );

        if ui.button("Cancel").clicked() {
            opt_state.cancel.store(true, Ordering::Relaxed);
            opt_state.running = false;
            opt_state.receiver = None;
            opt_state.thread_handle = None;
        }
    } else if ui.button("Run Optimization").clicked() {
        sim_state.paused = true;
        opt_state.opt_epoch = Some(sim_state.epoch);

        let (rx, handle, cancel) = launch_pole_optimizer(
            opt_state.config.clone(),
            sim_state.epoch,
            space.almanac.clone(),
        );
        opt_state.receiver = Some(Mutex::new(rx));
        opt_state.thread_handle = Some(Mutex::new(handle));
        opt_state.cancel = cancel;
        opt_state.running = true;
        opt_state.best_results.clear();
        opt_state.display_strips.clear();
        opt_state.final_result = None;
        opt_state.progress_done = 0;
        opt_state.progress_total = 0;
        opt_state.phase = "Starting...".into();
    }

    if !opt_state.best_results.is_empty() && !opt_state.running {
        if ui.button("Clear results").clicked() {
            opt_state.best_results.clear();
            opt_state.display_strips.clear();
            opt_state.final_result = None;
        }
    }

    if !opt_state.best_results.is_empty() {
        ui.separator();
        let running = opt_state.running;
        let mut to_adopt = Vec::new();
        egui::CollapsingHeader::new("Best Results")
            .default_open(true)
            .show(ui, |ui| {
                for (i, result) in opt_state.best_results.iter().enumerate() {
                    let color = POLE_COLORS[i % POLE_COLORS.len()];
                    let [r, g, b, _] = color.to_srgba().to_f32_array();
                    let label = egui::RichText::new(format!(
                        "#{}: {:.1}\u{00b0} @ {:.3} km/s",
                        i + 1,
                        result.angle_deg,
                        result.speed_km_s,
                    ))
                    .color(egui::Color32::from_rgb(
                        (r * 255.0) as u8,
                        (g * 255.0) as u8,
                        (b * 255.0) as u8,
                    ));
                    ui.collapsing(label, |ui| {
                        ui.label(format!("Cost: {:.1}", result.cost));
                        ui.label(format!(
                            "Closest: {:.0} km, {:.3} km/s",
                            result.closest_km, result.arrival_v,
                        ));
                        if !running && ui.button("Add to sim").clicked() {
                            to_adopt.push(i);
                        }
                    });
                }
            });
        opt_state.adopt_requests.extend(to_adopt);
    }
}

/// System that processes adopt requests from the pole optimizer.
pub fn adopt_pole_optimizer_results(
    mut opt_state: ResMut<PoleOptimizerState>,
    mut user_bodies: ResMut<UserBodies>,
    space: Res<SpaceResources>,
    time: Res<Time>,
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    if opt_state.adopt_requests.is_empty() {
        return;
    }

    let requests: Vec<usize> = opt_state.adopt_requests.drain(..).collect();
    let Some(epoch) = opt_state.opt_epoch else { return };

    let marker_mesh = meshes.add(Sphere::new(0.05).mesh().uv(16, 8));

    for result_idx in requests {
        let Some(result) = opt_state.best_results.get(result_idx) else { continue };

        let opt_color = POLE_COLORS[result_idx % POLE_COLORS.len()];
        let Some(&(_, pos, vel)) = result.trail.first() else { continue };

        let spacecraft = orbit::create_orbit_cartesian(
            pos, vel, epoch, &space.almanac, orbit::ReferenceFrame::EarthJ2000,
        )
        .ok()
        .map(|orb| {
            Spacecraft::builder()
                .orbit(orb)
                .mass(Mass::from_dry_mass(1.0))
                .build()
        });

        let idx = user_bodies.bodies.len();
        let name = format!(
            "Pole#{} ({:.1}\u{00b0} {:.2} km/s)",
            result_idx + 1, result.angle_deg, result.speed_km_s
        );

        user_bodies.bodies.push(UserBody {
            name,
            spacecraft,
            trail: vec![(epoch, pos, vel)],
            color: opt_color,
            spawn_real_time: time.elapsed_secs_f64(),
        });

        let body_mat = materials.add(StandardMaterial {
            base_color: opt_color,
            emissive: LinearRgba::from(opt_color) * 2.0,
            unlit: true,
            ..default()
        });
        commands.spawn((
            Mesh3d(marker_mesh.clone()),
            MeshMaterial3d(body_mat),
            Transform::from_xyz(0.0, 0.0, 0.0),
            UserBodyMarker { index: idx },
        ));
    }
}
