//! Trajectory optimization module.
//!
//! Finds launch parameters (angle, speed) on the lunar equator such that
//! two payloads launched with opposite tangential velocities both arrive
//! at EML1 with minimal velocity.
//!
//! Runs optimization in a background thread, communicating results to the
//! Bevy main thread via channels.

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
    State,
};
use rayon::prelude::*;

use crate::{
    bodies,
    constants::*,
    lagrange,
    orbit,
    visualization::{
        SimulationState, SpaceResources, TrailFrame,
        UserBodies, UserBody, UserBodyMarker,
    },
};

// === Reusable GUI helpers ===

/// Draw a min/max range control with two DragValues.
pub(crate) fn range_control(
    ui: &mut egui::Ui,
    label: &str,
    min: &mut f64,
    max: &mut f64,
    speed: f64,
    suffix: &str,
) {
    ui.label(label);
    ui.horizontal(|ui| {
        ui.add(
            egui::DragValue::new(min)
                .speed(speed)
                .prefix("min: ")
                .suffix(suffix),
        );
        ui.add(
            egui::DragValue::new(max)
                .speed(speed)
                .prefix("max: ")
                .suffix(suffix),
        );
    });
}

/// Draw weight controls for the end condition (distance + velocity weights).
pub(crate) fn end_condition_ui(
    ui: &mut egui::Ui,
    distance_weight: &mut f64,
    velocity_weight: &mut f64,
) {
    ui.horizontal(|ui| {
        ui.label("Dist weight:");
        ui.add(egui::DragValue::new(distance_weight).speed(0.1).range(0.0..=1e6));
    });
    ui.horizontal(|ui| {
        ui.label("Vel weight:");
        ui.add(egui::DragValue::new(velocity_weight).speed(100.0).range(0.0..=1e6));
    });
}

// === Data Structures ===

/// User-editable optimizer configuration.
#[derive(Clone, Debug)]
pub struct OptimizerConfig {
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

impl Default for OptimizerConfig {
    fn default() -> Self {
        Self {
            speed_min: 2.3,
            speed_max: 2.5,
            angle_min: -60.0,
            angle_max: 60.0,
            max_prop_days: 7.0,
            distance_weight: 1.0,
            velocity_weight: 0.0,
            grid_n_speed: 10,
            grid_n_angle: 36,
            nm_max_iter: 100,
            target: lagrange::LagrangeId::L1,
        }
    }
}

/// Result of evaluating a single (angle, speed) parameter pair.
#[derive(Clone, Debug)]
pub struct EvalResult {
    pub angle_deg: f64,
    pub speed_km_s: f64,
    pub cost: f64,
    pub trail_a: Vec<(Epoch, [f64; 3], [f64; 3])>,
    pub trail_b: Vec<(Epoch, [f64; 3], [f64; 3])>,
    pub closest_a_km: f64,
    pub arrival_v_a: f64,
    pub closest_b_km: f64,
    pub arrival_v_b: f64,
}

/// Messages from optimizer thread to main thread.
pub enum OptimizerMsg {
    /// Current evaluation just completed — show it briefly.
    Current(EvalResult),
    /// New best result found (rank 0 = best).
    NewBest { rank: usize, result: EvalResult },
    /// Progress update.
    Progress { phase: String, done: usize, total: usize },
    /// Optimization finished.
    Finished { best: EvalResult },
}

/// Colors for optimizer trails (distinct from user body colors).
pub(crate) const OPTIMIZER_COLORS: [Color; 3] = [
    Color::srgb(1.0, 0.0, 1.0),   // magenta
    Color::srgb(1.0, 1.0, 0.0),   // yellow
    Color::srgb(0.0, 1.0, 1.0),   // cyan
];

/// Precomputed Bevy-space line strip for a single trail.
pub(crate) type BevyStrip = Vec<Vec3>;

/// Bevy resource holding optimizer state.
#[derive(Resource)]
pub struct OptimizerState {
    pub config: OptimizerConfig,
    pub running: bool,
    pub cancel: Arc<AtomicBool>,
    receiver: Option<Mutex<mpsc::Receiver<OptimizerMsg>>>,
    thread_handle: Option<Mutex<JoinHandle<()>>>,
    pub best_results: Vec<EvalResult>,
    /// Precomputed Bevy-space strips for best results: (color, strip_a, strip_b).
    pub display_strips: Vec<(Color, BevyStrip, BevyStrip)>,
    pub phase: String,
    pub progress_done: usize,
    pub progress_total: usize,
    pub final_result: Option<EvalResult>,
    /// Precomputed Bevy-space strips for the current evaluation.
    pub current_strips: Option<(BevyStrip, BevyStrip)>,
    /// The epoch at which the optimizer was started (None if never run).
    pub opt_epoch: Option<Epoch>,
    /// Trail frame used when strips were last computed (recompute if changed).
    trail_frame_cached: TrailFrame,
    /// Pending requests to adopt optimizer results into the live simulation.
    /// Each entry is a result index; both payloads (A and B) are added.
    pub adopt_requests: Vec<usize>,
}

impl Default for OptimizerState {
    fn default() -> Self {
        Self {
            config: OptimizerConfig::default(),
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
            current_strips: None,
            opt_epoch: None,
            trail_frame_cached: TrailFrame::default(),
            adopt_requests: Vec::new(),
        }
    }
}

// === Core Functions ===

/// Build the Moon-centered equatorial frame axes in J2000.
/// Returns (away, north, prograde) unit vectors.
pub(crate) fn moon_frame(
    moon_pos: &bodies::J2000Position,
    moon_vel: &bodies::J2000Velocity,
) -> ([f64; 3], [f64; 3], [f64; 3]) {
    let moon_mag = moon_pos.magnitude();
    let away = [moon_pos.x / moon_mag, moon_pos.y / moon_mag, moon_pos.z / moon_mag];

    let hx = moon_pos.y * moon_vel.z - moon_pos.z * moon_vel.y;
    let hy = moon_pos.z * moon_vel.x - moon_pos.x * moon_vel.z;
    let hz = moon_pos.x * moon_vel.y - moon_pos.y * moon_vel.x;
    let h_mag = (hx * hx + hy * hy + hz * hz).sqrt();
    let north = [hx / h_mag, hy / h_mag, hz / h_mag];

    let prograde = [
        north[1] * away[2] - north[2] * away[1],
        north[2] * away[0] - north[0] * away[2],
        north[0] * away[1] - north[1] * away[0],
    ];

    (away, north, prograde)
}

/// Create a pair of spacecraft launched from the lunar equator at `angle_deg`
/// with opposite tangential velocity vectors of magnitude `speed_km_s`.
fn create_opposing_pair(
    angle_deg: f64,
    speed_km_s: f64,
    epoch: Epoch,
    almanac: &Almanac,
) -> Option<(Spacecraft, Spacecraft)> {
    let moon_pos = bodies::moon_position(almanac, epoch).ok()?;
    let moon_vel = bodies::moon_velocity(almanac, epoch).ok()?;
    let (away, north, prograde) = moon_frame(&moon_pos, &moon_vel);

    // Surface normal at equator at angle_deg azimuth
    let az = angle_deg.to_radians();
    let normal = [
        az.cos() * away[0] + az.sin() * prograde[0],
        az.cos() * away[1] + az.sin() * prograde[1],
        az.cos() * away[2] + az.sin() * prograde[2],
    ];

    // Position on Moon surface
    let pos = [
        moon_pos.x + normal[0] * MOON_RADIUS_KM,
        moon_pos.y + normal[1] * MOON_RADIUS_KM,
        moon_pos.z + normal[2] * MOON_RADIUS_KM,
    ];

    // Tangent direction in equatorial plane: north × normal
    let tangent = [
        north[1] * normal[2] - north[2] * normal[1],
        north[2] * normal[0] - north[0] * normal[2],
        north[0] * normal[1] - north[1] * normal[0],
    ];

    let base_vel = [moon_vel.x, moon_vel.y, moon_vel.z];

    let vel_a = [
        base_vel[0] + speed_km_s * tangent[0],
        base_vel[1] + speed_km_s * tangent[1],
        base_vel[2] + speed_km_s * tangent[2],
    ];
    let vel_b = [
        base_vel[0] - speed_km_s * tangent[0],
        base_vel[1] - speed_km_s * tangent[1],
        base_vel[2] - speed_km_s * tangent[2],
    ];

    let make_sc = |vel: [f64; 3]| -> Option<Spacecraft> {
        let orb = orbit::create_orbit_cartesian(
            pos, vel, epoch, almanac, orbit::ReferenceFrame::EarthJ2000,
        ).ok()?;
        Some(Spacecraft::builder()
            .orbit(orb)
            .mass(Mass::from_dry_mass(1.0))
            .build())
    };

    Some((make_sc(vel_a)?, make_sc(vel_b)?))
}

/// Propagate a spacecraft and return a trail in the standard format.
/// Samples at `sample_step_s` intervals.
pub(crate) fn propagate_to_trail(
    sc: Spacecraft,
    propagator: &Propagator<SpacecraftDynamics>,
    almanac: Arc<Almanac>,
    duration: Duration,
    sample_step_s: f64,
) -> Option<Vec<(Epoch, [f64; 3], [f64; 3])>> {
    let start_epoch = sc.epoch();
    let end_epoch = start_epoch + duration;

    let (_, traj) = propagator
        .with(sc, almanac)
        .until_epoch_with_traj(end_epoch)
        .ok()?;

    let total_s = duration.to_seconds();
    let n_samples = (total_s / sample_step_s) as usize;
    let mut trail = Vec::with_capacity(n_samples + 1);

    for i in 0..=n_samples {
        let t = start_epoch + Duration::from_seconds(i as f64 * sample_step_s);
        if let Ok(state) = traj.at(t) {
            let pos = state.orbit.radius_km;
            let vel = state.orbit.velocity_km_s;
            trail.push((t, [pos.x, pos.y, pos.z], [vel.x, vel.y, vel.z]));
        }
    }

    Some(trail)
}


/// Find the closest approach of a trail to any Lagrange point.
/// Returns (distance_km, relative_velocity_km_s) at the point of closest approach.
/// The velocity is relative to the Lagrange point (computed via finite difference).
pub(crate) fn closest_approach_to_lagrange(
    trail: &[(Epoch, [f64; 3], [f64; 3])],
    almanac: &Almanac,
    target: lagrange::LagrangeId,
) -> Option<(f64, f64)> {
    let mut best_dist = f64::MAX;
    let mut best_rel_vel = 0.0;

    let dt = hifitime::Duration::from_seconds(1.0);

    for &(epoch, pos, vel) in trail {
        let Ok(lp) = lagrange::lagrange_position(target, almanac, epoch) else {
            continue;
        };
        let dx = pos[0] - lp.x;
        let dy = pos[1] - lp.y;
        let dz = pos[2] - lp.z;
        let dist = (dx * dx + dy * dy + dz * dz).sqrt();
        if dist < best_dist {
            best_dist = dist;
            // Compute LP velocity via finite difference
            if let Ok(lp2) = lagrange::lagrange_position(target, almanac, epoch + dt) {
                let lp_vx = lp2.x - lp.x; // km/s (dt = 1s)
                let lp_vy = lp2.y - lp.y;
                let lp_vz = lp2.z - lp.z;
                let rvx = vel[0] - lp_vx;
                let rvy = vel[1] - lp_vy;
                let rvz = vel[2] - lp_vz;
                best_rel_vel = (rvx * rvx + rvy * rvy + rvz * rvz).sqrt();
            } else {
                best_rel_vel = (vel[0] * vel[0] + vel[1] * vel[1] + vel[2] * vel[2]).sqrt();
            }
        }
    }

    if best_dist < f64::MAX {
        Some((best_dist, best_rel_vel))
    } else {
        None
    }
}

/// Evaluate a single (angle, speed) parameter pair.
/// Propagates both opposing payloads and computes cost.
fn evaluate(
    angle_deg: f64,
    speed_km_s: f64,
    epoch: Epoch,
    almanac: &Arc<Almanac>,
    propagator: &Propagator<SpacecraftDynamics>,
    config: &OptimizerConfig,
) -> Option<EvalResult> {
    let (sc_a, sc_b) = create_opposing_pair(angle_deg, speed_km_s, epoch, almanac)?;

    let duration = Duration::from_seconds(config.max_prop_days * 86400.0);
    let sample_step = 60.0; // 1 sample per minute

    let trail_a = propagate_to_trail(sc_a, propagator, almanac.clone(), duration, sample_step)?;
    let trail_b = propagate_to_trail(sc_b, propagator, almanac.clone(), duration, sample_step)?;

    let (closest_a_km, arrival_v_a) = closest_approach_to_lagrange(&trail_a, almanac, config.target)?;
    let (closest_b_km, arrival_v_b) = closest_approach_to_lagrange(&trail_b, almanac, config.target)?;

    let cost = config.distance_weight * (closest_a_km.powi(2) + closest_b_km.powi(2))
        + config.velocity_weight * (arrival_v_a.powi(2) + arrival_v_b.powi(2));

    Some(EvalResult {
        angle_deg,
        speed_km_s,
        cost,
        trail_a,
        trail_b,
        closest_a_km,
        arrival_v_a,
        closest_b_km,
        arrival_v_b,
    })
}

// === Optimization Algorithms ===

/// Uniform grid scan over the parameter space.
/// Uses rayon for parallel evaluation. Sends best results via channel.
fn grid_scan(
    config: &OptimizerConfig,
    epoch: Epoch,
    almanac: &Arc<Almanac>,
    propagator: &Propagator<SpacecraftDynamics>,
    sender: &mpsc::Sender<OptimizerMsg>,
    cancel: &AtomicBool,
) -> Vec<EvalResult> {
    let total = config.grid_n_speed * config.grid_n_angle;

    // Build parameter grid
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

    // Evaluate in parallel using rayon
    let done_count = std::sync::atomic::AtomicUsize::new(0);
    let results: Vec<EvalResult> = params
        .par_iter()
        .filter_map(|&(angle, speed)| {
            if cancel.load(Ordering::Relaxed) {
                return None;
            }
            let result = evaluate(angle, speed, epoch, almanac, propagator, config);
            let done = done_count.fetch_add(1, Ordering::Relaxed) + 1;
            if let Some(ref r) = result {
                let _ = sender.send(OptimizerMsg::Current(r.clone()));
            }
            let _ = sender.send(OptimizerMsg::Progress {
                phase: "Grid scan".into(),
                done,
                total,
            });
            result
        })
        .collect();

    // Sort by cost
    let mut results = results;
    results.sort_by(|a, b| a.cost.partial_cmp(&b.cost).unwrap_or(std::cmp::Ordering::Equal));

    // Send top-3 as NewBest
    for (rank, r) in results.iter().take(3).enumerate() {
        let _ = sender.send(OptimizerMsg::NewBest {
            rank,
            result: r.clone(),
        });
    }

    results
}

/// 2D Nelder-Mead optimizer operating on [angle_deg, speed_km_s].
/// Standard reflect/expand/contract/shrink.
fn nelder_mead_2d(
    initial_simplex: [[f64; 2]; 3],
    config: &OptimizerConfig,
    epoch: Epoch,
    almanac: &Arc<Almanac>,
    propagator: &Propagator<SpacecraftDynamics>,
    sender: &mpsc::Sender<OptimizerMsg>,
    cancel: &AtomicBool,
) -> Option<EvalResult> {
    let clamp = |p: [f64; 2]| -> [f64; 2] {
        [
            p[0].clamp(config.angle_min, config.angle_max),
            p[1].clamp(config.speed_min, config.speed_max),
        ]
    };

    let eval_point = |p: [f64; 2]| -> (f64, Option<EvalResult>) {
        let p = clamp(p);
        let result = evaluate(p[0], p[1], epoch, almanac, propagator, config);
        if let Some(ref r) = result {
            let _ = sender.send(OptimizerMsg::Current(r.clone()));
        }
        let cost = result.as_ref().map(|r| r.cost).unwrap_or(f64::INFINITY);
        (cost, result)
    };

    // Initialize simplex: (point, cost, result)
    let mut simplex: Vec<([f64; 2], f64, Option<EvalResult>)> = initial_simplex
        .iter()
        .map(|&p| {
            let (c, r) = eval_point(p);
            (p, c, r)
        })
        .collect();

    // Standard Nelder-Mead coefficients
    let alpha = 1.0; // reflection
    let gamma = 2.0; // expansion
    let rho = 0.5; // contraction
    let sigma = 0.5; // shrink

    for iter in 0..config.nm_max_iter {
        if cancel.load(Ordering::Relaxed) {
            break;
        }

        // Sort by cost (ascending)
        simplex.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        let _ = sender.send(OptimizerMsg::Progress {
            phase: "Nelder-Mead".into(),
            done: iter,
            total: config.nm_max_iter,
        });

        // Send current best
        if let Some(ref r) = simplex[0].2 {
            let _ = sender.send(OptimizerMsg::NewBest {
                rank: 0,
                result: r.clone(),
            });
        }

        let n = simplex.len() - 1; // = 2

        // Centroid of all points except worst
        let centroid = [
            simplex[..n].iter().map(|s| s.0[0]).sum::<f64>() / n as f64,
            simplex[..n].iter().map(|s| s.0[1]).sum::<f64>() / n as f64,
        ];

        let worst = simplex[n].0;

        // Reflection
        let reflected = [
            centroid[0] + alpha * (centroid[0] - worst[0]),
            centroid[1] + alpha * (centroid[1] - worst[1]),
        ];
        let (fr, rr) = eval_point(reflected);

        if fr < simplex[0].1 {
            // Try expansion
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
            // Contraction
            let contracted = [
                centroid[0] + rho * (worst[0] - centroid[0]),
                centroid[1] + rho * (worst[1] - centroid[1]),
            ];
            let (fc, rc) = eval_point(contracted);
            if fc < simplex[n].1 {
                simplex[n] = (contracted, fc, rc);
            } else {
                // Shrink toward best
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

        // Convergence check
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

/// Launch the optimizer on a background thread.
fn launch_optimizer(
    config: OptimizerConfig,
    epoch: Epoch,
    almanac: Arc<Almanac>,
) -> (mpsc::Receiver<OptimizerMsg>, JoinHandle<()>, Arc<AtomicBool>) {
    let (sender, receiver) = mpsc::channel();
    let cancel = Arc::new(AtomicBool::new(false));
    let cancel_clone = cancel.clone();

    let handle = std::thread::spawn(move || {
        // Recreate propagator on this thread (avoids Clone requirement)
        let propagator = match orbit::setup_propagator(&almanac) {
            Ok(p) => p,
            Err(_) => return,
        };

        // Phase 1: Grid scan
        let grid_results = grid_scan(&config, epoch, &almanac, &propagator, &sender, &cancel_clone);
        if cancel_clone.load(Ordering::Relaxed) || grid_results.is_empty() {
            return;
        }

        // Phase 2: Nelder-Mead refinement
        if grid_results.len() >= 3 {
            let initial = [
                [grid_results[0].angle_deg, grid_results[0].speed_km_s],
                [grid_results[1].angle_deg, grid_results[1].speed_km_s],
                [grid_results[2].angle_deg, grid_results[2].speed_km_s],
            ];
            if let Some(best) = nelder_mead_2d(
                initial, &config, epoch, &almanac, &propagator, &sender, &cancel_clone,
            ) {
                let _ = sender.send(OptimizerMsg::Finished { best });
                return;
            }
        }

        // Fallback: just return the best grid result
        if let Some(best) = grid_results.into_iter().next() {
            let _ = sender.send(OptimizerMsg::Finished { best });
        }
    });

    (receiver, handle, cancel)
}

// === Bevy Systems ===

/// Precompute a J2000 trail into a Bevy-space line strip.
pub(crate) fn precompute_strip(
    trail: &[(Epoch, [f64; 3], [f64; 3])],
    frame: TrailFrame,
    almanac: &Almanac,
    moon_now: [f64; 3],
) -> BevyStrip {
    trail.iter().map(|(epoch, j2000, _vel)| {
        opt_trail_to_bevy(*j2000, *epoch, frame, almanac, moon_now)
    }).collect()
}

/// Recompute all display strips (called when trail_frame changes or new results arrive).
fn recompute_all_strips(opt_state: &mut OptimizerState, almanac: &Almanac, moon_now: [f64; 3]) {
    let frame = opt_state.trail_frame_cached;
    opt_state.display_strips.clear();
    for (i, result) in opt_state.best_results.iter().enumerate() {
        let color = OPTIMIZER_COLORS[i % OPTIMIZER_COLORS.len()];
        let strip_a = precompute_strip(&result.trail_a, frame, almanac, moon_now);
        let strip_b = precompute_strip(&result.trail_b, frame, almanac, moon_now);
        opt_state.display_strips.push((color, strip_a, strip_b));
    }
}

/// Poll the optimizer channel each frame and update state.
pub fn poll_optimizer(
    mut opt_state: ResMut<OptimizerState>,
    sim_state: Res<SimulationState>,
    space: Res<SpaceResources>,
) {
    let frame = sim_state.trail_frame;
    let moon_now = bodies::moon_position(&space.almanac, sim_state.epoch)
        .map(|m| [m.x, m.y, m.z])
        .unwrap_or([0.0; 3]);

    // If trail frame changed, recompute all cached strips
    if frame != opt_state.trail_frame_cached && !opt_state.best_results.is_empty() {
        opt_state.trail_frame_cached = frame;
        recompute_all_strips(&mut opt_state, &space.almanac, moon_now);
    }

    // Collect messages from the channel (hold lock briefly)
    let messages: Vec<OptimizerMsg> = {
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

    // Process collected messages
    let mut finished = false;
    for msg in messages {
        match msg {
            OptimizerMsg::Current(result) => {
                let strip_a = precompute_strip(&result.trail_a, frame, &space.almanac, moon_now);
                let strip_b = precompute_strip(&result.trail_b, frame, &space.almanac, moon_now);
                opt_state.current_strips = Some((strip_a, strip_b));
            }
            OptimizerMsg::NewBest { rank, result } => {
                let color = OPTIMIZER_COLORS[rank % OPTIMIZER_COLORS.len()];
                let strip_a = precompute_strip(&result.trail_a, frame, &space.almanac, moon_now);
                let strip_b = precompute_strip(&result.trail_b, frame, &space.almanac, moon_now);

                if rank < opt_state.best_results.len() {
                    opt_state.best_results[rank] = result;
                    opt_state.display_strips[rank] = (color, strip_a, strip_b);
                } else {
                    opt_state.best_results.push(result);
                    opt_state.display_strips.push((color, strip_a, strip_b));
                }
                opt_state.best_results.truncate(3);
                opt_state.display_strips.truncate(3);
            }
            OptimizerMsg::Progress { phase, done, total } => {
                opt_state.phase = phase;
                opt_state.progress_done = done;
                opt_state.progress_total = total;
            }
            OptimizerMsg::Finished { best } => {
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
        opt_state.current_strips = None;
        // Keep sim paused so user can adopt results at the optimization epoch.
    }
}

/// Transform an optimizer trail point to Bevy coordinates.
/// Unlike `trail_transform_point`, this computes Moon positions from the almanac
/// so it works for future epochs (where body_trails has no data).
pub(crate) fn opt_trail_to_bevy(
    j2000: [f64; 3],
    epoch: Epoch,
    frame: TrailFrame,
    almanac: &Almanac,
    moon_now: [f64; 3],
) -> Vec3 {
    match frame {
        TrailFrame::J2000 => {
            let p = bodies::J2000Position { x: j2000[0], y: j2000[1], z: j2000[2] };
            Vec3::from_array(p.to_bevy(VIS_SCALE_F64))
        }
        TrailFrame::MoonCentered => {
            let moon_t = bodies::moon_position(almanac, epoch)
                .map(|m| [m.x, m.y, m.z])
                .unwrap_or(moon_now);
            let rel = [
                j2000[0] - moon_t[0] + moon_now[0],
                j2000[1] - moon_t[1] + moon_now[1],
                j2000[2] - moon_t[2] + moon_now[2],
            ];
            let p = bodies::J2000Position { x: rel[0], y: rel[1], z: rel[2] };
            Vec3::from_array(p.to_bevy(VIS_SCALE_F64))
        }
        TrailFrame::Synodic | TrailFrame::Pulsating => {
            let moon_t = bodies::moon_position(almanac, epoch)
                .map(|m| [m.x, m.y, m.z])
                .unwrap_or(moon_now);

            let r_t = (moon_t[0] * moon_t[0] + moon_t[1] * moon_t[1] + moon_t[2] * moon_t[2]).sqrt();
            let r_now = (moon_now[0] * moon_now[0] + moon_now[1] * moon_now[1] + moon_now[2] * moon_now[2]).sqrt();

            if r_t < 1.0 || r_now < 1.0 {
                let p = bodies::J2000Position { x: j2000[0], y: j2000[1], z: j2000[2] };
                return Vec3::from_array(p.to_bevy(VIS_SCALE_F64));
            }

            let dir_t = [moon_t[0] / r_t, moon_t[1] / r_t, moon_t[2] / r_t];
            let dir_now = [moon_now[0] / r_now, moon_now[1] / r_now, moon_now[2] / r_now];

            // Rotation axis = dir_t × dir_now
            let cx = dir_t[1] * dir_now[2] - dir_t[2] * dir_now[1];
            let cy = dir_t[2] * dir_now[0] - dir_t[0] * dir_now[2];
            let cz = dir_t[0] * dir_now[1] - dir_t[1] * dir_now[0];
            let sin_a = (cx * cx + cy * cy + cz * cz).sqrt();
            let cos_a = dir_t[0] * dir_now[0] + dir_t[1] * dir_now[1] + dir_t[2] * dir_now[2];

            if sin_a < 1e-12 {
                let p = bodies::J2000Position { x: j2000[0], y: j2000[1], z: j2000[2] };
                return Vec3::from_array(p.to_bevy(VIS_SCALE_F64));
            }

            // Rodrigues' rotation
            let k = [cx / sin_a, cy / sin_a, cz / sin_a];
            let dot_kv = k[0] * j2000[0] + k[1] * j2000[1] + k[2] * j2000[2];
            let kxv = [
                k[1] * j2000[2] - k[2] * j2000[1],
                k[2] * j2000[0] - k[0] * j2000[2],
                k[0] * j2000[1] - k[1] * j2000[0],
            ];

            let mut rot = [
                j2000[0] * cos_a + kxv[0] * sin_a + k[0] * dot_kv * (1.0 - cos_a),
                j2000[1] * cos_a + kxv[1] * sin_a + k[1] * dot_kv * (1.0 - cos_a),
                j2000[2] * cos_a + kxv[2] * sin_a + k[2] * dot_kv * (1.0 - cos_a),
            ];

            // Pulsating: additionally scale by R(now)/R(t)
            if matches!(frame, TrailFrame::Pulsating) {
                let scale = r_now / r_t;
                rot[0] *= scale;
                rot[1] *= scale;
                rot[2] *= scale;
            }

            let p = bodies::J2000Position { x: rot[0], y: rot[1], z: rot[2] };
            Vec3::from_array(p.to_bevy(VIS_SCALE_F64))
        }
    }
}

/// Draw optimizer trails with gizmos from precomputed Bevy-space strips.
pub fn draw_optimizer_trails(
    opt_state: Res<OptimizerState>,
    mut gizmos: Gizmos,
) {
    let has_current = opt_state.current_strips.is_some();
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

    // Draw current evaluation in dim gray
    if let Some((ref strip_a, ref strip_b)) = opt_state.current_strips {
        let dim = Color::srgba(0.4, 0.4, 0.4, 0.6);
        draw_strip(&mut gizmos, strip_a, dim);
        draw_strip(&mut gizmos, strip_b, dim);
    }

    // Draw best results in bright colors
    for (color, strip_a, strip_b) in &opt_state.display_strips {
        draw_strip(&mut gizmos, strip_a, *color);
        draw_strip(&mut gizmos, strip_b, *color);
    }
}

/// Draw equator optimizer UI content into the given `Ui`.
pub(crate) fn optimizer_ui_content(
    ui: &mut egui::Ui,
    opt_state: &mut OptimizerState,
    sim_state: &mut SimulationState,
    space: &SpaceResources,
) {
    let config = &mut opt_state.config;

    egui::CollapsingHeader::new("Parameters")
        .default_open(false)
        .show(ui, |ui| {
            range_control(ui, "Launch speed:", &mut config.speed_min, &mut config.speed_max, 0.01, " km/s");
            range_control(ui, "Launch angle:", &mut config.angle_min, &mut config.angle_max, 1.0, " deg");
            ui.horizontal(|ui| {
                ui.label("Max propagation:");
                ui.add(
                    egui::DragValue::new(&mut config.max_prop_days)
                        .range(1.0..=30.0)
                        .speed(0.5)
                        .suffix(" days"),
                );
            });
        });

    egui::CollapsingHeader::new("Cost Function")
        .default_open(false)
        .show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.label("Target:");
                egui::ComboBox::from_id_salt("eq_target")
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

        let (rx, handle, cancel) = launch_optimizer(
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
                    let color = OPTIMIZER_COLORS[i % OPTIMIZER_COLORS.len()];
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
                            "A: {:.0} km, {:.3} km/s",
                            result.closest_a_km, result.arrival_v_a
                        ));
                        ui.label(format!(
                            "B: {:.0} km, {:.3} km/s",
                            result.closest_b_km, result.arrival_v_b
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

/// System that processes adopt requests: creates real UserBody entries from optimizer results.
pub fn adopt_optimizer_results(
    mut opt_state: ResMut<OptimizerState>,
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

        // Use the optimizer trail color so adopted bodies match their preview
        let opt_color = OPTIMIZER_COLORS[result_idx % OPTIMIZER_COLORS.len()];

        // Create spacecraft from the first trail point (initial state in J2000)
        for (suffix, trail) in [("A", &result.trail_a), ("B", &result.trail_b)] {
            let Some(&(_, pos, vel)) = trail.first() else { continue };

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
                "Opt#{} {suffix} ({:.1}° {:.2} km/s)",
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
}
