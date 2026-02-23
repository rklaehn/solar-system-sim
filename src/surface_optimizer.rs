//! Single-trajectory optimizer: Moon surface → Lagrange point.
//!
//! Core function: `trajectory()` — given a launch site (longitude, latitude on
//! the Moon) and search ranges for velocity & angle, find the single best
//! trajectory to a target Lagrange point.
//!
//! Designed to be called many times in parallel (no internal parallelism).
//! Uses multi-resolution search: coarse grid → fine grid → golden-section.

use std::sync::Arc;

use anise::prelude::Almanac;
use hifitime::{Duration, Epoch};
use nyx_space::{
    cosmic::{Mass, Spacecraft},
    dynamics::SpacecraftDynamics,
    propagators::{ErrorControl, IntegratorMethod, IntegratorOptions, Propagator},
};
use nyx_space::State;

use crate::{
    bodies,
    constants::*,
    lagrange,
    optimizer::moon_frame,
    orbit,
};

/// Result of a single trajectory optimization.
#[derive(Clone, Debug)]
pub struct TrajectoryResult {
    pub velocity: f64,
    pub angle: f64,
    pub closest_km: f64,
    pub time_to_closest_s: f64,
    pub trail: Vec<(Epoch, [f64; 3], [f64; 3])>,
}

/// Create a spacecraft on the Moon surface at the given selenographic longitude/latitude,
/// with a launch velocity in the tangent plane at `launch_angle_deg`
/// relative to the "away from Moon center" projected direction.
///
/// Standard selenographic coordinates:
/// longitude_deg: 0 = sub-Earth point, 90°E = prograde (east/leading limb), 180 = anti-Earth (far side)
/// latitude_deg: 0 = equator, +90 = north pole
/// launch_angle_deg: direction in the tangent plane (0 = east, 90 = north-ish)
fn create_surface_spacecraft(
    longitude_deg: f64,
    latitude_deg: f64,
    launch_angle_deg: f64,
    speed_km_s: f64,
    epoch: Epoch,
    almanac: &Almanac,
) -> Option<Spacecraft> {
    let moon_pos = bodies::moon_position(almanac, epoch).ok()?;
    let moon_vel = bodies::moon_velocity(almanac, epoch).ok()?;
    let (away, north, prograde) = moon_frame(&moon_pos, &moon_vel);

    // Standard selenographic: lon=0° toward Earth, lon=90°E prograde, lat=0° equator
    let toward = [-away[0], -away[1], -away[2]];

    let lon = longitude_deg.to_radians();
    let lat = latitude_deg.to_radians();

    // Surface normal in J2000 (spherical coordinates in selenographic frame)
    // toward = "x" (toward Earth), prograde = "y" (east), north = "z" (north)
    let normal = [
        lat.cos() * lon.cos() * toward[0] + lat.cos() * lon.sin() * prograde[0] + lat.sin() * north[0],
        lat.cos() * lon.cos() * toward[1] + lat.cos() * lon.sin() * prograde[1] + lat.sin() * north[1],
        lat.cos() * lon.cos() * toward[2] + lat.cos() * lon.sin() * prograde[2] + lat.sin() * north[2],
    ];

    // Position on Moon surface
    let pos = [
        moon_pos.x + normal[0] * MOON_RADIUS_KM,
        moon_pos.y + normal[1] * MOON_RADIUS_KM,
        moon_pos.z + normal[2] * MOON_RADIUS_KM,
    ];

    // Build tangent plane basis at this surface point.
    // east_dir: derivative of normal w.r.t. longitude (normalized)
    // d(normal)/d(lon) is tangent to the surface pointing east (increasing longitude).
    let east_unnorm = [
        lat.cos() * (-lon.sin()) * toward[0] + lat.cos() * lon.cos() * prograde[0],
        lat.cos() * (-lon.sin()) * toward[1] + lat.cos() * lon.cos() * prograde[1],
        lat.cos() * (-lon.sin()) * toward[2] + lat.cos() * lon.cos() * prograde[2],
    ];
    let east_mag = (east_unnorm[0].powi(2) + east_unnorm[1].powi(2) + east_unnorm[2].powi(2)).sqrt();
    // At the poles east is degenerate; fallback to prograde
    let east = if east_mag > 1e-10 {
        [east_unnorm[0] / east_mag, east_unnorm[1] / east_mag, east_unnorm[2] / east_mag]
    } else {
        prograde
    };

    // north_tangent = normal × east (points roughly toward north pole in tangent plane)
    let north_t = [
        normal[1] * east[2] - normal[2] * east[1],
        normal[2] * east[0] - normal[0] * east[2],
        normal[0] * east[1] - normal[1] * east[0],
    ];

    // Launch direction in tangent plane
    let la = launch_angle_deg.to_radians();
    let launch_dir = [
        la.cos() * east[0] + la.sin() * north_t[0],
        la.cos() * east[1] + la.sin() * north_t[1],
        la.cos() * east[2] + la.sin() * north_t[2],
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

/// Set up a fast propagator with looser tolerances for coarse sweeps.
fn fast_propagator(almanac: &Almanac, tolerance: f64, max_step_s: f64) -> Option<Propagator<SpacecraftDynamics>> {
    use anise::constants::{
        celestial_objects::{MOON, SUN},
        frames::EARTH_J2000,
    };
    use nyx_space::dynamics::orbital::{Harmonics, OrbitalDynamics, PointMasses};
    use nyx_space::io::gravity::HarmonicsMem;

    let earth_frame = almanac.frame_from_uid(EARTH_J2000).ok()?;
    let earth_sph_harm = HarmonicsMem::from_j2(EARTH_J2);
    let harmonics = Harmonics::from_stor(earth_frame, earth_sph_harm);
    let point_masses = PointMasses::new(vec![MOON, SUN]);
    let orbital_dyn = OrbitalDynamics::new(vec![harmonics, point_masses]);
    let dynamics = SpacecraftDynamics::new(orbital_dyn);

    let opts = IntegratorOptions::with_adaptive_step(
        Duration::from_seconds(0.1),
        Duration::from_seconds(max_step_s),
        tolerance,
        ErrorControl::RSSCartesianStep,
    );

    Some(Propagator::new(
        dynamics,
        IntegratorMethod::DormandPrince78,
        opts,
    ))
}

/// Precomputed Lagrange point positions at regular time steps.
/// Avoids repeated almanac lookups during grid search.
struct LpCache {
    positions: Vec<[f64; 3]>,
    sample_step_s: f64,
}

impl LpCache {
    fn new(
        epoch: Epoch,
        duration: Duration,
        sample_step_s: f64,
        target: lagrange::LagrangeId,
        almanac: &Almanac,
    ) -> Option<Self> {
        let n = (duration.to_seconds() / sample_step_s) as usize + 1;
        let mut positions = Vec::with_capacity(n);
        for i in 0..n {
            let t = epoch + Duration::from_seconds(i as f64 * sample_step_s);
            let lp = lagrange::lagrange_position(target, almanac, t).ok()?;
            positions.push([lp.x, lp.y, lp.z]);
        }
        Some(Self { positions, sample_step_s })
    }

    /// Get the closest LP position for a given time offset (linearly interpolated).
    fn at(&self, t_s: f64) -> [f64; 3] {
        let idx_f = t_s / self.sample_step_s;
        let idx = idx_f as usize;
        if idx + 1 >= self.positions.len() {
            return *self.positions.last().unwrap_or(&[0.0; 3]);
        }
        let frac = idx_f - idx as f64;
        let a = &self.positions[idx];
        let b = &self.positions[idx + 1];
        [
            a[0] + frac * (b[0] - a[0]),
            a[1] + frac * (b[1] - a[1]),
            a[2] + frac * (b[2] - a[2]),
        ]
    }
}

/// Propagate and find closest approach to a Lagrange point using precomputed LP cache.
/// Returns (closest_distance_km, time_to_closest_s) or None on failure.
fn propagate_closest(
    sc: Spacecraft,
    propagator: &Propagator<SpacecraftDynamics>,
    almanac: &Arc<Almanac>,
    duration: Duration,
    sample_step_s: f64,
    lp_cache: &LpCache,
) -> Option<(f64, f64)> {
    let start_epoch = sc.epoch();
    let end_epoch = start_epoch + duration;

    let (_, traj) = propagator
        .with(sc.clone(), almanac.clone())
        .until_epoch_with_traj(end_epoch)
        .ok()?;

    let total_s = duration.to_seconds();
    let n_samples = (total_s / sample_step_s) as usize;

    let mut best_dist = f64::MAX;
    let mut best_time = 0.0;

    for i in 0..=n_samples {
        let t_s = i as f64 * sample_step_s;
        let t = start_epoch + Duration::from_seconds(t_s);
        if let Ok(state) = traj.at(t) {
            let pos = state.orbit.radius_km;
            let lp = lp_cache.at(t_s);
            let dx = pos.x - lp[0];
            let dy = pos.y - lp[1];
            let dz = pos.z - lp[2];
            let dist = (dx * dx + dy * dy + dz * dz).sqrt();
            if dist < best_dist {
                best_dist = dist;
                best_time = t_s;
            }
        }
    }

    if best_dist < f64::MAX {
        Some((best_dist, best_time))
    } else {
        None
    }
}

/// Propagate and return the full trail + closest approach info.
fn propagate_full(
    sc: Spacecraft,
    propagator: &Propagator<SpacecraftDynamics>,
    almanac: &Arc<Almanac>,
    duration: Duration,
    sample_step_s: f64,
    target: lagrange::LagrangeId,
) -> Option<(f64, f64, Vec<(Epoch, [f64; 3], [f64; 3])>)> {
    let start_epoch = sc.epoch();
    let end_epoch = start_epoch + duration;

    let (_, traj) = propagator
        .with(sc.clone(), almanac.clone())
        .until_epoch_with_traj(end_epoch)
        .ok()?;

    let total_s = duration.to_seconds();
    let n_samples = (total_s / sample_step_s) as usize;

    let mut best_dist = f64::MAX;
    let mut best_time = 0.0;
    let mut trail = Vec::with_capacity(n_samples + 1);

    for i in 0..=n_samples {
        let t_s = i as f64 * sample_step_s;
        let t = start_epoch + Duration::from_seconds(t_s);
        if let Ok(state) = traj.at(t) {
            let pos = state.orbit.radius_km;
            let vel = state.orbit.velocity_km_s;
            trail.push((t, [pos.x, pos.y, pos.z], [vel.x, vel.y, vel.z]));
            if let Ok(lp) = lagrange::lagrange_position(target, almanac, t) {
                let dx = pos.x - lp.x;
                let dy = pos.y - lp.y;
                let dz = pos.z - lp.z;
                let dist = (dx * dx + dy * dy + dz * dz).sqrt();
                if dist < best_dist {
                    best_dist = dist;
                    best_time = t_s;
                }
            }
        }
    }

    if best_dist < f64::MAX {
        Some((best_dist, best_time, trail))
    } else {
        None
    }
}

/// Options for the coarse grid search phase.
#[derive(Clone, Debug)]
pub struct GridSearchOptions {
    pub n_vel: usize,
    pub n_ang: usize,
    pub tolerance: f64,
    pub max_step_s: f64,
    pub sample_step_s: f64,
}

impl Default for GridSearchOptions {
    fn default() -> Self {
        Self {
            n_vel: 10,
            n_ang: 10,
            tolerance: 1e-4,
            max_step_s: 2000.0,
            sample_step_s: 3600.0,
        }
    }
}

/// Options for the Nelder-Mead refinement phase.
#[derive(Clone, Debug)]
pub struct NelderMeadOptions {
    pub max_iter: usize,
    pub tolerance: f64,
    pub max_step_s: f64,
    pub sample_step_s: f64,
    pub converge_angle_deg: f64,
    pub converge_speed_km_s: f64,
}

impl Default for NelderMeadOptions {
    fn default() -> Self {
        Self {
            max_iter: 100,
            tolerance: 1e-9,
            max_step_s: 300.0,
            sample_step_s: 300.0,
            converge_angle_deg: 0.01,
            converge_speed_km_s: 0.0001,
        }
    }
}

/// Combined options for trajectory optimization.
#[derive(Clone, Debug)]
pub struct TrajectoryOptions {
    pub velocity_range: (f64, f64),
    pub angle_range: (f64, f64),
    pub integration_time_days: f64,
    pub l_point: lagrange::LagrangeId,
    pub grid: GridSearchOptions,
    pub nm: NelderMeadOptions,
}

impl Default for TrajectoryOptions {
    fn default() -> Self {
        Self {
            velocity_range: (2.3, 2.5),
            angle_range: (-30.0, 30.0),
            integration_time_days: 10.0,
            l_point: lagrange::LagrangeId::L1,
            grid: GridSearchOptions::default(),
            nm: NelderMeadOptions::default(),
        }
    }
}

/// Shared evaluation context for grid search and Nelder-Mead.
struct Evaluator<'a> {
    longitude_deg: f64,
    latitude_deg: f64,
    epoch: Epoch,
    almanac: &'a Arc<Almanac>,
    propagator: &'a Propagator<SpacecraftDynamics>,
    duration: Duration,
    sample_step_s: f64,
    lp_cache: &'a LpCache,
    velocity_range: (f64, f64),
    angle_range: (f64, f64),
}

impl Evaluator<'_> {
    fn eval(&self, angle_deg: f64, speed: f64) -> f64 {
        let a = angle_deg.clamp(self.angle_range.0, self.angle_range.1);
        let v = speed.clamp(self.velocity_range.0, self.velocity_range.1);
        let sc = create_surface_spacecraft(
            self.longitude_deg, self.latitude_deg, a, v, self.epoch, self.almanac,
        );
        let Some(sc) = sc else { return f64::MAX };
        propagate_closest(sc, self.propagator, self.almanac, self.duration, self.sample_step_s, self.lp_cache)
            .map(|(d, _)| d)
            .unwrap_or(f64::MAX)
    }
}

/// Internal: coarse grid search over (angle, velocity).
/// Returns sorted Vec<(angle_deg, speed_km_s, cost)>, best first.
fn grid_search(
    longitude_deg: f64,
    latitude_deg: f64,
    epoch: Epoch,
    almanac: &Arc<Almanac>,
    opts: &TrajectoryOptions,
    duration: Duration,
    lp_cache: &LpCache,
) -> Option<Vec<(f64, f64, f64)>> {
    let prop = fast_propagator(almanac, opts.grid.tolerance, opts.grid.max_step_s)?;
    let eval = Evaluator {
        longitude_deg, latitude_deg, epoch, almanac,
        propagator: &prop, duration, sample_step_s: opts.grid.sample_step_s,
        lp_cache, velocity_range: opts.velocity_range, angle_range: opts.angle_range,
    };

    let n_vel = opts.grid.n_vel;
    let n_ang = opts.grid.n_ang;
    let mut results = Vec::with_capacity(n_vel * n_ang);
    for iv in 0..n_vel {
        let v = opts.velocity_range.0
            + (opts.velocity_range.1 - opts.velocity_range.0) * iv as f64
                / (n_vel - 1).max(1) as f64;
        for ia in 0..n_ang {
            let a = opts.angle_range.0
                + (opts.angle_range.1 - opts.angle_range.0) * ia as f64
                    / (n_ang - 1).max(1) as f64;
            let cost = eval.eval(a, v);
            results.push((a, v, cost));
        }
    }
    results.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal));
    Some(results)
}

/// Internal: Nelder-Mead refinement starting from a grid-search best point.
/// Returns (best_angle_deg, best_speed_km_s, best_cost).
fn refine_nelder_mead(
    grid_best: (f64, f64),
    grid_step: (f64, f64),
    longitude_deg: f64,
    latitude_deg: f64,
    epoch: Epoch,
    almanac: &Arc<Almanac>,
    opts: &TrajectoryOptions,
    duration: Duration,
    lp_cache: &LpCache,
) -> Option<(f64, f64, f64)> {
    let nm = &opts.nm;
    let prop = fast_propagator(almanac, nm.tolerance, nm.max_step_s)?;
    let eval = Evaluator {
        longitude_deg, latitude_deg, epoch, almanac,
        propagator: &prop, duration, sample_step_s: nm.sample_step_s,
        lp_cache, velocity_range: opts.velocity_range, angle_range: opts.angle_range,
    };

    let (ba, bv) = grid_best;
    let (da, dv) = grid_step;
    let initial = [[ba, bv], [ba + da, bv], [ba, bv + dv]];

    let mut simplex: Vec<([f64; 2], f64)> = initial
        .iter()
        .map(|&p| (p, eval.eval(p[0], p[1])))
        .collect();

    let (alpha, gamma, rho, sigma) = (1.0, 2.0, 0.5, 0.5);

    for _ in 0..nm.max_iter {
        simplex.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

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
        let fr = eval.eval(reflected[0], reflected[1]);

        if fr < simplex[0].1 {
            let expanded = [
                centroid[0] + gamma * (reflected[0] - centroid[0]),
                centroid[1] + gamma * (reflected[1] - centroid[1]),
            ];
            let fe = eval.eval(expanded[0], expanded[1]);
            simplex[n] = if fe < fr { (expanded, fe) } else { (reflected, fr) };
        } else if fr < simplex[n - 1].1 {
            simplex[n] = (reflected, fr);
        } else {
            let contracted = [
                centroid[0] + rho * (worst[0] - centroid[0]),
                centroid[1] + rho * (worst[1] - centroid[1]),
            ];
            let fc = eval.eval(contracted[0], contracted[1]);
            if fc < simplex[n].1 {
                simplex[n] = (contracted, fc);
            } else {
                let best = simplex[0].0;
                for i in 1..simplex.len() {
                    let p = [
                        best[0] + sigma * (simplex[i].0[0] - best[0]),
                        best[1] + sigma * (simplex[i].0[1] - best[1]),
                    ];
                    simplex[i] = (p, eval.eval(p[0], p[1]));
                }
            }
        }

        // Convergence
        let da = (simplex[0].0[0] - simplex[n].0[0]).abs();
        let dv = (simplex[0].0[1] - simplex[n].0[1]).abs();
        if da < nm.converge_angle_deg && dv < nm.converge_speed_km_s {
            break;
        }
    }

    simplex.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    let best_a = simplex[0].0[0].clamp(opts.angle_range.0, opts.angle_range.1);
    let best_v = simplex[0].0[1].clamp(opts.velocity_range.0, opts.velocity_range.1);
    Some((best_a, best_v, simplex[0].1))
}

/// Find the best single trajectory from a Moon surface point to a Lagrange point.
///
/// Uses multi-resolution search: coarse grid → Nelder-Mead refinement.
/// Precomputes Lagrange point positions to avoid redundant almanac lookups.
/// No internal parallelism — designed to be called in parallel from outside.
pub fn trajectory(
    epoch: Epoch,
    longitude_deg: f64,
    latitude_deg: f64,
    opts: &TrajectoryOptions,
    almanac: &Arc<Almanac>,
) -> Option<TrajectoryResult> {
    let duration = Duration::from_seconds(opts.integration_time_days * 86400.0);
    let lp_cache = LpCache::new(epoch, duration, 600.0, opts.l_point, almanac)?;

    // Phase 1: coarse grid
    let grid = grid_search(longitude_deg, latitude_deg, epoch, almanac, opts, duration, &lp_cache)?;
    if grid.is_empty() { return None; }

    let grid_step_a = (opts.angle_range.1 - opts.angle_range.0) / opts.grid.n_ang as f64;
    let grid_step_v = (opts.velocity_range.1 - opts.velocity_range.0) / opts.grid.n_vel as f64;

    // Phase 2: Nelder-Mead refinement
    let (best_ang, best_vel, _) = refine_nelder_mead(
        (grid[0].0, grid[0].1), (grid_step_a, grid_step_v),
        longitude_deg, latitude_deg, epoch, almanac, opts, duration, &lp_cache,
    )?;

    // Final: high-precision propagation with trail
    let final_prop = fast_propagator(almanac, 1e-11, 300.0)?;
    let sc = create_surface_spacecraft(longitude_deg, latitude_deg, best_ang, best_vel, epoch, almanac)?;
    let (closest_km, time_s, trail) = propagate_full(sc, &final_prop, almanac, duration, 60.0, opts.l_point)?;

    Some(TrajectoryResult {
        velocity: best_vel,
        angle: best_ang,
        closest_km,
        time_to_closest_s: time_s,
        trail,
    })
}

/// Fast variant: returns only (velocity, angle, closest_km, time_s) — no trail.
/// Used for the surface map where we don't need trail data.
pub fn trajectory_fast(
    epoch: Epoch,
    longitude_deg: f64,
    latitude_deg: f64,
    opts: &TrajectoryOptions,
    almanac: &Arc<Almanac>,
) -> Option<(f64, f64, f64, f64)> {
    let duration = Duration::from_seconds(opts.integration_time_days * 86400.0);
    let lp_cache = LpCache::new(epoch, duration, 600.0, opts.l_point, almanac)?;

    let grid = grid_search(longitude_deg, latitude_deg, epoch, almanac, opts, duration, &lp_cache)?;
    if grid.is_empty() { return None; }

    let grid_step_a = (opts.angle_range.1 - opts.angle_range.0) / opts.grid.n_ang as f64;
    let grid_step_v = (opts.velocity_range.1 - opts.velocity_range.0) / opts.grid.n_vel as f64;

    let (best_ang, best_vel, _) = refine_nelder_mead(
        (grid[0].0, grid[0].1), (grid_step_a, grid_step_v),
        longitude_deg, latitude_deg, epoch, almanac, opts, duration, &lp_cache,
    )?;

    // Re-eval at fine precision to get distance and time
    let prop = fast_propagator(almanac, opts.nm.tolerance, opts.nm.max_step_s)?;
    let sc = create_surface_spacecraft(longitude_deg, latitude_deg, best_ang, best_vel, epoch, almanac)?;
    let (dist, time) = propagate_closest(sc, &prop, almanac, duration, opts.nm.sample_step_s, &lp_cache)?;

    Some((best_vel, best_ang, dist, time))
}

// === Continuation map ===

/// Per-point result with both eastward and westward launch trajectories.
/// Each field is (velocity_km_s, angle_deg, closest_km, time_s).
#[derive(Clone, Debug, Default)]
pub struct DualResult {
    pub east: Option<(f64, f64, f64, f64)>,
    pub west: Option<(f64, f64, f64, f64)>,
}

/// 2D map of optimal trajectories over the lunar surface.
/// Row-major: results[el_idx * n_az + az_idx].
#[derive(Clone, Debug)]
pub struct ContinuationMap {
    pub az_min: f64,
    pub az_max: f64,
    pub el_min: f64,
    pub el_max: f64,
    pub n_az: usize,
    pub n_el: usize,
    pub results: Vec<DualResult>,
}

/// Seed optimization: full grid search + NM at a single point.
/// Returns (velocity, angle, closest_km, time_s).
fn seed_optimize(
    longitude_deg: f64,
    latitude_deg: f64,
    epoch: Epoch,
    almanac: &Arc<Almanac>,
    opts: &TrajectoryOptions,
    duration: Duration,
    lp_cache: &LpCache,
) -> Option<(f64, f64, f64, f64)> {
    let grid = grid_search(longitude_deg, latitude_deg, epoch, almanac, opts, duration, lp_cache)?;
    if grid.is_empty() { return None; }

    let grid_step_a = (opts.angle_range.1 - opts.angle_range.0) / opts.grid.n_ang as f64;
    let grid_step_v = (opts.velocity_range.1 - opts.velocity_range.0) / opts.grid.n_vel as f64;

    let (best_ang, best_vel, _) = refine_nelder_mead(
        (grid[0].0, grid[0].1), (grid_step_a, grid_step_v),
        longitude_deg, latitude_deg, epoch, almanac, opts, duration, lp_cache,
    )?;

    let prop = fast_propagator(almanac, opts.nm.tolerance, opts.nm.max_step_s)?;
    let sc = create_surface_spacecraft(longitude_deg, latitude_deg, best_ang, best_vel, epoch, almanac)?;
    let (dist, time) = propagate_closest(sc, &prop, almanac, duration, opts.nm.sample_step_s, lp_cache)?;
    Some((best_vel, best_ang, dist, time))
}

/// NM continuation from a previous solution (no grid search).
/// Returns (velocity, angle, closest_km, time_s).
fn continue_from_point(
    prev_angle: f64,
    prev_velocity: f64,
    longitude_deg: f64,
    latitude_deg: f64,
    epoch: Epoch,
    almanac: &Arc<Almanac>,
    opts: &TrajectoryOptions,
    duration: Duration,
    lp_cache: &LpCache,
) -> Option<(f64, f64, f64, f64)> {
    let (best_ang, best_vel, _) = refine_nelder_mead(
        (prev_angle, prev_velocity),
        (1.0, 0.005), // small perturbations: 1°, 5 m/s
        longitude_deg, latitude_deg,
        epoch, almanac, opts, duration, lp_cache,
    )?;

    let prop = fast_propagator(almanac, opts.nm.tolerance, opts.nm.max_step_s)?;
    let sc = create_surface_spacecraft(longitude_deg, latitude_deg, best_ang, best_vel, epoch, almanac)?;
    let (dist, time) = propagate_closest(sc, &prop, almanac, duration, opts.nm.sample_step_s, lp_cache)?;
    Some((best_vel, best_ang, dist, time))
}

/// March one arm of the continuation (positive or negative longitude direction).
fn march_arm(
    label: &str,
    seed: Option<(f64, f64, f64, f64)>,
    seed_longitude_deg: f64,
    step: f64, // +1.0 or -1.0
    n_steps: usize,
    latitude_deg: f64,
    epoch: Epoch,
    almanac: &Arc<Almanac>,
    opts: &TrajectoryOptions,
    duration: Duration,
    lp_cache: &LpCache,
) -> Vec<(usize, Option<(f64, f64, f64, f64)>)> {
    let mut prev = seed;
    let mut out = Vec::with_capacity(n_steps);
    for i in 1..=n_steps {
        let az = seed_longitude_deg + step * i as f64;
        let result = prev.and_then(|(v, a, _, _)| {
            continue_from_point(a, v, az, latitude_deg, epoch, almanac, opts, duration, lp_cache)
        });
        if let Some((v, a, d, _)) = result {
            println!("  {label} [{i}/{n_steps}] az={az:.0}° v={v:.4} a={a:.1}° d={d:.0}km");
        } else {
            println!("  {label} [{i}/{n_steps}] az={az:.0}° FAILED");
        }
        out.push((i, result));
        if result.is_some() { prev = result; }
    }
    out
}

/// Compute a continuation map along the lunar equator.
///
/// Starts from `seed_longitude_deg` with full grid+NM optimization, then marches
/// in 1° longitude steps using each solution as the NM starting point for the next.
/// Computes both eastward and westward launch trajectories at each point.
pub fn compute_continuation_map(
    epoch: Epoch,
    almanac: &Arc<Almanac>,
    seed_longitude_deg: f64,
    az_half_range_deg: f64,
    opts: &TrajectoryOptions,
    east_angle_range: (f64, f64),
    west_angle_range: (f64, f64),
) -> ContinuationMap {
    let n_steps = az_half_range_deg as usize;
    let n_az = 2 * n_steps + 1;
    let n_el = 1;
    let el = 0.0;
    let az_min = seed_longitude_deg - az_half_range_deg;
    let az_max = seed_longitude_deg + az_half_range_deg;

    let duration = Duration::from_seconds(opts.integration_time_days * 86400.0);
    let lp_cache = LpCache::new(epoch, duration, 600.0, opts.l_point, almanac)
        .expect("Failed to create LP cache");

    let east_opts = TrajectoryOptions { angle_range: east_angle_range, ..opts.clone() };
    let west_opts = TrajectoryOptions { angle_range: west_angle_range, ..opts.clone() };

    let mut results = vec![DualResult::default(); n_az * n_el];
    let seed_idx = n_steps;

    // Seed: full grid search + NM
    println!("Seed at az={seed_longitude_deg:.0}°...");
    let east_seed = seed_optimize(seed_longitude_deg, el, epoch, almanac, &east_opts, duration, &lp_cache);
    let west_seed = seed_optimize(seed_longitude_deg, el, epoch, almanac, &west_opts, duration, &lp_cache);

    if let Some((v, a, d, t)) = east_seed {
        println!("  East: v={v:.4} a={a:.1}° d={d:.0}km t={:.1}h", t / 3600.0);
    } else {
        println!("  East: no solution");
    }
    if let Some((v, a, d, t)) = west_seed {
        println!("  West: v={v:.4} a={a:.1}° d={d:.0}km t={:.1}h", t / 3600.0);
    } else {
        println!("  West: no solution");
    }
    results[seed_idx] = DualResult { east: east_seed, west: west_seed };

    // March both directions: 4 independent arms in parallel
    let (e_pos, e_neg, w_pos, w_neg) = std::thread::scope(|s| {
        let h_ep = s.spawn(|| march_arm("E+", east_seed, seed_longitude_deg, 1.0, n_steps, el, epoch, almanac, &east_opts, duration, &lp_cache));
        let h_en = s.spawn(|| march_arm("E-", east_seed, seed_longitude_deg, -1.0, n_steps, el, epoch, almanac, &east_opts, duration, &lp_cache));
        let h_wp = s.spawn(|| march_arm("W+", west_seed, seed_longitude_deg, 1.0, n_steps, el, epoch, almanac, &west_opts, duration, &lp_cache));
        let h_wn = s.spawn(|| march_arm("W-", west_seed, seed_longitude_deg, -1.0, n_steps, el, epoch, almanac, &west_opts, duration, &lp_cache));
        (h_ep.join().unwrap(), h_en.join().unwrap(), h_wp.join().unwrap(), h_wn.join().unwrap())
    });

    // Merge into results
    for (i, east) in &e_pos {
        results[seed_idx + i].east = *east;
    }
    for (i, east) in &e_neg {
        results[seed_idx - i].east = *east;
    }
    for (i, west) in &w_pos {
        results[seed_idx + i].west = *west;
    }
    for (i, west) in &w_neg {
        results[seed_idx - i].west = *west;
    }

    // Print summary
    for idx in 0..n_az {
        let az = az_min + idx as f64;
        let r = &results[idx];
        print!("az={az:6.0}°");
        if let Some((v, a, d, _)) = r.east {
            print!("  E: v={v:.4} a={a:+6.1}° d={d:8.0}km");
        } else {
            print!("  E: ---");
        }
        if let Some((v, a, d, _)) = r.west {
            print!("  W: v={v:.4} a={a:+6.1}° d={d:8.0}km");
        } else {
            print!("  W: ---");
        }
        println!();
    }

    ContinuationMap {
        az_min, az_max, el_min: 0.0, el_max: 0.0,
        n_az, n_el, results,
    }
}

/// Result of a surface map computation for a single grid point.
#[derive(Clone, Debug)]
pub struct SurfaceMapPoint {
    pub longitude_deg: f64,
    pub latitude_deg: f64,
    pub best_velocity: f64,
    pub best_angle: f64,
    pub closest_km: f64,
    pub time_to_closest_s: f64,
}

/// Compute a surface map: for each (longitude, latitude) grid point on the Moon,
/// find the best trajectory to the target Lagrange point.
/// Uses rayon for parallelism across grid points.
pub fn compute_surface_map(
    epoch: Epoch,
    almanac: &Arc<Almanac>,
    longitude_range: (f64, f64),
    latitude_range: (f64, f64),
    n_longitude: usize,
    n_latitude: usize,
    opts: &TrajectoryOptions,
    progress: &std::sync::atomic::AtomicUsize,
) -> Vec<SurfaceMapPoint> {
    use rayon::prelude::*;

    let params: Vec<(f64, f64)> = (0..n_latitude)
        .flat_map(|il| {
            let lat = if n_latitude <= 1 {
                (latitude_range.0 + latitude_range.1) / 2.0
            } else {
                latitude_range.0 + (latitude_range.1 - latitude_range.0) * il as f64 / (n_latitude - 1) as f64
            };
            (0..n_longitude).map(move |ia| {
                let lon = if n_longitude <= 1 {
                    (longitude_range.0 + longitude_range.1) / 2.0
                } else {
                    longitude_range.0 + (longitude_range.1 - longitude_range.0) * ia as f64 / (n_longitude - 1) as f64
                };
                (lon, lat)
            })
        })
        .collect();

    params
        .par_iter()
        .filter_map(|&(lon, lat)| {
            let (vel, ang, dist, time) = trajectory_fast(
                epoch, lon, lat, opts, almanac,
            )?;
            progress.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            Some(SurfaceMapPoint {
                longitude_deg: lon,
                latitude_deg: lat,
                best_velocity: vel,
                best_angle: ang,
                closest_km: dist,
                time_to_closest_s: time,
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trajectory_equator_to_eml1() {
        let almanac = crate::setup_almanac().expect("Failed to load almanac");
        let epoch = Epoch::from_gregorian_utc(2025, 6, 21, 12, 0, 0, 0);

        let opts = TrajectoryOptions {
            velocity_range: (2.3, 2.5),
            angle_range: (-30.0, 30.0),
            integration_time_days: 10.0,
            l_point: lagrange::LagrangeId::L1,
            ..Default::default()
        };

        // Far side on equator (best for EML1)
        let result = trajectory(
            epoch,
            180.0, // longitude: anti-Earth (far side)
            0.0,   // latitude: equator
            &opts,
            &almanac,
        );

        let result = result.expect("Should find a trajectory");
        println!("velocity: {:.4} km/s", result.velocity);
        println!("angle: {:.1} deg", result.angle);
        println!("closest: {:.0} km", result.closest_km);
        println!("time: {:.1} hours", result.time_to_closest_s / 3600.0);
        println!("trail points: {}", result.trail.len());

        // Should get within ~10,000 km of EML1 at least
        assert!(result.closest_km < 50_000.0,
            "Expected < 50,000 km, got {:.0} km", result.closest_km);
    }

    #[test]
    fn test_continuation_map() {
        let almanac = crate::setup_almanac().expect("Failed to load almanac");
        let epoch = Epoch::from_gregorian_utc(2025, 6, 21, 12, 0, 0, 0);

        let opts = TrajectoryOptions {
            velocity_range: (2.3, 2.5),
            angle_range: (-30.0, 30.0), // overridden per-direction
            integration_time_days: 10.0,
            l_point: lagrange::LagrangeId::L1,
            grid: GridSearchOptions::default(),
            nm: NelderMeadOptions {
                max_iter: 30,
                tolerance: 1e-6,
                max_step_s: 600.0,
                sample_step_s: 600.0,
                converge_angle_deg: 0.1,
                converge_speed_km_s: 0.001,
            },
        };

        let map = compute_continuation_map(
            epoch, &almanac,
            180.0,  // seed: anti-Earth (far side, best for EML1)
            45.0,   // ±45°
            &opts,
            (-30.0, 30.0),   // east: launch ~eastward (prograde)
            (150.0, 210.0),  // west: launch ~westward (retrograde)
        );

        println!("\nMap: {} points, az=[{:.0}°, {:.0}°]", map.results.len(), map.az_min, map.az_max);
        let valid_east = map.results.iter().filter(|r| r.east.is_some()).count();
        let valid_west = map.results.iter().filter(|r| r.west.is_some()).count();
        println!("Valid: east={valid_east}, west={valid_west}");

        assert!(valid_east > 0, "Should find at least one east trajectory");
        assert!(valid_west > 0, "Should find at least one west trajectory");
    }

    #[test]
    fn test_full_grid_csv() {
        use rayon::prelude::*;
        use std::io::Write;
        use std::sync::atomic::{AtomicUsize, Ordering};

        let almanac = crate::setup_almanac().expect("Failed to load almanac");
        let epoch = Epoch::from_gregorian_utc(2025, 6, 21, 12, 0, 0, 0);

        // Longitude: far side ±45° → 135° to 225°, 10° steps = 10 values
        // Latitude: 0° to 45°, 5° steps = 10 values (skip southern, near-symmetric)
        let lon_min = 135.0_f64;
        let lon_max = 225.0_f64;
        let lon_step = 10.0_f64;
        let lat_min = 0.0_f64;
        let lat_max = 45.0_f64;
        let lat_step = 5.0_f64;

        let lons: Vec<f64> = {
            let mut v = Vec::new();
            let mut lon = lon_min;
            while lon <= lon_max + 0.001 { v.push(lon); lon += lon_step; }
            v
        };
        let lats: Vec<f64> = {
            let mut v = Vec::new();
            let mut lat = lat_min;
            while lat <= lat_max + 0.001 { v.push(lat); lat += lat_step; }
            v
        };

        let east_opts = TrajectoryOptions {
            velocity_range: (2.3, 2.5),
            angle_range: (-30.0, 30.0),
            integration_time_days: 10.0,
            l_point: lagrange::LagrangeId::L1,
            ..Default::default()
        };
        let west_opts = TrajectoryOptions {
            angle_range: (150.0, 210.0),
            ..east_opts.clone()
        };

        // Build all (lon, lat) pairs
        let points: Vec<(f64, f64)> = lats.iter()
            .flat_map(|&lat| lons.iter().map(move |&lon| (lon, lat)))
            .collect();
        let total = points.len() * 2; // east + west per point
        let done = AtomicUsize::new(0);

        println!("Computing {} points ({} optimizations)...", points.len(), total);

        // Full grid+NM for each point, parallelized with rayon
        let results: Vec<(f64, f64, Option<(f64, f64, f64, f64)>, Option<(f64, f64, f64, f64)>)> =
            points.par_iter().map(|&(lon, lat)| {
                let east = trajectory_fast(epoch, lon, lat, &east_opts, &almanac);
                let d = done.fetch_add(1, Ordering::Relaxed) + 1;
                if let Some((v, a, dist, _)) = east {
                    println!("[{d}/{total}] lon={lon:.0}° lat={lat:.0}° E: v={v:.4} a={a:.1}° d={dist:.0}km");
                } else {
                    println!("[{d}/{total}] lon={lon:.0}° lat={lat:.0}° E: FAILED");
                }

                let west = trajectory_fast(epoch, lon, lat, &west_opts, &almanac);
                let d = done.fetch_add(1, Ordering::Relaxed) + 1;
                if let Some((v, a, dist, _)) = west {
                    println!("[{d}/{total}] lon={lon:.0}° lat={lat:.0}° W: v={v:.4} a={a:.1}° d={dist:.0}km");
                } else {
                    println!("[{d}/{total}] lon={lon:.0}° lat={lat:.0}° W: FAILED");
                }

                (lon, lat, east, west)
            }).collect();

        // Write CSV
        // Angle convention: 0° = east (prograde), 90° = north, 180° = west (retrograde)
        let csv_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("eml1_grid.csv");
        let mut f = std::fs::File::create(&csv_path).expect("create csv");
        writeln!(f, "lat,lon,east_vel,east_angle,east_dist_km,west_vel,west_angle,west_dist_km").unwrap();
        for &(lon, lat, ref east, ref west) in &results {
            let (ev, ea, ed) = east.map(|(v, a, d, _)| (v, a, d)).unwrap_or((f64::NAN, f64::NAN, f64::NAN));
            let (wv, wa, wd) = west.map(|(v, a, d, _)| (v, a, d)).unwrap_or((f64::NAN, f64::NAN, f64::NAN));
            writeln!(f, "{lat:.1},{lon:.1},{ev:.6},{ea:.2},{ed:.1},{wv:.6},{wa:.2},{wd:.1}").unwrap();
        }

        println!("\nCSV written to {}", csv_path.display());

        // Summary
        let valid_e = results.iter().filter(|r| r.2.is_some()).count();
        let valid_w = results.iter().filter(|r| r.3.is_some()).count();
        println!("Valid: east={valid_e}/{}, west={valid_w}/{}", results.len(), results.len());
    }
}

// === Bevy integration ===

use std::sync::{
    Mutex,
    atomic::{AtomicBool, AtomicUsize, Ordering},
    mpsc,
};
use std::thread::JoinHandle;

use bevy::prelude::*;
use bevy_egui::egui;

use crate::optimizer::{BevyStrip, precompute_strip, range_control};
use crate::visualization::{SimulationState, SpaceResources, TrailFrame};

/// Messages from surface optimizer thread to main thread.
pub enum SurfaceOptimizerMsg {
    /// Single trajectory result (for "test single" mode).
    SingleResult(TrajectoryResult),
    /// Surface map progress update.
    MapProgress { done: usize, total: usize },
    /// Surface map complete.
    MapFinished(Vec<SurfaceMapPoint>),
}

/// Configuration for the surface optimizer UI.
#[derive(Clone, Debug)]
pub struct SurfaceOptimizerConfig {
    pub longitude_deg: f64,
    pub latitude_deg: f64,
    pub opts: TrajectoryOptions,
    // Map settings
    pub map_lon_min: f64,
    pub map_lon_max: f64,
    pub map_lat_min: f64,
    pub map_lat_max: f64,
    pub map_n_lon: usize,
    pub map_n_lat: usize,
}

impl Default for SurfaceOptimizerConfig {
    fn default() -> Self {
        Self {
            longitude_deg: 180.0,
            latitude_deg: 0.0,
            opts: TrajectoryOptions::default(),
            map_lon_min: -180.0,
            map_lon_max: 180.0,
            map_lat_min: -90.0,
            map_lat_max: 90.0,
            map_n_lon: 36,
            map_n_lat: 18,
        }
    }
}

/// Bevy resource for the surface optimizer.
#[derive(Resource)]
pub struct SurfaceOptimizerState {
    pub config: SurfaceOptimizerConfig,
    pub running: bool,
    pub cancel: Arc<AtomicBool>,
    receiver: Option<Mutex<mpsc::Receiver<SurfaceOptimizerMsg>>>,
    thread_handle: Option<Mutex<JoinHandle<()>>>,
    pub single_result: Option<TrajectoryResult>,
    pub display_strip: Option<(Color, BevyStrip)>,
    pub map_results: Option<Vec<SurfaceMapPoint>>,
    pub progress_done: usize,
    pub progress_total: usize,
    pub opt_epoch: Option<Epoch>,
    trail_frame_cached: TrailFrame,
    pub mode: SurfaceOptMode,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SurfaceOptMode {
    Single,
    Map,
}

impl Default for SurfaceOptimizerState {
    fn default() -> Self {
        Self {
            config: SurfaceOptimizerConfig::default(),
            running: false,
            cancel: Arc::new(AtomicBool::new(false)),
            receiver: None,
            thread_handle: None,
            single_result: None,
            display_strip: None,
            map_results: None,
            progress_done: 0,
            progress_total: 0,
            opt_epoch: None,
            trail_frame_cached: TrailFrame::default(),
            mode: SurfaceOptMode::Single,
        }
    }
}

fn launch_single(
    config: SurfaceOptimizerConfig,
    epoch: Epoch,
    almanac: Arc<Almanac>,
) -> (mpsc::Receiver<SurfaceOptimizerMsg>, JoinHandle<()>, Arc<AtomicBool>) {
    let (sender, receiver) = mpsc::channel();
    let cancel = Arc::new(AtomicBool::new(false));

    let handle = std::thread::spawn(move || {
        if let Some(result) = trajectory(
            epoch,
            config.longitude_deg,
            config.latitude_deg,
            &config.opts,
            &almanac,
        ) {
            let _ = sender.send(SurfaceOptimizerMsg::SingleResult(result));
        }
    });

    (receiver, handle, cancel)
}

fn launch_map(
    config: SurfaceOptimizerConfig,
    epoch: Epoch,
    almanac: Arc<Almanac>,
) -> (mpsc::Receiver<SurfaceOptimizerMsg>, JoinHandle<()>, Arc<AtomicBool>) {
    let (sender, receiver) = mpsc::channel();
    let cancel = Arc::new(AtomicBool::new(false));

    let total = config.map_n_lon * config.map_n_lat;
    let handle = std::thread::spawn(move || {
        let progress = AtomicUsize::new(0);

        let sender2 = sender.clone();
        let progress_ref = &progress;
        std::thread::scope(|s| {
            let reporter = s.spawn(|| {
                loop {
                    let done = progress_ref.load(Ordering::Relaxed);
                    let _ = sender2.send(SurfaceOptimizerMsg::MapProgress { done, total });
                    if done >= total {
                        break;
                    }
                    std::thread::sleep(std::time::Duration::from_millis(250));
                }
            });

            let results = compute_surface_map(
                epoch,
                &almanac,
                (config.map_lon_min, config.map_lon_max),
                (config.map_lat_min, config.map_lat_max),
                config.map_n_lon,
                config.map_n_lat,
                &config.opts,
                &progress,
            );

            progress.store(total, Ordering::Relaxed);
            let _ = reporter.join();

            let _ = sender.send(SurfaceOptimizerMsg::MapFinished(results));
        });
    });

    (receiver, handle, cancel)
}

/// Poll the surface optimizer channel each frame.
pub fn poll_surface_optimizer(
    mut state: ResMut<SurfaceOptimizerState>,
    sim_state: Res<SimulationState>,
    space: Res<SpaceResources>,
) {
    let frame = sim_state.trail_frame;
    let moon_now = bodies::moon_position(&space.almanac, sim_state.epoch)
        .map(|m| [m.x, m.y, m.z])
        .unwrap_or([0.0; 3]);

    // Recompute strip if trail frame changed
    if frame != state.trail_frame_cached {
        state.trail_frame_cached = frame;
        if let Some(ref result) = state.single_result {
            let strip = precompute_strip(&result.trail, frame, &space.almanac, moon_now);
            state.display_strip = Some((Color::srgb(0.0, 1.0, 0.5), strip));
        }
    }

    let messages: Vec<SurfaceOptimizerMsg> = {
        let Some(ref receiver) = state.receiver else { return };
        let Ok(receiver) = receiver.lock() else { return };
        let mut msgs = Vec::new();
        while let Ok(msg) = receiver.try_recv() {
            msgs.push(msg);
        }
        msgs
    };

    if messages.is_empty() {
        return;
    }

    for msg in messages {
        match msg {
            SurfaceOptimizerMsg::SingleResult(result) => {
                let strip = precompute_strip(&result.trail, frame, &space.almanac, moon_now);
                state.display_strip = Some((Color::srgb(0.0, 1.0, 0.5), strip));
                state.single_result = Some(result);
                state.running = false;
                state.receiver = None;
                state.thread_handle = None;
            }
            SurfaceOptimizerMsg::MapProgress { done, total } => {
                state.progress_done = done;
                state.progress_total = total;
            }
            SurfaceOptimizerMsg::MapFinished(results) => {
                state.map_results = Some(results);
                state.running = false;
                state.receiver = None;
                state.thread_handle = None;
            }
        }
    }
}

/// Draw surface optimizer trails.
pub fn draw_surface_optimizer_trails(
    state: Res<SurfaceOptimizerState>,
    mut gizmos: Gizmos,
) {
    if let Some((color, ref strip)) = state.display_strip {
        if strip.len() >= 2 {
            gizmos.linestrip(strip.iter().copied(), color);
        }
    }
}

/// Draw the surface map as a colored overlay on the Moon.
pub fn draw_surface_map(
    state: Res<SurfaceOptimizerState>,
    sim_state: Res<SimulationState>,
    space: Res<SpaceResources>,
    mut gizmos: Gizmos,
) {
    let Some(ref map) = state.map_results else { return };
    if map.is_empty() { return; }

    let moon_pos = bodies::moon_position(&space.almanac, sim_state.epoch)
        .map(|m| bodies::J2000Position { x: m.x, y: m.y, z: m.z })
        .unwrap_or(bodies::J2000Position { x: 0.0, y: 0.0, z: 0.0 });
    let moon_vel = bodies::moon_velocity(&space.almanac, sim_state.epoch)
        .unwrap_or(bodies::J2000Velocity { x: 0.0, y: 0.0, z: 0.0 });
    let (away, north, prograde) = moon_frame(&moon_pos, &moon_vel);

    // Find the range of closest_km for color mapping (use log scale)
    let min_dist = map.iter().map(|p| p.closest_km).fold(f64::MAX, f64::min);
    let max_dist = map.iter().map(|p| p.closest_km).fold(0.0_f64, f64::max);
    let log_min = (min_dist.max(1.0)).ln();
    let log_max = (max_dist.max(2.0)).ln();
    let log_range = (log_max - log_min).max(0.01);

    let r = MOON_RADIUS_KM + 50.0; // Slightly above the surface

    // Standard selenographic: lon=0 toward Earth
    let toward = [-away[0], -away[1], -away[2]];

    for point in map.iter() {
        let lon = point.longitude_deg.to_radians();
        let lat = point.latitude_deg.to_radians();

        let normal = [
            lat.cos() * lon.cos() * toward[0] + lat.cos() * lon.sin() * prograde[0] + lat.sin() * north[0],
            lat.cos() * lon.cos() * toward[1] + lat.cos() * lon.sin() * prograde[1] + lat.sin() * north[1],
            lat.cos() * lon.cos() * toward[2] + lat.cos() * lon.sin() * prograde[2] + lat.sin() * north[2],
        ];

        let pos_j2000 = [
            moon_pos.x + normal[0] * r,
            moon_pos.y + normal[1] * r,
            moon_pos.z + normal[2] * r,
        ];

        let bevy_pos = bodies::J2000Position {
            x: pos_j2000[0], y: pos_j2000[1], z: pos_j2000[2],
        }.to_bevy(VIS_SCALE_F64);

        // Color: green (good/close) → red (bad/far), log scale
        let t = ((point.closest_km.max(1.0)).ln() - log_min) / log_range;
        let t = t.clamp(0.0, 1.0);
        // green → yellow → red
        let (cr, cg) = if t < 0.5 {
            (t * 2.0, 1.0)
        } else {
            (1.0, (1.0 - t) * 2.0)
        };
        let color = Color::srgb(cr as f32, cg as f32, 0.0);

        gizmos.sphere(
            Isometry3d::from_translation(Vec3::from_array(bevy_pos)),
            20.0 * VIS_SCALE,
            color,
        );
    }
}

/// UI panel content for the surface optimizer.
pub(crate) fn surface_optimizer_ui_content(
    ui: &mut egui::Ui,
    state: &mut SurfaceOptimizerState,
    sim_state: &mut SimulationState,
    space: &SpaceResources,
) {
    egui::CollapsingHeader::new("Launch Site")
        .default_open(false)
        .show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.label("Longitude:");
                ui.add(egui::DragValue::new(&mut state.config.longitude_deg).speed(1.0).suffix("\u{00b0}").range(-180.0..=360.0));
            });
            ui.horizontal(|ui| {
                ui.label("Latitude:");
                ui.add(egui::DragValue::new(&mut state.config.latitude_deg).speed(1.0).suffix("\u{00b0}").range(-90.0..=90.0));
            });
        });

    egui::CollapsingHeader::new("Search Ranges")
        .default_open(false)
        .show(ui, |ui| {
            range_control(ui, "Velocity:", &mut state.config.opts.velocity_range.0, &mut state.config.opts.velocity_range.1, 0.01, " km/s");
            range_control(ui, "Launch angle:", &mut state.config.opts.angle_range.0, &mut state.config.opts.angle_range.1, 1.0, "\u{00b0}");
            ui.horizontal(|ui| {
                ui.label("Max propagation:");
                ui.add(egui::DragValue::new(&mut state.config.opts.integration_time_days).range(1.0..=30.0).speed(0.5).suffix(" days"));
            });
            ui.horizontal(|ui| {
                ui.label("Target:");
                egui::ComboBox::from_id_salt("surf_target")
                    .selected_text(state.config.opts.l_point.label())
                    .show_ui(ui, |ui| {
                        for lp in lagrange::LagrangeId::ALL {
                            ui.selectable_value(&mut state.config.opts.l_point, lp, lp.label());
                        }
                    });
            });
        });

    ui.separator();

    if state.running {
        let frac = if state.progress_total > 0 {
            state.progress_done as f32 / state.progress_total as f32
        } else {
            0.0
        };
        let label = match state.mode {
            SurfaceOptMode::Single => "Computing trajectory...".to_string(),
            SurfaceOptMode::Map => format!("Map: {}/{}", state.progress_done, state.progress_total),
        };
        ui.add(egui::ProgressBar::new(frac).text(label));
        if ui.button("Cancel").clicked() {
            state.cancel.store(true, Ordering::Relaxed);
            state.running = false;
            state.receiver = None;
            state.thread_handle = None;
        }
    } else {
        let find_clicked = ui.button("Find Trajectory").clicked();
        if find_clicked {
            sim_state.paused = true;
            state.opt_epoch = Some(sim_state.epoch);
            state.mode = SurfaceOptMode::Single;
            let (rx, handle, cancel) = launch_single(
                state.config.clone(), sim_state.epoch, space.almanac.clone(),
            );
            state.receiver = Some(Mutex::new(rx));
            state.thread_handle = Some(Mutex::new(handle));
            state.cancel = cancel;
            state.running = true;
            state.single_result = None;
            state.display_strip = None;
        }

        egui::CollapsingHeader::new("Surface Map")
            .default_open(false)
            .show(ui, |ui| {
                range_control(ui, "Longitude:", &mut state.config.map_lon_min, &mut state.config.map_lon_max, 1.0, "\u{00b0}");
                range_control(ui, "Latitude:", &mut state.config.map_lat_min, &mut state.config.map_lat_max, 1.0, "\u{00b0}");
                ui.horizontal(|ui| {
                    ui.label("Grid:");
                    ui.add(egui::DragValue::new(&mut state.config.map_n_lon).range(2..=72).prefix("az:"));
                    ui.label("\u{00d7}");
                    ui.add(egui::DragValue::new(&mut state.config.map_n_lat).range(2..=36).prefix("el:"));
                });
                if ui.button("Compute Map").clicked() {
                    sim_state.paused = true;
                    state.opt_epoch = Some(sim_state.epoch);
                    state.mode = SurfaceOptMode::Map;
                    let (rx, handle, cancel) = launch_map(
                        state.config.clone(), sim_state.epoch, space.almanac.clone(),
                    );
                    state.receiver = Some(Mutex::new(rx));
                    state.thread_handle = Some(Mutex::new(handle));
                    state.cancel = cancel;
                    state.running = true;
                    state.map_results = None;
                    state.progress_done = 0;
                    state.progress_total = state.config.map_n_lon * state.config.map_n_lat;
                }
            });
    }

    // Show single result
    if let Some(ref result) = state.single_result {
        ui.separator();
        ui.label(format!(
            "Best: v={:.4} km/s, angle={:.1}\u{00b0}",
            result.velocity, result.angle,
        ));
        ui.label(format!(
            "Closest: {:.0} km at t={:.1}h",
            result.closest_km, result.time_to_closest_s / 3600.0,
        ));
    }

    // Show map summary
    if let Some(ref map) = state.map_results {
        ui.separator();
        let best = map.iter().min_by(|a, b| a.closest_km.partial_cmp(&b.closest_km).unwrap());
        if let Some(best) = best {
            ui.label(format!("Map: {} points computed", map.len()));
            ui.label(format!(
                "Best: az={:.0}\u{00b0} el={:.0}\u{00b0} → {:.0} km",
                best.longitude_deg, best.latitude_deg, best.closest_km,
            ));
            ui.label(format!(
                "  v={:.4} km/s, angle={:.1}\u{00b0}, t={:.1}h",
                best.best_velocity, best.best_angle, best.time_to_closest_s / 3600.0,
            ));
        }
        if ui.button("Clear map").clicked() {
            state.map_results = None;
        }
    }
}
