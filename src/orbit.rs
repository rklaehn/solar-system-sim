//! Orbit creation and propagation using nyx-space.

use std::sync::Arc;

use anise::{
    constants::{
        celestial_objects::{MOON, SUN},
        frames::{EARTH_J2000, MOON_J2000},
    },
    prelude::Almanac,
};
use hifitime::{Duration, Epoch};
use nyx_space::State;
use nyx_space::{
    cosmic::{Mass, Orbit, Spacecraft},
    dynamics::{
        orbital::{Harmonics, OrbitalDynamics, PointMasses},
        SpacecraftDynamics,
    },
    io::gravity::HarmonicsMem,
    md::Trajectory,
    propagators::{ErrorControl, IntegratorMethod, IntegratorOptions, Propagator},
};

use crate::constants::EARTH_J2;

/// Reference frame for orbit definition.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ReferenceFrame {
    EarthJ2000,
    MoonJ2000,
}

/// Orbital elements for creating an orbit.
#[derive(Clone, Debug)]
pub struct OrbitalElements {
    pub semi_major_axis_km: f64,
    pub eccentricity: f64,
    pub inclination_deg: f64,
    pub raan_deg: f64,
    pub arg_periapsis_deg: f64,
    pub true_anomaly_deg: f64,
    pub frame: ReferenceFrame,
}

impl OrbitalElements {
    /// Create a circular orbit.
    pub fn circular(radius_km: f64, inclination_deg: f64, raan_deg: f64, frame: ReferenceFrame) -> Self {
        Self {
            semi_major_axis_km: radius_km,
            eccentricity: 0.0,
            inclination_deg,
            raan_deg,
            arg_periapsis_deg: 0.0,
            true_anomaly_deg: 0.0,
            frame,
        }
    }
}

/// Create an nyx-space Orbit from orbital elements.
pub fn create_orbit(
    elements: &OrbitalElements,
    epoch: Epoch,
    almanac: &Almanac,
) -> anyhow::Result<Orbit> {
    let frame_uid = match elements.frame {
        ReferenceFrame::EarthJ2000 => EARTH_J2000,
        ReferenceFrame::MoonJ2000 => MOON_J2000,
    };
    let frame = almanac.frame_from_uid(frame_uid)?;

    let orbit = Orbit::keplerian(
        elements.semi_major_axis_km,
        elements.eccentricity,
        elements.inclination_deg,
        elements.raan_deg,
        elements.arg_periapsis_deg,
        elements.true_anomaly_deg,
        epoch,
        frame,
    );

    Ok(orbit)
}

/// Create an nyx-space Orbit from Cartesian state vectors in the given frame.
pub fn create_orbit_cartesian(
    pos_km: [f64; 3],
    vel_km_s: [f64; 3],
    epoch: Epoch,
    almanac: &Almanac,
    ref_frame: ReferenceFrame,
) -> anyhow::Result<Orbit> {
    let frame_uid = match ref_frame {
        ReferenceFrame::EarthJ2000 => EARTH_J2000,
        ReferenceFrame::MoonJ2000 => MOON_J2000,
    };
    let frame = almanac.frame_from_uid(frame_uid)?;
    Ok(Orbit::cartesian(
        pos_km[0], pos_km[1], pos_km[2],
        vel_km_s[0], vel_km_s[1], vel_km_s[2],
        epoch, frame,
    ))
}

/// Set up a propagator with Earth J2 harmonics and Moon/Sun point-mass perturbations.
pub fn setup_propagator(
    almanac: &Almanac,
) -> anyhow::Result<Propagator<SpacecraftDynamics>> {
    let earth_frame = almanac.frame_from_uid(EARTH_J2000)?;

    // Earth J2 harmonics
    let earth_sph_harm = HarmonicsMem::from_j2(EARTH_J2);
    let harmonics = Harmonics::from_stor(earth_frame, earth_sph_harm);

    // Build dynamics with J2 + point masses from Moon and Sun
    let point_masses = PointMasses::new(vec![MOON, SUN]);
    let orbital_dyn = OrbitalDynamics::new(vec![harmonics, point_masses]);
    let dynamics = SpacecraftDynamics::new(orbital_dyn);

    // Adaptive step Dormand-Prince 7-8: fast for coasting arcs,
    // accurate near gravitational wells.  Trajectory sampling uses
    // interpolation anyway, so step size doesn't affect output.
    let opts = IntegratorOptions::with_adaptive_step(
        Duration::from_seconds(0.1),    // min step
        Duration::from_seconds(300.0),  // max step (5 min)
        1e-11,                          // tolerance
        ErrorControl::RSSCartesianStep,
    );

    Ok(Propagator::new(
        dynamics,
        IntegratorMethod::DormandPrince78,
        opts,
    ))
}

/// Propagate an orbit for the given duration and return the trajectory.
pub fn propagate(
    orbit: Orbit,
    propagator: &Propagator<SpacecraftDynamics>,
    almanac: Arc<Almanac>,
    duration: Duration,
) -> anyhow::Result<Trajectory> {
    let spacecraft = Spacecraft::builder()
        .orbit(orbit)
        .mass(Mass::from_dry_mass(1.0)) // 1 kg test mass
        .build();

    let (_, trajectory) = propagator
        .with(spacecraft, almanac)
        .until_epoch_with_traj(orbit.epoch + duration)?;

    Ok(trajectory)
}

/// Sample positions from a trajectory at regular intervals.
/// Returns positions in J2000 km.
pub fn sample_trajectory(
    trajectory: &Trajectory,
    num_samples: usize,
) -> Vec<[f64; 3]> {
    let start = trajectory.first().epoch();
    let end = trajectory.last().epoch();
    let total_seconds = (end - start).to_seconds();

    (0..num_samples)
        .filter_map(|i| {
            let t = start + Duration::from_seconds(total_seconds * i as f64 / (num_samples - 1) as f64);
            trajectory.at(t).ok().map(|state| {
                let orbit = state.orbit;
                [orbit.radius_km.x, orbit.radius_km.y, orbit.radius_km.z]
            })
        })
        .collect()
}
