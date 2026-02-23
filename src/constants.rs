//! Physical constants for the solar system simulation.

// === Earth ===
/// Earth's equatorial radius (km)
pub const EARTH_RADIUS_KM: f64 = 6378.137;
/// Earth's gravitational parameter (km^3/s^2)
pub const EARTH_MU: f64 = 398600.4418;
/// Earth's J2 oblateness coefficient
pub const EARTH_J2: f64 = 1.08262668e-3;
/// Earth's rotation rate (rad/s) - one rotation per sidereal day
pub const EARTH_ROTATION_RATE: f64 = 7.2921159e-5;

// === Moon ===
/// Moon's mean radius (km)
pub const MOON_RADIUS_KM: f64 = 1737.4;
/// Moon's gravitational parameter (km^3/s^2)
pub const MOON_MU: f64 = 4902.799;
/// Moon's mean distance from Earth (km)
pub const MOON_SEMI_MAJOR_AXIS_KM: f64 = 384400.0;

// === Earth-Moon System ===
/// Mass ratio: Moon / (Earth + Moon)
pub const EARTH_MOON_MASS_RATIO: f64 = MOON_MU / (EARTH_MU + MOON_MU);

// === Visualization ===
/// Scale factor: km to Bevy render units
pub const VIS_SCALE: f32 = 0.001;
/// Scale factor as f64 for computation
pub const VIS_SCALE_F64: f64 = 0.001;
