//! Celestial body state queries via ANISE almanac.

use anise::{
    constants::{
        celestial_objects::{MOON, SUN},
        frames::EARTH_J2000,
    },
    prelude::Almanac,
};
use hifitime::Epoch;

/// Position in km (J2000 Earth-centered inertial frame).
#[derive(Clone, Copy, Debug)]
pub struct J2000Position {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl J2000Position {
    /// Convert J2000 position to Bevy coordinates (scaled).
    /// J2000: X=vernal equinox, Z=celestial north
    /// Bevy: Y=up, so (x, y, z) -> (x, z, -y)
    pub fn to_bevy(&self, scale: f64) -> [f32; 3] {
        [
            (self.x * scale) as f32,
            (self.z * scale) as f32,
            (-self.y * scale) as f32,
        ]
    }

    /// Magnitude in km.
    pub fn magnitude(&self) -> f64 {
        (self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
    }

    /// Unit vector.
    pub fn normalized(&self) -> J2000Position {
        let mag = self.magnitude();
        J2000Position {
            x: self.x / mag,
            y: self.y / mag,
            z: self.z / mag,
        }
    }
}

/// Velocity in km/s (J2000 Earth-centered inertial frame).
#[derive(Clone, Copy, Debug)]
pub struct J2000Velocity {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

/// Query Moon position in Earth J2000 frame at the given epoch.
pub fn moon_position(almanac: &Almanac, epoch: Epoch) -> anyhow::Result<J2000Position> {
    let earth_frame = almanac.frame_from_uid(EARTH_J2000)?;
    let state = almanac.state_of(MOON, earth_frame, epoch, None)?;
    Ok(J2000Position {
        x: state.radius_km.x,
        y: state.radius_km.y,
        z: state.radius_km.z,
    })
}

/// Query Moon velocity in Earth J2000 frame at the given epoch.
pub fn moon_velocity(almanac: &Almanac, epoch: Epoch) -> anyhow::Result<J2000Velocity> {
    let earth_frame = almanac.frame_from_uid(EARTH_J2000)?;
    let state = almanac.state_of(MOON, earth_frame, epoch, None)?;
    Ok(J2000Velocity {
        x: state.velocity_km_s.x,
        y: state.velocity_km_s.y,
        z: state.velocity_km_s.z,
    })
}

/// Query Sun position in Earth J2000 frame at the given epoch.
pub fn sun_position(almanac: &Almanac, epoch: Epoch) -> anyhow::Result<J2000Position> {
    let earth_frame = almanac.frame_from_uid(EARTH_J2000)?;
    let state = almanac.state_of(SUN, earth_frame, epoch, None)?;
    Ok(J2000Position {
        x: state.radius_km.x,
        y: state.radius_km.y,
        z: state.radius_km.z,
    })
}
