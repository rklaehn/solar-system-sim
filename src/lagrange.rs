//! Earth-Moon Lagrange point computation.
//!
//! Uses the Circular Restricted Three-Body Problem (CR3BP) to find
//! all five Lagrange points (L1-L5) of the Earth-Moon system.

use crate::bodies::{self, J2000Position};
use crate::constants::{EARTH_MU, MOON_MU};
use anise::prelude::Almanac;
use hifitime::Epoch;

/// Compute the CR3BP L1 distance ratio (from Moon, toward Earth).
/// Returns the distance from Moon to L1 as a fraction of the Earth-Moon distance.
fn l1_distance_ratio(mu: f64) -> f64 {
    // L1 is between Earth and Moon.
    // Newton's method on the quintic equation for the collinear points.
    // Using Hill's approximation as initial guess: r ~ (mu/3)^(1/3)
    let mut r = (mu / 3.0).powf(1.0 / 3.0);
    for _ in 0..50 {
        let r2 = r * r;
        let r3 = r2 * r;
        let f = (1.0 - mu) / (1.0 - r).powi(2) - mu / r2 - (1.0 - r);
        let df = 2.0 * (1.0 - mu) / (1.0 - r).powi(3) + 2.0 * mu / r3 + 1.0;
        let dr = f / df;
        r -= dr;
        if dr.abs() < 1e-15 {
            break;
        }
    }
    r
}

/// Compute the CR3BP L2 distance ratio (from Moon, away from Earth).
/// Returns the distance from Moon to L2 as a fraction of the Earth-Moon distance.
fn l2_distance_ratio(mu: f64) -> f64 {
    // L2 is beyond the Moon (farther from Earth).
    let mut r = (mu / 3.0).powf(1.0 / 3.0);
    for _ in 0..50 {
        let r2 = r * r;
        let r3 = r2 * r;
        let f = (1.0 - mu) / (1.0 + r).powi(2) + mu / r2 - (1.0 + r);
        let df = -2.0 * (1.0 - mu) / (1.0 + r).powi(3) - 2.0 * mu / r3 - 1.0;
        let dr = f / df;
        r -= dr;
        if dr.abs() < 1e-15 {
            break;
        }
    }
    r
}

/// Compute the CR3BP L3 distance ratio (from Earth, opposite side from Moon).
/// Returns the distance from Earth to L3 as a fraction of the Earth-Moon distance.
/// L3 is approximately at the Moon's orbital distance, on the opposite side.
fn l3_distance_ratio(mu: f64) -> f64 {
    // L3 is on the far side of Earth from Moon.
    // In the rotating frame, L3 is at x ~ -(1 + r) where r is small.
    // The equilibrium condition: -(1-mu)/(1+r)^2 - mu/(2+r)^2 + (1-mu+r) = 0
    // Initial guess from series expansion
    let mut r = 1.0 - 7.0 * mu / 12.0;
    for _ in 0..50 {
        // Position in rotating frame: x = -(r) from barycenter (opposite side)
        // Rewritten: distance from primary = r, distance from secondary = 1+r
        let f = (1.0 - mu) / r.powi(2) - mu / (1.0 + r).powi(2) - r + mu;
        let df = -2.0 * (1.0 - mu) / r.powi(3) + 2.0 * mu / (1.0 + r).powi(3) - 1.0;
        let dr = f / df;
        r -= dr;
        if dr.abs() < 1e-15 {
            break;
        }
    }
    r
}

/// All five Lagrange points as an enum.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LagrangeId {
    L1,
    L2,
    L3,
    L4,
    L5,
}

impl LagrangeId {
    pub fn label(&self) -> &'static str {
        match self {
            LagrangeId::L1 => "EML1",
            LagrangeId::L2 => "EML2",
            LagrangeId::L3 => "EML3",
            LagrangeId::L4 => "EML4",
            LagrangeId::L5 => "EML5",
        }
    }

    pub const ALL: [LagrangeId; 5] = [
        LagrangeId::L1,
        LagrangeId::L2,
        LagrangeId::L3,
        LagrangeId::L4,
        LagrangeId::L5,
    ];
}

/// Compute any Lagrange point position in Earth J2000 frame.
pub fn lagrange_position(
    id: LagrangeId,
    almanac: &Almanac,
    epoch: Epoch,
) -> anyhow::Result<J2000Position> {
    match id {
        LagrangeId::L1 => eml1_position(almanac, epoch),
        LagrangeId::L2 => eml2_position(almanac, epoch),
        LagrangeId::L3 => eml3_position(almanac, epoch),
        LagrangeId::L4 => eml4_position(almanac, epoch),
        LagrangeId::L5 => eml5_position(almanac, epoch),
    }
}

/// Compute EML1 position in Earth J2000 frame at the given epoch.
/// L1 is between Earth and Moon.
pub fn eml1_position(almanac: &Almanac, epoch: Epoch) -> anyhow::Result<J2000Position> {
    let moon_pos = bodies::moon_position(almanac, epoch)?;
    let moon_dist = moon_pos.magnitude();
    let moon_dir = moon_pos.normalized();

    let mu = MOON_MU / (EARTH_MU + MOON_MU);
    let r_ratio = l1_distance_ratio(mu);

    // L1 is at (1 - r_ratio) * moon_distance from Earth, along the Earth-Moon line
    let l1_dist = moon_dist * (1.0 - r_ratio);

    Ok(J2000Position {
        x: moon_dir.x * l1_dist,
        y: moon_dir.y * l1_dist,
        z: moon_dir.z * l1_dist,
    })
}

/// Compute EML2 position in Earth J2000 frame at the given epoch.
/// L2 is beyond the Moon (farther from Earth).
pub fn eml2_position(almanac: &Almanac, epoch: Epoch) -> anyhow::Result<J2000Position> {
    let moon_pos = bodies::moon_position(almanac, epoch)?;
    let moon_dist = moon_pos.magnitude();
    let moon_dir = moon_pos.normalized();

    let mu = MOON_MU / (EARTH_MU + MOON_MU);
    let r_ratio = l2_distance_ratio(mu);

    // L2 is at (1 + r_ratio) * moon_distance from Earth, along the Earth-Moon line
    let l2_dist = moon_dist * (1.0 + r_ratio);

    Ok(J2000Position {
        x: moon_dir.x * l2_dist,
        y: moon_dir.y * l2_dist,
        z: moon_dir.z * l2_dist,
    })
}

/// Compute EML3 position in Earth J2000 frame at the given epoch.
/// L3 is on the opposite side of Earth from the Moon.
pub fn eml3_position(almanac: &Almanac, epoch: Epoch) -> anyhow::Result<J2000Position> {
    let moon_pos = bodies::moon_position(almanac, epoch)?;
    let moon_dist = moon_pos.magnitude();
    let moon_dir = moon_pos.normalized();

    let mu = MOON_MU / (EARTH_MU + MOON_MU);
    let r_ratio = l3_distance_ratio(mu);

    // L3 is on the opposite side of Earth from Moon
    let l3_dist = moon_dist * r_ratio;

    Ok(J2000Position {
        x: -moon_dir.x * l3_dist,
        y: -moon_dir.y * l3_dist,
        z: -moon_dir.z * l3_dist,
    })
}

/// Compute EML4 position in Earth J2000 frame at the given epoch.
/// L4 is 60 degrees ahead of Moon in its orbit (leading triangular point).
pub fn eml4_position(almanac: &Almanac, epoch: Epoch) -> anyhow::Result<J2000Position> {
    triangular_lagrange(almanac, epoch, 60.0_f64.to_radians())
}

/// Compute EML5 position in Earth J2000 frame at the given epoch.
/// L5 is 60 degrees behind Moon in its orbit (trailing triangular point).
pub fn eml5_position(almanac: &Almanac, epoch: Epoch) -> anyhow::Result<J2000Position> {
    triangular_lagrange(almanac, epoch, -60.0_f64.to_radians())
}

/// Compute a triangular Lagrange point by rotating the Moon position
/// by `angle` radians around the orbital angular momentum vector.
fn triangular_lagrange(
    almanac: &Almanac,
    epoch: Epoch,
    angle: f64,
) -> anyhow::Result<J2000Position> {
    let moon_pos = bodies::moon_position(almanac, epoch)?;
    let moon_vel = bodies::moon_velocity(almanac, epoch)?;

    // Angular momentum direction: h = r x v (normalized)
    let hx = moon_pos.y * moon_vel.z - moon_pos.z * moon_vel.y;
    let hy = moon_pos.z * moon_vel.x - moon_pos.x * moon_vel.z;
    let hz = moon_pos.x * moon_vel.y - moon_pos.y * moon_vel.x;
    let h_mag = (hx * hx + hy * hy + hz * hz).sqrt();
    let (ux, uy, uz) = (hx / h_mag, hy / h_mag, hz / h_mag);

    // Rodrigues' rotation formula: rotate moon_pos by `angle` around h-hat
    let cos_a = angle.cos();
    let sin_a = angle.sin();
    let dot = ux * moon_pos.x + uy * moon_pos.y + uz * moon_pos.z;

    // v_rot = v*cos(a) + (k x v)*sin(a) + k*(k.v)*(1-cos(a))
    let cx = uy * moon_pos.z - uz * moon_pos.y;
    let cy = uz * moon_pos.x - ux * moon_pos.z;
    let cz = ux * moon_pos.y - uy * moon_pos.x;

    Ok(J2000Position {
        x: moon_pos.x * cos_a + cx * sin_a + ux * dot * (1.0 - cos_a),
        y: moon_pos.y * cos_a + cy * sin_a + uy * dot * (1.0 - cos_a),
        z: moon_pos.z * cos_a + cz * sin_a + uz * dot * (1.0 - cos_a),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constants::EARTH_MOON_MASS_RATIO;

    fn almanac() -> std::sync::Arc<anise::prelude::Almanac> {
        crate::setup_almanac().expect("Failed to load almanac")
    }

    fn epoch() -> Epoch {
        Epoch::from_gregorian_utc(2025, 6, 21, 12, 0, 0, 0)
    }

    #[test]
    fn test_l1_distance() {
        let mu = EARTH_MOON_MASS_RATIO;
        let r = l1_distance_ratio(mu);
        let l1_from_moon_km = r * 384400.0;
        println!("L1 distance ratio: {r:.6}");
        println!("L1 from Moon: {l1_from_moon_km:.0} km");
        assert!(l1_from_moon_km > 50000.0 && l1_from_moon_km < 70000.0);
    }

    #[test]
    fn test_l2_distance() {
        let mu = EARTH_MOON_MASS_RATIO;
        let r = l2_distance_ratio(mu);
        let l2_from_moon_km = r * 384400.0;
        println!("L2 distance ratio: {r:.6}");
        println!("L2 from Moon: {l2_from_moon_km:.0} km");
        assert!(l2_from_moon_km > 55000.0 && l2_from_moon_km < 75000.0);
    }

    #[test]
    fn test_l3_distance() {
        let mu = EARTH_MOON_MASS_RATIO;
        let r = l3_distance_ratio(mu);
        // L3 should be approximately at the Moon's orbital distance
        let l3_from_earth_km = r * 384400.0;
        println!("L3 distance ratio: {r:.6}");
        println!("L3 from Earth: {l3_from_earth_km:.0} km");
        assert!(l3_from_earth_km > 370000.0 && l3_from_earth_km < 400000.0);
    }

    #[test]
    fn test_l4_l5_equilateral() {
        let a = almanac();
        let e = epoch();

        let moon = bodies::moon_position(&a, e).unwrap();
        let l4 = eml4_position(&a, e).unwrap();
        let l5 = eml5_position(&a, e).unwrap();

        let moon_dist = moon.magnitude();
        let l4_dist = l4.magnitude();
        let l5_dist = l5.magnitude();

        println!("Moon dist: {moon_dist:.0} km");
        println!("L4 dist: {l4_dist:.0} km");
        println!("L5 dist: {l5_dist:.0} km");

        // L4 and L5 should be at roughly the same distance as the Moon
        assert!((l4_dist - moon_dist).abs() / moon_dist < 0.01);
        assert!((l5_dist - moon_dist).abs() / moon_dist < 0.01);

        // The angle between Moon and L4/L5 should be ~60 degrees
        let dot_l4 = (moon.x * l4.x + moon.y * l4.y + moon.z * l4.z) / (moon_dist * l4_dist);
        let dot_l5 = (moon.x * l5.x + moon.y * l5.y + moon.z * l5.z) / (moon_dist * l5_dist);
        let angle_l4 = dot_l4.acos().to_degrees();
        let angle_l5 = dot_l5.acos().to_degrees();
        println!("Angle Moon-L4: {angle_l4:.1} deg");
        println!("Angle Moon-L5: {angle_l5:.1} deg");
        assert!((angle_l4 - 60.0).abs() < 1.0);
        assert!((angle_l5 - 60.0).abs() < 1.0);
    }

    #[test]
    fn test_all_lagrange_points() {
        let a = almanac();
        let e = epoch();

        for id in LagrangeId::ALL {
            let pos = lagrange_position(id, &a, e).unwrap();
            println!("{}: ({:.0}, {:.0}, {:.0}) km  dist={:.0} km",
                id.label(), pos.x, pos.y, pos.z, pos.magnitude());
        }
    }
}
