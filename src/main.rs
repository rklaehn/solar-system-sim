//! Solar System Simulator - Earth/Moon System
//!
//! Interactive 3D visualization of the Earth-Moon system with:
//! - Accurate celestial body positions from ANISE/JPL ephemerides
//! - Earth rotation (GMST-based)
//! - Moon orbital motion
//! - Sun direction lighting
//! - Fixed star background
//! - Lagrange point markers (EML1/EML2)
//! - Add orbital bodies with nyx-space propagation

use std::sync::Arc;

use anise::prelude::Almanac;
use hifitime::Epoch;

mod bodies;
mod constants;
mod lagrange;
mod optimizer;
mod orbit;
mod pole_optimizer;
mod stars;
mod surface_optimizer;
mod visualization;

/// Set up the ANISE almanac with planetary ephemerides.
pub fn setup_almanac() -> anyhow::Result<Arc<Almanac>> {
    let almanac = Arc::new(
        Almanac::new("data/01_planetary/pck08.pca")?
            .load("data/01_planetary/de440s.bsp")?,
    );
    Ok(almanac)
}

fn parse_epoch(s: &str) -> anyhow::Result<Epoch> {
    // Try "YYYY-MM-DD" or "YYYY-MM-DDTHH:MM:SS"
    let parts: Vec<&str> = s.split('T').collect();
    let date_parts: Vec<u32> = parts[0]
        .split('-')
        .map(|p| p.parse())
        .collect::<Result<Vec<_>, _>>()?;
    anyhow::ensure!(date_parts.len() == 3, "Expected YYYY-MM-DD");
    let (y, m, d) = (date_parts[0] as i32, date_parts[1] as u8, date_parts[2] as u8);
    let (h, min, sec) = if parts.len() > 1 {
        let time_parts: Vec<u32> = parts[1]
            .split(':')
            .map(|p| p.parse())
            .collect::<Result<Vec<_>, _>>()?;
        (
            *time_parts.first().unwrap_or(&0) as u8,
            *time_parts.get(1).unwrap_or(&0) as u8,
            *time_parts.get(2).unwrap_or(&0) as u8,
        )
    } else {
        (12, 0, 0)
    };
    Ok(Epoch::from_gregorian_utc(y, m, d, h, min, sec, 0))
}

fn main() -> anyhow::Result<()> {
    println!("Solar System Simulator - Earth/Moon System");
    println!("==========================================\n");

    let almanac = setup_almanac()?;
    let epoch = match std::env::args().nth(1) {
        Some(s) => {
            let e = parse_epoch(&s)?;
            println!("Using epoch from CLI: {e}");
            e
        }
        None => {
            println!("Usage: solar-system-sim [YYYY-MM-DD | YYYY-MM-DDTHH:MM:SS]");
            Epoch::from_gregorian_utc(2025, 6, 21, 12, 0, 0, 0)
        }
    };

    // Print initial state
    let moon_pos = bodies::moon_position(&almanac, epoch)?;
    let sun_pos = bodies::sun_position(&almanac, epoch)?;
    println!("Epoch: {epoch}");
    println!(
        "Moon position (J2000): ({:.0}, {:.0}, {:.0}) km  distance: {:.0} km",
        moon_pos.x, moon_pos.y, moon_pos.z, moon_pos.magnitude()
    );
    println!(
        "Sun direction (J2000): ({:.4}, {:.4}, {:.4})",
        sun_pos.normalized().x,
        sun_pos.normalized().y,
        sun_pos.normalized().z,
    );

    // Compute all 5 Lagrange points
    println!("\nLagrange points:");
    for id in lagrange::LagrangeId::ALL {
        let pos = lagrange::lagrange_position(id, &almanac, epoch)?;
        println!(
            "  {} (J2000): ({:.0}, {:.0}, {:.0}) km  distance: {:.0} km",
            id.label(), pos.x, pos.y, pos.z, pos.magnitude()
        );
    }

    println!("\nControls:");
    println!("  Mouse drag: Rotate camera");
    println!("  Mouse wheel: Zoom");
    println!("  Click trail: Inspect (orbital elements)");
    println!("  +/-: Speed up/slow down time");
    println!("  P: Pause/resume");
    println!("  T: Cycle trail (OFF → 1h → 2h → 6h → 1d → 3d → 7d → OFF)");
    println!("  L: Toggle Lagrange point markers");
    println!("  S: Toggle star background");
    println!("  C/V: Cycle camera target (Earth, Moon, EML1-5)");
    println!("  M: Open add body menu");
    println!("  R: Reset");
    println!("  H: Toggle side panels");
    println!("  F: Screenshot");
    println!("  G: Toggle GIF recording");

    println!("\nLaunching 3D visualization...\n");
    visualization::run(almanac, epoch)?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_moon_position() {
        let almanac = setup_almanac().expect("Failed to load almanac");
        let epoch = Epoch::from_gregorian_utc(2025, 6, 21, 12, 0, 0, 0);
        let moon = bodies::moon_position(&almanac, epoch).unwrap();
        let dist = moon.magnitude();
        println!("Moon distance: {dist:.0} km");
        // Moon should be roughly 356,000 - 407,000 km from Earth
        assert!(dist > 350_000.0 && dist < 410_000.0, "Moon distance {dist} out of range");
    }

    #[test]
    fn test_lagrange_points() {
        let almanac = setup_almanac().expect("Failed to load almanac");
        let epoch = Epoch::from_gregorian_utc(2025, 6, 21, 12, 0, 0, 0);

        let moon = bodies::moon_position(&almanac, epoch).unwrap();
        let moon_dist = moon.magnitude();

        // Check all 5 points exist and are at reasonable distances
        for id in lagrange::LagrangeId::ALL {
            let pos = lagrange::lagrange_position(id, &almanac, epoch).unwrap();
            let dist = pos.magnitude();
            println!("{}: dist={dist:.0} km", id.label());
            // All Lagrange points should be within ~2x Moon distance
            assert!(dist > moon_dist * 0.5 && dist < moon_dist * 1.5,
                "{} distance {dist:.0} out of range", id.label());
        }

        // L1 between Earth and Moon
        let eml1 = lagrange::eml1_position(&almanac, epoch).unwrap();
        assert!(eml1.magnitude() < moon_dist, "EML1 should be closer than Moon");

        // L2 beyond Moon
        let eml2 = lagrange::eml2_position(&almanac, epoch).unwrap();
        assert!(eml2.magnitude() > moon_dist, "EML2 should be farther than Moon");

        // L3 on opposite side
        let eml3 = lagrange::eml3_position(&almanac, epoch).unwrap();
        let dot = moon.x * eml3.x + moon.y * eml3.y + moon.z * eml3.z;
        assert!(dot < 0.0, "EML3 should be opposite the Moon");
    }

    #[test]
    fn test_orbit_creation() {
        let almanac = setup_almanac().expect("Failed to load almanac");
        let epoch = Epoch::from_gregorian_utc(2025, 6, 21, 12, 0, 0, 0);

        let elements = orbit::OrbitalElements::circular(
            constants::EARTH_RADIUS_KM + 400.0,
            51.6,
            0.0,
            orbit::ReferenceFrame::EarthJ2000,
        );

        let orb = orbit::create_orbit(&elements, epoch, &almanac).unwrap();
        println!("LEO orbit: SMA={:.1} km, ecc={:.4}, inc={:.2} deg",
            orb.sma_km().unwrap(), orb.ecc().unwrap(), orb.inc_deg().unwrap());

        assert!((orb.sma_km().unwrap() - 6778.137).abs() < 1.0);
    }

    #[test]
    fn test_propagation() {
        let almanac = setup_almanac().expect("Failed to load almanac");
        let epoch = Epoch::from_gregorian_utc(2025, 6, 21, 12, 0, 0, 0);

        let elements = orbit::OrbitalElements::circular(
            constants::EARTH_RADIUS_KM + 400.0,
            51.6,
            0.0,
            orbit::ReferenceFrame::EarthJ2000,
        );

        let orb = orbit::create_orbit(&elements, epoch, &almanac).unwrap();
        let propagator = orbit::setup_propagator(&almanac).unwrap();

        // Propagate for one orbital period (~92 min for 400 km LEO)
        let period = hifitime::Duration::from_seconds(5554.0);
        let traj = orbit::propagate(orb, &propagator, almanac.clone(), period).unwrap();

        let positions = orbit::sample_trajectory(&traj, 10);
        println!("Sampled {} positions", positions.len());
        assert!(positions.len() >= 8);

        // Check that the orbit stays roughly at the right altitude
        for pos in &positions {
            let r = (pos[0] * pos[0] + pos[1] * pos[1] + pos[2] * pos[2]).sqrt();
            assert!(r > 6700.0 && r < 6900.0, "Radius {r:.0} out of expected range");
        }
    }
}
