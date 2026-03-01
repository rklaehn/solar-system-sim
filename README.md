# Solar System Simulator

Interactive 3D visualization of the Earth-Moon system, built to explore
trajectories for [lunar sling launchers](https://blog.klaehn.org/blog/lunar-sling-launchers/).

The GUI is vibe coded with Claude. The trajectory propagation is solid — it uses
[nyx-space](https://nyxspace.com/) for orbit propagation and
[ANISE](https://docs.rs/anise) with JPL ephemerides (DE440s) for celestial body
positions.

## What it does

- Accurate Earth, Moon and Sun positions from JPL ephemerides
- All 5 Earth-Moon Lagrange points (EML1-5)
- Earth rotation (GMST-based)
- Adaptive step Dormand-Prince 7-8 propagator (via nyx-space)
- Launch fans from the lunar surface with configurable azimuth, elevation and speed
- Launch fans from Lagrange points
- Trail display in multiple reference frames (J2000, pulsating, synodic)
- Closest approach analysis to EML1
- Trajectory optimization (grid search + Nelder-Mead)
- Time parameterized by lunar perigee phase
- Screenshot and GIF recording

## Building

```
cargo run
```

Or with a specific epoch:

```
cargo run -- 2025-12-25
cargo run -- 2025-12-25T18:30:00
```

Requires the ANISE data files in `data/01_planetary/`:
- `pck08.pca` (planetary constants)
- `de440s.bsp` (planetary ephemerides)

## Controls

| Key | Action |
|-----|--------|
| Mouse drag | Rotate camera |
| Mouse wheel | Zoom |
| Click trail | Inspect orbital elements |
| +/- | Speed up/slow down time |
| P | Pause/resume |
| T | Cycle trail length |
| L | Toggle Lagrange point markers |
| S | Toggle star background |
| C/V | Cycle camera target |
| M | Open add body menu |
| R | Reset time |
| H | Toggle side panels |
| F | Screenshot |
| G | Toggle GIF recording |

## Getting started (no Rust experience needed)

1. Install Rust: https://rustup.rs/
2. Clone and run:
   ```
   git clone <this repo>
   cd solar-system-sim
   cargo run --release
   ```
   The first build takes a few minutes. Subsequent runs are fast.

## Dependencies

- [bevy](https://bevyengine.org/) 0.17 — rendering and ECS
- [bevy_egui](https://github.com/vladbat00/bevy_egui) — immediate mode GUI
- [nyx-space](https://nyxspace.com/) — high fidelity orbit propagation
- [anise](https://docs.rs/anise) — JPL ephemeris access
- [hifitime](https://docs.rs/hifitime) — high precision time handling

## Blog post

[Lunar Sling Launchers](https://blog.klaehn.org/blog/lunar-sling-launchers/)

## Credits

Created by [Rüdiger Klaehn](https://blog.klaehn.org/) using [Claude](https://claude.ai/).
