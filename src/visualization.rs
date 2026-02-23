//! 3D visualization of the Earth/Moon system using Bevy.

use std::sync::{Arc, Mutex};

use anise::prelude::Almanac;
use bevy::{
    input::{
        keyboard::{Key, KeyboardInput},
        mouse::{MouseMotion, MouseWheel},
    },
    prelude::*,
    render::view::screenshot::{save_to_disk, Screenshot, ScreenshotCaptured},
    window::{CursorIcon, SystemCursorIcon},
};
use bevy_egui::{
    EguiContexts, EguiPlugin, EguiPrimaryContextPass, egui,
    input::{egui_wants_any_keyboard_input, egui_wants_any_pointer_input},
};
use hifitime::Epoch;
use nyx_space::cosmic::Spacecraft;
use nyx_space::State;
use rayon::prelude::*;

use crate::{
    bodies,
    constants::*,
    lagrange,
    orbit::{self, OrbitalElements, ReferenceFrame},
    stars::{magnitude_to_brightness, magnitude_to_size, BRIGHT_STARS},
};

/// Compute Greenwich Mean Sidereal Time (GMST) in radians for a given epoch.
fn compute_gmst(epoch: Epoch) -> f64 {
    const J2000_JD: f64 = 2451545.0;
    let jd = epoch.to_jde_utc_days();
    let d = jd - J2000_JD;
    let gmst_deg = 280.46061837 + 360.98564736629 * d;
    (gmst_deg.to_radians()).rem_euclid(std::f64::consts::TAU)
}

/// Find the next `count` lunar perigees starting from `start`.
/// Uses coarse sampling + golden-section refinement on the Moon distance.
fn find_lunar_perigees(almanac: &Almanac, start: Epoch, count: usize) -> Vec<(Epoch, f64)> {
    use hifitime::Duration;

    let step = Duration::from_hours(1.0);
    // ~28 days per anomalistic month, search a bit more than count * 28 days
    let search_dur = Duration::from_days((count as f64) * 29.0 + 5.0);

    let dist = |e: Epoch| -> f64 {
        bodies::moon_position(almanac, e)
            .map(|p| p.magnitude())
            .unwrap_or(f64::MAX)
    };

    // Coarse pass: find local minima in hourly samples
    let mut perigees = Vec::new();
    let mut prev2 = dist(start);
    let mut prev1 = dist(start + step);
    let mut t = start + step + step;
    while t < start + search_dur && perigees.len() < count {
        let cur = dist(t);
        if prev1 < prev2 && prev1 < cur {
            // prev1 is a local minimum — refine with golden section
            let mut a = t - step - step;
            let mut b = t;
            let gr = 0.381966011250105; // (3 - sqrt(5)) / 2
            for _ in 0..50 {
                let c = a + gr * (b - a);
                let d = b - gr * (b - a);
                if dist(c) < dist(d) {
                    b = d;
                } else {
                    a = c;
                }
            }
            let mid = a + (b - a) * 0.5;
            perigees.push((mid, dist(mid)));
        }
        prev2 = prev1;
        prev1 = cur;
        t = t + step;
    }
    perigees.truncate(count);
    perigees
}

// === Resources ===

/// A point in the scene that the camera can be positioned at or look at.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CameraEndpoint {
    Free,
    Earth,
    Moon,
    MoonEML1,
    EML1,
    EML2,
    EML3,
    EML4,
    EML5,
    Satellite(usize),
}

impl CameraEndpoint {
    /// All fixed endpoints (for dropdown menus).
    const FIXED: &[CameraEndpoint] = &[
        CameraEndpoint::Free,
        CameraEndpoint::Earth,
        CameraEndpoint::Moon,
        CameraEndpoint::MoonEML1,
        CameraEndpoint::EML1,
        CameraEndpoint::EML2,
        CameraEndpoint::EML3,
        CameraEndpoint::EML4,
        CameraEndpoint::EML5,
    ];

    fn display_name(&self, user_bodies: &UserBodies) -> String {
        match self {
            CameraEndpoint::Free => "Free".into(),
            CameraEndpoint::Earth => "Earth".into(),
            CameraEndpoint::Moon => "Moon".into(),
            CameraEndpoint::MoonEML1 => "Moon-EML1".into(),
            CameraEndpoint::EML1 => "EML1".into(),
            CameraEndpoint::EML2 => "EML2".into(),
            CameraEndpoint::EML3 => "EML3".into(),
            CameraEndpoint::EML4 => "EML4".into(),
            CameraEndpoint::EML5 => "EML5".into(),
            CameraEndpoint::Satellite(i) => {
                user_bodies.bodies.get(*i)
                    .map(|b| b.name.clone())
                    .unwrap_or_else(|| format!("Sat {i}"))
            }
        }
    }

    /// Cycle through non-Free targets (for C/V keyboard shortcuts).
    fn cycle(current: CameraEndpoint, num_sats: usize, forward: bool) -> CameraEndpoint {
        let mut targets: Vec<CameraEndpoint> = Self::FIXED.iter()
            .filter(|e| **e != CameraEndpoint::Free)
            .copied()
            .collect();
        for i in 0..num_sats {
            targets.push(CameraEndpoint::Satellite(i));
        }
        let idx = targets.iter().position(|e| *e == current).unwrap_or(0);
        if forward {
            targets[(idx + 1) % targets.len()]
        } else {
            targets[(idx + targets.len() - 1) % targets.len()]
        }
    }

    /// Get the Bevy world-space position. Returns `None` for `Free`.
    fn position(&self, almanac: &Almanac, epoch: Epoch, user_bodies: &UserBodies) -> Option<Vec3> {
        match self {
            CameraEndpoint::Free => None,
            CameraEndpoint::Earth => Some(Vec3::ZERO),
            CameraEndpoint::Moon => {
                bodies::moon_position(almanac, epoch)
                    .map(|p| Vec3::from_array(p.to_bevy(VIS_SCALE_F64)))
                    .ok()
            }
            CameraEndpoint::MoonEML1 => {
                let moon = bodies::moon_position(almanac, epoch)
                    .map(|p| Vec3::from_array(p.to_bevy(VIS_SCALE_F64)))
                    .unwrap_or(Vec3::ZERO);
                let eml1 = lagrange_bevy_pos(lagrange::LagrangeId::L1, almanac, epoch);
                Some((moon + eml1) * 0.5)
            }
            CameraEndpoint::EML1 => Some(lagrange_bevy_pos(lagrange::LagrangeId::L1, almanac, epoch)),
            CameraEndpoint::EML2 => Some(lagrange_bevy_pos(lagrange::LagrangeId::L2, almanac, epoch)),
            CameraEndpoint::EML3 => Some(lagrange_bevy_pos(lagrange::LagrangeId::L3, almanac, epoch)),
            CameraEndpoint::EML4 => Some(lagrange_bevy_pos(lagrange::LagrangeId::L4, almanac, epoch)),
            CameraEndpoint::EML5 => Some(lagrange_bevy_pos(lagrange::LagrangeId::L5, almanac, epoch)),
            CameraEndpoint::Satellite(i) => {
                user_bodies.bodies.get(*i).and_then(|body| {
                    body.spacecraft.as_ref().map(|sc| {
                        let pos = sc.orbit.radius_km;
                        let j2000 = bodies::J2000Position { x: pos.x, y: pos.y, z: pos.z };
                        Vec3::from_array(j2000.to_bevy(VIS_SCALE_F64))
                    })
                })
            }
        }
    }
}

fn lagrange_bevy_pos(id: lagrange::LagrangeId, almanac: &Almanac, epoch: Epoch) -> Vec3 {
    lagrange::lagrange_position(id, almanac, epoch)
        .map(|p| Vec3::from_array(p.to_bevy(VIS_SCALE_F64)))
        .unwrap_or(Vec3::ZERO)
}

/// Camera up-vector / Y-axis choice.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum CameraUp {
    /// Mouse-controlled orbit rotation (default Y up).
    #[default]
    Free,
    /// J2000 Z axis → Bevy Y.
    EarthNorth,
    /// Normal to the Earth-Moon orbital plane (from Moon's angular momentum).
    MoonOrbitNormal,
    /// Ecliptic north pole.
    EclipticNorth,
}

impl CameraUp {
    const ALL: &[CameraUp] = &[
        CameraUp::Free,
        CameraUp::EarthNorth,
        CameraUp::MoonOrbitNormal,
        CameraUp::EclipticNorth,
    ];

    fn label(&self) -> &'static str {
        match self {
            CameraUp::Free => "Free",
            CameraUp::EarthNorth => "Earth North",
            CameraUp::MoonOrbitNormal => "Moon Orbit",
            CameraUp::EclipticNorth => "Ecliptic North",
        }
    }

    /// Resolve to a Bevy world-space up vector. Returns `None` for `Free`.
    fn resolve(&self, almanac: &Almanac, epoch: Epoch) -> Option<Vec3> {
        match self {
            CameraUp::Free => None,
            CameraUp::EarthNorth => Some(Vec3::Y),
            CameraUp::MoonOrbitNormal => {
                use anise::constants::{celestial_objects::MOON, frames::EARTH_J2000};
                let frame = almanac.frame_from_uid(EARTH_J2000).ok()?;
                let moon = almanac.state_of(MOON, frame, epoch, None).ok()?;
                let r = moon.radius_km;
                let v = moon.velocity_km_s;
                let h = [
                    r.y * v.z - r.z * v.y,
                    r.z * v.x - r.x * v.z,
                    r.x * v.y - r.y * v.x,
                ];
                let h_mag = (h[0] * h[0] + h[1] * h[1] + h[2] * h[2]).sqrt();
                // J2000 → Bevy: (x,y,z) → (x,z,-y)
                Some(Vec3::new(
                    (h[0] / h_mag) as f32,
                    (h[2] / h_mag) as f32,
                    (-h[1] / h_mag) as f32,
                ))
            }
            CameraUp::EclipticNorth => {
                let obliquity: f64 = 23.4392911_f64.to_radians();
                // Ecliptic north in J2000: (0, -sin(ε), cos(ε))
                // J2000 → Bevy: (x,y,z) → (x,z,-y)
                Some(Vec3::new(
                    0.0,
                    obliquity.cos() as f32,
                    obliquity.sin() as f32,
                ))
            }
        }
    }
}

/// Compute camera offset from a target in spherical coordinates relative to an up vector.
fn orbit_offset(distance: f32, azimuth: f32, elevation: f32, up: Vec3) -> Vec3 {
    let up = up.normalize();
    let right = if up.cross(Vec3::Z).length() > 0.01 {
        up.cross(Vec3::Z).normalize()
    } else {
        up.cross(Vec3::X).normalize()
    };
    let forward = right.cross(up);
    let x = azimuth.cos() * elevation.cos();
    let y = elevation.sin();
    let z = azimuth.sin() * elevation.cos();
    (right * x + up * y + forward * z) * distance
}

/// Reference frame for trail display.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum TrailFrame {
    /// Earth-centered J2000 inertial — trails shown as-is.
    #[default]
    J2000,
    /// Moon-centered inertial — subtract Moon position at each epoch.
    MoonCentered,
    /// Co-rotating — rotate trail points from moon_dir(t) to moon_dir(now).
    Synodic,
    /// Pulsating synodic — rotate + scale by R(now)/R(t). L1-L5 truly stationary.
    Pulsating,
}

/// How to position the camera along the From→To line in fixed mode.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum CameraOrigin {
    #[default]
    Fraction,
    DistToTo,
    DistFromFrom,
}

#[derive(Resource)]
pub struct SimulationState {
    pub epoch: Epoch,
    pub speed: f64,
    pub paused: bool,
    pub camera_from: CameraEndpoint,
    pub camera_to: CameraEndpoint,
    pub camera_up: CameraUp,
    pub camera_distance: f32,
    pub camera_azimuth: f32,
    pub camera_elevation: f32,
    pub camera_origin: CameraOrigin,
    pub camera_origin_value: f32,
    pub perspective: bool,
    pub show_lagrange: bool,
    pub show_stars: bool,
    pub trail_duration: f64,
    pub trail_frame: TrailFrame,
    pub main_panel_open: bool,
    pub opt_panel_open: bool,
}

impl SimulationState {
    /// Reset to default viewing state.
    fn reset(&mut self, initial_epoch: Epoch) {
        *self = Self {
            epoch: initial_epoch,
            ..Default::default()
        };
    }

    /// The body the camera effectively orbits/looks at.
    /// In orbit mode (one Free), returns the non-Free body.
    /// In fixed mode (both bodies), returns camera_to.
    fn orbit_center(&self) -> CameraEndpoint {
        match (self.camera_from, self.camera_to) {
            (CameraEndpoint::Free, body) => body,
            (body, CameraEndpoint::Free) => body,
            (_, to) => to,
        }
    }
}

impl Default for SimulationState {
    fn default() -> Self {
        Self {
            epoch: Epoch::from_gregorian_utc(2025, 6, 21, 12, 0, 0, 0),
            speed: 3600.0, // 1 hour per second (good for seeing Moon move)
            paused: true,
            camera_from: CameraEndpoint::Free,
            camera_to: CameraEndpoint::EML1,
            camera_up: CameraUp::default(),
            camera_distance: 500.0, // See Earth-Moon system
            camera_azimuth: 0.0,
            camera_elevation: 1.5, // top-down: perpendicular to orbital plane
            camera_origin: CameraOrigin::default(),
            camera_origin_value: 0.0,
            perspective: true,
            show_lagrange: true,
            show_stars: false,
            trail_duration: 86400.0, // 1 day default
            trail_frame: TrailFrame::Pulsating,
            main_panel_open: true,
            opt_panel_open: true,
        }
    }
}

#[derive(Resource)]
pub struct SpaceResources {
    pub almanac: Arc<Almanac>,
    pub initial_epoch: Epoch,
    pub initial_gmst: f64,
    pub propagator: Option<nyx_space::propagators::Propagator<nyx_space::dynamics::SpacecraftDynamics>>,
}

/// Stored trajectories for celestial bodies (Earth, Moon) in J2000 frame.
/// Used for frame transformations and future trail rendering.
#[derive(Resource, Default)]
pub struct BodyTrails {
    /// (epoch, J2000 pos km, J2000 vel km/s) — Earth is always origin but stored for uniformity.
    pub earth: Vec<(Epoch, [f64; 3], [f64; 3])>,
    pub moon: Vec<(Epoch, [f64; 3], [f64; 3])>,
}

/// A user-added orbital body with its propagated trajectory.
#[derive(Resource, Default)]
pub struct UserBodies {
    pub bodies: Vec<UserBody>,
}

pub struct UserBody {
    pub name: String,
    pub spacecraft: Option<Spacecraft>,
    pub trail: Vec<(Epoch, [f64; 3], [f64; 3])>, // (epoch, J2000 pos km, J2000 vel km/s)
    pub color: Color,
    pub spawn_real_time: f64,
}

// === Components ===

#[derive(Component)]
struct Earth;

#[derive(Component)]
struct Moon;

#[derive(Component)]
struct SunLight;

#[derive(Component)]
struct MainCamera;

#[derive(Component)]
struct StarField;

#[derive(Component)]
struct LagrangeMarker {
    point: lagrange::LagrangeId,
}

#[derive(Component)]
pub(crate) struct UserBodyMarker {
    pub(crate) index: usize,
}

/// Reference body for trajectory analysis.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
enum AnalysisRef {
    #[default]
    Earth,
    Moon,
}

impl AnalysisRef {
    fn radius_km(&self) -> f64 {
        match self {
            AnalysisRef::Earth => EARTH_RADIUS_KM,
            AnalysisRef::Moon => MOON_RADIUS_KM,
        }
    }

    fn mu(&self) -> f64 {
        match self {
            AnalysisRef::Earth => EARTH_MU,
            AnalysisRef::Moon => MOON_MU,
        }
    }

    fn frame(&self) -> orbit::ReferenceFrame {
        match self {
            AnalysisRef::Earth => ReferenceFrame::EarthJ2000,
            AnalysisRef::Moon => ReferenceFrame::MoonJ2000,
        }
    }
}

/// Result of a closest approach search.
struct ClosestApproach {
    epoch: Epoch,
    distance_km: f64,
    pos_km: [f64; 3],
    vel_km_s: [f64; 3],
}

/// How to display orbital state in the analysis panel.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
enum OrbitalDisplay {
    #[default]
    Keplerian,
    Euclidean,
}

/// Resource for trajectory analysis panel.
#[derive(Resource, Default)]
struct AnalysisState {
    /// Which satellite is selected (index into UserBodies.bodies).
    selected_body: Option<usize>,
    /// Reference body for orbital elements.
    reference: AnalysisRef,
    /// Display mode for orbital parameters.
    display: OrbitalDisplay,
    /// Closest approach result from trail scan.
    closest_approach: Option<ClosestApproach>,
}

/// Tracks mouse press position and drag distance to distinguish clicks from drags.
#[derive(Resource, Default)]
struct ClickTracker {
    drag_distance: f32,
}

/// Tracks whether the cursor is currently hovering over a trail.
#[derive(Resource, Default)]
struct TrailHoverState {
    hovering: bool,
}

/// Cached lunar perigee epochs computed from initial_epoch.
#[derive(Resource)]
struct PerigeeCache {
    /// Perigee epochs and distances, computed from SpaceResources::initial_epoch.
    perigees: Vec<(Epoch, f64)>,
    /// Current phase value shown in the UI.
    phase: f64,
}

impl Default for PerigeeCache {
    fn default() -> Self {
        Self {
            perigees: Vec::new(),
            phase: 0.0,
        }
    }
}

impl PerigeeCache {
    /// Epoch for a given phase. Integer part = perigee index, fractional = lerp to next.
    fn epoch_for_phase(&self, phase: f64) -> Option<Epoch> {
        let idx = phase.floor() as usize;
        let frac = phase - phase.floor();
        if idx + 1 < self.perigees.len() {
            let a = self.perigees[idx].0;
            let b = self.perigees[idx + 1].0;
            Some(a + (b - a) * frac)
        } else if idx < self.perigees.len() {
            Some(self.perigees[idx].0)
        } else {
            None
        }
    }

    /// Distance (km) at perigee for integer part of phase.
    fn perigee_dist(&self, phase: f64) -> Option<f64> {
        let idx = phase.floor() as usize;
        self.perigees.get(idx).map(|(_, d)| *d)
    }
}

/// State for screenshot and GIF recording.
#[derive(Resource)]
struct RecordingState {
    recording: bool,
    frame_counter: u32,
    /// Capture every N render frames (~10fps at 60fps).
    frame_skip: u32,
    /// Collected frames (downsampled RGBA).
    frames: Vec<Vec<u8>>,
    gif_width: u32,
    gif_height: u32,
    max_frames: usize,
    /// Status message shown in UI (shared with encoding thread).
    status: Arc<Mutex<String>>,
    screenshot_counter: u32,
}

impl Default for RecordingState {
    fn default() -> Self {
        Self {
            recording: false,
            frame_counter: 0,
            frame_skip: 6,
            frames: Vec::new(),
            gif_width: 0,
            gif_height: 0,
            max_frames: 3000,
            status: Arc::new(Mutex::new(String::new())),
            screenshot_counter: 0,
        }
    }
}

/// Downsample RGBA image by 2× (nearest-neighbor).
fn downsample_2x(data: &[u8], width: u32, height: u32) -> (Vec<u8>, u32, u32) {
    let nw = width / 2;
    let nh = height / 2;
    let mut out = vec![0u8; (nw * nh * 4) as usize];
    for y in 0..nh {
        for x in 0..nw {
            let src = ((y * 2 * width + x * 2) * 4) as usize;
            let dst = ((y * nw + x) * 4) as usize;
            out[dst..dst + 4].copy_from_slice(&data[src..src + 4]);
        }
    }
    (out, nw, nh)
}

/// Encode collected frames as an animated GIF.
/// Quantization is parallelized with rayon; only sequential GIF writing is serial.
fn save_recording_gif(
    frames: Vec<Vec<u8>>,
    width: u32,
    height: u32,
    frame_skip: u32,
    status: Arc<Mutex<String>>,
) {
    use image::{codecs::gif::{GifEncoder, Repeat}, Frame, Delay, RgbaImage};
    use rayon::prelude::*;
    use std::io::BufWriter;

    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    // Format as ISO-ish timestamp: YYYYMMDD-HHMMSS
    let secs_per_day = 86400u64;
    let days = now / secs_per_day;
    let rem = now % secs_per_day;
    let (h, m, s) = (rem / 3600, (rem % 3600) / 60, rem % 60);
    // Days since epoch → date (simplified, good enough for filenames)
    let (year, month, day) = {
        let mut y = 1970i32;
        let mut d = days as i32;
        loop {
            let yd = if y % 4 == 0 && (y % 100 != 0 || y % 400 == 0) { 366 } else { 365 };
            if d < yd { break; }
            d -= yd;
            y += 1;
        }
        let leap = y % 4 == 0 && (y % 100 != 0 || y % 400 == 0);
        let mdays = [31, if leap {29} else {28}, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
        let mut mo = 0u32;
        for &md in &mdays {
            if d < md { break; }
            d -= md;
            mo += 1;
        }
        (y, mo + 1, d + 1)
    };
    let path = format!("recording-{year:04}{month:02}{day:02}T{h:02}{m:02}{s:02}.gif");
    let file = match std::fs::File::create(&path) {
        Ok(f) => f,
        Err(e) => {
            *status.lock().unwrap() = format!("Error: {e}");
            return;
        }
    };
    let mut encoder = GifEncoder::new_with_speed(BufWriter::new(file), 30);
    let _ = encoder.set_repeat(Repeat::Infinite);
    // delay per frame in ms: frame_skip / 60fps * 1000
    let delay_ms = (frame_skip as u32 * 1000 / 60).max(10);

    *status.lock().unwrap() = format!("Quantizing {} frames...", frames.len());

    // Parallel quantization: convert RGBA → image::Frame (does NeuQuant per frame)
    let quantized: Vec<Option<Frame>> = frames
        .into_par_iter()
        .map(|data| {
            RgbaImage::from_raw(width, height, data)
                .map(|rgba| Frame::from_parts(rgba, 0, 0, Delay::from_numer_denom_ms(delay_ms, 1)))
        })
        .collect();

    let total = quantized.len();
    *status.lock().unwrap() = format!("Writing {total} frames...");

    for (i, frame) in quantized.into_iter().enumerate() {
        if let Some(frame) = frame {
            if encoder.encode_frame(frame).is_err() {
                *status.lock().unwrap() = "Encode error".into();
                return;
            }
        }
        if (i + 1) % 50 == 0 {
            *status.lock().unwrap() = format!("Writing {}/{}...", i + 1, total);
        }
    }
    *status.lock().unwrap() = format!("Saved {path}");
}

/// System: spawn screenshot entities while recording, and auto-save when limit is hit.
fn capture_recording_frames(
    mut recording: ResMut<RecordingState>,
    mut commands: Commands,
) {
    // Auto-save when limit was reached (recording stopped by observer but frames not yet saved)
    if !recording.recording && !recording.frames.is_empty() {
        let frames = std::mem::take(&mut recording.frames);
        let w = recording.gif_width;
        let h = recording.gif_height;
        let skip = recording.frame_skip;
        let status = recording.status.clone();
        *status.lock().unwrap() = format!("Encoding {} frames (limit reached)...", frames.len());
        std::thread::spawn(move || {
            save_recording_gif(frames, w, h, skip, status);
        });
        return;
    }
    if !recording.recording {
        return;
    }
    recording.frame_counter += 1;
    if recording.frame_counter % recording.frame_skip == 0 {
        commands.spawn(Screenshot::primary_window());
    }
}

/// Global observer: collect screenshot data into recording buffer.
fn collect_recording_frame(
    trigger: On<ScreenshotCaptured>,
    mut recording: ResMut<RecordingState>,
) {
    if !recording.recording {
        return;
    }
    let img = trigger.event();
    let w = img.width();
    let h = img.height();
    if let Some(ref raw) = img.data {
        recording.gif_width = w;
        recording.gif_height = h;
        recording.frames.push(raw.clone());
        if recording.frames.len() >= recording.max_frames {
            recording.recording = false;
        }
    }
}

// === App Entry ===

pub fn run(almanac: Arc<Almanac>, epoch: Epoch) -> anyhow::Result<()> {
    let initial_gmst = compute_gmst(epoch);
    let propagator = orbit::setup_propagator(&almanac).ok();
    let perigees = find_lunar_perigees(&almanac, epoch, 20);
    let perigee_cache = PerigeeCache {
        perigees,
        phase: 0.0,
    };

    // Compute initial azimuth so screen-up is perpendicular to Earth-Moon axis
    let initial_azimuth = bodies::moon_position(&almanac, epoch)
        .map(|m| {
            let b = m.to_bevy(1.0);
            // Moon direction angle in Bevy XZ plane; offset 90° so moon line is horizontal
            b[2].atan2(b[0]) + std::f32::consts::FRAC_PI_2
        })
        .unwrap_or(0.0);

    App::new()
        .add_plugins(DefaultPlugins
            .set(WindowPlugin {
                primary_window: Some(Window {
                    title: "Solar System Simulator - Earth/Moon System".into(),
                    resolution: (1400_u32, 900_u32).into(),
                    ..default()
                }),
                ..default()
            })
            .set(bevy::log::LogPlugin {
                filter: std::env::var("RUST_LOG")
                    .unwrap_or_else(|_| "nyx_space=warn,solar_system_sim=info".into()),
                ..default()
            })
        )
        .add_plugins(EguiPlugin::default())
        .insert_resource(SpaceResources {
            almanac,
            initial_epoch: epoch,
            initial_gmst,
            propagator,
        })
        .insert_resource(SimulationState {
            epoch,
            camera_azimuth: initial_azimuth,
            ..default()
        })
        .insert_resource(UserBodies::default())
        .insert_resource(BodyTrails::default())
        .insert_resource(AnalysisState::default())
        .insert_resource(AddBodyMenuState::default())
        .insert_resource(ClickTracker::default())
        .insert_resource(TrailHoverState::default())
        .insert_resource(crate::optimizer::OptimizerState::default())
        .insert_resource(crate::pole_optimizer::PoleOptimizerState::default())
        .insert_resource(crate::surface_optimizer::SurfaceOptimizerState::default())
        .insert_resource(ClearColor(Color::BLACK))
        .insert_resource(perigee_cache)
        .insert_resource(RecordingState::default())
        .add_observer(collect_recording_frame)
        .add_systems(Startup, setup_scene)
        .add_systems(
            EguiPrimaryContextPass,
            (egui_side_panel, optimizers_panel, egui_bottom_panel).chain(),
        )
        .add_systems(
            Update,
            (
                (
                    handle_keyboard_input.run_if(not(egui_wants_any_keyboard_input)),
                    (
                        handle_mouse_input.run_if(not(egui_wants_any_pointer_input)),
                        handle_trail_click.run_if(not(egui_wants_any_pointer_input)),
                        update_trail_hover.run_if(not(egui_wants_any_pointer_input)),
                    ).chain(),
                ),
                update_simulation_time,
                (
                    update_earth_rotation,
                    update_moon_position,
                    update_sun_direction,
                    update_lagrange_markers,
                    update_user_bodies,
                ),
                (draw_trails, crate::optimizer::draw_optimizer_trails, crate::pole_optimizer::draw_pole_optimizer_trails, crate::surface_optimizer::draw_surface_optimizer_trails, crate::surface_optimizer::draw_surface_map, update_camera, update_projection, sync_star_visibility),
                (crate::optimizer::poll_optimizer, crate::optimizer::adopt_optimizer_results, crate::pole_optimizer::poll_pole_optimizer, crate::pole_optimizer::adopt_pole_optimizer_results, crate::surface_optimizer::poll_surface_optimizer),
                capture_recording_frames,
            )
                .chain(),
        )
        .run();

    Ok(())
}

// === Setup ===

fn setup_scene(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    space: Res<SpaceResources>,
    state: Res<SimulationState>,
    asset_server: Res<AssetServer>,
) {
    let earth_radius_scaled = EARTH_RADIUS_KM as f32 * VIS_SCALE;
    let moon_radius_scaled = MOON_RADIUS_KM as f32 * VIS_SCALE;

    // --- Earth ---
    let earth_texture: Handle<Image> = asset_server.load("textures/earth.jpg");
    let initial_gmst = space.initial_gmst as f32;
    commands.spawn((
        Mesh3d(meshes.add(Sphere::new(earth_radius_scaled).mesh().uv(64, 32))),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color_texture: Some(earth_texture),
            perceptual_roughness: 0.9,
            reflectance: 0.1,
            ..default()
        })),
        Transform::from_xyz(0.0, 0.0, 0.0)
            .with_rotation(Quat::from_rotation_y(initial_gmst)),
        Earth,
    ));

    // --- Moon ---
    let moon_texture: Handle<Image> = asset_server.load("textures/moon.jpg");
    let moon_pos = bodies::moon_position(&space.almanac, state.epoch)
        .unwrap_or(bodies::J2000Position { x: MOON_SEMI_MAJOR_AXIS_KM, y: 0.0, z: 0.0 });
    let moon_bevy = moon_pos.to_bevy(VIS_SCALE_F64);

    commands.spawn((
        Mesh3d(meshes.add(Sphere::new(moon_radius_scaled).mesh().uv(32, 16))),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color_texture: Some(moon_texture),
            base_color: Color::srgb(0.8, 0.8, 0.8),
            emissive: LinearRgba::new(0.3, 0.3, 0.3, 1.0),
            perceptual_roughness: 0.95,
            reflectance: 0.05,
            ..default()
        })),
        Transform::from_xyz(moon_bevy[0], moon_bevy[1], moon_bevy[2]),
        Moon,
    ));

    // --- All 5 Lagrange point markers (initially hidden) ---
    let marker_mesh = meshes.add(Sphere::new(1.5).mesh().uv(32, 16));
    let lagrange_mat = materials.add(StandardMaterial {
        base_color: Color::srgba(0.2, 1.0, 0.6, 0.1),
        emissive: LinearRgba::new(0.04, 0.2, 0.12, 1.0),
        alpha_mode: AlphaMode::Blend,
        unlit: true,
        ..default()
    });
    for id in lagrange::LagrangeId::ALL {
        commands.spawn((
            Mesh3d(marker_mesh.clone()),
            MeshMaterial3d(lagrange_mat.clone()),
            Transform::from_xyz(0.0, 0.0, 0.0),
            if state.show_lagrange { Visibility::Visible } else { Visibility::Hidden },
            LagrangeMarker { point: id },
        ));
    }

    // --- Sun light ---
    commands.spawn((
        DirectionalLight {
            illuminance: 40_000.0,
            shadows_enabled: true,
            ..default()
        },
        Transform::default().looking_at(Vec3::NEG_X, Vec3::Y),
        SunLight,
    ));

    commands.insert_resource(AmbientLight {
        color: Color::WHITE,
        brightness: 200.0,
        affects_lightmapped_meshes: false,
    });

    // --- Star field ---
    // Stars are spheres on a far shell. They need to be large enough to span
    // at least a few pixels at that distance, so we scale radius proportionally.
    let star_distance = 5000.0;
    for star in BRIGHT_STARS {
        let (x, y, z) = star.to_bevy_direction();
        let pos = Vec3::new(x, y, z) * star_distance;
        // Size needs to be visible at star_distance — angular size matters.
        // magnitude_to_size returns 0.2..1.0; multiply by star_distance fraction
        // so the brightest stars subtend ~1 degree.
        let size = magnitude_to_size(star.magnitude) * star_distance * 0.005;
        let brightness = magnitude_to_brightness(star.magnitude);
        let (r, g, b) = star.spectral.to_color();

        commands.spawn((
            Mesh3d(meshes.add(Sphere::new(size).mesh().uv(8, 4))),
            MeshMaterial3d(materials.add(StandardMaterial {
                base_color: Color::WHITE,
                emissive: LinearRgba::new(
                    50.0 * brightness * r,
                    50.0 * brightness * g,
                    50.0 * brightness * b,
                    1.0,
                ),
                unlit: true,
                ..default()
            })),
            Transform::from_translation(pos),
            StarField,
        ));
    }

    // --- Camera ---
    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(0.0, 100.0, 500.0).looking_at(Vec3::ZERO, Vec3::Y),
        MainCamera,
    ));

}

// === Menu items ===

/// A menu entry in the Add Body panel.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum AddBodyEntry {
    Orbit(usize),
    LagrangeGrid(lagrange::LagrangeId),
    LagrangeFan(lagrange::LagrangeId),
    LunarFan,
}

impl AddBodyEntry {
    fn label(&self) -> String {
        match self {
            AddBodyEntry::Orbit(i) => ORBIT_PRESETS.get(*i).unwrap_or(&"?").to_string(),
            AddBodyEntry::LagrangeGrid(id) => format!("{} Grid", id.label()),
            AddBodyEntry::LagrangeFan(id) => format!("{} Fan", id.label()),
            AddBodyEntry::LunarFan => "Lunar Surface Fan".into(),
        }
    }

    fn has_fan_config(&self) -> bool {
        matches!(self, AddBodyEntry::LagrangeFan(_) | AddBodyEntry::LunarFan)
    }

    fn has_lunar_config(&self) -> bool {
        matches!(self, AddBodyEntry::LunarFan)
    }

    fn default_speed_km_s(&self) -> f64 {
        match self {
            AddBodyEntry::LagrangeFan(_) => 0.1,
            AddBodyEntry::LunarFan => 2.335,
            _ => 0.0,
        }
    }

    const ALL: &[AddBodyEntry] = &[
        AddBodyEntry::Orbit(0), AddBodyEntry::Orbit(1), AddBodyEntry::Orbit(2),
        AddBodyEntry::Orbit(3), AddBodyEntry::Orbit(4), AddBodyEntry::Orbit(5),
        AddBodyEntry::LagrangeGrid(lagrange::LagrangeId::L1),
        AddBodyEntry::LagrangeFan(lagrange::LagrangeId::L1),
        AddBodyEntry::LagrangeGrid(lagrange::LagrangeId::L2),
        AddBodyEntry::LagrangeFan(lagrange::LagrangeId::L2),
        AddBodyEntry::LagrangeGrid(lagrange::LagrangeId::L3),
        AddBodyEntry::LagrangeFan(lagrange::LagrangeId::L3),
        AddBodyEntry::LagrangeGrid(lagrange::LagrangeId::L4),
        AddBodyEntry::LagrangeFan(lagrange::LagrangeId::L4),
        AddBodyEntry::LagrangeGrid(lagrange::LagrangeId::L5),
        AddBodyEntry::LagrangeFan(lagrange::LagrangeId::L5),
        AddBodyEntry::LunarFan,
    ];
}

/// Persistent state for the Add Body menu.
#[derive(Resource)]
struct AddBodyMenuState {
    selected: usize,
    fan_count: u32,
    fan_speed_km_s: f64,
    lunar_az_deg: f64,
    lunar_el_deg: f64,
}

impl Default for AddBodyMenuState {
    fn default() -> Self {
        Self {
            selected: 0,
            fan_count: 36,
            fan_speed_km_s: 2.335,
            lunar_az_deg: 150.0,
            lunar_el_deg: 0.0,
        }
    }
}

const ORBIT_PRESETS: &[&str] = &[
    "LEO (400 km, 51.6°)",
    "GTO (250 x 35786 km)",
    "Lunar Transfer",
    "Low Lunar Orbit (100 km)",
    "EML1 Halo",
    "EML2 Halo",
];

fn menu_orbital_elements(index: usize, almanac: &Almanac, epoch: Epoch) -> Option<(String, OrbitalElements)> {
    match index {
        0 => Some(("LEO".into(), OrbitalElements {
            semi_major_axis_km: EARTH_RADIUS_KM + 400.0,
            eccentricity: 0.0,
            inclination_deg: 51.6,
            raan_deg: 0.0,
            arg_periapsis_deg: 0.0,
            true_anomaly_deg: 0.0,
            frame: ReferenceFrame::EarthJ2000,
        })),
        1 => Some(("GTO".into(), OrbitalElements {
            semi_major_axis_km: (EARTH_RADIUS_KM + 250.0 + EARTH_RADIUS_KM + 35786.0) / 2.0,
            eccentricity: 1.0 - (EARTH_RADIUS_KM + 250.0) / ((EARTH_RADIUS_KM + 250.0 + EARTH_RADIUS_KM + 35786.0) / 2.0),
            inclination_deg: 28.5,
            raan_deg: 0.0,
            arg_periapsis_deg: 180.0,
            true_anomaly_deg: 0.0,
            frame: ReferenceFrame::EarthJ2000,
        })),
        2 => {
            // Lunar transfer: rough Hohmann-like orbit
            // Perigee at ~200 km, apogee near Moon distance
            let rp = EARTH_RADIUS_KM + 200.0;
            let ra = MOON_SEMI_MAJOR_AXIS_KM * 0.95;
            let sma = (rp + ra) / 2.0;
            let ecc = (ra - rp) / (ra + rp);
            Some(("Lunar Transfer".into(), OrbitalElements {
                semi_major_axis_km: sma,
                eccentricity: ecc,
                inclination_deg: 28.5,
                raan_deg: 0.0,
                arg_periapsis_deg: 0.0,
                true_anomaly_deg: 0.0,
                frame: ReferenceFrame::EarthJ2000,
            }))
        }
        3 => Some(("LLO".into(), OrbitalElements {
            semi_major_axis_km: MOON_RADIUS_KM + 100.0,
            eccentricity: 0.0,
            inclination_deg: 90.0,
            raan_deg: 0.0,
            arg_periapsis_deg: 0.0,
            true_anomaly_deg: 0.0,
            frame: ReferenceFrame::MoonJ2000,
        })),
        4 => {
            // EML1 halo - approximate as orbit around EML1 position
            // Use Earth-centered orbit that reaches near EML1
            if let Ok(eml1) = lagrange::eml1_position(almanac, epoch) {
                let dist = eml1.magnitude();
                Some(("EML1 Orbit".into(), OrbitalElements {
                    semi_major_axis_km: dist,
                    eccentricity: 0.02,
                    inclination_deg: 5.0,
                    raan_deg: 0.0,
                    arg_periapsis_deg: 0.0,
                    true_anomaly_deg: 0.0,
                    frame: ReferenceFrame::EarthJ2000,
                }))
            } else {
                None
            }
        }
        5 => {
            // EML2 - approximate
            if let Ok(eml2) = lagrange::eml2_position(almanac, epoch) {
                let dist = eml2.magnitude();
                Some(("EML2 Orbit".into(), OrbitalElements {
                    semi_major_axis_km: dist,
                    eccentricity: 0.02,
                    inclination_deg: 5.0,
                    raan_deg: 0.0,
                    arg_periapsis_deg: 0.0,
                    true_anomaly_deg: 0.0,
                    frame: ReferenceFrame::EarthJ2000,
                }))
            } else {
                None
            }
        }
        _ => None,
    }
}

/// Local frame and co-rotation velocity at a Lagrange point.
struct LagrangeLocalFrame {
    pos: bodies::J2000Position,
    x_axis: [f64; 3],
    y_axis: [f64; 3],
    z_axis: [f64; 3],
    base_vel: [f64; 3],
}

/// Compute local coordinate frame at a Lagrange point.
/// X = towards Earth, Y = orbit normal, Z = X × Y.
/// Base velocity = co-rotation ω × r.
fn lagrange_local_frame(
    id: lagrange::LagrangeId,
    almanac: &Almanac,
    epoch: Epoch,
) -> Option<LagrangeLocalFrame> {
    let moon_pos = bodies::moon_position(almanac, epoch).ok()?;
    let moon_vel = bodies::moon_velocity(almanac, epoch).ok()?;
    let lp = lagrange::lagrange_position(id, almanac, epoch).ok()?;

    let lp_mag = lp.magnitude();
    let x_axis = [-lp.x / lp_mag, -lp.y / lp_mag, -lp.z / lp_mag];

    let hx = moon_pos.y * moon_vel.z - moon_pos.z * moon_vel.y;
    let hy = moon_pos.z * moon_vel.x - moon_pos.x * moon_vel.z;
    let hz = moon_pos.x * moon_vel.y - moon_pos.y * moon_vel.x;
    let h_mag = (hx * hx + hy * hy + hz * hz).sqrt();
    let y_axis = [hx / h_mag, hy / h_mag, hz / h_mag];

    let z_axis = [
        x_axis[1] * y_axis[2] - x_axis[2] * y_axis[1],
        x_axis[2] * y_axis[0] - x_axis[0] * y_axis[2],
        x_axis[0] * y_axis[1] - x_axis[1] * y_axis[0],
    ];

    let moon_dist = moon_pos.magnitude();
    let omega_mag = h_mag / (moon_dist * moon_dist);
    let omega = [y_axis[0] * omega_mag, y_axis[1] * omega_mag, y_axis[2] * omega_mag];
    let base_vel = [
        omega[1] * lp.z - omega[2] * lp.y,
        omega[2] * lp.x - omega[0] * lp.z,
        omega[0] * lp.y - omega[1] * lp.x,
    ];

    Some(LagrangeLocalFrame { pos: lp, x_axis, y_axis, z_axis, base_vel })
}

/// Create a spacecraft at a Lagrange point with a local-frame velocity perturbation.
fn create_spacecraft_at_lagrange(
    frame: &LagrangeLocalFrame,
    dv_local: [f64; 3],
    epoch: Epoch,
    almanac: &Almanac,
) -> Option<Spacecraft> {
    let [dvx, dvy, dvz] = dv_local;
    let vx = frame.base_vel[0] + dvx * frame.x_axis[0] + dvy * frame.y_axis[0] + dvz * frame.z_axis[0];
    let vy = frame.base_vel[1] + dvx * frame.x_axis[1] + dvy * frame.y_axis[1] + dvz * frame.z_axis[1];
    let vz = frame.base_vel[2] + dvx * frame.x_axis[2] + dvy * frame.y_axis[2] + dvz * frame.z_axis[2];

    let orb = orbit::create_orbit_cartesian(
        [frame.pos.x, frame.pos.y, frame.pos.z], [vx, vy, vz],
        epoch, almanac, ReferenceFrame::EarthJ2000,
    ).ok()?;
    Some(nyx_space::cosmic::Spacecraft::builder()
        .orbit(orb)
        .mass(nyx_space::cosmic::Mass::from_dry_mass(1.0))
        .build())
}

/// Create 6 spacecraft with +/- dv in local X, Y, Z at a Lagrange point.
fn create_lagrange_grid(
    id: lagrange::LagrangeId,
    almanac: &Almanac,
    epoch: Epoch,
    dv_km_s: f64,
) -> Vec<(String, Spacecraft)> {
    let frame = match lagrange_local_frame(id, almanac, epoch) {
        Some(f) => f,
        None => return Vec::new(),
    };
    let label = id.label();
    let perturbations: [(&str, [f64; 3]); 6] = [
        ("+X", [dv_km_s, 0.0, 0.0]),
        ("-X", [-dv_km_s, 0.0, 0.0]),
        ("+Y", [0.0, dv_km_s, 0.0]),
        ("-Y", [0.0, -dv_km_s, 0.0]),
        ("+Z", [0.0, 0.0, dv_km_s]),
        ("-Z", [0.0, 0.0, -dv_km_s]),
    ];
    let mut result = Vec::new();
    for (dir, dv) in perturbations {
        if let Some(sc) = create_spacecraft_at_lagrange(&frame, dv, epoch, almanac) {
            result.push((format!("{label} {dir}"), sc));
        }
    }
    result
}

/// Create 36 spacecraft with velocity perturbations in a circle in the local XZ plane
/// at a Lagrange point.
fn create_lagrange_circle(
    id: lagrange::LagrangeId,
    almanac: &Almanac,
    epoch: Epoch,
    dv_km_s: f64,
    count: u32,
) -> Vec<(String, Spacecraft)> {
    let frame = match lagrange_local_frame(id, almanac, epoch) {
        Some(f) => f,
        None => return Vec::new(),
    };
    let label = id.label();

    let mut result = Vec::new();
    for i in 0..count {
        let angle_deg = i as f64 * (360.0 / count as f64);
        let angle_rad = angle_deg.to_radians();
        let dv = [dv_km_s * angle_rad.cos(), 0.0, dv_km_s * angle_rad.sin()];
        if let Some(sc) = create_spacecraft_at_lagrange(&frame, dv, epoch, almanac) {
            result.push((format!("{label} {angle_deg:.0}°"), sc));
        }
    }
    result
}

/// Create 36 spacecraft on the Moon's surface at a given surface normal direction,
/// with lunar escape velocity fanned in the tangent plane.
/// `surface_normal` must be a unit vector in J2000 (outward from Moon center).
/// 0° in the fan = direction of `ref_dir` projected onto the tangent plane.
fn create_lunar_surface_fan(
    label: &str,
    surface_normal: [f64; 3],
    ref_dir: [f64; 3],
    moon_pos: &bodies::J2000Position,
    moon_vel: &bodies::J2000Velocity,
    epoch: Epoch,
    almanac: &Almanac,
    speed_km_s: f64,
    count: u32,
) -> Vec<(String, Spacecraft)> {
    // Position: Moon center + MOON_RADIUS_KM along surface normal
    let pos = [
        moon_pos.x + surface_normal[0] * MOON_RADIUS_KM,
        moon_pos.y + surface_normal[1] * MOON_RADIUS_KM,
        moon_pos.z + surface_normal[2] * MOON_RADIUS_KM,
    ];

    // X_local = ref_dir projected onto tangent plane (perpendicular to surface_normal)
    let dot = ref_dir[0] * surface_normal[0] + ref_dir[1] * surface_normal[1] + ref_dir[2] * surface_normal[2];
    let x_raw = [
        ref_dir[0] - dot * surface_normal[0],
        ref_dir[1] - dot * surface_normal[1],
        ref_dir[2] - dot * surface_normal[2],
    ];
    let x_mag = (x_raw[0].powi(2) + x_raw[1].powi(2) + x_raw[2].powi(2)).sqrt();
    if x_mag < 1e-10 {
        return Vec::new(); // ref_dir parallel to surface normal, degenerate
    }
    let x_axis = [x_raw[0] / x_mag, x_raw[1] / x_mag, x_raw[2] / x_mag];
    // Y_local = surface_normal × X
    let y_axis = [
        surface_normal[1] * x_axis[2] - surface_normal[2] * x_axis[1],
        surface_normal[2] * x_axis[0] - surface_normal[0] * x_axis[2],
        surface_normal[0] * x_axis[1] - surface_normal[1] * x_axis[0],
    ];

    let base_vel = [moon_vel.x, moon_vel.y, moon_vel.z];

    let mut result = Vec::new();
    for i in 0..count {
        let angle_deg = i as f64 * (360.0 / count as f64);
        let angle_rad = angle_deg.to_radians();
        let dvx = speed_km_s * angle_rad.cos();
        let dvy = speed_km_s * angle_rad.sin();
        let vel = [
            base_vel[0] + dvx * x_axis[0] + dvy * y_axis[0],
            base_vel[1] + dvx * x_axis[1] + dvy * y_axis[1],
            base_vel[2] + dvx * x_axis[2] + dvy * y_axis[2],
        ];
        if let Ok(orb) = orbit::create_orbit_cartesian(
            pos, vel, epoch, almanac, ReferenceFrame::EarthJ2000,
        ) {
            let sc = nyx_space::cosmic::Spacecraft::builder()
                .orbit(orb)
                .mass(nyx_space::cosmic::Mass::from_dry_mass(1.0))
                .build();
            result.push((format!("{label} {angle_deg:.0}°"), sc));
        }
    }
    result
}

/// Create fan on Moon surface at given azimuth/elevation.
/// Azimuth: 0° = away from Earth, 90° = prograde.
/// Elevation: 0° = equator, 90° = north pole, -90° = south pole.
/// Fan reference direction for 0° in the fan: toward north pole (projected onto tangent).
fn create_lunar_fan(
    almanac: &Almanac, epoch: Epoch,
    az_deg: f64, el_deg: f64,
    speed_km_s: f64, count: u32,
) -> Vec<(String, Spacecraft)> {
    let (Ok(moon_pos), Ok(moon_vel)) = (
        bodies::moon_position(almanac, epoch),
        bodies::moon_velocity(almanac, epoch),
    ) else { return Vec::new() };

    // Moon-centered frame axes in J2000:
    // away = Moon position direction (away from Earth)
    let moon_mag = moon_pos.magnitude();
    let away = [moon_pos.x / moon_mag, moon_pos.y / moon_mag, moon_pos.z / moon_mag];
    // north = orbital angular momentum (approximate Moon north pole)
    let hx = moon_pos.y * moon_vel.z - moon_pos.z * moon_vel.y;
    let hy = moon_pos.z * moon_vel.x - moon_pos.x * moon_vel.z;
    let hz = moon_pos.x * moon_vel.y - moon_pos.y * moon_vel.x;
    let h_mag = (hx * hx + hy * hy + hz * hz).sqrt();
    let north = [hx / h_mag, hy / h_mag, hz / h_mag];
    // prograde = north × away
    let prograde = [
        north[1] * away[2] - north[2] * away[1],
        north[2] * away[0] - north[0] * away[2],
        north[0] * away[1] - north[1] * away[0],
    ];

    // Standard selenographic: lon=0 toward Earth, lon=90 prograde
    let toward = [-away[0], -away[1], -away[2]];
    let lon = az_deg.to_radians();
    let lat = el_deg.to_radians();
    let cos_lat = lat.cos();
    let sin_lat = lat.sin();
    let cos_lon = lon.cos();
    let sin_lon = lon.sin();
    let normal = [
        cos_lat * (cos_lon * toward[0] + sin_lon * prograde[0]) + sin_lat * north[0],
        cos_lat * (cos_lon * toward[1] + sin_lon * prograde[1]) + sin_lat * north[1],
        cos_lat * (cos_lon * toward[2] + sin_lon * prograde[2]) + sin_lat * north[2],
    ];

    let label = format!("Moon({az_deg:.0},{el_deg:.0})");
    // Pick ref_dir as whichever of "north" or "away" is less aligned with the surface normal,
    // so the tangent plane projection doesn't collapse at poles (normal≈north) or far side (normal≈away).
    let cross_mag_sq = |a: [f64; 3], b: [f64; 3]| {
        let cx = a[1] * b[2] - a[2] * b[1];
        let cy = a[2] * b[0] - a[0] * b[2];
        let cz = a[0] * b[1] - a[1] * b[0];
        cx * cx + cy * cy + cz * cz
    };
    let ref_dir = if cross_mag_sq(normal, north) > cross_mag_sq(normal, away) {
        north
    } else {
        away
    };
    create_lunar_surface_fan(&label, normal, ref_dir, &moon_pos, &moon_vel, epoch, almanac, speed_km_s, count)
}

pub(crate) static BODY_COLORS: &[Color] = &[
    Color::srgb(0.2, 0.8, 1.0),  // cyan
    Color::srgb(1.0, 0.4, 0.2),  // orange
    Color::srgb(0.4, 1.0, 0.4),  // green
    Color::srgb(1.0, 0.8, 0.2),  // yellow
    Color::srgb(0.8, 0.4, 1.0),  // purple
    Color::srgb(1.0, 0.2, 0.6),  // pink
];

// === Shared helpers ===

/// Trail duration options for combo box.
const TRAIL_OPTIONS: &[(f64, &str)] = &[
    (0.0, "OFF"),
    (3600.0, "1h"),
    (7200.0, "2h"),
    (21600.0, "6h"),
    (86400.0, "1d"),
    (172800.0, "2d"),
    (259200.0, "3d"),
    (604800.0, "7d"),
];

fn format_speed(speed: f64, paused: bool) -> String {
    if paused {
        "PAUSED".to_string()
    } else if speed >= 86400.0 {
        format!("{:.1} d/s", speed / 86400.0)
    } else if speed >= 3600.0 {
        format!("{:.1} hr/s", speed / 3600.0)
    } else {
        format!("{}x", speed as i64)
    }
}

fn format_trail(duration: f64) -> String {
    if duration <= 0.0 {
        "OFF".to_string()
    } else if duration >= 86400.0 {
        format!("{:.0}d", duration / 86400.0)
    } else if duration >= 3600.0 {
        format!("{:.0}h", duration / 3600.0)
    } else {
        format!("{:.0}m", duration / 60.0)
    }
}

fn format_epoch_short(epoch: &Epoch) -> String {
    let s = format!("{}", epoch);
    if let Some(dot_pos) = s.find('.') {
        let end = (dot_pos + 4).min(s.len());
        let rest = &s[end..];
        let skip = rest.find(|c: char| !c.is_ascii_digit()).unwrap_or(rest.len());
        format!("{}{}", &s[..end], &rest[skip..])
    } else {
        s
    }
}

/// Show Keplerian elements in an egui Grid.
fn show_keplerian(ui: &mut egui::Ui, orb: &nyx_space::cosmic::Orbit, body_radius_km: f64, mu: f64) {
    if let (Ok(sma), Ok(ecc), Ok(inc), Ok(raan), Ok(aop), Ok(ta)) = (
        orb.sma_km(), orb.ecc(), orb.inc_deg(), orb.raan_deg(), orb.aop_deg(), orb.ta_deg(),
    ) {
        egui::Grid::new(ui.next_auto_id()).show(ui, |ui| {
            ui.label("SMA:"); ui.label(format!("{sma:.1} km")); ui.end_row();
            ui.label("Ecc:"); ui.label(format!("{ecc:.6}")); ui.end_row();
            ui.label("Inc:"); ui.label(format!("{inc:.2} deg")); ui.end_row();
            ui.label("RAAN:"); ui.label(format!("{raan:.2} deg")); ui.end_row();
            ui.label("AoP:"); ui.label(format!("{aop:.2} deg")); ui.end_row();
            ui.label("TA:"); ui.label(format!("{ta:.2} deg")); ui.end_row();
        });

        let r = (orb.radius_km.x.powi(2) + orb.radius_km.y.powi(2) + orb.radius_km.z.powi(2)).sqrt();
        let alt = r - body_radius_km;
        ui.label(format!("Alt: {alt:.1} km"));

        if ecc < 1.0 && sma > 0.0 {
            let period_s = std::f64::consts::TAU * (sma.powi(3) / mu).sqrt();
            if period_s < 86400.0 {
                ui.label(format!("Period: {:.1} min", period_s / 60.0));
            } else {
                ui.label(format!("Period: {:.2} days", period_s / 86400.0));
            }
        } else {
            ui.label("Period: (hyperbolic)");
        }
    } else {
        ui.label("(elements undefined)");
    }
}

/// Show Euclidean state (position + velocity) in an egui Grid.
fn show_euclidean(ui: &mut egui::Ui, pos: [f64; 3], vel: [f64; 3], body_radius_km: f64) {
    egui::Grid::new(ui.next_auto_id()).show(ui, |ui| {
        ui.label("x:"); ui.label(format!("{:.3} km", pos[0])); ui.end_row();
        ui.label("y:"); ui.label(format!("{:.3} km", pos[1])); ui.end_row();
        ui.label("z:"); ui.label(format!("{:.3} km", pos[2])); ui.end_row();
        ui.label("vx:"); ui.label(format!("{:.6} km/s", vel[0])); ui.end_row();
        ui.label("vy:"); ui.label(format!("{:.6} km/s", vel[1])); ui.end_row();
        ui.label("vz:"); ui.label(format!("{:.6} km/s", vel[2])); ui.end_row();
    });
    let r = (pos[0].powi(2) + pos[1].powi(2) + pos[2].powi(2)).sqrt();
    let v = (vel[0].powi(2) + vel[1].powi(2) + vel[2].powi(2)).sqrt();
    let alt = r - body_radius_km;
    ui.label(format!("r={r:.1} km  v={v:.4} km/s  alt={alt:.1} km"));
}

/// Compute relative state (pos, vel) of a satellite w.r.t. a reference body.
/// Returns (relative_pos_km, relative_vel_km_s).
fn relative_state(
    pos_km: [f64; 3],
    vel_km_s: [f64; 3],
    reference: AnalysisRef,
    almanac: &Almanac,
    epoch: Epoch,
) -> Option<([f64; 3], [f64; 3])> {
    match reference {
        AnalysisRef::Earth => Some((pos_km, vel_km_s)),
        AnalysisRef::Moon => {
            let mp = bodies::moon_position(almanac, epoch).ok()?;
            let mv = bodies::moon_velocity(almanac, epoch).ok()?;
            Some((
                [pos_km[0] - mp.x, pos_km[1] - mp.y, pos_km[2] - mp.z],
                [vel_km_s[0] - mv.x, vel_km_s[1] - mv.y, vel_km_s[2] - mv.z],
            ))
        }
    }
}

/// Scan trail to find the closest approach to a reference body.
fn find_closest_approach(
    trail: &[(Epoch, [f64; 3], [f64; 3])],
    reference: AnalysisRef,
    body_trails: &BodyTrails,
) -> Option<ClosestApproach> {
    if trail.is_empty() {
        return None;
    }

    let ref_trail = match reference {
        AnalysisRef::Earth => &body_trails.earth,
        AnalysisRef::Moon => &body_trails.moon,
    };

    let mut best_idx = 0;
    let mut best_dist = f64::MAX;
    let mut ref_search_start = 0;

    for (i, (epoch, pos, _)) in trail.iter().enumerate() {
        // Find matching epoch in reference trail (both sorted by time)
        let ref_pos = if ref_trail.is_empty() {
            [0.0, 0.0, 0.0]
        } else {
            // Linear scan forward since both trails are time-sorted
            while ref_search_start + 1 < ref_trail.len()
                && ref_trail[ref_search_start + 1].0 <= *epoch
            {
                ref_search_start += 1;
            }
            ref_trail[ref_search_start].1
        };

        let dx = pos[0] - ref_pos[0];
        let dy = pos[1] - ref_pos[1];
        let dz = pos[2] - ref_pos[2];
        let dist = (dx * dx + dy * dy + dz * dz).sqrt();
        if dist < best_dist {
            best_dist = dist;
            best_idx = i;
        }
    }

    let (epoch, pos_km, vel_km_s) = trail[best_idx];
    Some(ClosestApproach {
        epoch,
        distance_km: best_dist,
        pos_km,
        vel_km_s,
    })
}

/// Add bodies for a given menu entry with fan parameters.
/// Add bodies for a given menu entry with fan parameters.
fn add_body_entry(
    entry: AddBodyEntry,
    menu: &AddBodyMenuState,
    state: &mut SimulationState,
    space: &SpaceResources,
    user_bodies: &mut UserBodies,
    time: &Time,
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    materials: &mut Assets<StandardMaterial>,
) {
    let mut to_add: Vec<(String, Option<Spacecraft>)> = Vec::new();

    match entry {
        AddBodyEntry::LagrangeGrid(id) => {
            for (name, sc) in create_lagrange_grid(id, &space.almanac, state.epoch, menu.fan_speed_km_s) {
                to_add.push((name, Some(sc)));
            }
        }
        AddBodyEntry::LagrangeFan(id) => {
            for (name, sc) in create_lagrange_circle(id, &space.almanac, state.epoch, menu.fan_speed_km_s, menu.fan_count) {
                to_add.push((name, Some(sc)));
            }
        }
        AddBodyEntry::Orbit(index) => {
            if let Some((name, elements)) =
                menu_orbital_elements(index, &space.almanac, state.epoch)
            {
                let spacecraft = orbit::create_orbit(&elements, state.epoch, &space.almanac)
                    .ok()
                    .map(|orb| {
                        nyx_space::cosmic::Spacecraft::builder()
                            .orbit(orb)
                            .mass(nyx_space::cosmic::Mass::from_dry_mass(1.0))
                            .build()
                    });
                to_add.push((name, spacecraft));
            }
        }
        AddBodyEntry::LunarFan => {
            for (name, sc) in create_lunar_fan(
                &space.almanac, state.epoch,
                menu.lunar_az_deg, menu.lunar_el_deg,
                menu.fan_speed_km_s, menu.fan_count,
            ) {
                to_add.push((name, Some(sc)));
            }
        }
    }

    let marker_mesh = meshes.add(Sphere::new(0.05).mesh().uv(16, 8));
    for (name, spacecraft) in to_add {
        let color = BODY_COLORS[user_bodies.bodies.len() % BODY_COLORS.len()];
        let idx = user_bodies.bodies.len();

        let initial_trail = if let Some(ref sc) = spacecraft {
            let pos = sc.orbit.radius_km;
            let vel = sc.orbit.velocity_km_s;
            vec![(state.epoch, [pos.x, pos.y, pos.z], [vel.x, vel.y, vel.z])]
        } else {
            Vec::new()
        };

        user_bodies.bodies.push(UserBody {
            name,
            spacecraft,
            trail: initial_trail,
            color,
            spawn_real_time: time.elapsed_secs_f64(),
        });

        let body_mat = materials.add(StandardMaterial {
            base_color: color,
            emissive: LinearRgba::from(color) * 2.0,
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

// === Update Systems ===

fn handle_keyboard_input(
    mut keyboard_events: MessageReader<KeyboardInput>,
    mut state: ResMut<SimulationState>,
    user_bodies: Res<UserBodies>,
    space: Res<SpaceResources>,
    mut lagrange_query: Query<&mut Visibility, (With<LagrangeMarker>, Without<StarField>)>,
    mut star_query: Query<&mut Visibility, (With<StarField>, Without<LagrangeMarker>)>,
    mut recording: ResMut<RecordingState>,
    mut commands: Commands,
) {
    for event in keyboard_events.read() {
        if !event.state.is_pressed() {
            continue;
        }

        if let Key::Character(c) = &event.logical_key {
            match c.as_str() {
                "+" | "=" => state.speed *= 2.0,
                "-" => state.speed = (state.speed / 2.0).max(1.0),
                "p" => state.paused = !state.paused,
                "l" => {
                    state.show_lagrange = !state.show_lagrange;
                    let vis = if state.show_lagrange {
                        Visibility::Visible
                    } else {
                        Visibility::Hidden
                    };
                    for mut v in lagrange_query.iter_mut() {
                        *v = vis;
                    }
                }
                "s" => {
                    state.show_stars = !state.show_stars;
                    let vis = if state.show_stars && state.perspective {
                        Visibility::Visible
                    } else {
                        Visibility::Hidden
                    };
                    for mut v in star_query.iter_mut() {
                        *v = vis;
                    }
                }
                "m" | "M" => {
                }
                "c" => {
                    state.camera_to = CameraEndpoint::cycle(
                        state.camera_to, user_bodies.bodies.len(), true,
                    );
                }
                "v" => {
                    state.camera_to = CameraEndpoint::cycle(
                        state.camera_to, user_bodies.bodies.len(), false,
                    );
                }
                "t" => {
                    let cur = state.trail_duration;
                    let next = TRAIL_OPTIONS.iter()
                        .find(|&&(s, _)| s > cur + 1.0)
                        .map(|&(s, _)| s)
                        .unwrap_or(0.0);
                    state.trail_duration = next;
                }
                "r" => {
                    if user_bodies.bodies.is_empty() {
                        state.reset(space.initial_epoch);
                    }
                }
                "h" => {
                    state.main_panel_open = !state.main_panel_open;
                    state.opt_panel_open = !state.opt_panel_open;
                }
                "g" => {
                    // Toggle GIF recording
                    if recording.recording {
                        recording.recording = false;
                        let frames = std::mem::take(&mut recording.frames);
                        let w = recording.gif_width;
                        let h = recording.gif_height;
                        let skip = recording.frame_skip;
                        let status = recording.status.clone();
                        if !frames.is_empty() {
                            *status.lock().unwrap() = format!("Encoding {} frames...", frames.len());
                            std::thread::spawn(move || {
                                save_recording_gif(frames, w, h, skip, status);
                            });
                        }
                    } else {
                        recording.recording = true;
                        recording.frames.clear();
                        recording.frame_counter = 0;
                        *recording.status.lock().unwrap() = String::new();
                    }
                }
                "f" => {
                    // Screenshot
                    let n = recording.screenshot_counter;
                    recording.screenshot_counter += 1;
                    commands
                        .spawn(Screenshot::primary_window())
                        .observe(save_to_disk(format!("screenshot_{n}.png")));
                }
                _ => {}
            }
        }

    }
}

fn handle_mouse_input(
    mut state: ResMut<SimulationState>,
    mouse_button: Res<ButtonInput<MouseButton>>,
    mut mouse_motion: MessageReader<MouseMotion>,
    mut mouse_wheel: MessageReader<MouseWheel>,
    mut click_tracker: ResMut<ClickTracker>,
) {
    if mouse_button.just_pressed(MouseButton::Left) {
        click_tracker.drag_distance = 0.0;
    }

    // Orbit controls only when at least one endpoint is Free
    let is_orbit = state.camera_from == CameraEndpoint::Free
        || state.camera_to == CameraEndpoint::Free;

    if mouse_button.pressed(MouseButton::Left) {
        for event in mouse_motion.read() {
            click_tracker.drag_distance += event.delta.length();
            if is_orbit {
                state.camera_azimuth += event.delta.x * 0.005;
                state.camera_elevation =
                    (state.camera_elevation + event.delta.y * 0.005).clamp(-1.5, 1.5);
            }
        }
    } else {
        mouse_motion.clear();
    }

    for event in mouse_wheel.read() {
        if is_orbit {
            let zoom_factor = 1.0 - event.y * 0.05;
            state.camera_distance = (state.camera_distance * zoom_factor).clamp(5.0, 3000.0);
        }
    }
}

fn update_simulation_time(time: Res<Time>, mut state: ResMut<SimulationState>) {
    if state.paused {
        return;
    }
    let dt = time.delta_secs_f64();
    let speed = state.speed;
    state.epoch += hifitime::Duration::from_seconds(dt * speed);
}

fn update_earth_rotation(
    state: Res<SimulationState>,
    space: Res<SpaceResources>,
    mut query: Query<&mut Transform, With<Earth>>,
) {
    let elapsed = (state.epoch - space.initial_epoch).to_seconds();
    let current_gmst = space.initial_gmst + EARTH_ROTATION_RATE * elapsed;

    for mut transform in query.iter_mut() {
        transform.rotation = Quat::from_rotation_y(current_gmst as f32);
    }
}

fn update_moon_position(
    state: Res<SimulationState>,
    space: Res<SpaceResources>,
    mut query: Query<&mut Transform, With<Moon>>,
) {
    let Ok(moon_pos) = bodies::moon_position(&space.almanac, state.epoch) else {
        return;
    };
    let bevy_pos = moon_pos.to_bevy(VIS_SCALE_F64);

    for mut transform in query.iter_mut() {
        transform.translation = Vec3::from_array(bevy_pos);
    }
}

fn update_sun_direction(
    state: Res<SimulationState>,
    space: Res<SpaceResources>,
    mut query: Query<&mut Transform, With<SunLight>>,
) {
    let Ok(sun_pos) = bodies::sun_position(&space.almanac, state.epoch) else {
        return;
    };
    let bevy_pos = sun_pos.to_bevy(1.0); // Just need direction, not scaled position
    let sun_dir = Vec3::from_array(bevy_pos).normalize();

    for mut transform in query.iter_mut() {
        *transform = Transform::default().looking_at(-sun_dir, Vec3::Y);
    }
}

fn update_lagrange_markers(
    state: Res<SimulationState>,
    space: Res<SpaceResources>,
    mut query: Query<(&mut Transform, &LagrangeMarker)>,
) {
    if !state.show_lagrange {
        return;
    }

    for (mut transform, marker) in query.iter_mut() {
        if let Ok(pos) = lagrange::lagrange_position(marker.point, &space.almanac, state.epoch) {
            let bevy_pos = pos.to_bevy(VIS_SCALE_F64);
            transform.translation = Vec3::from_array(bevy_pos);
        }
    }
}

fn update_user_bodies(
    state: Res<SimulationState>,
    time: Res<Time>,
    space: Res<SpaceResources>,
    mut user_bodies: ResMut<UserBodies>,
    mut body_trails: ResMut<BodyTrails>,
    mut body_query: Query<(&mut Transform, &UserBodyMarker)>,
) {
    // Always record celestial body positions at current epoch (needed for frame transforms)
    {
        let epoch = state.epoch;
        // If epoch went backward (reset, perigee jump), clear stale trail data
        if body_trails.moon.last().is_some_and(|(e, _, _)| *e > epoch) {
            body_trails.moon.clear();
            body_trails.earth.clear();
            for body in &mut user_bodies.bodies {
                body.trail.clear();
            }
        }
        let dominated = body_trails.moon.last().is_some_and(|(e, _, _)| *e >= epoch);
        if !dominated {
            body_trails.earth.push((epoch, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]));
            if let Ok(moon_pos) = bodies::moon_position(&space.almanac, epoch) {
                if let Ok(moon_vel) = bodies::moon_velocity(&space.almanac, epoch) {
                    body_trails.moon.push((
                        epoch,
                        [moon_pos.x, moon_pos.y, moon_pos.z],
                        [moon_vel.x, moon_vel.y, moon_vel.z],
                    ));
                }
            }
        }
    }

    // Propagate each body forward by the frame's sim dt
    if !state.paused {
        let dt_sim = time.delta_secs_f64() * state.speed;

        if let Some(ref propagator) = space.propagator {
            user_bodies.bodies.par_iter_mut().for_each(|body| {
                if let Some(ref sc) = body.spacecraft {
                    let target_epoch = sc.epoch() + hifitime::Duration::from_seconds(dt_sim);
                    match propagator
                        .with(*sc, space.almanac.clone())
                        .until_epoch(target_epoch)
                    {
                        Ok(new_sc) => {
                            let pos = new_sc.orbit.radius_km;
                            let vel = new_sc.orbit.velocity_km_s;
                            let epoch = new_sc.epoch();
                            body.trail.push((epoch, [pos.x, pos.y, pos.z], [vel.x, vel.y, vel.z]));
                            body.spacecraft = Some(new_sc);
                        }
                        Err(_) => {}
                    }
                }
            });
        }
    }

    // Update transforms from current spacecraft state
    for (mut transform, marker) in body_query.iter_mut() {
        if let Some(body) = user_bodies.bodies.get(marker.index) {
            if let Some(ref sc) = body.spacecraft {
                let pos = sc.orbit.radius_km;
                let j2000 = bodies::J2000Position { x: pos.x, y: pos.y, z: pos.z };
                transform.translation = Vec3::from_array(j2000.to_bevy(VIS_SCALE_F64));
            }
        }
    }
}

/// Build orthogonal Y and Z axes for a synodic frame given X = moon direction.
/// Z is approximately ecliptic normal, orthogonalized against X. Y = Z × X.
fn synodic_yz(xh: &[f64; 3]) -> ([f64; 3], [f64; 3]) {
    let raw_z = [0.0_f64, 0.0, 1.0];
    let dot_zx = raw_z[0] * xh[0] + raw_z[1] * xh[1] + raw_z[2] * xh[2];
    let zo = [raw_z[0] - dot_zx * xh[0], raw_z[1] - dot_zx * xh[1], raw_z[2] - dot_zx * xh[2]];
    let zm = (zo[0] * zo[0] + zo[1] * zo[1] + zo[2] * zo[2]).sqrt();
    let zh = [zo[0] / zm, zo[1] / zm, zo[2] / zm];
    let yh = [
        zh[1] * xh[2] - zh[2] * xh[1],
        zh[2] * xh[0] - zh[0] * xh[2],
        zh[0] * xh[1] - zh[1] * xh[0],
    ];
    (zh, yh)
}

/// Transform a J2000 position into the selected trail display frame.
///
/// - J2000: identity (no transform)
/// - MoonCentered: subtract moon_pos(t), re-add moon_pos(now) — shows Moon-relative
///   motion displayed at the Moon's current location
/// - Synodic: co-rotating display — Rodrigues rotation from moon_dir(t) to moon_dir(now)
///
/// `moon_now` is the current Moon J2000 position (from body_trails).
pub(crate) fn trail_transform_point(
    j2000: [f64; 3],
    epoch: Epoch,
    frame: TrailFrame,
    body_trails: &BodyTrails,
    moon_now: [f64; 3],
    moon_search_hint: &mut usize,
) -> Vec3 {
    match frame {
        TrailFrame::J2000 => {
            let p = bodies::J2000Position { x: j2000[0], y: j2000[1], z: j2000[2] };
            Vec3::from_array(p.to_bevy(VIS_SCALE_F64))
        }
        TrailFrame::MoonCentered => {
            let moon_t = lookup_body_trail(&body_trails.moon, epoch, moon_search_hint);
            let rel = [
                j2000[0] - moon_t[0] + moon_now[0],
                j2000[1] - moon_t[1] + moon_now[1],
                j2000[2] - moon_t[2] + moon_now[2],
            ];
            let p = bodies::J2000Position { x: rel[0], y: rel[1], z: rel[2] };
            Vec3::from_array(p.to_bevy(VIS_SCALE_F64))
        }
        TrailFrame::Synodic => {
            // Co-rotating: rotate each trail point so moon_dir(t) -> moon_dir(now).
            // Keeps everything in J2000 coords where the camera can see it.
            let moon_t = lookup_body_trail(&body_trails.moon, epoch, moon_search_hint);

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

            // Rodrigues' rotation: v' = v*cos(a) + (k×v)*sin(a) + k*(k·v)*(1-cos(a))
            let k = [cx / sin_a, cy / sin_a, cz / sin_a];
            let dot_kv = k[0] * j2000[0] + k[1] * j2000[1] + k[2] * j2000[2];
            let kxv = [
                k[1] * j2000[2] - k[2] * j2000[1],
                k[2] * j2000[0] - k[0] * j2000[2],
                k[0] * j2000[1] - k[1] * j2000[0],
            ];

            let rot = [
                j2000[0] * cos_a + kxv[0] * sin_a + k[0] * dot_kv * (1.0 - cos_a),
                j2000[1] * cos_a + kxv[1] * sin_a + k[1] * dot_kv * (1.0 - cos_a),
                j2000[2] * cos_a + kxv[2] * sin_a + k[2] * dot_kv * (1.0 - cos_a),
            ];

            let p = bodies::J2000Position { x: rot[0], y: rot[1], z: rot[2] };
            Vec3::from_array(p.to_bevy(VIS_SCALE_F64))
        }
        TrailFrame::Pulsating => {
            // Pulsating synodic: rotate + scale so Moon is always at the same
            // normalized position. Correctly handles elliptical Moon orbit.
            // 1) Build synodic frame at epoch t, project, normalize by R(t)
            // 2) Un-normalize by R(now), rotate back to J2000 at now
            let moon_t = lookup_body_trail(&body_trails.moon, epoch, moon_search_hint);

            let r_t = (moon_t[0] * moon_t[0] + moon_t[1] * moon_t[1] + moon_t[2] * moon_t[2]).sqrt();
            let r_now = (moon_now[0] * moon_now[0] + moon_now[1] * moon_now[1] + moon_now[2] * moon_now[2]).sqrt();

            if r_t < 1.0 || r_now < 1.0 {
                let p = bodies::J2000Position { x: j2000[0], y: j2000[1], z: j2000[2] };
                return Vec3::from_array(p.to_bevy(VIS_SCALE_F64));
            }

            // Synodic frame at epoch t: x_hat = moon_dir(t)
            let xh_t = [moon_t[0] / r_t, moon_t[1] / r_t, moon_t[2] / r_t];
            let (zh_t, yh_t) = synodic_yz(&xh_t);

            // Synodic frame at now: x_hat = moon_dir(now)
            let xh_now = [moon_now[0] / r_now, moon_now[1] / r_now, moon_now[2] / r_now];
            let (zh_now, yh_now) = synodic_yz(&xh_now);

            // Project into synodic at t (R_t * p), then normalize by r_t
            let sx = (xh_t[0] * j2000[0] + xh_t[1] * j2000[1] + xh_t[2] * j2000[2]) / r_t;
            let sy = (yh_t[0] * j2000[0] + yh_t[1] * j2000[1] + yh_t[2] * j2000[2]) / r_t;
            let sz = (zh_t[0] * j2000[0] + zh_t[1] * j2000[1] + zh_t[2] * j2000[2]) / r_t;

            // Un-normalize by r_now and rotate back to J2000: p = R_now^T * (s * r_now)
            let px = (sx * xh_now[0] + sy * yh_now[0] + sz * zh_now[0]) * r_now;
            let py = (sx * xh_now[1] + sy * yh_now[1] + sz * zh_now[1]) * r_now;
            let pz = (sx * xh_now[2] + sy * yh_now[2] + sz * zh_now[2]) * r_now;

            let p = bodies::J2000Position { x: px, y: py, z: pz };
            Vec3::from_array(p.to_bevy(VIS_SCALE_F64))
        }
    }
}

/// Look up a body position from its trail at the given epoch.
/// Uses a forward-scanning hint for efficiency with sorted trails.
fn lookup_body_trail(
    trail: &[(Epoch, [f64; 3], [f64; 3])],
    epoch: Epoch,
    hint: &mut usize,
) -> [f64; 3] {
    if trail.is_empty() {
        return [0.0, 0.0, 0.0];
    }
    // Scan forward from hint
    while *hint + 1 < trail.len() && trail[*hint + 1].0 <= epoch {
        *hint += 1;
    }
    trail[*hint].1
}

fn draw_trails(
    state: Res<SimulationState>,
    time: Res<Time>,
    user_bodies: Res<UserBodies>,
    body_trails: Res<BodyTrails>,
    analysis: Res<AnalysisState>,
    mut gizmos: Gizmos,
) {
    // No trails when duration is OFF (0)
    if state.trail_duration <= 0.0 {
        return;
    }

    let moon_now = body_trails.moon.last().map(|t| t.1).unwrap_or([0.0; 3]);
    let has_selection = analysis.selected_body.is_some();
    for (i, body) in user_bodies.bodies.iter().enumerate() {
        if body.trail.len() < 2 {
            continue;
        }

        // Suppress trails for the first real-time second after spawn
        if time.elapsed_secs_f64() - body.spawn_real_time < 1.0 {
            continue;
        }

        let cutoff_epoch = state.epoch - hifitime::Duration::from_seconds(state.trail_duration);

        let selected = analysis.selected_body == Some(i);
        let brightness = if has_selection && !selected { 0.5 } else { 1.0 };
        let base_color = LinearRgba::from(body.color);
        let mut moon_hint: usize = 0;
        let points = body.trail.iter()
            .filter(|(epoch, _, _)| *epoch >= cutoff_epoch)
            .map(|(epoch, j2000, _vel)| {
            let age = (state.epoch - *epoch).to_seconds();
            let alpha = (1.0 - age / state.trail_duration).clamp(0.0, 1.0) as f32;

            let display_pos = trail_transform_point(
                *j2000, *epoch, state.trail_frame, &body_trails, moon_now, &mut moon_hint,
            );

            (
                display_pos,
                Color::LinearRgba(LinearRgba::new(
                    base_color.red * brightness,
                    base_color.green * brightness,
                    base_color.blue * brightness,
                    alpha * brightness,
                )),
            )
        });
        gizmos.linestrip_gradient(points);
    }
}

fn update_camera(
    state: Res<SimulationState>,
    space: Res<SpaceResources>,
    user_bodies: Res<UserBodies>,
    mut query: Query<&mut Transform, With<MainCamera>>,
) {
    let from_pos = state.camera_from.position(&space.almanac, state.epoch, &user_bodies);
    let to_pos = state.camera_to.position(&space.almanac, state.epoch, &user_bodies);
    let up = state.camera_up.resolve(&space.almanac, state.epoch).unwrap_or(Vec3::Y);

    for mut transform in query.iter_mut() {
        match (from_pos, to_pos) {
            // Both bodies set: fixed camera along from→to line, looking at to
            (Some(from), Some(to)) => {
                if from.distance_squared(to) > 0.001 {
                    let camera_pos = match state.camera_origin {
                        CameraOrigin::Fraction => {
                            from.lerp(to, state.camera_origin_value.clamp(0.0, 1.0))
                        }
                        CameraOrigin::DistToTo => {
                            let dir = (from - to).normalize();
                            to + dir * state.camera_origin_value * VIS_SCALE
                        }
                        CameraOrigin::DistFromFrom => {
                            let dir = (to - from).normalize();
                            from + dir * state.camera_origin_value * VIS_SCALE
                        }
                    };
                    transform.translation = camera_pos;
                    transform.look_at(to, up);
                } else {
                    // Same point — fall back to orbit
                    let offset = orbit_offset(
                        state.camera_distance, state.camera_azimuth, state.camera_elevation, up,
                    );
                    transform.translation = from + offset;
                    transform.look_at(from, up);
                }
            }
            // One Free: orbit around the non-Free body
            (None, Some(target)) | (Some(target), None) => {
                let offset = orbit_offset(
                    state.camera_distance, state.camera_azimuth, state.camera_elevation, up,
                );
                transform.translation = target + offset;
                transform.look_at(target, up);
            }
            // Both Free: shouldn't happen (disallowed), fall back to orbit Earth
            (None, None) => {
                let offset = orbit_offset(
                    state.camera_distance, state.camera_azimuth, state.camera_elevation, up,
                );
                transform.translation = offset;
                transform.look_at(Vec3::ZERO, up);
            }
        }
    }
}

fn update_projection(
    state: Res<SimulationState>,
    windows: Query<&Window>,
    mut query: Query<&mut Projection, With<MainCamera>>,
) {
    let Ok(window) = windows.single() else { return };
    let Ok(mut proj) = query.single_mut() else { return };

    if state.perspective {
        if !matches!(*proj, Projection::Perspective(_)) {
            *proj = Projection::Perspective(PerspectiveProjection {
                far: 20000.0,
                ..default()
            });
        }
    } else {
        // Match perspective view size at camera distance
        let fov = std::f32::consts::PI / 4.0;
        let ortho_scale = 2.0 * state.camera_distance * (fov / 2.0).tan() / window.height();
        match &mut *proj {
            Projection::Orthographic(ortho) => {
                ortho.scale = ortho_scale;
            }
            _ => {
                *proj = Projection::Orthographic(OrthographicProjection {
                    scale: ortho_scale,
                    near: 0.1,
                    far: 20000.0,
                    ..OrthographicProjection::default_3d()
                });
            }
        }
    }
}

fn sync_star_visibility(
    state: Res<SimulationState>,
    mut query: Query<&mut Visibility, With<StarField>>,
) {
    let target = if state.show_stars && state.perspective {
        Visibility::Visible
    } else {
        Visibility::Hidden
    };
    for mut vis in query.iter_mut() {
        if *vis != target {
            *vis = target;
        }
    }
}

/// Round a value to the nearest "nice" number in the 1-2-5 sequence.
fn round_to_nice(value: f32) -> f32 {
    let magnitude = 10.0_f32.powf(value.log10().floor());
    let normalized = value / magnitude;
    let nice = if normalized < 1.5 {
        1.0
    } else if normalized < 3.5 {
        2.0
    } else if normalized < 7.5 {
        5.0
    } else {
        10.0
    };
    nice * magnitude
}

/// Format a km value for the scale bar.
fn format_km(km: f32) -> String {
    if km >= 1_000_000.0 {
        format!("{:.0}M km", km / 1_000_000.0)
    } else if km >= 1000.0 {
        format!("{:.0}k km", km / 1000.0)
    } else {
        format!("{:.0} km", km)
    }
}

// === egui UI Systems ===

/// Helper: draw a vertical tab for a collapsed panel.
fn vertical_tab(ui: &mut egui::Ui, label: &str) -> bool {
    let text: String = label.chars().flat_map(|c| [c, '\n']).collect();
    ui.add(egui::Button::new(text.trim()).frame(false)).clicked()
}

/// Combined optimizer panel on the right side, with collapsible sections.
fn optimizers_panel(
    mut contexts: EguiContexts,
    mut eq_state: ResMut<crate::optimizer::OptimizerState>,
    mut pole_state: ResMut<crate::pole_optimizer::PoleOptimizerState>,
    mut surf_state: ResMut<crate::surface_optimizer::SurfaceOptimizerState>,
    mut sim_state: ResMut<SimulationState>,
    space: Res<SpaceResources>,
) {
    let Ok(ctx) = contexts.ctx_mut() else {
        return;
    };

    if !sim_state.opt_panel_open {
        return;
    }

    egui::SidePanel::right("optimizers_panel")
        .default_width(280.0)
        .resizable(true)
        .show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.heading("Optimizers");
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui.small_button(">").on_hover_text("Collapse").clicked() {
                        sim_state.opt_panel_open = false;
                    }
                });
            });

            let eq_label = format!("Equator \u{2192} {}", eq_state.config.target.label());
            egui::CollapsingHeader::new(eq_label)
                .default_open(false)
                .show(ui, |ui| {
                    crate::optimizer::optimizer_ui_content(
                        ui, &mut eq_state, &mut sim_state, &space,
                    );
                });

            ui.separator();

            let pole_label = format!("North Pole \u{2192} {}", pole_state.config.target.label());
            egui::CollapsingHeader::new(pole_label)
                .default_open(false)
                .show(ui, |ui| {
                    crate::pole_optimizer::pole_optimizer_ui_content(
                        ui, &mut pole_state, &mut sim_state, &space,
                    );
                });

            ui.separator();

            let surf_label = format!("Surface \u{2192} {}", surf_state.config.opts.l_point.label());
            egui::CollapsingHeader::new(surf_label)
                .default_open(false)
                .show(ui, |ui| {
                    crate::surface_optimizer::surface_optimizer_ui_content(
                        ui, &mut surf_state, &mut sim_state, &space,
                    );
                });
        });
}

#[allow(clippy::too_many_arguments)]
fn egui_side_panel(
    mut contexts: EguiContexts,
    mut state: ResMut<SimulationState>,
    space: Res<SpaceResources>,
    mut user_bodies: ResMut<UserBodies>,
    mut body_trails: ResMut<BodyTrails>,
    mut analysis: ResMut<AnalysisState>,
    mut add_menu: ResMut<AddBodyMenuState>,
    time: Res<Time>,
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut lagrange_query: Query<&mut Visibility, With<LagrangeMarker>>,
    mut recording: ResMut<RecordingState>,
    mut perigee_cache: ResMut<PerigeeCache>,
    body_marker_query: Query<Entity, With<UserBodyMarker>>,
) {
    let Ok(ctx) = contexts.ctx_mut() else { return };

    // Dark style matching the old panel
    let mut visuals = egui::Visuals::dark();
    visuals.panel_fill = egui::Color32::from_rgba_unmultiplied(3, 5, 13, 204);
    ctx.set_visuals(visuals);

    // Collapsed-panel tabs on the right edge
    let main_closed = !state.main_panel_open;
    let opt_closed = !state.opt_panel_open;
    if main_closed || opt_closed {
        egui::SidePanel::right("collapsed_tabs")
            .exact_width(22.0)
            .resizable(false)
            .show(ctx, |ui| {
                if main_closed && vertical_tab(ui, "Main") {
                    state.main_panel_open = true;
                }
                if opt_closed && vertical_tab(ui, "Opt") {
                    state.opt_panel_open = true;
                }
            });
    }

    if !state.main_panel_open {
        return;
    }

    egui::SidePanel::right("control_panel")
        .default_width(280.0)
        .resizable(false)
        .show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.heading("Earth / Moon System");
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui.small_button(">").on_hover_text("Collapse").clicked() {
                        state.main_panel_open = false;
                    }
                });
            });
            ui.separator();

            // === Simulation (always visible, compact) ===
            ui.horizontal(|ui| {
                let pause_text = if state.paused { "\u{25b6}" } else { "\u{23f8}" };
                if ui.small_button(pause_text)
                    .on_hover_text(if state.paused { "Play (P)" } else { "Pause (P)" })
                    .clicked()
                {
                    state.paused = !state.paused;
                }
                let has_bodies = !user_bodies.bodies.is_empty();
                // Reset: disabled when bodies exist
                if ui.add_enabled(!has_bodies, egui::Button::new("\u{23ee}").small())
                    .on_hover_text(if has_bodies { "Clear bodies first" } else { "Reset (R)" })
                    .clicked()
                {
                    state.reset(space.initial_epoch);
                }
                // Clear all bodies
                if has_bodies && ui.small_button("X").on_hover_text("Clear all bodies").clicked() {
                    user_bodies.bodies.clear();
                    body_trails.moon.clear();
                    body_trails.earth.clear();
                    analysis.selected_body = None;
                    analysis.closest_approach = None;
                    for entity in body_marker_query.iter() {
                        commands.entity(entity).despawn();
                    }
                }
                if ui.small_button("-").clicked() {
                    state.speed = (state.speed / 2.0).max(1.0);
                }
                ui.label(format_speed(state.speed, state.paused));
                if ui.small_button("+").clicked() {
                    state.speed *= 2.0;
                }
                ui.separator();
                // Screenshot
                if ui.small_button("Snap").on_hover_text("Screenshot (F)").clicked() {
                    let n = recording.screenshot_counter;
                    recording.screenshot_counter += 1;
                    commands
                        .spawn(Screenshot::primary_window())
                        .observe(save_to_disk(format!("screenshot_{n}.png")));
                }
                // Record / stop
                if recording.recording {
                    let btn = egui::Button::new("\u{23f9}")
                        .fill(egui::Color32::from_rgb(180, 40, 40));
                    if ui.add(btn).on_hover_text("Stop recording (G)").clicked() {
                        recording.recording = false;
                        let frames = std::mem::take(&mut recording.frames);
                        let w = recording.gif_width;
                        let h = recording.gif_height;
                        let skip = recording.frame_skip;
                        let status = recording.status.clone();
                        if !frames.is_empty() {
                            *status.lock().unwrap() = format!("Encoding {} frames...", frames.len());
                            std::thread::spawn(move || {
                                save_recording_gif(frames, w, h, skip, status);
                            });
                        }
                    }
                } else if ui.small_button("\u{23fa}").on_hover_text("Record GIF (G)").clicked() {
                    recording.recording = true;
                    recording.frames.clear();
                    recording.frame_counter = 0;
                    *recording.status.lock().unwrap() = String::new();
                }
            });
            // Recording status
            {
                let msg = recording.status.lock().unwrap().clone();
                if !msg.is_empty() {
                    ui.label(&msg);
                }
            }

            // Perigee phase selector (only when no bodies are in the sim)
            if user_bodies.bodies.is_empty() && !perigee_cache.perigees.is_empty() {
                ui.separator();
                ui.horizontal(|ui| {
                    ui.label("Perigee:");
                    let max = (perigee_cache.perigees.len() - 1) as f64;
                    let old_phase = perigee_cache.phase;
                    ui.add(
                        egui::DragValue::new(&mut perigee_cache.phase)
                            .range(0.0..=max)
                            .speed(0.01)
                            .min_decimals(2)
                            .max_decimals(2),
                    );
                    if (perigee_cache.phase - old_phase).abs() > 1e-6 {
                        if let Some(new_epoch) = perigee_cache.epoch_for_phase(perigee_cache.phase) {
                            state.epoch = new_epoch;
                        }
                    }
                });
                if let Some(dist) = perigee_cache.perigee_dist(perigee_cache.phase) {
                    ui.label(
                        egui::RichText::new(format!(
                            "perigee dist: {dist:.0} km  alt: {:.0} km",
                            dist - crate::constants::MOON_RADIUS_KM
                        ))
                        .weak(),
                    );
                }
            }

            egui::CollapsingHeader::new("Display")
                .default_open(false)
                .show(ui, |ui| {
                    // Trail duration
                    ui.horizontal(|ui| {
                        ui.label("Trail:");
                        egui::ComboBox::from_id_salt("trail_duration")
                            .selected_text(format_trail(state.trail_duration))
                            .show_ui(ui, |ui| {
                                for &(val, label) in TRAIL_OPTIONS {
                                    let selected = (state.trail_duration - val).abs() < 1.0;
                                    if ui.selectable_label(selected, label).clicked() {
                                        state.trail_duration = val;
                                    }
                                }
                            });
                    });

                    // Trail frame
                    ui.horizontal(|ui| {
                        ui.label("Frame:");
                        ui.selectable_value(&mut state.trail_frame, TrailFrame::J2000, "J2000");
                        ui.selectable_value(&mut state.trail_frame, TrailFrame::MoonCentered, "Moon");
                        ui.selectable_value(&mut state.trail_frame, TrailFrame::Synodic, "Synodic");
                        ui.selectable_value(&mut state.trail_frame, TrailFrame::Pulsating, "Pulsating");
                    });

                    // Lagrange toggle
                    let mut show = state.show_lagrange;
                    if ui.checkbox(&mut show, "Lagrange Points (L)").changed() {
                        state.show_lagrange = show;
                        let vis = if show { Visibility::Visible } else { Visibility::Hidden };
                        for mut v in lagrange_query.iter_mut() {
                            *v = vis;
                        }
                    }

                    // Stars toggle
                    let mut show_s = state.show_stars;
                    if ui.checkbox(&mut show_s, "Stars (S)").changed() {
                        state.show_stars = show_s;
                    }
                });

            ui.separator();

            // === Camera ===
            egui::CollapsingHeader::new("Camera")
                .default_open(false)
                .show(ui, |ui| {
                    // Build endpoint list: fixed + satellites
                    let mut endpoints: Vec<CameraEndpoint> = CameraEndpoint::FIXED.to_vec();
                    for i in 0..user_bodies.bodies.len() {
                        endpoints.push(CameraEndpoint::Satellite(i));
                    }

                    // From
                    ui.horizontal(|ui| {
                        ui.label("From:");
                        let from_label = state.camera_from.display_name(&user_bodies);
                        egui::ComboBox::from_id_salt("camera_from")
                            .selected_text(from_label)
                            .show_ui(ui, |ui| {
                                for ep in &endpoints {
                                    let label = ep.display_name(&user_bodies);
                                    if ui.selectable_value(&mut state.camera_from, *ep, label).changed() {
                                        // Prevent both Free
                                        if state.camera_from == CameraEndpoint::Free
                                            && state.camera_to == CameraEndpoint::Free
                                        {
                                            state.camera_to = CameraEndpoint::Earth;
                                        }
                                    }
                                }
                            });
                    });

                    // To
                    ui.horizontal(|ui| {
                        ui.label("To:");
                        let to_label = state.camera_to.display_name(&user_bodies);
                        egui::ComboBox::from_id_salt("camera_to")
                            .selected_text(to_label)
                            .show_ui(ui, |ui| {
                                for ep in &endpoints {
                                    let label = ep.display_name(&user_bodies);
                                    if ui.selectable_value(&mut state.camera_to, *ep, label).changed() {
                                        // Prevent both Free
                                        if state.camera_from == CameraEndpoint::Free
                                            && state.camera_to == CameraEndpoint::Free
                                        {
                                            state.camera_from = CameraEndpoint::Earth;
                                        }
                                    }
                                }
                            });
                    });

                    // Up vector
                    ui.horizontal(|ui| {
                        ui.label("Up:");
                        egui::ComboBox::from_id_salt("camera_up")
                            .selected_text(state.camera_up.label())
                            .show_ui(ui, |ui| {
                                for &up in CameraUp::ALL {
                                    ui.selectable_value(&mut state.camera_up, up, up.label());
                                }
                            });
                    });

                    // Perspective toggle
                    ui.checkbox(&mut state.perspective, "Perspective");

                    // Camera origin controls (fixed mode only)
                    let fixed_mode = state.camera_from != CameraEndpoint::Free
                        && state.camera_to != CameraEndpoint::Free;
                    if fixed_mode {
                        ui.horizontal(|ui| {
                            ui.label("Origin:");
                            let origin_label = match state.camera_origin {
                                CameraOrigin::Fraction => "Fraction",
                                CameraOrigin::DistToTo => "km to To",
                                CameraOrigin::DistFromFrom => "km from From",
                            };
                            egui::ComboBox::from_id_salt("camera_origin")
                                .selected_text(origin_label)
                                .show_ui(ui, |ui| {
                                    ui.selectable_value(&mut state.camera_origin, CameraOrigin::Fraction, "Fraction");
                                    ui.selectable_value(&mut state.camera_origin, CameraOrigin::DistToTo, "km to To");
                                    ui.selectable_value(&mut state.camera_origin, CameraOrigin::DistFromFrom, "km from From");
                                });
                        });

                        match state.camera_origin {
                            CameraOrigin::Fraction => {
                                ui.add(egui::Slider::new(&mut state.camera_origin_value, 0.0..=1.0).text("fraction"));
                            }
                            CameraOrigin::DistToTo | CameraOrigin::DistFromFrom => {
                                ui.add(egui::DragValue::new(&mut state.camera_origin_value)
                                    .speed(100.0)
                                    .suffix(" km")
                                    .range(0.0..=f32::MAX));
                            }
                        }
                    }

                    // Mode hint
                    let mode = if fixed_mode {
                        "Fixed"
                    } else {
                        "Orbit (drag + scroll)"
                    };
                    ui.label(egui::RichText::new(mode).weak());
                });

            ui.separator();

            // === Add Body ===
            egui::CollapsingHeader::new("Add Body")
                .default_open(false)
                .show(ui, |ui| {
                    ui.horizontal(|ui| {
                        // Left: selectable list
                        ui.vertical(|ui| {
                            ui.set_width(120.0);
                            for (i, entry) in AddBodyEntry::ALL.iter().enumerate() {
                                let selected = add_menu.selected == i;
                                if ui.selectable_label(selected, entry.label()).clicked() {
                                    add_menu.selected = i;
                                    add_menu.fan_speed_km_s = entry.default_speed_km_s();
                                }
                            }
                        });

                        ui.separator();

                        // Right: config + Add button
                        ui.vertical(|ui| {
                            let entry = AddBodyEntry::ALL[add_menu.selected.min(AddBodyEntry::ALL.len() - 1)];

                            if entry.has_fan_config() {
                                ui.horizontal(|ui| {
                                    ui.label("Count:");
                                    ui.add(egui::DragValue::new(&mut add_menu.fan_count)
                                        .range(1..=360)
                                        .speed(1));
                                });
                                ui.horizontal(|ui| {
                                    ui.label("Speed:");
                                    ui.add(egui::DragValue::new(&mut add_menu.fan_speed_km_s)
                                        .range(0.001..=100.0)
                                        .speed(0.01)
                                        .suffix(" km/s")
                                        .min_decimals(3));
                                });
                            }

                            if entry.has_lunar_config() {
                                ui.horizontal(|ui| {
                                    ui.label("Az:");
                                    ui.add(egui::DragValue::new(&mut add_menu.lunar_az_deg)
                                        .range(-180.0..=180.0)
                                        .speed(5.0)
                                        .suffix("°"));
                                });
                                ui.horizontal(|ui| {
                                    ui.label("El:");
                                    ui.add(egui::DragValue::new(&mut add_menu.lunar_el_deg)
                                        .range(-90.0..=90.0)
                                        .speed(5.0)
                                        .suffix("°"));
                                });
                                ui.label(egui::RichText::new("Lon: 0°=sub-Earth, 180°=far side").weak());
                                ui.label(egui::RichText::new("Lat: 0°=equator, 90°=north pole").weak());
                            }

                            if ui.button("Add").clicked() {
                                add_body_entry(
                                    entry,
                                    &add_menu,
                                    &mut state, &space, &mut user_bodies,
                                    &time, &mut commands, &mut meshes, &mut materials,
                                );
                            }
                        });
                    });
                });

            // === Analysis ===
            if !user_bodies.bodies.is_empty() {
                // Auto-select first body if none selected
                if analysis.selected_body.is_none() {
                    analysis.selected_body = Some(0);
                }
                // Clamp selection to valid range
                if let Some(idx) = analysis.selected_body {
                    if idx >= user_bodies.bodies.len() {
                        analysis.selected_body = Some(user_bodies.bodies.len() - 1);
                    }
                }

                ui.separator();
                egui::CollapsingHeader::new("Analysis")
                    .default_open(false)
                    .show(ui, |ui| {
                        // Body selector
                        ui.horizontal(|ui| {
                            ui.label("Body:");
                            let sel_name = analysis.selected_body
                                .and_then(|i| user_bodies.bodies.get(i))
                                .map(|b| b.name.as_str())
                                .unwrap_or("(none)");
                            egui::ComboBox::from_id_salt("analysis_body")
                                .selected_text(sel_name)
                                .show_ui(ui, |ui| {
                                    for (i, body) in user_bodies.bodies.iter().enumerate() {
                                        if ui.selectable_value(
                                            &mut analysis.selected_body,
                                            Some(i),
                                            &body.name,
                                        ).changed() {
                                            analysis.closest_approach = None;
                                        }
                                    }
                                });
                        });

                        // Reference body toggle
                        ui.horizontal(|ui| {
                            ui.label("Ref:");
                            let old_ref = analysis.reference;
                            ui.selectable_value(&mut analysis.reference, AnalysisRef::Earth, "Earth");
                            ui.selectable_value(&mut analysis.reference, AnalysisRef::Moon, "Moon");
                            if analysis.reference != old_ref {
                                analysis.closest_approach = None;
                            }
                        });

                        // Display mode toggle
                        ui.horizontal(|ui| {
                            ui.label("Display:");
                            ui.selectable_value(&mut analysis.display, OrbitalDisplay::Keplerian, "Keplerian");
                            ui.selectable_value(&mut analysis.display, OrbitalDisplay::Euclidean, "Euclidean");
                        });

                        // Current state
                        if let Some(body) = analysis.selected_body
                            .and_then(|i| user_bodies.bodies.get(i))
                        {
                            if let Some(ref sc) = body.spacecraft {
                                let pos = [sc.orbit.radius_km.x, sc.orbit.radius_km.y, sc.orbit.radius_km.z];
                                let vel = [sc.orbit.velocity_km_s.x, sc.orbit.velocity_km_s.y, sc.orbit.velocity_km_s.z];
                                let epoch = state.epoch;

                                ui.separator();
                                ui.label(egui::RichText::new(format!("{} — Current", body.name)).strong());
                                ui.label(format_epoch_short(&epoch));

                                if let Some((rel_pos, rel_vel)) = relative_state(
                                    pos, vel, analysis.reference, &space.almanac, epoch,
                                ) {
                                    match analysis.display {
                                        OrbitalDisplay::Keplerian => {
                                            let r = (rel_pos[0].powi(2) + rel_pos[1].powi(2) + rel_pos[2].powi(2)).sqrt();
                                            let v = (rel_vel[0].powi(2) + rel_vel[1].powi(2) + rel_vel[2].powi(2)).sqrt();
                                            let alt = r - analysis.reference.radius_km();
                                            ui.label(format!("r={r:.0} km  v={v:.3} km/s  alt={alt:.0} km"));
                                            if let Ok(orb) = orbit::create_orbit_cartesian(
                                                rel_pos, rel_vel, epoch, &space.almanac,
                                                analysis.reference.frame(),
                                            ) {
                                                show_keplerian(ui, &orb, analysis.reference.radius_km(), analysis.reference.mu());
                                            }
                                        }
                                        OrbitalDisplay::Euclidean => {
                                            show_euclidean(ui, rel_pos, rel_vel, analysis.reference.radius_km());
                                        }
                                    }
                                }
                            }

                            // Closest approach
                            ui.separator();
                            ui.label(egui::RichText::new("Closest Approach").strong());

                            let ref_copy = analysis.reference;
                            ui.horizontal(|ui| {
                                if ui.button("Find in trail").clicked() {
                                    analysis.closest_approach = find_closest_approach(
                                        &body.trail, ref_copy, &body_trails,
                                    );
                                }
                                if ui.button("Find best").clicked() {
                                    let mut best_idx = None;
                                    let mut best_ca: Option<ClosestApproach> = None;
                                    for (i, b) in user_bodies.bodies.iter().enumerate() {
                                        if let Some(ca) = find_closest_approach(
                                            &b.trail, ref_copy, &body_trails,
                                        ) {
                                            if best_ca.as_ref().is_none_or(|prev| ca.distance_km < prev.distance_km) {
                                                best_idx = Some(i);
                                                best_ca = Some(ca);
                                            }
                                        }
                                    }
                                    if let Some(idx) = best_idx {
                                        analysis.selected_body = Some(idx);
                                        analysis.closest_approach = best_ca;
                                    }
                                }
                            });

                            if let Some(ref ca) = analysis.closest_approach {
                                ui.label(format_epoch_short(&ca.epoch));
                                let alt = ca.distance_km - ref_copy.radius_km();
                                ui.label(format!("dist={:.0} km  alt={alt:.0} km", ca.distance_km));

                                if let Some((rel_pos, rel_vel)) = relative_state(
                                    ca.pos_km, ca.vel_km_s, ref_copy, &space.almanac, ca.epoch,
                                ) {
                                    match analysis.display {
                                        OrbitalDisplay::Keplerian => {
                                            if let Ok(orb) = orbit::create_orbit_cartesian(
                                                rel_pos, rel_vel, ca.epoch, &space.almanac,
                                                ref_copy.frame(),
                                            ) {
                                                show_keplerian(ui, &orb, ref_copy.radius_km(), ref_copy.mu());
                                            }
                                        }
                                        OrbitalDisplay::Euclidean => {
                                            show_euclidean(ui, rel_pos, rel_vel, ref_copy.radius_km());
                                        }
                                    }
                                }
                            }
                        } else {
                            ui.label(egui::RichText::new("Add a body to analyze").weak());
                        }
                    });
            }
        });
}

fn egui_bottom_panel(
    mut contexts: EguiContexts,
    state: Res<SimulationState>,
    windows: Query<&Window>,
) {
    let Ok(ctx) = contexts.ctx_mut() else { return };

    egui::TopBottomPanel::bottom("epoch_footer").show(ctx, |ui| {
        ui.horizontal(|ui| {
            let epoch_str = format_epoch_short(&state.epoch);
            let speed_str = format_speed(state.speed, state.paused);
            ui.label(
                egui::RichText::new(format!("{epoch_str}  {speed_str}"))
                    .monospace()
                    .size(14.0),
            );

            // Scale bar (orthographic mode only)
            if !state.perspective {
                if let Ok(window) = windows.single() {
                    let fov = std::f32::consts::PI / 4.0;
                    let ortho_scale =
                        2.0 * state.camera_distance * (fov / 2.0).tan() / window.height();
                    let km_per_pixel = ortho_scale / VIS_SCALE;
                    let target_px = 150.0_f32;
                    let raw_km = target_px * km_per_pixel;
                    let nice_km = round_to_nice(raw_km);
                    let bar_px = nice_km / km_per_pixel;

                    // Right-align the scale bar
                    let avail = ui.available_rect_before_wrap();
                    let bar_right = avail.max.x - 10.0;
                    let bar_left = bar_right - bar_px;
                    let bar_y = avail.center().y;
                    let painter = ui.painter();

                    // Main bar line
                    painter.line_segment(
                        [egui::pos2(bar_left, bar_y), egui::pos2(bar_right, bar_y)],
                        egui::Stroke::new(2.0, egui::Color32::WHITE),
                    );
                    // End ticks
                    painter.line_segment(
                        [
                            egui::pos2(bar_left, bar_y - 4.0),
                            egui::pos2(bar_left, bar_y + 4.0),
                        ],
                        egui::Stroke::new(1.5, egui::Color32::WHITE),
                    );
                    painter.line_segment(
                        [
                            egui::pos2(bar_right, bar_y - 4.0),
                            egui::pos2(bar_right, bar_y + 4.0),
                        ],
                        egui::Stroke::new(1.5, egui::Color32::WHITE),
                    );
                    // Label centered on bar with background
                    let label = format_km(nice_km);
                    let font = egui::FontId::monospace(11.0);
                    let center = egui::pos2((bar_left + bar_right) / 2.0, bar_y);
                    let galley = painter.layout_no_wrap(label, font.clone(), egui::Color32::WHITE);
                    let text_rect = egui::Align2::CENTER_CENTER.anchor_size(center, galley.size());
                    let bg = text_rect.expand2(egui::vec2(3.0, 1.0));
                    painter.rect_filled(bg, 0.0, egui::Color32::BLACK);
                    painter.galley(text_rect.min, galley, egui::Color32::WHITE);
                }
            }
        });
    });
}

fn handle_trail_click(
    mouse_button: Res<ButtonInput<MouseButton>>,
    windows: Query<&Window>,
    camera_q: Query<(&Camera, &GlobalTransform), With<MainCamera>>,
    state: Res<SimulationState>,
    body_trails: Res<BodyTrails>,
    user_bodies: Res<UserBodies>,
    mut analysis: ResMut<AnalysisState>,
    click_tracker: Res<ClickTracker>,
) {
    // Left-click release with minimal drag (not a camera rotation)
    if !mouse_button.just_released(MouseButton::Left) {
        return;
    }
    if click_tracker.drag_distance > 5.0 {
        return;
    }

    let Ok(window) = windows.single() else { return };
    let Some(cursor_pos) = window.cursor_position() else { return };
    let Ok((camera, camera_transform)) = camera_q.single() else { return };
    // Only pick visible trail points
    if state.trail_duration <= 0.0 {
        return;
    }

    let cutoff_epoch = state.epoch - hifitime::Duration::from_seconds(state.trail_duration);
    let moon_now = body_trails.moon.last().map(|t| t.1).unwrap_or([0.0; 3]);

    let mut best_screen_dist = f32::MAX;
    let mut best_body_idx: Option<usize> = None;

    for (body_idx, body) in user_bodies.bodies.iter().enumerate() {
        let step = if body.trail.len() > 1000 { 10 } else { 1 };
        let mut moon_hint: usize = 0;
        for (i, (epoch, pos, _vel)) in body.trail.iter().enumerate() {
            if i % step != 0 { continue; }
            if *epoch < cutoff_epoch { continue; }

            let display_pos = trail_transform_point(
                *pos, *epoch, state.trail_frame, &body_trails, moon_now, &mut moon_hint,
            );

            let Ok(screen_pos) = camera.world_to_viewport(camera_transform, display_pos) else {
                continue;
            };
            let dist = screen_pos.distance(cursor_pos);

            if dist < best_screen_dist {
                best_screen_dist = dist;
                best_body_idx = Some(body_idx);
            }
        }
    }

    // 30 pixel threshold — select the body for analysis
    if best_screen_dist < 30.0 {
        if analysis.selected_body != best_body_idx {
            analysis.selected_body = best_body_idx;
            analysis.closest_approach = None;
        }
    } else {
        // Don't deselect on miss — keep current selection
    }
}

/// Update cursor icon when hovering over a trail point.
fn update_trail_hover(
    windows: Query<(Entity, &Window)>,
    mut commands: Commands,
    camera_q: Query<(&Camera, &GlobalTransform), With<MainCamera>>,
    state: Res<SimulationState>,
    body_trails: Res<BodyTrails>,
    user_bodies: Res<UserBodies>,
    mut hover_state: ResMut<TrailHoverState>,
) {
    let Ok((window_entity, window)) = windows.single() else { return };
    let Some(cursor_pos) = window.cursor_position() else {
        if hover_state.hovering {
            hover_state.hovering = false;
            commands.entity(window_entity).insert(CursorIcon::default());
        }
        return;
    };
    let Ok((camera, camera_transform)) = camera_q.single() else { return };

    // Skip if no trails visible or no bodies
    if state.trail_duration <= 0.0 || user_bodies.bodies.is_empty() {
        if hover_state.hovering {
            hover_state.hovering = false;
            commands.entity(window_entity).insert(CursorIcon::default());
        }
        return;
    }

    let cutoff_epoch = state.epoch - hifitime::Duration::from_seconds(state.trail_duration);
    let moon_now = body_trails.moon.last().map(|t| t.1).unwrap_or([0.0; 3]);

    let mut near = false;
    'outer: for body in &user_bodies.bodies {
        let step = if body.trail.len() > 500 { 20 } else { 3 };
        let mut moon_hint: usize = 0;
        for (i, (epoch, pos, _vel)) in body.trail.iter().enumerate() {
            if i % step != 0 { continue; }
            if *epoch < cutoff_epoch { continue; }

            let display_pos = trail_transform_point(
                *pos, *epoch, state.trail_frame, &body_trails, moon_now, &mut moon_hint,
            );

            if let Ok(screen_pos) = camera.world_to_viewport(camera_transform, display_pos) {
                if screen_pos.distance(cursor_pos) < 30.0 {
                    near = true;
                    break 'outer;
                }
            }
        }
    }

    if near != hover_state.hovering {
        hover_state.hovering = near;
        if near {
            commands.entity(window_entity).insert(
                CursorIcon::System(SystemCursorIcon::Pointer),
            );
        } else {
            commands.entity(window_entity).insert(CursorIcon::default());
        }
    }
}
