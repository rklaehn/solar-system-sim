//! Star catalog for the skybox.
//! Contains the brightest stars with their J2000 equatorial coordinates and spectral types.

/// Spectral class of a star (determines color)
#[derive(Clone, Copy)]
pub enum SpectralClass {
    O, // Blue (very hot, >30,000K)
    B, // Blue-white (10,000-30,000K)
    A, // White (7,500-10,000K)
    F, // Yellow-white (6,000-7,500K)
    G, // Yellow (5,200-6,000K) - like our Sun
    K, // Orange (3,700-5,200K)
    M, // Red (2,400-3,700K)
}

impl SpectralClass {
    /// Get the RGB color for this spectral class (normalized 0-1)
    pub fn to_color(&self) -> (f32, f32, f32) {
        match self {
            SpectralClass::O => (0.6, 0.7, 1.0),   // Deep blue
            SpectralClass::B => (0.7, 0.8, 1.0),   // Blue-white
            SpectralClass::A => (0.9, 0.92, 1.0),  // White with slight blue
            SpectralClass::F => (1.0, 0.98, 0.9),  // Yellow-white
            SpectralClass::G => (1.0, 0.95, 0.7),  // Yellow
            SpectralClass::K => (1.0, 0.8, 0.5),   // Orange
            SpectralClass::M => (1.0, 0.6, 0.4),   // Red-orange
        }
    }
}

/// A star with its position, magnitude, and spectral type.
pub struct Star {
    /// Right Ascension in degrees (J2000)
    pub ra_deg: f64,
    /// Declination in degrees (J2000)
    pub dec_deg: f64,
    /// Apparent visual magnitude (lower = brighter)
    pub magnitude: f32,
    /// Spectral class (determines color)
    pub spectral: SpectralClass,
}

impl Star {
    /// Convert RA/Dec to unit vector in J2000 Cartesian coordinates.
    /// X points to vernal equinox, Z points to celestial north pole.
    pub fn to_j2000_unit_vector(&self) -> (f64, f64, f64) {
        let ra_rad = self.ra_deg.to_radians();
        let dec_rad = self.dec_deg.to_radians();

        let x = dec_rad.cos() * ra_rad.cos();
        let y = dec_rad.cos() * ra_rad.sin();
        let z = dec_rad.sin();

        (x, y, z)
    }

    /// Convert to Bevy coordinates (J2000: x→x, y→-z, z→y becomes Bevy: x, y_up, z)
    pub fn to_bevy_direction(&self) -> (f32, f32, f32) {
        let (x, y, z) = self.to_j2000_unit_vector();
        // J2000 to Bevy: (x, y, z) → (x, z, -y)
        (x as f32, z as f32, -y as f32)
    }
}

use SpectralClass::*;

/// Catalog of the brightest stars visible from Earth.
/// Data from the Yale Bright Star Catalog (J2000 epoch).
pub const BRIGHT_STARS: &[Star] = &[
    // Top 20 brightest stars
    Star { ra_deg: 101.287, dec_deg: -16.716, magnitude: -1.46, spectral: A }, // Sirius
    Star { ra_deg: 95.988, dec_deg: -52.696, magnitude: -0.74, spectral: F },  // Canopus
    Star { ra_deg: 219.899, dec_deg: -60.834, magnitude: -0.27, spectral: G }, // Alpha Centauri
    Star { ra_deg: 213.915, dec_deg: 19.182, magnitude: -0.05, spectral: K },  // Arcturus
    Star { ra_deg: 279.235, dec_deg: 38.784, magnitude: 0.03, spectral: A },   // Vega
    Star { ra_deg: 79.172, dec_deg: 45.998, magnitude: 0.08, spectral: G },    // Capella
    Star { ra_deg: 78.634, dec_deg: -8.202, magnitude: 0.13, spectral: B },    // Rigel
    Star { ra_deg: 114.826, dec_deg: 5.225, magnitude: 0.34, spectral: F },    // Procyon
    Star { ra_deg: 24.429, dec_deg: -57.237, magnitude: 0.46, spectral: B },   // Achernar
    Star { ra_deg: 88.793, dec_deg: 7.407, magnitude: 0.50, spectral: M },     // Betelgeuse
    Star { ra_deg: 210.956, dec_deg: -60.373, magnitude: 0.61, spectral: B },  // Hadar
    Star { ra_deg: 297.696, dec_deg: 8.868, magnitude: 0.77, spectral: A },    // Altair
    Star { ra_deg: 186.650, dec_deg: -63.099, magnitude: 0.76, spectral: B },  // Acrux
    Star { ra_deg: 68.980, dec_deg: 16.509, magnitude: 0.85, spectral: K },    // Aldebaran
    Star { ra_deg: 247.352, dec_deg: -26.432, magnitude: 0.96, spectral: M },  // Antares
    Star { ra_deg: 201.298, dec_deg: -11.161, magnitude: 0.97, spectral: B },  // Spica
    Star { ra_deg: 116.329, dec_deg: 28.026, magnitude: 1.14, spectral: K },   // Pollux
    Star { ra_deg: 344.413, dec_deg: -29.622, magnitude: 1.16, spectral: A },  // Fomalhaut
    Star { ra_deg: 310.358, dec_deg: 45.280, magnitude: 1.25, spectral: A },   // Deneb
    Star { ra_deg: 191.930, dec_deg: -59.689, magnitude: 1.25, spectral: B },  // Mimosa

    // Stars magnitude 1.3 - 2.0
    Star { ra_deg: 152.093, dec_deg: 11.967, magnitude: 1.35, spectral: B },   // Regulus
    Star { ra_deg: 104.656, dec_deg: -28.972, magnitude: 1.50, spectral: B },  // Adhara
    Star { ra_deg: 113.650, dec_deg: 31.888, magnitude: 1.57, spectral: A },   // Castor
    Star { ra_deg: 187.791, dec_deg: -57.113, magnitude: 1.63, spectral: M },  // Gacrux
    Star { ra_deg: 263.402, dec_deg: -37.104, magnitude: 1.63, spectral: B },  // Shaula
    Star { ra_deg: 81.283, dec_deg: 6.350, magnitude: 1.64, spectral: B },     // Bellatrix
    Star { ra_deg: 81.573, dec_deg: 28.608, magnitude: 1.65, spectral: B },    // Elnath
    Star { ra_deg: 138.300, dec_deg: -69.717, magnitude: 1.68, spectral: A },  // Miaplacidus
    Star { ra_deg: 84.053, dec_deg: -1.202, magnitude: 1.69, spectral: B },    // Alnilam
    Star { ra_deg: 332.058, dec_deg: -46.961, magnitude: 1.74, spectral: B },  // Alnair
    Star { ra_deg: 85.190, dec_deg: -1.943, magnitude: 1.77, spectral: O },    // Alnitak
    Star { ra_deg: 193.507, dec_deg: 55.960, magnitude: 1.77, spectral: A },   // Alioth
    Star { ra_deg: 165.932, dec_deg: 61.751, magnitude: 1.79, spectral: K },   // Dubhe
    Star { ra_deg: 51.081, dec_deg: 49.861, magnitude: 1.80, spectral: F },    // Mirfak
    Star { ra_deg: 107.098, dec_deg: -26.393, magnitude: 1.84, spectral: F },  // Wezen
    Star { ra_deg: 264.330, dec_deg: -42.998, magnitude: 1.87, spectral: F },  // Sargas
    Star { ra_deg: 276.043, dec_deg: -34.384, magnitude: 1.85, spectral: B },  // Kaus Australis
    Star { ra_deg: 125.629, dec_deg: -59.509, magnitude: 1.86, spectral: K },  // Avior
    Star { ra_deg: 206.885, dec_deg: 49.313, magnitude: 1.86, spectral: B },   // Alkaid
    Star { ra_deg: 89.882, dec_deg: 44.948, magnitude: 1.90, spectral: A },    // Menkalinan
    Star { ra_deg: 252.166, dec_deg: -69.028, magnitude: 1.92, spectral: K },  // Atria
    Star { ra_deg: 99.428, dec_deg: 16.399, magnitude: 1.93, spectral: A },    // Alhena
    Star { ra_deg: 306.412, dec_deg: -56.735, magnitude: 1.94, spectral: B },  // Peacock
    Star { ra_deg: 131.176, dec_deg: -54.709, magnitude: 1.96, spectral: A },  // Alsephina
    Star { ra_deg: 95.675, dec_deg: -17.956, magnitude: 1.98, spectral: B },   // Mirzam
    Star { ra_deg: 37.954, dec_deg: 89.264, magnitude: 2.02, spectral: F },    // Polaris
    Star { ra_deg: 141.897, dec_deg: -8.659, magnitude: 1.98, spectral: K },   // Alphard

    // Stars magnitude 2.0 - 2.5
    Star { ra_deg: 31.793, dec_deg: 23.463, magnitude: 2.00, spectral: K },    // Hamal
    Star { ra_deg: 10.897, dec_deg: -17.987, magnitude: 2.02, spectral: K },   // Diphda
    Star { ra_deg: 200.981, dec_deg: 54.925, magnitude: 2.04, spectral: A },   // Mizar
    Star { ra_deg: 283.816, dec_deg: -26.297, magnitude: 2.05, spectral: B },  // Nunki
    Star { ra_deg: 211.671, dec_deg: -36.370, magnitude: 2.06, spectral: K },  // Menkent
    Star { ra_deg: 17.433, dec_deg: 35.621, magnitude: 2.06, spectral: M },    // Mirach
    Star { ra_deg: 2.097, dec_deg: 29.091, magnitude: 2.06, spectral: B },     // Alpheratz
    Star { ra_deg: 263.734, dec_deg: 12.560, magnitude: 2.07, spectral: A },   // Rasalhague
    Star { ra_deg: 222.676, dec_deg: 74.156, magnitude: 2.08, spectral: K },   // Kochab
    Star { ra_deg: 86.939, dec_deg: -9.670, magnitude: 2.09, spectral: B },    // Saiph
    Star { ra_deg: 177.265, dec_deg: 14.572, magnitude: 2.11, spectral: A },   // Denebola
    Star { ra_deg: 47.042, dec_deg: 40.956, magnitude: 2.12, spectral: B },    // Algol
    Star { ra_deg: 340.667, dec_deg: -46.885, magnitude: 2.14, spectral: K },  // Tiaki
    Star { ra_deg: 190.379, dec_deg: -48.960, magnitude: 2.17, spectral: A },  // Muhlifain
    Star { ra_deg: 139.273, dec_deg: -59.275, magnitude: 2.21, spectral: A },  // Aspidiske
    Star { ra_deg: 136.999, dec_deg: -43.433, magnitude: 2.21, spectral: K },  // Suhail
    Star { ra_deg: 233.672, dec_deg: 26.715, magnitude: 2.23, spectral: A },   // Alphecca
    Star { ra_deg: 83.002, dec_deg: -0.299, magnitude: 2.23, spectral: O },    // Mintaka
    Star { ra_deg: 305.557, dec_deg: 40.257, magnitude: 2.23, spectral: F },   // Sadr
    Star { ra_deg: 269.152, dec_deg: 51.489, magnitude: 2.23, spectral: K },   // Eltanin
    Star { ra_deg: 10.127, dec_deg: 56.537, magnitude: 2.23, spectral: K },    // Schedar
    Star { ra_deg: 120.896, dec_deg: -40.003, magnitude: 2.25, spectral: O },  // Naos
    Star { ra_deg: 30.975, dec_deg: 42.330, magnitude: 2.26, spectral: K },    // Almach
    Star { ra_deg: 2.295, dec_deg: 59.150, magnitude: 2.27, spectral: F },     // Caph
    Star { ra_deg: 221.247, dec_deg: 27.074, magnitude: 2.29, spectral: K },   // Izar
    Star { ra_deg: 240.083, dec_deg: -22.622, magnitude: 2.32, spectral: B },  // Dschubba
    Star { ra_deg: 252.541, dec_deg: -34.293, magnitude: 2.29, spectral: K },  // Larawag
    Star { ra_deg: 165.460, dec_deg: 56.382, magnitude: 2.37, spectral: A },   // Merak
    Star { ra_deg: 6.571, dec_deg: -42.306, magnitude: 2.37, spectral: K },    // Ankaa
    Star { ra_deg: 265.622, dec_deg: -39.030, magnitude: 2.41, spectral: B },  // Girtab
    Star { ra_deg: 326.046, dec_deg: 9.875, magnitude: 2.39, spectral: K },    // Enif
    Star { ra_deg: 345.944, dec_deg: 28.083, magnitude: 2.42, spectral: M },   // Scheat
    Star { ra_deg: 257.595, dec_deg: -15.725, magnitude: 2.43, spectral: A },  // Sabik
    Star { ra_deg: 178.458, dec_deg: 53.695, magnitude: 2.44, spectral: A },   // Phecda
    Star { ra_deg: 111.024, dec_deg: -29.303, magnitude: 2.45, spectral: B },  // Aludra
    Star { ra_deg: 140.528, dec_deg: -55.011, magnitude: 2.45, spectral: B },  // Markeb
    Star { ra_deg: 14.177, dec_deg: 60.717, magnitude: 2.47, spectral: B },    // Navi
    Star { ra_deg: 346.190, dec_deg: 15.205, magnitude: 2.49, spectral: B },   // Markab
    Star { ra_deg: 311.553, dec_deg: 33.970, magnitude: 2.48, spectral: K },   // Aljanah
    Star { ra_deg: 241.359, dec_deg: -19.805, magnitude: 2.50, spectral: B },  // Acrab

    // Stars magnitude 2.5 - 3.0
    Star { ra_deg: 188.371, dec_deg: -23.397, magnitude: 2.55, spectral: K },  // Algorab
    Star { ra_deg: 95.740, dec_deg: -17.822, magnitude: 2.58, spectral: F },   // Furud
    Star { ra_deg: 305.253, dec_deg: -14.781, magnitude: 2.60, spectral: A },  // Sadalsuud
    Star { ra_deg: 186.265, dec_deg: -16.516, magnitude: 2.58, spectral: K },  // Minkar
    Star { ra_deg: 62.165, dec_deg: 47.788, magnitude: 2.62, spectral: G },    // Menkhib
    Star { ra_deg: 122.383, dec_deg: -47.337, magnitude: 2.59, spectral: K },  // Tureis
    Star { ra_deg: 168.527, dec_deg: 20.524, magnitude: 2.56, spectral: A },   // Zosma
    Star { ra_deg: 322.165, dec_deg: -5.571, magnitude: 2.59, spectral: G },   // Sadalmelik
    Star { ra_deg: 283.626, dec_deg: -26.990, magnitude: 2.60, spectral: K },  // Ascella
    Star { ra_deg: 156.523, dec_deg: -16.194, magnitude: 2.61, spectral: G },  // Alkes
    Star { ra_deg: 187.466, dec_deg: -16.516, magnitude: 2.65, spectral: G },  // Gienah
    Star { ra_deg: 243.586, dec_deg: -3.694, magnitude: 2.56, spectral: A },   // Yed Prior
    Star { ra_deg: 275.249, dec_deg: -29.828, magnitude: 2.70, spectral: K },  // Kaus Media
    Star { ra_deg: 230.182, dec_deg: -40.648, magnitude: 2.55, spectral: K },  // Eta Lupi
    Star { ra_deg: 130.821, dec_deg: 3.398, magnitude: 2.64, spectral: K },    // Algieba
    Star { ra_deg: 87.740, dec_deg: -35.768, magnitude: 2.65, spectral: K },   // Phact
    Star { ra_deg: 267.464, dec_deg: -37.043, magnitude: 2.70, spectral: F },  // Kaus Borealis
    Star { ra_deg: 248.971, dec_deg: -28.216, magnitude: 2.62, spectral: B },  // Wei
    Star { ra_deg: 127.566, dec_deg: 26.896, magnitude: 2.61, spectral: K },   // Ras Elased Australis
    Star { ra_deg: 116.112, dec_deg: -37.097, magnitude: 2.45, spectral: O },  // Naos (Zeta Pup)
    Star { ra_deg: 56.871, dec_deg: 24.105, magnitude: 2.64, spectral: B },    // Atik
    Star { ra_deg: 154.993, dec_deg: 19.842, magnitude: 2.98, spectral: A },   // Adhafera
    Star { ra_deg: 28.660, dec_deg: 20.808, magnitude: 2.64, spectral: F },    // Sheratan
    Star { ra_deg: 63.500, dec_deg: -42.294, magnitude: 2.69, spectral: B },   // Arneb
    Star { ra_deg: 186.650, dec_deg: -63.100, magnitude: 2.80, spectral: B },  // Gacrux B
    Star { ra_deg: 181.312, dec_deg: -54.491, magnitude: 2.68, spectral: A },  // Iota Carinae
    Star { ra_deg: 94.138, dec_deg: -6.275, magnitude: 2.68, spectral: F },    // Muliphein

    // Stars magnitude 3.0 - 3.5
    Star { ra_deg: 123.286, dec_deg: 9.186, magnitude: 3.01, spectral: A },    // Alterf
    Star { ra_deg: 251.493, dec_deg: -34.293, magnitude: 3.03, spectral: F },  // Lesath
    Star { ra_deg: 266.433, dec_deg: -27.670, magnitude: 3.03, spectral: G },  // Alnasl
    Star { ra_deg: 319.645, dec_deg: 62.586, magnitude: 3.07, spectral: K },   // Errai
    Star { ra_deg: 290.418, dec_deg: 3.115, magnitude: 3.08, spectral: G },    // Alshain
    Star { ra_deg: 260.502, dec_deg: -24.999, magnitude: 3.11, spectral: B },  // Polis
    Star { ra_deg: 13.429, dec_deg: 15.346, magnitude: 3.04, spectral: K },    // Alrescha
    Star { ra_deg: 33.250, dec_deg: 8.847, magnitude: 3.00, spectral: A },     // Mesartim
    Star { ra_deg: 289.949, dec_deg: -21.107, magnitude: 3.05, spectral: B },  // Albaldah
    Star { ra_deg: 211.097, dec_deg: -26.682, magnitude: 3.02, spectral: K },  // Syrma
    Star { ra_deg: 102.460, dec_deg: -32.509, magnitude: 3.02, spectral: K },  // Sigma CMa
    Star { ra_deg: 280.759, dec_deg: -38.324, magnitude: 2.98, spectral: A },  // Rukbat
    Star { ra_deg: 247.728, dec_deg: 21.490, magnitude: 3.14, spectral: K },   // Kornephoros
    Star { ra_deg: 269.757, dec_deg: 29.247, magnitude: 3.14, spectral: G },   // Rastaban
    Star { ra_deg: 183.857, dec_deg: -58.749, magnitude: 2.77, spectral: A },  // Theta Carinae
    Star { ra_deg: 159.326, dec_deg: -59.229, magnitude: 2.92, spectral: B },  // Upsilon Carinae
    Star { ra_deg: 226.280, dec_deg: -47.388, magnitude: 3.13, spectral: B },  // Zeta Lupi
    Star { ra_deg: 228.071, dec_deg: -52.099, magnitude: 2.30, spectral: B },  // Alpha Lupi
    Star { ra_deg: 245.297, dec_deg: -25.593, magnitude: 2.89, spectral: B },  // Pi Scorpii
    Star { ra_deg: 252.968, dec_deg: -38.047, magnitude: 2.82, spectral: B },  // Mu Scorpii
    Star { ra_deg: 234.256, dec_deg: -29.214, magnitude: 2.90, spectral: B },  // Rho Scorpii
    Star { ra_deg: 85.255, dec_deg: -1.201, magnitude: 3.36, spectral: O },    // Sigma Orionis
    Star { ra_deg: 92.985, dec_deg: 14.768, magnitude: 3.19, spectral: A },    // Tejat
    Star { ra_deg: 83.858, dec_deg: 9.934, magnitude: 3.33, spectral: B },     // Meissa
    Star { ra_deg: 86.821, dec_deg: -14.822, magnitude: 3.02, spectral: A },   // Pi3 Orionis
    Star { ra_deg: 243.859, dec_deg: -19.461, magnitude: 2.74, spectral: B },  // Graffias
    Star { ra_deg: 236.067, dec_deg: 6.426, magnitude: 2.77, spectral: A },    // Unukalhai
    Star { ra_deg: 270.161, dec_deg: 2.932, magnitude: 3.24, spectral: A },    // Cebalrai
    Star { ra_deg: 314.293, dec_deg: 41.167, magnitude: 3.20, spectral: K },   // Rukh
    Star { ra_deg: 21.006, dec_deg: -8.183, magnitude: 3.47, spectral: K },    // Deneb Kaitos Shemali
    Star { ra_deg: 340.750, dec_deg: 30.221, magnitude: 3.41, spectral: G },   // Matar
    Star { ra_deg: 299.077, dec_deg: 35.083, magnitude: 3.24, spectral: B },   // Tarazed
    Star { ra_deg: 53.233, dec_deg: -9.458, magnitude: 3.00, spectral: F },    // Zaurak
    Star { ra_deg: 40.825, dec_deg: 3.236, magnitude: 3.00, spectral: M },     // Menkar
    Star { ra_deg: 26.021, dec_deg: -15.939, magnitude: 3.47, spectral: M },   // Mira
    Star { ra_deg: 75.492, dec_deg: 43.823, magnitude: 3.03, spectral: G },    // Hassaleh
    Star { ra_deg: 74.248, dec_deg: 33.166, magnitude: 2.87, spectral: A },    // Menkib
    Star { ra_deg: 84.412, dec_deg: -2.600, magnitude: 3.69, spectral: O },    // 42 Orionis
    Star { ra_deg: 81.119, dec_deg: -2.397, magnitude: 3.35, spectral: M },    // Hatsya
    Star { ra_deg: 68.499, dec_deg: -55.045, magnitude: 2.85, spectral: A },   // Gamma Pictoris
    Star { ra_deg: 136.039, dec_deg: -47.065, magnitude: 2.25, spectral: A },  // Chi Carinae
    Star { ra_deg: 146.312, dec_deg: -65.072, magnitude: 2.21, spectral: A },  // Omega Carinae
    Star { ra_deg: 161.264, dec_deg: -69.135, magnitude: 2.97, spectral: B },  // PP Carinae
    Star { ra_deg: 171.221, dec_deg: -80.540, magnitude: 3.85, spectral: A },  // Beta Chamaeleontis
    Star { ra_deg: 93.714, dec_deg: -66.937, magnitude: 2.86, spectral: B },   // Beta Pictoris
    Star { ra_deg: 317.585, dec_deg: -53.450, magnitude: 2.87, spectral: B },  // Alpha Gruis
    Star { ra_deg: 43.564, dec_deg: -51.487, magnitude: 3.24, spectral: A },   // Acamar
    Star { ra_deg: 66.009, dec_deg: 17.543, magnitude: 3.53, spectral: B },    // Epsilon Tauri
    Star { ra_deg: 64.948, dec_deg: 15.628, magnitude: 3.65, spectral: G },    // Gamma Tauri
    Star { ra_deg: 67.154, dec_deg: 15.871, magnitude: 3.40, spectral: K },    // Delta Tauri
    Star { ra_deg: 60.170, dec_deg: 12.490, magnitude: 3.91, spectral: A },    // Theta Tauri
    Star { ra_deg: 67.166, dec_deg: 19.181, magnitude: 3.77, spectral: G },    // Epsilon Tauri
    Star { ra_deg: 56.457, dec_deg: 24.113, magnitude: 3.70, spectral: B },    // Zeta Persei
    Star { ra_deg: 45.570, dec_deg: 40.010, magnitude: 3.80, spectral: K },    // Delta Persei
    Star { ra_deg: 46.294, dec_deg: 38.840, magnitude: 3.01, spectral: B },    // Epsilon Persei
    Star { ra_deg: 55.731, dec_deg: 47.788, magnitude: 3.77, spectral: G },    // Kappa Persei
    Star { ra_deg: 39.871, dec_deg: 0.329, magnitude: 3.60, spectral: M },     // Omicron Ceti
    Star { ra_deg: 35.621, dec_deg: -17.987, magnitude: 3.50, spectral: K },   // Tau Ceti
    Star { ra_deg: 24.428, dec_deg: -57.237, magnitude: 3.24, spectral: A },   // Phi Eridani
];

/// Get the size multiplier for a star based on its magnitude.
/// Brighter stars (lower magnitude) get larger sizes.
pub fn magnitude_to_size(magnitude: f32) -> f32 {
    // Map magnitude range [-1.5, 2.5] to size range [1.0, 0.2]
    let normalized = ((magnitude + 1.5) / 4.0).clamp(0.0, 1.0);
    1.0 - normalized * 0.8
}

/// Get the brightness multiplier for a star based on its magnitude.
/// Uses the astronomical magnitude scale (2.512x per magnitude).
pub fn magnitude_to_brightness(magnitude: f32) -> f32 {
    // Reference: magnitude 0 = brightness 1.0
    // Each magnitude step is 2.512x dimmer
    let base_brightness = 2.512_f32.powf(-magnitude);
    // Scale to reasonable range for rendering
    (base_brightness * 1.5).clamp(0.2, 8.0)
}
