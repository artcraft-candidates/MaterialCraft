// SPDX-License-Identifier: MIT OR Apache-2.0
//! Our own grunge map family (twenty-four procedural looks: specks, drips, blotches, scratches, crazing, cracks, pits,
//! stains, scuffs, peeling patches).
//! Each recipe builds a tileable "dirt" field from our noise primitives; `GrungeP` then thresholds it with balance /
//! contrast like a grunge map. The recipes are our own procedural designs.
use crate::noise;

#[derive(Clone, Copy)]
pub struct GrungeP {
    pub map: u32,
    pub seed: u32,
    pub balance: f32,
    pub contrast: f32,
    pub brush: f32,
    pub cracks: f32,
    pub disorder: f32,
}

fn smooth(a: f32, b: f32, x: f32) -> f32 {
    let t = ((x - a) / (b - a).max(1e-6)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Coverage threshold of each map (dirt where the field exceeds it); tuned so the default looks have a similar
/// amount of dirt to the originals.
pub const T0: [f32; 24] = [0.846, 0.733, 0.695, 0.214, 0.849, 0.775, 0.672, 0.504, 0.695, 0.405, 0.432, 0.797, 0.473, 0.701, 0.783, 0.648, 0.816, 0.669, 0.693, 0.833, 0.695, 1.009, 0.337, 0.802];

fn blot<N: Src>(n: &N, a: f32, b: f32, px: i32, seed: u32) -> f32 {
    n.fbm(a, b, px, px, 5, 0.55, seed)
}

fn streak<N: Src>(n: &N, a: f32, b: f32, px: i32, py: i32, seed: u32) -> f32 {
    n.fbm(a, b, px, py, 4, 0.55, seed)
}

fn vband(b: f32, centre: f32) -> f32 {
    0.5 + 0.5 * (std::f32::consts::TAU * (b - centre)).cos()
}

fn crack<N: Src>(n: &N, a: f32, b: f32, px: i32, seed: u32, width: f32) -> f32 {
    // jagged hairline cracks: a fine warp roughens the cell borders, the width wanders along each crack, short gaps
    // break them up and a second, smaller network adds branches
    let jx = n.fbm(a, b, 48, 48, 3, 0.6, seed.wrapping_add(61)) - 0.5;
    let jy = n.fbm(a, b, 48, 48, 3, 0.6, seed.wrapping_add(62)) - 0.5;
    let (ja, jb) = (noise::wrap01(a + jx * 0.03), noise::wrap01(b + jy * 0.03));
    let wv = 0.35 + 1.3 * n.fbm(a, b, 36, 36, 2, 0.5, seed.wrapping_add(63));
    let gap = smooth(0.28, 0.5, n.fbm(a, b, 28, 28, 3, 0.55, seed.wrapping_add(64)));
    let (f1, f2, _) = noise::voronoi(ja, jb, px, px, seed, 1.0);
    let main = 1.0 - smooth(0.0, width * wv, f2 - f1);
    let (g1, g2, _) = noise::voronoi(ja, jb, px * 2 + 1, px * 2 + 1, seed.wrapping_add(65), 1.0);
    let branch = (1.0 - smooth(0.0, width * 0.5 * wv, g2 - g1)) * smooth(0.45, 0.65, n.fbm(a, b, 14, 14, 3, 0.55, seed.wrapping_add(66)));
    (main.max(branch * 0.9)) * gap
}

/// Dirt field in 0..1 (higher = more dirt) for map 1..=24.
pub fn dirt<N: Src>(n: &N, p: &GrungeP, u: f32, v: f32) -> f32 {
    let s = p.seed;
    let (mut a, mut b) = (u, v);
    if p.disorder > 0.0 {
        let w = n.fbm(u, v, 4, 4, 3, 0.5, s.wrapping_add(900)) - 0.5;
        let w2 = n.fbm(u, v, 4, 4, 3, 0.5, s.wrapping_add(901)) - 0.5;
        a = noise::wrap01(u + w * p.disorder * 0.2);
        b = noise::wrap01(v + w2 * p.disorder * 0.2);
    }
    // every map gets a ragged edge: a small turbulent warp plus fine grain, so nothing looks like a clean cell or disc
    let rag = if matches!(p.map, 9..=11 | 13) { 0.02 } else { 0.03 };
    let (wu, wv) = (n.fbm(a, b, 20, 20, 4, 0.55, s.wrapping_add(940)) - 0.5, n.fbm(a, b, 20, 20, 4, 0.55, s.wrapping_add(941)) - 0.5);
    let (a, b) = (noise::wrap01(a + wu * rag * 2.0), noise::wrap01(b + wv * rag * 2.0));
    let grain = (n.fbm(a, b, 200, 200, 2, 0.6, s.wrapping_add(950)) - 0.5) * 0.25;
    let d = match p.map {
        1 => 0.55 * blot(n, a, b, 6, s) + 0.45 * n.fbm(a, b, 96, 96, 2, 0.5, s + 1),
        2 => 0.6 * streak(n, a, b, 48, 3, s) + 0.4 * blot(n, a, b, 4, s + 2),
        3 => 0.65 * streak(n, a, b, 24, 2, s) + 0.35 * blot(n, a, b, 3, s + 3),
        4 => 1.0 - noise::spots(a, b, 7, 7, s, |id| (0.2, 0.3 + id * 0.35), 0.0) * smooth(0.25, 0.5, blot(n, a, b, 3, s + 4)).max(0.35),
        5 => streak(n, a, b, 96, 2, s),
        6 => 0.5 * blot(n, a, b, 4, s) + 0.5 * vband(b, 0.95).powi(2),
        7 => {
            let w = blot(n, a, b, 5, s + 6);
            0.7 * blot(n, noise::wrap01(a + w * 0.1), noise::wrap01(b + w * 0.1), 7, s) + 0.3 * n.fbm(a, b, 40, 40, 3, 0.5, s + 1)
        }
        8 => noise::spots(a, b, 40, 40, s, |id| (0.04, 0.08 + id * 0.1), 0.0) * 0.6 + 0.6 * blot(n, a, b, 6, s + 7) - 0.2,
        9 => 0.5 * blot(n, a, b, 8, s) + 0.5 * n.fbm(a, b, 128, 128, 2, 0.6, s + 8),
        10 => 0.7 * crack(n, a, b, 14, s, 0.05) * smooth(0.35, 0.6, blot(n, a, b, 5, s + 20)) + 0.65 * blot(n, a, b, 6, s + 9) - 0.15,
        11 => p.cracks * 1.0 * crack(n, a, b, 10, s, 0.06) * smooth(0.3, 0.55, blot(n, a, b, 4, s + 21)) + 0.5 * blot(n, a, b, 6, s + 10),
        12 => noise::spots(a, b, 56, 56, s, |id| (0.03, 0.06 + id * 0.05), 0.0) * 0.7 + 0.45 * blot(n, a, b, 8, s + 12),
        13 => p.cracks * 0.9 * crack(n, a, b, 6, s, 0.04) * smooth(0.4, 0.6, blot(n, a, b, 4, s + 22)) + 0.45 * blot(n, a, b, 6, s + 11),
        14 => 0.6 * streak(n, a, b, 30, 2, s) + 0.4 * blot(n, a, b, 5, s + 13) * 1.2,
        15 => streak(n, a, b, 120, 6, s) * (0.55 + 0.6 * vband(b, 0.0)),
        // rain streaks: long vertical runs that start from a few sources and fade towards the bottom
        16 => (0.65 * streak(n, a, b, 72, 2, s + 16) + 0.35 * blot(n, a, b, 3, s + 17)) * (0.6 + 0.5 * vband(b, 0.0)),
        // sponge: open pores of different sizes
        17 => {
            let (f1, _, id) = noise::voronoi(a, b, 14, 14, s + 18, 1.0);
            0.65 * (1.0 - smooth(0.05, 0.3 + id * 0.25, f1)) + 0.4 * blot(n, a, b, 6, s + 19)
        }
        // chalky dust: fine grain with a broad uneven coat
        18 => 0.55 * n.fbm(a, b, 160, 160, 4, 0.6, s + 20) + 0.45 * blot(n, a, b, 3, s + 21),
        // corrosion pits: small deep pits inside patchy corroded areas
        19 => noise::spots(a, b, 26, 26, s + 22, |id| (0.05, 0.12 + id * 0.2), 0.0) * smooth(0.3, 0.55, blot(n, a, b, 4, s + 23)) + 0.5 * blot(n, a, b, 7, s + 24) - 0.1,
        // ring stains: dried rims of spills
        20 => {
            let (f1, _, id) = noise::voronoi(a, b, 5, 5, s + 25, 1.0);
            let r = f1 / (0.45 + 0.4 * id);
            smooth(0.55, 0.9, r) * (1.0 - smooth(0.9, 1.1, r)) * 0.9 + 0.35 * blot(n, a, b, 5, s + 26)
        }
        // scuffs: short diagonal scrapes
        21 => {
            let (da, db) = (noise::wrap01(a + b), noise::wrap01(b - a));
            0.7 * streak(n, da, db, 56, 2, s + 27) + 0.3 * blot(n, a, b, 4, s + 28)
        }
        // peeling patches: irregular flakes with ragged borders
        22 => {
            let (_, _, id) = noise::voronoi(a, b, 6, 6, s + 29, 1.0);
            let edge = n.fbm(a, b, 30, 30, 4, 0.55, s + 30) - 0.5;
            smooth(0.4, 0.6, id + edge * 0.5) * 0.8 + 0.3 * blot(n, a, b, 7, s + 31)
        }
        // salt and speckle: tiny crisp dots under a loose coat
        23 => noise::spots(a, b, 96, 96, s + 32, |id| (0.04, 0.09 + id * 0.1), 0.0) * smooth(0.3, 0.6, blot(n, a, b, 3, s + 33)) + 0.35 * blot(n, a, b, 5, s + 34),
        // weathering: broad uneven staining, heavier along one band
        _ => 0.55 * blot(n, a, b, 2, s + 35) + 0.45 * vband(b, 0.5) * (0.6 + 0.8 * blot(n, a, b, 4, s + 36)),
    };
    let d = if p.brush > 0.0 { d * (1.0 + p.brush * (streak(n, a, b, 60, 4, s + 31) - 0.5) * 1.5) } else { d };
    (d + grain).clamp(0.0, 1.5)
}

/// Where the recipes get their fractal noise from (always fBm mode 0).
pub trait Src {
    fn fbm(&self, a: f32, b: f32, px: i32, py: i32, octaves: u32, gain: f32, seed: u32) -> f32;
}

/// Plain per-pixel noise (hashes every lattice corner).
pub struct Plain;

impl Src for Plain {
    #[inline]
    fn fbm(&self, a: f32, b: f32, px: i32, py: i32, octaves: u32, gain: f32, seed: u32) -> f32 {
        noise::fbm(a, b, px, py, octaves, gain, seed, 0)
    }
}

type Key = (i32, i32, u32, u32, u32);

/// Notes which noises a map uses (they depend on the settings only, never on the pixel).
struct Rec(std::cell::RefCell<Vec<Key>>);

impl Src for Rec {
    fn fbm(&self, a: f32, b: f32, px: i32, py: i32, octaves: u32, gain: f32, seed: u32) -> f32 {
        let k = (px, py, octaves, gain.to_bits(), seed);
        let mut v = self.0.borrow_mut();
        if !v.contains(&k) {
            v.push(k);
        }
        noise::fbm(a, b, px, py, octaves, gain, seed, 0)
    }
}

/// The noises of one grunge map with their gradient lattices built once, so a pixel only interpolates. Gives the
/// same values as [`Plain`].
pub struct Kit {
    e: Vec<(Key, noise::Fbm)>,
}

impl Kit {
    pub fn new(p: &GrungeP) -> Kit {
        let rec = Rec(std::cell::RefCell::new(Vec::new()));
        // every noise a map uses is chosen by its settings, so one sample discovers all of them
        let _ = dirt(&rec, p, 0.37, 0.61);
        let e = rec.0.into_inner().into_iter().map(|k| (k, noise::Fbm::new(k.0, k.1, k.2, f32::from_bits(k.3), k.4, 0))).collect();
        Kit { e }
    }
}

impl Src for Kit {
    #[inline]
    fn fbm(&self, a: f32, b: f32, px: i32, py: i32, octaves: u32, gain: f32, seed: u32) -> f32 {
        let k = (px, py, octaves, gain.to_bits(), seed);
        for (key, f) in &self.e {
            if *key == k {
                return f.sample(a, b);
            }
        }
        noise::fbm(a, b, px, py, octaves, gain, seed, 0)
    }
}

/// Final grunge value: white where clean, black where dirty (before inversion).
pub fn grunge_map(p: &GrungeP, u: f32, v: f32, t0: f32) -> f32 {
    grunge_map_with(&Plain, p, u, v, t0)
}

/// [`grunge_map`] with a chosen noise source (a [`Kit`] for whole images).
pub fn grunge_map_with<N: Src>(n: &N, p: &GrungeP, u: f32, v: f32, t0: f32) -> f32 {
    let d = dirt(n, p, u, v);
    let x = 0.5 + (t0 - d) * 1.5 + (p.balance - 0.5) * 0.5;
    let w = (1.0 - p.contrast.clamp(0.0, 1.0)) * 0.45 + 0.01;
    smooth(0.5 - w, 0.5 + w, x)
}
