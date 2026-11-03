// SPDX-License-Identifier: MIT OR Apache-2.0
//! Behaviour of the noises, patterns and effects added in Milestone 2 (batch 3).
use materialcraft::graph::Graph;
use materialcraft::tex::Tex;
use std::collections::HashMap;

fn run(size: usize, text: &str) -> HashMap<String, Tex> {
    let g = Graph::parse(&format!("size {size}\n{text}")).unwrap_or_else(|e| panic!("parse: {e}"));
    g.evaluate(Some(size)).unwrap_or_else(|e| panic!("eval: {e}")).into_iter().collect()
}

fn mean(t: &Tex) -> f32 {
    t.px.iter().map(|p| p[0]).sum::<f32>() / t.px.len() as f32
}

fn range(t: &Tex) -> (f32, f32) {
    t.px.iter().fold((f32::MAX, f32::MIN), |a, p| (a.0.min(p[0]), a.1.max(p[0])))
}

fn var(t: &Tex) -> f32 {
    let m = mean(t);
    t.px.iter().map(|p| (p[0] - m) * (p[0] - m)).sum::<f32>() / t.px.len() as f32
}

fn px(t: &Tex, u: f32, v: f32) -> f32 {
    t.at((u * t.w as f32) as i64, (v * t.h as f32) as i64)[0]
}

/// Share of the signal's energy in the lowest spatial frequencies: the mean of the image blurred by a few pixels,
/// relative to the whole variance.
fn low_freq_share(t: &Tex) -> f32 {
    let (w, h) = (t.w as i64, t.h as i64);
    let mut blurred = vec![0.0f32; t.px.len()];
    for y in 0..h {
        for x in 0..w {
            let mut a = 0.0;
            for dy in -2..=2 {
                for dx in -2..=2 {
                    a += t.at(x + dx, y + dy)[0];
                }
            }
            blurred[(y * w + x) as usize] = a / 25.0;
        }
    }
    let m = mean(t);
    let v: f32 = blurred.iter().map(|b| (b - m) * (b - m)).sum::<f32>() / blurred.len() as f32;
    v / var(t).max(1e-9)
}

#[test]
fn blue_noise_has_less_low_frequency_energy_than_white() {
    let o = run(128, "node 1 blue_noise scale=128 seed=3\nnode 2 white_noise seed=3\noutput blue 1\noutput white 2\n");
    let (lo, hi) = range(&o["blue"]);
    assert!(lo >= 0.0 && hi <= 1.0 && hi - lo > 0.9, "flat histogram over 0..1");
    assert!((mean(&o["blue"]) - 0.5).abs() < 0.02);
    let (b, w) = (low_freq_share(&o["blue"]), low_freq_share(&o["white"]));
    assert!(b < w * 0.5, "blue {b} white {w}");
}

#[test]
fn fibers_and_scratches_draw_thin_marks() {
    let o = run(128, "node 1 fibers seed=1\nnode 2 scratches seed=1\nnode 3 scratches seed=1 count=0\noutput f 1\noutput s 2\noutput none 3\n");
    for k in ["f", "s"] {
        let m = mean(&o[k]);
        assert!(m > 0.005 && m < 0.35, "{k}: coverage {m}");
        assert!(range(&o[k]).1 > 0.5);
    }
    assert!(mean(&o["none"]) == 0.0, "no strands, no marks");
    // different seeds give different marks
    let b = run(128, "node 1 fibers seed=2\noutput f 1\n");
    assert!(o["f"].px.iter().zip(&b["f"].px).any(|(p, q)| (p[0] - q[0]).abs() > 0.5));
}

#[test]
fn caustics_are_bright_lines_on_dark() {
    let o = run(128, "node 1 caustics scale=4 seed=2\noutput o 1\n");
    let (lo, hi) = range(&o["o"]);
    assert!(lo >= 0.0 && hi <= 1.0 + 1e-5 && hi > 0.8 && lo < 0.3, "{lo} {hi}");
    assert!(mean(&o["o"]) < 0.6, "mostly darker than the lines: {}", mean(&o["o"]));
    let flat = run(64, "node 1 caustics scale=4 strength=0\noutput o 1\n");
    assert!(var(&flat["o"]) < 1e-6, "no refraction, no pattern");
}

#[test]
fn liquid_noise_modes() {
    let o = run(64, "node 1 liquid_noise seed=2\nnode 2 liquid_noise seed=2 mode=1\nnode 3 liquid_noise seed=2 warp=0\nnode 4 fbm scale=3 octaves=4 seed=2\noutput liquid 1\noutput marble 2\noutput plain 3\noutput fbm 4\n");
    assert!(var(&o["liquid"]) > 0.001 && var(&o["marble"]) > 0.01);
    assert!(o["liquid"].px.iter().zip(&o["plain"].px).any(|(p, q)| (p[0] - q[0]).abs() > 0.05), "warp changes the noise");
}

#[test]
fn perlin_3d_loops_in_depth_and_varies_between_slices() {
    let o = run(64, "node 1 perlin_3d scale=4 depth=0 seed=5\nnode 2 perlin_3d scale=4 depth=0.25 seed=5\nnode 3 perlin_3d scale=4 depth=1 seed=5\noutput a 1\noutput b 2\noutput c 3\n");
    let d = |x: &Tex, y: &Tex| x.px.iter().zip(&y.px).map(|(p, q)| (p[0] - q[0]).abs()).sum::<f32>() / x.px.len() as f32;
    assert!(d(&o["a"], &o["c"]) < 1e-3, "depth 1 is depth 0 again");
    assert!(d(&o["a"], &o["b"]) > 0.02, "different slices differ");
    let (lo, hi) = range(&o["a"]);
    assert!(lo > 0.0 && hi < 1.0 && hi - lo > 0.4, "{lo} {hi}");
}

#[test]
fn paving_outputs() {
    let o = run(128, "node 1 paving scale=6 seed=4\noutput h 1\noutput id 1.cell_id\noutput gaps 1.gaps\n");
    let gaps = mean(&o["gaps"]);
    assert!(gaps > 0.02 && gaps < 0.5, "gap coverage {gaps}");
    // gaps are low, stones are high
    let (mut in_gap, mut n_gap, mut on_stone, mut n_stone) = (0.0, 0.0, 0.0, 0.0);
    for i in 0..o["h"].px.len() {
        if o["gaps"].px[i][0] > 0.9 {
            in_gap += o["h"].px[i][0];
            n_gap += 1.0;
        } else if o["gaps"].px[i][0] < 0.01 {
            on_stone += o["h"].px[i][0];
            n_stone += 1.0;
        }
    }
    assert!(n_gap > 0.0 && n_stone > 0.0 && in_gap / n_gap < on_stone / n_stone * 0.3);
    assert!(var(&o["id"]) > 0.02, "each stone has its own id");
}

#[test]
fn wire_mesh_has_wires_and_gaps() {
    let o = run(128, "node 1 wire_mesh count_x=8 count_y=8 wire_width=0.3\nnode 2 wire_mesh count_x=8 count_y=8 mode=1\noutput sq 1\noutput dia 2\n");
    for k in ["sq", "dia"] {
        let (lo, hi) = range(&o[k]);
        assert!(lo == 0.0 && hi > 0.5, "{k}: {lo} {hi}");
        let m = mean(&o[k]);
        assert!(m > 0.03 && m < 0.7, "{k}: coverage {m}");
    }
    assert!(o["sq"].px.iter().zip(&o["dia"].px).any(|(p, q)| (p[0] - q[0]).abs() > 0.3));
}

#[test]
fn halftone_dot_size_follows_brightness() {
    let o = run(128, "node 1 pixel_processor expr=\"u\"\nnode 2 halftone in=@1 cells=16\nnode 3 halftone in=@1 cells=16 mode=1\noutput g 2\noutput d 3\n");
    let side = |t: &Tex, a: usize, b: usize| (a..b).map(|x| (0..t.h).map(|y| t.px[y * t.w + x][0]).sum::<f32>()).sum::<f32>() / ((b - a) * t.h) as f32;
    for k in ["g", "d"] {
        assert!(side(&o[k], 96, 128) > side(&o[k], 0, 32) * 2.0, "{k}: bright side has bigger dots");
    }
    let black = run(64, "node 1 color r=0 g=0 b=0\nnode 2 halftone in=@1\noutput o 2\n");
    assert!(mean(&black["o"]) == 0.0);
}

#[test]
fn reaction_diffusion_grows_a_pattern() {
    let o = run(64, "node 1 sdf_shape shape=0 size=0.1\nnode 2 sdf_mask in=@1 offset=0 softness=0.001\nnode 3 reaction_diffusion in=@2 resolution=64 iterations=400\nnode 4 color r=0 g=0 b=0\nnode 5 reaction_diffusion in=@4 resolution=64 iterations=400 spots=0\noutput rd 3\noutput none 5\n");
    assert!(var(&o["rd"]) > 0.005, "pattern variance {}", var(&o["rd"]));
    assert!(var(&o["none"]) < 1e-8, "nothing to grow from");
}

#[test]
fn fxaa_smooths_hard_edges_only() {
    let o = run(64, "node 0 gradient angle=0.08\nnode 1 threshold in=@0 threshold=0.5\nnode 2 fxaa in=@1\nnode 3 color r=0.4 g=0.4 b=0.4\nnode 4 fxaa in=@3\nnode 5 fxaa in=@1 strength=0\noutput src 1\noutput aa 2\noutput flat 4\noutput off 5\noutput c 3\n");
    let grey = |t: &Tex| t.px.iter().filter(|p| p[0] > 0.02 && p[0] < 0.98).count();
    assert!(grey(&o["aa"]) > grey(&o["src"]), "edges get intermediate values");
    assert!(o["flat"].px.iter().zip(&o["c"].px).all(|(p, q)| (p[0] - q[0]).abs() < 1e-6), "flat areas are untouched");
    assert!(o["off"].px.iter().zip(&o["src"].px).all(|(p, q)| (p[0] - q[0]).abs() < 1e-6), "strength 0 is the identity");
}

#[test]
fn bilateral_blur_keeps_edges_but_smooths_noise() {
    let o = run(64, "node 1 checker tiles=2\nnode 2 white_noise seed=2\nnode 3 blend fg=@2 bg=@1 mode=1 opacity=0.08\nnode 4 bilateral_blur in=@3 radius=32 range=0.1\nnode 5 blur in=@3 radius=4\noutput noisy 3\noutput bil 4\noutput gauss 5\n");
    // inside one checker square (away from the edges) the bilateral blur removes the grain
    let rough = |t: &Tex| (8..24usize).map(|y| (8..23usize).map(|x| (t.px[y * t.w + x][0] - t.px[y * t.w + x + 1][0]).abs()).sum::<f32>()).sum::<f32>();
    assert!(rough(&o["bil"]) < rough(&o["noisy"]) * 0.5, "smoother than the input: {} vs {}", rough(&o["bil"]), rough(&o["noisy"]));
    // across the checker edge the jump survives much better than with a gaussian blur
    let jump = |t: &Tex| (px(t, 0.52, 0.25) - px(t, 0.48, 0.25)).abs();
    assert!(jump(&o["bil"]) > jump(&o["gauss"]), "bilateral {} gauss {}", jump(&o["bil"]), jump(&o["gauss"]));
}
