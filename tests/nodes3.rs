// SPDX-License-Identifier: MIT OR Apache-2.0
//! Behaviour of the adjustment, transform and normal-map nodes added in Milestone 2 (batch 1).
use materialcraft::graph::Graph;
use materialcraft::tex::Tex;
use std::collections::HashMap;

fn run(size: usize, text: &str) -> HashMap<String, Tex> {
    let g = Graph::parse(&format!("size {size}\n{text}")).unwrap_or_else(|e| panic!("parse: {e}"));
    g.evaluate(Some(size)).unwrap_or_else(|e| panic!("eval: {e}")).into_iter().collect()
}

fn mean(t: &Tex, c: usize) -> f32 {
    t.px.iter().map(|p| p[c]).sum::<f32>() / t.px.len() as f32
}

fn diff(a: &Tex, b: &Tex) -> f32 {
    a.px.iter().zip(&b.px).map(|(p, q)| (0..3).map(|c| (p[c] - q[c]).abs()).fold(0.0f32, f32::max)).sum::<f32>() / a.px.len() as f32
}

fn px(t: &Tex, u: f32, v: f32) -> [f32; 4] {
    t.at((u * t.w as f32) as i64, (v * t.h as f32) as i64)
}

const HILL: &str = "node 1 pixel_processor expr=\"0.5+0.5*cos((u-0.5)*6.2831853)*cos((v-0.5)*6.2831853)\"\n";

#[test]
fn srgb_curves_round_trip_and_keep_alpha() {
    let o = run(32, "node 1 pixel_processor expr=\"u\"\nnode 2 srgb_to_linear in=@1\nnode 3 linear_to_srgb in=@2\nnode 4 color r=0.5 g=0.5 b=0.5 a=0.25\nnode 5 srgb_to_linear in=@4\noutput src 1\nlin 2\noutput back 3\noutput half 5\n".replace("\nlin 2\n", "\noutput lin 2\n").as_str());
    assert!(diff(&o["src"], &o["back"]) < 1e-3);
    let p = px(&o["half"], 0.5, 0.5);
    assert!((p[0] - 0.214).abs() < 0.002, "0.5 sRGB is 0.214 linear, got {}", p[0]);
    assert!((p[3] - 0.25).abs() < 1e-6, "alpha untouched");
    assert!(mean(&o["lin"], 0) < mean(&o["src"], 0), "decoding darkens mid-tones");
}

#[test]
fn contrast_and_luminosity() {
    let o = run(16, "node 1 color r=0.75 g=0.75 b=0.75\nnode 2 contrast_luminosity in=@1 contrast=2\nnode 3 contrast_luminosity in=@1 luminosity=0.1\noutput a 2\noutput b 3\n");
    assert!((px(&o["a"], 0.5, 0.5)[0] - 1.0).abs() < 1e-4);
    assert!((px(&o["b"], 0.5, 0.5)[0] - 0.85).abs() < 1e-4);
}

#[test]
fn premultiply_round_trips() {
    let o = run(16, "node 1 color r=0.8 g=0.4 b=0.2 a=0.5\nnode 2 premultiply in=@1\nnode 3 premultiply in=@2 mode=1\noutput pre 2\noutput back 3\n");
    let p = px(&o["pre"], 0.5, 0.5);
    assert!((p[0] - 0.4).abs() < 1e-5 && (p[3] - 0.5).abs() < 1e-6);
    assert!((px(&o["back"], 0.5, 0.5)[0] - 0.8).abs() < 1e-5);
}

#[test]
fn histogram_equalize_spreads_values() {
    let o = run(64, "node 1 pixel_processor expr=\"0.3+0.1*u\"\nnode 2 histogram_equalize in=@1\nnode 3 histogram_equalize in=@1 amount=0\noutput src 1\noutput eq 2\noutput none 3\n");
    let range = |t: &Tex| {
        let (lo, hi) = t.px.iter().fold((f32::MAX, f32::MIN), |a, p| (a.0.min(p[0]), a.1.max(p[0])));
        hi - lo
    };
    assert!(range(&o["src"]) < 0.11 && range(&o["eq"]) > 0.9, "{} -> {}", range(&o["src"]), range(&o["eq"]));
    assert!(diff(&o["src"], &o["none"]) < 1e-5, "amount 0 changes nothing");
    assert!((mean(&o["eq"], 0) - 0.5).abs() < 0.05);
}

#[test]
fn min_max_picks_per_channel() {
    let o = run(16, "node 1 pixel_processor expr=\"u\"\nnode 2 pixel_processor expr=\"1-u\"\nnode 3 min_max a=@1 b=@2 mode=0\nnode 4 min_max a=@1 b=@2 mode=1\noutput lo 3\noutput hi 4\n");
    assert!(o["lo"].px.iter().all(|p| p[0] <= 0.5 + 1e-4));
    assert!(o["hi"].px.iter().all(|p| p[0] >= 0.5 - 1e-4));
}

#[test]
fn morphology_grows_shrinks_and_wraps() {
    let dot = "node 1 sdf_shape shape=0 size=0.12 x=-0.5 y=-0.5\nnode 2 sdf_mask in=@1 offset=0 softness=0.001\n";
    let o = run(64, &format!("{dot}node 3 morphology in=@2 mode=0 radius=32\nnode 4 morphology in=@2 mode=1 radius=16\nnode 5 morphology in=@4 mode=0 radius=16\noutput src 2\noutput dil 3\noutput ero 4\noutput open 5\n"));
    let (s, d, e) = (mean(&o["src"], 0), mean(&o["dil"], 0), mean(&o["ero"], 0));
    assert!(s > 0.0 && d > s * 1.5 && e < s, "src {s} dil {d} ero {e}");
    // the dot straddles the corner, so growth must continue across the wrapped border
    assert!(px(&o["dil"], 0.99, 0.5)[0] >= px(&o["src"], 0.99, 0.5)[0]);
    assert!(mean(&o["open"], 0) <= s + 1e-4, "opening never adds");
}

#[test]
fn multi_switch_selects_input() {
    let o = run(16, "node 1 color r=1 g=0 b=0\nnode 2 color r=0 g=1 b=0\nnode 3 multi_switch a=@1 c=@2 index=2\nnode 4 multi_switch a=@1 c=@2 index=5\noutput pick 3\noutput fallback 4\n");
    assert!((px(&o["pick"], 0.5, 0.5)[1] - 1.0).abs() < 1e-6);
    assert!((px(&o["fallback"], 0.5, 0.5)[0] - 1.0).abs() < 1e-6, "a missing input falls back to the first connected one");
}

#[test]
fn lookup_table_maps_values() {
    let o = run(32, "node 1 pixel_processor expr=\"u\"\nnode 2 pixel_processor expr=\"1-u\"\nnode 3 lookup_table in=@1 lut=@2\nnode 4 lookup_table in=@1 lut=@1\noutput inv 3\noutput same 4\n");
    assert!(px(&o["inv"], 0.1, 0.5)[0] > 0.8 && px(&o["inv"], 0.9, 0.5)[0] < 0.2, "reversed ramp reverses the image");
    assert!(diff(&o["same"], &run(32, "node 1 pixel_processor expr=\"u\"\noutput o 1\n")["o"]) < 0.02, "identity ramp is the identity");
}

#[test]
fn frequency_bands_recombine() {
    let o = run(64, "node 1 fbm scale=6 octaves=5 seed=3\nnode 2 frequency_split in=@1 radius=6\nnode 3 frequency_merge low=@2 high=@2.high\noutput src 1\noutput low 2\noutput high 2.high\noutput back 3\n");
    assert!(diff(&o["src"], &o["back"]) < 1e-4);
    assert!((mean(&o["high"], 0) - 0.5).abs() < 0.02, "detail band is centred on gray");
}

#[test]
fn auto_crop_fills_the_image() {
    let o = run(64, "node 1 sdf_shape shape=1 size=0.15\nnode 2 sdf_mask in=@1 offset=0 softness=0.001\nnode 3 auto_crop in=@2\noutput src 2\noutput crop 3\n");
    assert!(mean(&o["crop"], 0) > mean(&o["src"], 0) * 4.0, "{} vs {}", mean(&o["crop"], 0), mean(&o["src"], 0));
    let flat = run(32, "node 1 color r=0 g=0 b=0\nnode 2 auto_crop in=@1\noutput o 2\n");
    assert!(flat["o"].px.iter().all(|p| p[0] == 0.0), "an empty image stays empty");
}

#[test]
fn polar_convert_turns_rings_into_rows() {
    // distance from the centre: constant along each row after the cartesian-to-polar conversion
    let o = run(64, "node 1 pixel_processor expr=\"sqrt((u-0.5)*(u-0.5)+(v-0.5)*(v-0.5))\"\nnode 2 polar_convert in=@1 tile=0 radius=0.9\noutput o 2\n");
    let t = &o["o"];
    let row = |y: usize| (0..t.w).map(|x| t.px[y * t.w + x][0]).collect::<Vec<_>>();
    let r = row(10);
    let spread = r.iter().cloned().fold(f32::MIN, f32::max) - r.iter().cloned().fold(f32::MAX, f32::min);
    assert!(spread < 0.03, "row spread {spread}");
    assert!(t.px[2 * t.w][0] > t.px[30 * t.w][0], "the centre row is the centre of the image");
    let back = run(64, "node 1 pixel_processor expr=\"sqrt((u-0.5)*(u-0.5)+(v-0.5)*(v-0.5))\"\nnode 2 polar_convert in=@1 tile=0 radius=1\nnode 3 polar_convert in=@2 mode=1 tile=0 radius=1\noutput o 3\n");
    let b = &back["o"];
    assert!(px(b, 0.5, 0.5)[0] < px(b, 0.5, 0.15)[0], "distance still grows away from the centre after the round trip");
}

#[test]
fn clone_patch_copies_a_region() {
    let o = run(64, "node 1 pixel_processor expr=\"u\"\nnode 2 clone_patch in=@1 src_x=0.25 src_y=0.25 dst_x=0.75 dst_y=0.75 width=0.2 height=0.2 feather=0.1\noutput o 2\noutput src 1\n");
    assert!((px(&o["o"], 0.75, 0.75)[0] - 0.25).abs() < 0.03, "got {}", px(&o["o"], 0.75, 0.75)[0]);
    assert!((px(&o["o"], 0.9, 0.2)[0] - 0.9).abs() < 0.02, "outside the patch nothing changes");
}

#[test]
fn noise_upscale_smooths_and_blocks() {
    let src = "node 1 white_noise seed=3\n";
    let o = run(64, &format!("{src}node 2 noise_upscale in=@1 cells=4\nnode 3 noise_upscale in=@1 cells=4 mode=2\noutput src 1\noutput soft 2\noutput blocks 3\n"));
    let rough = |t: &Tex| (0..t.h).map(|y| (0..t.w - 1).map(|x| (t.px[y * t.w + x][0] - t.px[y * t.w + x + 1][0]).abs()).sum::<f32>()).sum::<f32>() / (t.w * t.h) as f32;
    assert!(rough(&o["soft"]) < rough(&o["src"]) * 0.2);
    let b = &o["blocks"];
    assert_eq!(b.px[0][0], b.px[1][0], "nearest mode keeps whole blocks");
}

#[test]
fn symmetry_slice_changes_the_image_and_tiles() {
    let o = run(64, "node 1 fbm scale=3 seed=5\nnode 2 symmetry_slice in=@1 slices=6\noutput src 1\noutput o 2\n");
    assert!(diff(&o["src"], &o["o"]) > 0.01);
    assert!(o["o"].px.iter().all(|p| p[0].is_finite()));
}

#[test]
fn quad_and_trapezoid_transforms() {
    let o = run(64, "node 1 fbm scale=3 seed=5\nnode 2 quad_transform in=@1 tile=0\nnode 3 quad_transform in=@1 tl_x=0.2 tr_x=-0.2 tile=0\nnode 4 trapezoid_transform in=@1 top=1 bottom=1 tile=0\nnode 5 trapezoid_transform in=@1 top=0.5 tile=0\noutput src 1\noutput same 2\noutput moved 3\noutput flat 4\noutput taper 5\n");
    assert!(diff(&o["src"], &o["same"]) < 1e-5, "no offsets = identity");
    assert!(diff(&o["src"], &o["flat"]) < 1e-5, "a rectangle = identity");
    assert!(diff(&o["src"], &o["moved"]) > 0.02 && diff(&o["src"], &o["taper"]) > 0.02);
}

#[test]
fn bent_normal_leans_away_from_walls() {
    let o = run(64, &format!("{HILL}node 2 bent_normal in=@1 relief=1 radius=24\nnode 3 normal in=@1 strength=2\nnode 4 constant_flat\noutput bent 2\noutput ao 2.ao\noutput n 3\n").replace("node 4 constant_flat\n", ""));
    // left flank of the hill: the surface faces left, and so does the bent normal
    let (b, n) = (px(&o["bent"], 0.3, 0.5), px(&o["n"], 0.3, 0.5));
    assert!(n[0] < 0.5 && b[0] < 0.5, "normal {:?} bent {:?}", n, b);
    // the pit at the corner is more occluded than the hill top
    assert!(px(&o["ao"], 0.5, 0.008)[0] < px(&o["ao"], 0.5, 0.5)[0], "pit {} top {}", px(&o["ao"], 0.5, 0.008)[0], px(&o["ao"], 0.5, 0.5)[0]);
    let flat = run(32, "node 1 color r=0.5 g=0.5 b=0.5\nnode 2 bent_normal in=@1\noutput n 2\noutput ao 2.ao\n");
    let p = px(&flat["n"], 0.5, 0.5);
    assert!((p[0] - 0.5).abs() < 0.01 && (p[1] - 0.5).abs() < 0.01 && p[2] > 0.99);
    assert!(px(&flat["ao"], 0.5, 0.5)[0] > 0.99, "a flat surface is not occluded");
}

#[test]
fn normal_facing_masks_directions() {
    let o = run(32, &format!("{HILL}node 2 normal in=@1 strength=3\nnode 3 normal_facing in=@2\nnode 4 normal_facing in=@2 tilt=0.6 angle=0.5\nnode 5 normal_facing in=@2 tilt=0.6 angle=0\noutput up 3\noutput left 4\noutput right 5\n"));
    // flat top faces up; the left flank faces left (angle 0.5 turns = -x)
    assert!(px(&o["up"], 0.5, 0.5)[0] > 0.99);
    assert!(px(&o["left"], 0.3, 0.5)[0] > px(&o["right"], 0.3, 0.5)[0]);
}

#[test]
fn height_normal_blend_adds_relief() {
    let o = run(64, &format!("{HILL}node 2 color r=0.5 g=0.5 b=1\nnode 3 height_normal_blend normal=@2 height=@1 strength=2\nnode 4 normal in=@1 strength=2\nnode 5 height_normal_blend normal=@2 height=@1 opacity=0\noutput blend 3\noutput direct 4\noutput base 5\noutput flat 2\n"));
    assert!(diff(&o["blend"], &o["direct"]) < 0.02, "a flat base normal gives the height's own normal map: {}", diff(&o["blend"], &o["direct"]));
    assert!(diff(&o["base"], &o["flat"]) < 1e-5, "opacity 0 keeps the base");
}

#[test]
fn world_unit_normals_scale_with_height() {
    let o = run(64, &format!("{HILL}node 2 height_to_normal_world in=@1 size=100 height=2\nnode 3 height_to_normal_world in=@1 size=100 height=8\nnode 4 height_to_normal_world in=@1 size=100 height=0\noutput low 2\noutput high 3\noutput zero 4\n"));
    let tilt = |t: &Tex| t.px.iter().map(|p| (p[0] - 0.5).abs()).sum::<f32>() / t.px.len() as f32;
    assert!(tilt(&o["high"]) > tilt(&o["low"]) * 2.0);
    assert!(tilt(&o["zero"]) < 1e-6);
}

#[test]
fn normal_transform_matches_transforming_the_height() {
    let h = "node 1 pixel_processor expr=\"0.5+0.25*sin(6.2831853*u)+0.15*sin(12.5663706*v)+0.1*sin(6.2831853*(u+v))\"\n";
    for dx in [0.0, 1.0] {
        let o = run(64, &format!("{h}node 2 normal in=@1 strength=2 directx={dx}\nnode 3 normal_transform in=@2 rotation=0.25 directx={dx}\nnode 4 transform in=@1 rotation=0.25\nnode 5 normal in=@4 strength=2 directx={dx}\noutput a 3\noutput b 5\noutput raw 2\n"));
        assert!(diff(&o["a"], &o["b"]) < 0.02, "directx={dx}: {}", diff(&o["a"], &o["b"]));
        assert!(diff(&o["a"], &o["raw"]) > 0.02, "the rotation changed something");
    }
}

#[test]
fn normal_sobel_finds_edges_and_curvature() {
    let o = run(64, &format!("{HILL}node 2 normal in=@1 strength=3\nnode 3 normal_sobel in=@2\nnode 4 normal_sobel in=@2 mode=1\nnode 5 color r=0.5 g=0.5 b=1\nnode 6 normal_sobel in=@5\nnode 7 normal_sobel in=@5 mode=1\noutput edge 3\noutput curv 4\noutput flat_edge 6\noutput flat_curv 7\n"));
    assert!(mean(&o["flat_edge"], 0) < 1e-6 && (mean(&o["flat_curv"], 0) - 0.5).abs() < 1e-6);
    assert!(mean(&o["edge"], 0) > 0.001);
    assert!(px(&o["curv"], 0.5, 0.5)[0] > 0.5, "hill top is convex");
    assert!(px(&o["curv"], 0.5, 0.008)[0] < 0.5, "pit is concave");
}
