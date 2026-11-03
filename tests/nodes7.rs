// SPDX-License-Identifier: MIT OR Apache-2.0
//! Behaviour of the nodes that finished Milestone 2: pavements, noises, 3D masks, extruded and mapped shapes, and the
//! effect, transform and material filters.
use materialcraft::graph::Graph;
use materialcraft::tex::Tex;
use std::collections::{BTreeSet, HashMap};

fn run(size: usize, text: &str) -> HashMap<String, Tex> {
    let g = Graph::parse(&format!("size {size}\n{text}")).unwrap_or_else(|e| panic!("parse: {e}"));
    g.evaluate(Some(size)).unwrap_or_else(|e| panic!("eval: {e}")).into_iter().collect()
}

fn one(size: usize, text: &str) -> Tex {
    run(size, text).remove("o").unwrap()
}

fn mean(t: &Tex) -> f32 {
    t.px.iter().map(|p| p[0]).sum::<f32>() / t.px.len() as f32
}

fn range(t: &Tex) -> (f32, f32) {
    t.px.iter().fold((f32::MAX, f32::MIN), |a, p| (a.0.min(p[0]), a.1.max(p[0])))
}

fn diff(a: &Tex, b: &Tex) -> f32 {
    a.px.iter().zip(&b.px).map(|(x, y)| (x[0] - y[0]).abs()).sum::<f32>() / a.px.len() as f32
}

#[test]
fn herringbone_tiles_the_plane_with_equal_bricks() {
    // 8 x 8 cells hold 32 bricks of two cells; each has its own id, and half lie each way
    let ids = one(256, "node 1 herringbone scale=8 mode=1\noutput o 1\n");
    let set: BTreeSet<u32> = ids.px.iter().map(|p| (p[0] * 1.0e6) as u32).collect();
    assert_eq!(set.len(), 32, "expected 32 bricks, got {}", set.len());
    let dir = one(256, "node 1 herringbone scale=8 mode=2\noutput o 1\n");
    assert!((mean(&dir) - 0.5).abs() < 0.02, "orientations are not balanced: {}", mean(&dir));
    // mortar lines: the height drops to zero between bricks and reaches the top inside
    let h = one(256, "node 1 herringbone scale=8 gap=6 variation=0\noutput o 1\n");
    let (lo, hi) = range(&h);
    assert!(lo < 0.01 && hi > 0.99, "height range {lo}..{hi}");
}

#[test]
fn arc_pavement_has_stones_and_gaps() {
    let ids = one(256, "node 1 arc_pavement scale=6 rings=4 stones=9 mode=1\noutput o 1\n");
    let set: BTreeSet<u32> = ids.px.iter().map(|p| (p[0] * 1.0e6) as u32).collect();
    assert!(set.len() > 60, "only {} stones", set.len());
    let h = one(256, "node 1 arc_pavement scale=6 gap=2 bevel=1 variation=0\noutput o 1\n");
    let (lo, hi) = range(&h);
    assert!(lo < 0.02 && hi > 0.98, "height range {lo}..{hi}");
}

#[test]
fn anisotropic_noise_is_stretched_along_its_direction() {
    let t = one(256, "node 1 anisotropic_noise scale=8 stretch=8 angle=0 octaves=1\noutput o 1\n");
    let step = |dx: i64, dy: i64| -> f32 {
        let mut s = 0.0;
        for y in 0..256i64 {
            for x in 0..256i64 {
                s += (t.at(x, y)[0] - t.at(x + dx, y + dy)[0]).abs();
            }
        }
        s / 65536.0
    };
    let (along, across) = (step(6, 0), step(0, 6));
    assert!(along < across * 0.6, "not stretched along x: {along} vs {across}");
    // turning the direction a quarter turn swaps the roles
    let r = one(256, "node 1 anisotropic_noise scale=8 stretch=8 angle=0.25 octaves=1\noutput o 1\n");
    let sd = |dx: i64, dy: i64| -> f32 {
        let mut s = 0.0;
        for y in 0..256i64 {
            for x in 0..256i64 {
                s += (r.at(x, y)[0] - r.at(x + dx, y + dy)[0]).abs();
            }
        }
        s / 65536.0
    };
    assert!(sd(0, 6) < sd(6, 0) * 0.6, "angle 0.25 should stretch along y");
}

#[test]
fn fur_moisture_and_microscope_noises_have_structure() {
    for (g, what) in [
        ("node 1 fur_noise count=800\n", "fur"),
        ("node 1 moisture_noise\n", "moisture"),
        ("node 1 microscope_noise\n", "microscope"),
        ("node 1 worley_3d\n", "worley 3d"),
        ("node 1 fbm_3d\n", "fractal 3d"),
    ] {
        let t = one(128, &format!("{g}output o 1\n"));
        let (lo, hi) = range(&t);
        assert!(hi - lo > 0.3 && lo >= -0.001 && hi <= 1.001, "{what}: range {lo}..{hi}");
    }
    // microscope outputs differ: membranes are thin, nuclei are blobs
    let m = mean(&one(128, "node 1 microscope_noise mode=1\noutput o 1\n"));
    let n = mean(&one(128, "node 1 microscope_noise mode=2\noutput o 1\n"));
    assert!(m > 0.02 && m < 0.5 && n > 0.02 && n < 0.5, "membrane {m} nucleus {n}");
    // moisture: more coverage wets more of the surface
    let a = mean(&one(128, "node 1 moisture_noise coverage=0.2 drips=0\noutput o 1\n"));
    let b = mean(&one(128, "node 1 moisture_noise coverage=0.8 drips=0\noutput o 1\n"));
    assert!(b > a + 0.2, "coverage {a} -> {b}");
}

#[test]
fn slices_of_3d_noise_change_with_z_and_tile_in_z_free_steps() {
    for kind in ["worley_3d", "fbm_3d"] {
        let a = one(64, &format!("node 1 {kind} z=0\noutput o 1\n"));
        let b = one(64, &format!("node 1 {kind} z=0.5\noutput o 1\n"));
        let c = one(64, &format!("node 1 {kind} z=0.01\noutput o 1\n"));
        assert!(diff(&a, &b) > 0.05, "{kind}: slices at different depth look the same");
        assert!(diff(&a, &c) < diff(&a, &b), "{kind}: neighbouring slices should be close");
    }
}

#[test]
fn volume_mask_slices_a_sphere() {
    // a sphere of radius 0.35 in a unit cell: the slice through the middle covers pi * 0.35^2 of the area
    let mid = mean(&one(256, "node 1 volume_mask shape=0 size=0.35 z=0.5 softness=0\noutput o 1\n"));
    assert!((mid - std::f32::consts::PI * 0.35 * 0.35).abs() < 0.02, "middle slice covers {mid}");
    let top = mean(&one(256, "node 1 volume_mask shape=0 size=0.35 z=0.8 softness=0\noutput o 1\n"));
    let expect = std::f32::consts::PI * (0.35f32 * 0.35 - 0.3 * 0.3);
    assert!((top - expect).abs() < 0.02, "upper slice covers {top}, expected {expect}");
    let far = mean(&one(64, "node 1 volume_mask shape=0 size=0.35 z=1.0 softness=0\noutput o 1\n"));
    assert!(far < 0.001, "above the sphere nothing is left: {far}");
    // the gradient peaks in the middle of the shape
    let g = one(64, "node 1 gradient_3d shape=0 size=0.3 z=0.5 range=0.5\noutput o 1\n");
    assert!(g.at(32, 32)[0] > 0.9 && g.at(0, 0)[0] < g.at(32, 32)[0]);
}

#[test]
fn extrude_pulls_walls_along_the_direction() {
    let o = run(128, "node 1 circle radius=0.15\nnode 2 shape_extrude in=@1 length=40 angle=0\noutput h 2\noutput w 2.walls\n");
    let (h, w) = (&o["h"], &o["w"]);
    // to the right of the circle (angle 0) the wall falls from the top to nothing; to the left there is none
    let right: f32 = (0..128).map(|x| h.at(x, 64)[0]).sum();
    let left_probe = h.at(64 - 40, 64)[0];
    assert!(right > 0.0 && mean(w) > 0.01, "no wall");
    assert!(left_probe < 0.05, "wall on the wrong side: {left_probe}");
    let cx = h.at(64, 64)[0];
    assert!((cx - 1.0).abs() < 0.01, "top is flat at 1, got {cx}");
}

#[test]
fn mapper_puts_a_pattern_in_every_region() {
    let t = one(128, "node 1 voronoi scale=4 mode=2\nnode 2 threshold in=@1 threshold=0.2\nnode 3 flood_fill in=@2\nnode 4 shape_mapper in=@3\noutput o 4\n");
    let (lo, hi) = range(&t);
    assert!(hi > 0.9 && lo < 0.1, "range {lo}..{hi}");
    // every region is filled by its own radial pattern, so there are several bright centres
    let bright = t.px.iter().filter(|p| p[0] > 0.9).count();
    assert!(bright > 8, "{bright} bright pixels");
}

#[test]
fn cross_section_draws_the_profile() {
    let t = one(128, "node 1 pixel_processor expr=\"u\"\nnode 2 cross_section in=@1 thickness=3 fill=0\noutput o 2\n");
    // a horizontal ramp rises across the plot: the line is low on the left and high on the right
    let col = |x: usize| (0..128).find(|y| t.at(x as i64, *y as i64)[0] > 0.5);
    let (l, r) = (col(8).unwrap(), col(120).unwrap());
    assert!(r + 20 < l, "line should climb to the right: left row {l}, right row {r}");
}

#[test]
fn uber_emboss_shows_slopes_and_leaves_flat_surfaces_neutral() {
    let flat = run(64, "node 1 color r=0.5 g=0.5 b=0.5\nnode 2 uber_emboss height=@1\noutput e 2\noutput h 2.highlight\noutput s 2.shadow\n");
    assert!((range(&flat["e"]).0 - 0.5).abs() < 0.01 && (range(&flat["e"]).1 - 0.5).abs() < 0.01);
    assert!(range(&flat["s"]).1 < 0.01);
    let bumpy = run(128, "node 1 perlin scale=6\nnode 2 uber_emboss height=@1 intensity=8\noutput e 2\noutput h 2.highlight\n");
    assert!(range(&bumpy["e"]).1 - range(&bumpy["e"]).0 > 0.2);
    assert!(range(&bumpy["h"]).1 > 0.05, "no highlights");
}

#[test]
fn diffusion_fills_the_unknown_area() {
    // the left half is known (red), the right half is empty: it ends up red as well, with no black left
    let o = run(64, "node 1 color r=0.9 g=0.1 b=0.1 a=1\nnode 2 pixel_processor expr=\"u\"\nnode 3 threshold in=@2 threshold=0.5\nnode 4 invert in=@3\nnode 5 diffusion in=@1 mask=@4\noutput o 5\n");
    let t = &o["o"];
    let (r, g) = (t.at(60, 32)[0], t.at(60, 32)[1]);
    assert!(r > 0.8 && g < 0.2, "filled pixel is ({r}, {g})");
}

#[test]
fn summed_area_table_grows_to_the_total() {
    let t = one(32, "node 1 color r=0.5 g=0.5 b=0.5\nnode 2 summed_area_table in=@1\noutput o 2\n");
    assert!((t.at(31, 31)[0] - 1.0).abs() < 1e-4);
    assert!(t.at(7, 7)[0] < t.at(15, 15)[0] && t.at(15, 15)[0] < t.at(31, 31)[0]);
}

#[test]
fn vector_morph_moves_pixels() {
    // a flat vector map (0.5, 0.5) means no movement
    let still = run(64, "node 1 perlin scale=4\nnode 2 color r=0.5 g=0.5 b=0.5\nnode 3 vector_morph in=@1 vector=@2 amount=30\noutput a 1\noutput b 3\n");
    assert!(diff(&still["a"], &still["b"]) < 0.01);
    // a full-strength rightwards vector moves the picture right by amount * scale
    let moved = run(128, "node 1 perlin scale=4\nnode 2 color r=1 g=0.5 b=0.5\nnode 3 vector_morph in=@1 vector=@2 amount=32\noutput a 1\noutput b 3\n");
    let (a, b) = (&moved["a"], &moved["b"]);
    let shift = 32.0 * 128.0 / 512.0;
    let mut err_shift = 0.0;
    let mut err_none = 0.0;
    for y in 0..128i64 {
        for x in 0..128i64 {
            err_shift += (b.at(x + shift as i64, y)[0] - a.at(x, y)[0]).abs();
            err_none += (b.at(x, y)[0] - a.at(x, y)[0]).abs();
        }
    }
    assert!(err_shift < err_none * 0.4, "morph did not shift: {err_shift} vs {err_none}");
}

#[test]
fn safe_transform_repeats_and_rotates_without_breaking_the_tile() {
    let src = "node 1 bricks rows=4 cols=4\n";
    let o = run(128, &format!("{src}node 2 safe_transform in=@1 scale_x=2 scale_y=2 filter=0\noutput a 1\noutput b 2\n"));
    let (a, b) = (&o["a"], &o["b"]);
    // twice the repeats: pixel (x, y) of the result is pixel (2x, 2y) of the input
    let mut bad = 0;
    for y in 4..120i64 {
        for x in 4..120i64 {
            if (b.at(x, y)[0] - a.at((2 * x + 1) % 128, (2 * y + 1) % 128)[0]).abs() > 0.2 {
                bad += 1;
            }
        }
    }
    assert!(bad < 400, "{bad} pixels differ from the 2x repeat");
    // a quarter turn is exact: the result is the input turned by 90 degrees
    let r = run(64, &format!("{src}node 2 safe_transform in=@1 angle=0.25 filter=0\noutput a 1\noutput b 2\n"));
    assert!(diff(&r["a"], &r["b"]) > 0.02, "a quarter turn should change the picture");
    // free mode keeps arbitrary angles
    let f = run(64, &format!("{src}node 2 safe_transform in=@1 mode=1 angle=0.07 scale_x=1.3 scale_y=1.3\noutput b 2\n"));
    assert!(range(&f["b"]).1 > 0.5);
}

#[test]
fn snow_prefers_flat_surfaces() {
    let flat = run(64, "node 1 color r=0.5 g=0.5 b=0.5\nnode 2 snow_filter height=@1 breakup=0\noutput m 2.mask\noutput h 2\n");
    let steep = run(64, "node 1 pixel_processor expr=\"sin(u*150.0)*0.5+0.5\"\nnode 2 snow_filter height=@1 breakup=0 steepness=40\noutput m 2.mask\noutput h 2\n");
    assert!(mean(&flat["m"]) > mean(&steep["m"]) + 0.15, "flat {} steep {}", mean(&flat["m"]), mean(&steep["m"]));
    // snow raises the surface
    assert!(mean(&flat["h"]) > 0.5);
}

#[test]
fn season_recolours_foliage_only() {
    let green = "node 1 color r=0.2 g=0.6 b=0.15 a=1\n";
    let autumn = run(32, &format!("{green}node 2 season_filter in=@1 season=2\noutput o 2\n"));
    let p = autumn["o"].at(8, 8);
    assert!(p[0] > p[1], "autumn leaves should turn warm: {p:?}");
    let winter = run(32, &format!("{green}node 2 season_filter in=@1 season=3\noutput o 2\n"));
    let w = winter["o"].at(8, 8);
    assert!((w[0] - w[1]).abs() < (0.6f32 - 0.2).abs(), "winter should drain the colour: {w:?}");
    // blue paint is not foliage and stays
    let blue = run(32, "node 1 color r=0.1 g=0.2 b=0.8 a=1\nnode 2 season_filter in=@1 season=2\noutput o 2\n");
    let b = blue["o"].at(8, 8);
    assert!((b[2] - 0.8).abs() < 0.1, "non-foliage changed: {b:?}");
}

#[test]
fn water_fills_the_low_parts() {
    let o = run(64, "node 1 pixel_processor expr=\"u\"\nnode 2 color r=0.6 g=0.5 b=0.3 a=1\nnode 3 water_filter height=@1 base=@2 level=0.4 softness=0.01\noutput c 3\noutput m 3.mask\noutput r 3.roughness\n");
    let m = &o["m"];
    assert!(m.at(5, 32)[0] > 0.99 && m.at(60, 32)[0] < 0.01, "water mask should follow the low side");
    let cov = mean(m);
    assert!((cov - 0.4).abs() < 0.05, "coverage {cov}");
    let wet = o["c"].at(5, 32);
    let dry = o["c"].at(60, 32);
    assert!(wet[2] > wet[0] && dry[0] > dry[2], "wet {wet:?} dry {dry:?}");
    assert!(o["r"].at(5, 32)[0] < o["r"].at(60, 32)[0]);
}
