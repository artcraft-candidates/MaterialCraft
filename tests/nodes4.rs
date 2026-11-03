// SPDX-License-Identifier: MIT OR Apache-2.0
//! Behaviour of the material utilities added in Milestone 2 (batch 2).
use materialcraft::graph::Graph;
use materialcraft::tex::Tex;
use std::collections::HashMap;

fn run(size: usize, text: &str) -> HashMap<String, Tex> {
    let g = Graph::parse(&format!("size {size}\n{text}")).unwrap_or_else(|e| panic!("parse: {e}"));
    g.evaluate(Some(size)).unwrap_or_else(|e| panic!("eval: {e}")).into_iter().collect()
}

fn px(t: &Tex, u: f32, v: f32) -> [f32; 4] {
    t.at((u * t.w as f32) as i64, (v * t.h as f32) as i64)
}

fn mean(t: &Tex, c: usize) -> f32 {
    t.px.iter().map(|p| p[c]).sum::<f32>() / t.px.len() as f32
}

fn diff(a: &Tex, b: &Tex) -> f32 {
    a.px.iter().zip(&b.px).map(|(p, q)| (0..3).map(|c| (p[c] - q[c]).abs()).fold(0.0f32, f32::max)).sum::<f32>() / a.px.len() as f32
}

#[test]
fn workflow_converters_round_trip() {
    let src = "node 1 color r=0.8 g=0.3 b=0.1\nnode 2 color r=0.9 g=0.9 b=0.9\nnode 3 color r=0.3 g=0.3 b=0.3\nnode 4 color r=0 g=0 b=0\nnode 5 color r=1 g=1 b=1\n";
    // a dielectric: diffuse = base, specular = 4 %, gloss = 1 - roughness
    let o = run(16, &format!("{src}node 10 metal_rough_to_spec_gloss base_color=@1 metallic=@4 roughness=@3\noutput d 10\noutput s 10.specular\noutput g 10.glossiness\n"));
    let (d, s, g) = (px(&o["d"], 0.5, 0.5), px(&o["s"], 0.5, 0.5), px(&o["g"], 0.5, 0.5));
    assert!((d[0] - 0.8).abs() < 1e-5 && (s[0] - 0.04).abs() < 1e-5 && (g[0] - 0.7).abs() < 1e-5);
    // a metal: no diffuse, the base colour becomes the specular colour
    let o = run(16, &format!("{src}node 10 metal_rough_to_spec_gloss base_color=@1 metallic=@5 roughness=@3\noutput d 10\noutput s 10.specular\n"));
    assert!(px(&o["d"], 0.5, 0.5)[0] < 1e-5 && (px(&o["s"], 0.5, 0.5)[0] - 0.8).abs() < 1e-5);
    // and back again
    for metal in ["@4", "@5"] {
        let o = run(16, &format!("{src}node 10 metal_rough_to_spec_gloss base_color=@1 metallic={metal} roughness=@3\nnode 11 spec_gloss_to_metal_rough diffuse=@10 specular=@10.specular glossiness=@10.glossiness\noutput base 11\noutput m 11.metallic\noutput r 11.roughness\n"));
        let (b, m, r) = (px(&o["base"], 0.5, 0.5), px(&o["m"], 0.5, 0.5)[0], px(&o["r"], 0.5, 0.5)[0]);
        let want_m = if metal == "@5" { 1.0 } else { 0.0 };
        assert!((m - want_m).abs() < 0.05, "metallic {m} want {want_m}");
        assert!((r - 0.3).abs() < 1e-4, "roughness {r}");
        if metal == "@5" {
            assert!((b[0] - 0.8).abs() < 0.03 && (b[1] - 0.3).abs() < 0.03, "base {b:?}");
        } else {
            assert!((b[0] - 0.8).abs() < 0.03, "base {b:?}");
        }
    }
}

#[test]
fn material_blend_follows_mask_and_height() {
    let mats = "node 1 color r=1 g=0 b=0\nnode 2 color r=0 g=0 b=1\nnode 3 color r=0.2 g=0.2 b=0.2\nnode 4 color r=0.9 g=0.9 b=0.9\nnode 5 pixel_processor expr=\"u\"\nnode 6 color r=0.5 g=0.5 b=0.5\n";
    let o = run(64, &format!("{mats}node 10 material_blend bg_base=@1 fg_base=@2 bg_rough=@3 fg_rough=@4 mask=@5 height_influence=0\noutput base 10\noutput rough 10.roughness\noutput nrm 10.normal\noutput ao 10.ao\n"));
    assert!(px(&o["base"], 0.05, 0.5)[0] > 0.9 && px(&o["base"], 0.95, 0.5)[2] > 0.9, "the mask moves the blend from background to foreground");
    assert!(px(&o["rough"], 0.05, 0.5)[0] < 0.3 && px(&o["rough"], 0.95, 0.5)[0] > 0.8);
    let n = px(&o["nrm"], 0.5, 0.5);
    assert!((n[2] - 1.0).abs() < 1e-3, "missing normals count as flat");
    assert!((px(&o["ao"], 0.5, 0.5)[0] - 1.0).abs() < 1e-5, "missing AO counts as unoccluded");
    // with a half mask, the material whose height is higher wins
    let o = run(16, &format!("{mats}node 7 color r=0.9 g=0.9 b=0.9\nnode 8 color r=0.1 g=0.1 b=0.1\nnode 10 material_blend bg_base=@1 fg_base=@2 bg_height=@7 fg_height=@8 mask=@6 height_influence=2\nnode 11 material_blend bg_base=@1 fg_base=@2 bg_height=@8 fg_height=@7 mask=@6 height_influence=2\noutput a 10\noutput b 11\n"));
    assert!(px(&o["a"], 0.5, 0.5)[0] > px(&o["b"], 0.5, 0.5)[0], "higher background shows through");
}

#[test]
fn height_combine_modes() {
    let o = run(16, "node 1 pixel_processor expr=\"u\"\nnode 2 pixel_processor expr=\"1-u\"\nnode 3 height_combine a=@1 b=@2 mode=0\nnode 4 height_combine a=@1 b=@2 mode=1\nnode 5 height_combine a=@1 b=@2 mode=3\noutput mx 3\noutput mn 4\noutput df 5\n");
    assert!(o["mx"].px.iter().all(|p| p[0] >= 0.5 - 1e-3));
    assert!(o["mn"].px.iter().all(|p| p[0] <= 0.5 + 1e-3));
    assert!(px(&o["df"], 0.0, 0.5)[0] > 0.9 && px(&o["df"], 0.5, 0.5)[0] < 0.1);
}

#[test]
fn convex_concave_split_a_bump() {
    let o = run(64, "node 1 pixel_processor expr=\"0.5+0.5*cos((u-0.5)*6.2831853)*cos((v-0.5)*6.2831853)\"\nnode 2 convex_concave in=@1 radius=24 gain=8\noutput convex 2\noutput concave 2.concave\n");
    assert!(px(&o["convex"], 0.5, 0.5)[0] > px(&o["concave"], 0.5, 0.5)[0]);
    assert!(px(&o["concave"], 0.5, 0.008)[0] > px(&o["convex"], 0.5, 0.008)[0]);
    let flat = run(32, "node 1 color r=0.5 g=0.5 b=0.5\nnode 2 convex_concave in=@1\noutput o 2\n");
    assert!(mean(&flat["o"], 0) < 1e-6);
}

#[test]
fn leak_streaks_run_on_downwards() {
    let o = run(64, "node 1 sdf_shape shape=0 size=0.05 y=-0.2\nnode 2 sdf_mask in=@1 offset=0 softness=0.001\nnode 3 leak_streaks in=@2 length=300 angle=0.25 breakup=0\nnode 4 leak_streaks in=@2 length=300 angle=0.75 breakup=0\noutput down 3\noutput up 4\noutput src 2\n");
    // the dot sits at y = 0.3; the trail goes to larger y
    assert!(px(&o["down"], 0.5, 0.55)[0] > 0.1 && px(&o["down"], 0.5, 0.55)[0] > px(&o["up"], 0.5, 0.55)[0]);
    assert!(px(&o["up"], 0.5, 0.1)[0] > px(&o["down"], 0.5, 0.1)[0]);
    assert!(mean(&o["down"], 0) > mean(&o["src"], 0));
}

#[test]
fn settled_dust_prefers_flat_tops_and_cavities() {
    let o = run(64, "node 1 pixel_processor expr=\"0.5+0.5*cos((u-0.5)*6.2831853)*cos((v-0.5)*6.2831853)\"\nnode 2 settled_dust in=@1 amount=1 slope=0.1 cavity=0.0 relief=8\nnode 3 settled_dust in=@1 amount=0 cavity=1 radius=24\noutput flat 2\noutput cav 3\n");
    assert!(px(&o["flat"], 0.5, 0.5)[0] > px(&o["flat"], 0.3, 0.5)[0], "the flat hilltop collects more than the flank");
    assert!(px(&o["cav"], 0.5, 0.008)[0] > px(&o["cav"], 0.5, 0.5)[0], "the pit collects more than the top");
}

#[test]
fn palette_tools() {
    let o = run(64, "node 1 pixel_processor expr=\"step(0.5,u)\" expr_g=\"0\" expr_b=\"0\"\nnode 2 palette_extract in=@1 count=2\nnode 3 palette_apply in=@1 palette=@2 count=2\noutput pal 2\noutput snap 3\noutput src 1\n");
    assert!(px(&o["pal"], 0.1, 0.5)[0] < 0.1 && px(&o["pal"], 0.9, 0.5)[0] > 0.9, "dark and light swatches");
    assert!(diff(&o["src"], &o["snap"]) < 1e-4, "an image of two colours is unchanged by its own palette");
}

#[test]
fn mirror_tiling_is_continuous() {
    let o = run(64, "node 1 fbm scale=3 seed=9\nnode 2 make_it_tile_mirror in=@1\nnode 3 make_it_tile_mirror in=@1 mode=1\noutput both 2\noutput x 3\noutput src 1\n");
    let t = &o["both"];
    let w = t.w as i64;
    let jump: f32 = (0..t.h as i64).map(|y| (t.at(0, y)[0] - t.at(w - 1, y)[0]).abs()).sum::<f32>() / t.h as f32;
    assert!(jump < 0.02, "left and right edges match: {jump}");
    assert!(diff(&o["x"], &o["src"]) > 0.01);
}
