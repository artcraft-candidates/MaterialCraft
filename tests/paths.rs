// SPDX-License-Identifier: MIT OR Apache-2.0
use materialcraft::graph::Graph;
use materialcraft::spline::{SPath, SPoint, Spline};

fn render(text: &str, size: usize) -> materialcraft::tex::Tex {
    let g = Graph::parse(text).unwrap_or_else(|e| panic!("{e}"));
    g.evaluate(Some(size)).unwrap_or_else(|e| panic!("{e}")).remove("o").unwrap()
}

fn sum(t: &materialcraft::tex::Tex) -> f32 {
    t.px.iter().map(|p| p[0]).sum::<f32>() / t.px.len() as f32
}

#[test]
fn spline_text_roundtrip_and_errors() {
    let mut sp = Spline::default();
    sp.paths.push(SPath { closed: true, smooth: false, pts: vec![SPoint::new(0.1, 0.2), SPoint { p: [0.5, 0.25], hin: Some([-0.1, 0.0]), hout: Some([0.1, 0.0]), w: 2.0, z: 1.0, tan: 0 }, SPoint::new(0.3, 0.9)] });
    sp.paths.push(SPath { pts: vec![SPoint::new(0.0, 0.0), SPoint::new(1.0, 1.0)], ..Default::default() });
    let text = sp.to_text();
    let back = Spline::parse(&text).unwrap();
    assert_eq!(back.paths.len(), 2);
    assert!(back.paths[0].closed && !back.paths[0].smooth);
    assert_eq!(back.paths[0].pts[1].hout, Some([0.1, 0.0]));
    assert!((back.paths[0].pts[1].w - 2.0).abs() < 1e-6);
    assert_eq!(back.point_count(), 5);
    for bad in ["pt 1 2", "spline 1\npath\npt a b", "spline 1\npath\npt 0 0 hin=1", "spline 1\npath\npt 0 0 bogus=1", "spline 1\nwhat", "hello", "spline 1\npath\npt nan 0"] {
        assert!(Spline::parse(bad).is_err(), "should reject {bad:?}");
    }
    assert!(Spline::from_bytes(&[0xff, 0xfe, 0x00]).is_err());
}

#[test]
fn flatten_resample_and_presets() {
    for k in 1..=7 {
        let sp = Spline::preset(k, 0.1, 3.0);
        let polys = sp.flatten(256.0, 128.0);
        assert!(!polys.is_empty() && polys[0].pts.len() >= 2, "preset {k}");
        assert!(polys[0].pts.iter().all(|p| p[0].is_finite() && p[1].is_finite()));
    }
    // a straight line across a 200 x 100 texture is 200 px long
    let line = Spline::preset(1, 0.0, 1.0).flatten(200.0, 100.0);
    assert!((line[0].length() - 200.0).abs() < 0.01);
    let s = line[0].resample(20.0, 0.0);
    assert_eq!(s.len(), 11);
    assert!((s[3].p[0] - 60.0).abs() < 0.01 && (s[3].t[0] - 1.0).abs() < 1e-5);
    // a closed circle does not repeat its first point
    let c = Spline::preset(4, 0.0, 1.0).flatten(256.0, 256.0);
    assert!(c[0].closed);
    let n = c[0].resample(10.0, 0.0).len() as f32;
    assert!((n - c[0].length() / 10.0).abs() < 1.5);
}

#[test]
fn svg_paths_become_a_spline() {
    let svg = r##"<svg viewBox="0 0 10 10"><path d="M1 1 L9 1 L9 9 Z"/><circle cx="5" cy="5" r="2"/></svg>"##;
    let sp = Spline::from_svg(svg).unwrap();
    assert_eq!(sp.paths.len(), 2);
    assert!(sp.paths[0].closed);
    assert!(sp.paths[0].pts.len() == 3, "triangle keeps its corners, got {}", sp.paths[0].pts.len());
    assert!(sp.paths.iter().all(|p| p.pts.iter().all(|q| (0.0..=1.0).contains(&q.p[0]) && (0.0..=1.0).contains(&q.p[1]))));
    assert!(Spline::from_svg("<svg viewBox=\"0 0 1 1\"></svg>").is_err());
    assert!(Spline::from_svg("not svg").is_err());
}

#[test]
fn stroke_width_and_tiling() {
    // horizontal line at y = 0.5, 20 px wide on a 256 px graph: about 20/256 of the image is covered
    let t = render("size 512\nnode 1 path_stroke preset=0 width=20\noutput o 1\n", 512);
    let cover = sum(&t);
    assert!((cover - 20.0 / 512.0).abs() < 0.005, "cover {cover}");
    assert!(t.at(10, 256)[0] > 0.9 && t.at(10, 10)[0] < 0.01);
    // a diagonal line wraps: both corners are on the line
    let d = render("size 512\nnode 1 path_stroke preset=1 width=10\noutput o 1\n", 512);
    assert!(d.at(256, 256)[0] > 0.9 && d.at(511, 511)[0] > 0.5 && d.at(0, 0)[0] > 0.5 && d.at(100, 400)[0] < 0.01);
    // taper makes the ends thinner than the middle (open wave, taper on both ends)
    let tp = render("size 512\nnode 1 path_stroke preset=0 width=30 taper_start=0.3 taper_end=0.3\noutput o 1\n", 512);
    let col = |x: usize| (0..512).map(|y| tp.at(x as i64, y as i64)[0]).sum::<f32>();
    assert!(col(256) > col(12) * 3.0, "{} vs {}", col(256), col(12));
}

#[test]
fn dashes_fit_the_path_and_repeat() {
    let t = render("size 512\nnode 1 path_stroke preset=0 width=8 dash_length=20 dash_gap=12\noutput o 1\n", 512);
    // along the line (y=256) count rising edges: 512 / 32 = 16 dashes
    let mut n = 0;
    let mut prev = t.at(511, 256)[0] > 0.5;
    for x in 0..512 {
        let on = t.at(x, 256)[0] > 0.5;
        if on && !prev {
            n += 1;
        }
        prev = on;
    }
    assert_eq!(n, 16, "dashes {n}");
}

#[test]
fn fill_and_distance_modes() {
    let fill = render("size 128\nnode 1 path_fill preset=3\noutput o 1\n", 128);
    // circle of radius 0.3 around the centre: area about pi * 0.09
    assert!((sum(&fill) - std::f32::consts::PI * 0.09).abs() < 0.03, "{}", sum(&fill));
    assert!(fill.at(64, 64)[0] > 0.99 && fill.at(2, 2)[0] < 0.01);
    let sd = render("size 128\nnode 1 path_distance preset=3 mode=1 range=40\noutput o 1\n", 128);
    assert!(sd.at(64, 64)[0] > 0.9, "inside is above the middle");
    assert!(sd.at(1, 1)[0] < 0.1, "far outside is below");
    let line = (sd.at(64 + 38, 64)[0] - 0.5).abs();
    assert!(line < 0.08, "on the path the value is about the middle: {line}");
    let inward = render("size 128\nnode 1 path_distance preset=3 mode=2 range=30\noutput o 1\n", 128);
    assert!(inward.at(64, 64)[0] > 0.99 && inward.at(1, 1)[0] < 0.01);
}

#[test]
fn border_scatter_track_cable_seam_have_content() {
    for (g, what) in [
        ("node 1 path_border preset=3 width=24", "border"),
        ("node 1 path_scatter preset=2 spacing=16 scale=10", "scatter"),
        ("node 1 path_track preset=0", "track"),
        ("node 1 path_cable preset=2", "cable"),
        ("node 1 path_seam preset=0", "seam"),
    ] {
        let t = render(&format!("size 128\n{g}\noutput o 1\n"), 128);
        let (lo, hi) = t.px.iter().fold((1.0f32, 0.0f32), |a, p| (a.0.min(p[0]), a.1.max(p[0])));
        assert!(hi - lo > 0.2, "{what} is flat ({lo}..{hi})");
    }
    // secondary outputs
    let g = Graph::parse("size 128\nnode 1 path_track preset=0\noutput rails 1.rails\noutput sleepers 1.sleepers\noutput height 1\n").unwrap();
    let out = g.evaluate(Some(128)).unwrap();
    assert!(sum(&out["rails"]) > 0.01 && sum(&out["sleepers"]) > 0.02);
    assert!(sum(&out["height"]) >= sum(&out["rails"]) * 0.3);
}

#[test]
fn scatter_uses_a_pattern_and_follows_the_path() {
    // stamps of a pattern: a dark input leaves nothing, a white one leaves marks
    let w = render("size 512\nnode 2 color r=1 g=1 b=1 a=1\nnode 1 path_scatter preset=0 spacing=20 scale=12 in=@2 scale_variation=0\noutput o 1\n", 512);
    let b = render("size 512\nnode 2 color r=0 g=0 b=0 a=1\nnode 1 path_scatter preset=0 spacing=20 scale=12 in=@2 scale_variation=0\noutput o 1\n", 512);
    assert!(sum(&w) > 0.01 && sum(&b) < 1e-6);
    // the stamps are the same size every time without variation: equal columns at the stamp centres
    let t = render("size 512\nnode 1 path_scatter preset=0 spacing=32 scale=16 scale_variation=0\noutput o 1\n", 512);
    assert!(t.at(32, 256)[0] > 0.9 && t.at(96, 256)[0] > 0.9 && t.at(16, 256)[0] < 0.2, "{} {} {}", t.at(32, 256)[0], t.at(96, 256)[0], t.at(16, 256)[0]);
}

#[test]
fn spline_resource_is_used_and_edits_show() {
    let mut g = Graph::parse("size 512\nnode 1 path_stroke file=\"a.spl\" width=6\noutput o 1\n").unwrap();
    g.resources.insert("a.spl".into(), std::sync::Arc::new(b"spline 1\npath smooth=0\npt 0.1 0.2\npt 0.9 0.2\n".to_vec()));
    let a = g.evaluate(Some(512)).unwrap().remove("o").unwrap();
    assert!(a.at(256, 102)[0] > 0.9 && a.at(256, 400)[0] < 0.01);
    g.resources.insert("a.spl".into(), std::sync::Arc::new(b"spline 1\npath smooth=0\npt 0.1 0.8\npt 0.9 0.8\n".to_vec()));
    let b = g.evaluate(Some(512)).unwrap().remove("o").unwrap();
    assert!(b.at(256, 102)[0] < 0.01 && b.at(256, 409)[0] > 0.9);
    // a missing or broken spline is an error, not a panic
    let mut bad = Graph::parse("size 64\nnode 1 path_stroke file=\"x.spl\"\noutput o 1\n").unwrap();
    assert!(bad.evaluate(Some(64)).is_err());
    bad.resources.insert("x.spl".into(), std::sync::Arc::new(b"garbage".to_vec()));
    assert!(bad.evaluate(Some(64)).is_err());
}

#[test]
fn non_square_and_size_independent() {
    let g = Graph::parse("size 256 128\nnode 1 path_cable preset=2\nnode 2 path_stroke preset=3 width=10\noutput o 1\noutput p 2\n").unwrap();
    let out = g.evaluate(Some(256)).unwrap();
    assert_eq!((out["o"].w, out["o"].h), (256, 128));
    // the same picture at half size (coverage is resolution independent)
    let big = render("size 512\nnode 1 path_stroke preset=3 width=24\noutput o 1\n", 512);
    let small = render("size 512\nnode 1 path_stroke preset=3 width=24\noutput o 1\n", 128);
    assert!((sum(&big) - sum(&small)).abs() < 0.01, "{} vs {}", sum(&big), sum(&small));
}

#[test]
fn splitting_a_segment_keeps_the_curve() {
    let mut p = Spline::preset(3, 0.2, 2.0).paths.remove(0);
    let before = p.flatten(512.0, 512.0);
    let (seg, t, d) = p.nearest([0.31, 0.5 + 0.2 * (0.31f32 * 2.0 * std::f32::consts::TAU).sin()], 512.0, 512.0).unwrap();
    assert!(d < 3.0, "nearest is on the curve ({d} px)");
    let n0 = p.pts.len();
    let idx = p.split(seg, t);
    assert_eq!(p.pts.len(), n0 + 1);
    assert_eq!(idx, seg + 1);
    let after = p.flatten(512.0, 512.0);
    // every flattened point of the old curve is still on the new one
    let worst = before.pts.iter().map(|a| after.pts.windows(2).map(|w| {
        let (dx, dy) = (w[1][0] - w[0][0], w[1][1] - w[0][1]);
        let l2 = (dx * dx + dy * dy).max(1e-9);
        let t = (((a[0] - w[0][0]) * dx + (a[1] - w[0][1]) * dy) / l2).clamp(0.0, 1.0);
        ((a[0] - w[0][0] - dx * t).powi(2) + (a[1] - w[0][1] - dy * t).powi(2)).sqrt()
    }).fold(f32::MAX, f32::min)).fold(0.0f32, f32::max);
    assert!(worst < 0.6, "curve moved by {worst} px");
    // closed path: splitting the closing segment appends the point
    let mut c = Spline::preset(4, 0.0, 1.0).paths.remove(0);
    let last = c.segment_count() - 1;
    let at = c.split(last, 0.5);
    assert_eq!(at, c.pts.len() - 1);
    let (hi, ho) = c.handles(0);
    assert!(hi[0].is_finite() && ho[1].is_finite());
}

fn graph_with(text: &str, files: &[(&str, &str)]) -> Graph {
    let mut g = Graph::parse(text).unwrap_or_else(|e| panic!("{e}"));
    for (k, v) in files {
        g.resources.insert((*k).into(), std::sync::Arc::new(v.as_bytes().to_vec()));
    }
    g
}

fn eval(g: &Graph, size: usize) -> materialcraft::tex::Tex {
    g.evaluate(Some(size)).unwrap_or_else(|e| panic!("{e}")).remove("o").unwrap()
}

const LINE: &str = "spline 1\npath smooth=0\npt 0.1 0.5\npt 0.9 0.5\n";

#[test]
fn tangent_modes_and_heights_are_saved() {
    let sp = Spline::parse("spline 1\npath\npt 0.1 0.2 tan=3 z=0.25\npt 0.5 0.5 tan=1\npt 0.9 0.2 tan=free\n").unwrap();
    assert_eq!(sp.paths[0].pts[0].tan, 3);
    assert_eq!(sp.paths[0].pts[2].tan, 2);
    assert!((sp.paths[0].pts[0].z - 0.25).abs() < 1e-6);
    let back = Spline::parse(&sp.to_text()).unwrap();
    assert_eq!(back, sp);
    assert!(Spline::parse("spline 1\npath\npt 0 0 tan=9").is_err());
    // a corner point has no automatic handle, so the curve through it bends sharply
    let mut a = sp.clone();
    a.paths[0].pts[0].tan = 0;
    let (ca, cb) = (a.flatten(100.0, 100.0), sp.flatten(100.0, 100.0));
    assert!(ca[0].pts.iter().zip(&cb[0].pts).any(|(p, q)| (p[0] - q[0]).abs() + (p[1] - q[1]).abs() > 0.1));
}

#[test]
fn transform_merge_and_reverse() {
    let sp = Spline::parse(LINE).unwrap();
    let moved = sp.transformed([1.0, 1.0], 0.0, [0.0, 0.25], (false, false), false);
    assert!((moved.paths[0].pts[0].p[1] - 0.75).abs() < 1e-6);
    let turned = sp.transformed([1.0, 1.0], 0.25, [0.0, 0.0], (false, false), false);
    assert!((turned.paths[0].pts[0].p[0] - 0.5).abs() < 1e-5 && (turned.paths[0].pts[0].p[1] - 0.1).abs() < 1e-5);
    let rev = sp.transformed([1.0, 1.0], 0.0, [0.0, 0.0], (false, false), true);
    assert!((rev.paths[0].pts[0].p[0] - 0.9).abs() < 1e-6);
    // mirroring flips the direction of travel; reverse puts it back
    let m = sp.transformed([1.0, 1.0], 0.0, [0.0, 0.0], (true, false), false);
    assert!((m.paths[0].pts[0].p[0] - 0.1).abs() < 1e-6, "mirror alone keeps the start index");
    let mut both = sp.clone();
    both.merge(moved);
    assert_eq!(both.paths.len(), 2);
}

#[test]
fn node_options_move_merge_and_pick_paths() {
    let base = "size 256\nnode 1 path_stroke file=\"a.spl\" width=8";
    let plain = eval(&graph_with(&format!("{base}\noutput o 1\n"), &[("a.spl", LINE)]), 256);
    assert!(plain.at(128, 128)[0] > 0.9 && plain.at(128, 64)[0] < 0.01);
    // move the path up a quarter of the texture
    let up = eval(&graph_with(&format!("{base} path_y=-0.25\noutput o 1\n"), &[("a.spl", LINE)]), 256);
    assert!(up.at(128, 64)[0] > 0.9 && up.at(128, 128)[0] < 0.01);
    // a second file adds a path
    let two = eval(&graph_with(&format!("{base} file2=\"b.spl\"\noutput o 1\n"), &[("a.spl", LINE), ("b.spl", "spline 1\npath smooth=0\npt 0.1 0.2\npt 0.9 0.2\n")]), 256);
    assert!(two.at(128, 128)[0] > 0.9 && two.at(128, 51)[0] > 0.9);
    // choose the second path of one file only
    let both = "spline 1\npath smooth=0\npt 0.1 0.5\npt 0.9 0.5\npath smooth=0\npt 0.1 0.2\npt 0.9 0.2\n";
    let second = eval(&graph_with(&format!("{base} path=1\noutput o 1\n"), &[("a.spl", both)]), 256);
    assert!(second.at(128, 51)[0] > 0.9 && second.at(128, 128)[0] < 0.01);
    let range = eval(&graph_with(&format!("{base} path=0 path_count=2\noutput o 1\n"), &[("a.spl", both)]), 256);
    assert!(range.at(128, 51)[0] > 0.9 && range.at(128, 128)[0] > 0.9);
    // the edited content of the second file changes the picture
    let mut g = graph_with(&format!("{base} file2=\"b.spl\"\noutput o 1\n"), &[("a.spl", LINE), ("b.spl", "spline 1\npath smooth=0\npt 0.1 0.2\npt 0.9 0.2\n")]);
    let a = eval(&g, 256);
    g.resources.insert("b.spl".into(), std::sync::Arc::new(b"spline 1\npath smooth=0\npt 0.1 0.8\npt 0.9 0.8\n".to_vec()));
    let b = eval(&g, 256);
    assert!(a.at(128, 51)[0] > 0.9 && b.at(128, 51)[0] < 0.01 && b.at(128, 205)[0] > 0.9);
}

#[test]
fn path_sample_gives_position_width_height_and_side() {
    let sp = "spline 1\npath smooth=0\npt 0.1 0.5 z=0.2 w=0.5\npt 0.9 0.5 z=1 w=2\n";
    let g = |o: &str| graph_with(&format!("size 256\nnode 1 path_sample file=\"a.spl\" width=40\noutput o 1.{o}\n"), &[("a.spl", sp)]);
    let pos = eval(&g("position"), 256);
    assert!(pos.at(60, 128)[0] < pos.at(190, 128)[0] && pos.at(128, 20)[0] < 0.01);
    let wid = eval(&g("width"), 256);
    assert!(wid.at(200, 128)[0] > wid.at(40, 128)[0]);
    let hgt = eval(&g("height"), 256);
    assert!(hgt.at(200, 128)[0] > hgt.at(40, 128)[0] + 0.3);
    let side = eval(&g("side"), 256);
    let (a, b) = (side.at(128, 120)[0], side.at(128, 136)[0]);
    assert!((a - b).abs() > 0.1 && (side.at(128, 10)[0] - 0.5).abs() < 1e-4, "{a} {b}");
}

#[test]
fn path_warp_moves_the_picture_near_the_path_only() {
    let src = "node 1 pixel_processor expr=\"u\"\n";
    let plain = eval(&graph_with(&format!("size 256\n{src}node 2 path_warp in=@1 file=\"a.spl\" width=60 amount=0\noutput o 2\n"), &[("a.spl", LINE)]), 256);
    let warped = eval(&graph_with(&format!("size 256\n{src}node 2 path_warp in=@1 file=\"a.spl\" width=60 amount=30 mode=1\noutput o 2\n"), &[("a.spl", LINE)]), 256);
    assert!((plain.at(128, 128)[0] - 0.5).abs() < 0.01);
    assert!((warped.at(128, 128)[0] - plain.at(128, 128)[0]).abs() > 0.05, "pixels on the path slide");
    assert!((warped.at(128, 10)[0] - plain.at(128, 10)[0]).abs() < 1e-4, "far from the path nothing moves");
    // push away moves pixels sideways: the image changes where the gradient has a vertical part
    let v = "node 1 pixel_processor expr=\"v\"\n";
    let push = eval(&graph_with(&format!("size 256\n{v}node 2 path_warp in=@1 file=\"a.spl\" width=60 amount=20 mode=0\noutput o 2\n"), &[("a.spl", LINE)]), 256);
    assert!((push.at(128, 118)[0] - 118.5 / 256.0).abs() > 0.02);
}

#[test]
fn splatter_makes_rows_and_listens_to_a_mask() {
    let base = "size 256\nnode 1 path_splatter file=\"a.spl\" rows=3 row_spacing=30 spacing=16 scale=12 seed=3";
    let rows = eval(&graph_with(&format!("{base}\noutput o 1\n"), &[("a.spl", LINE)]), 256);
    let one = eval(&graph_with("size 256\nnode 1 path_splatter file=\"a.spl\" rows=1 spacing=16 scale=12 seed=3\noutput o 1\n", &[("a.spl", LINE)]), 256);
    assert!(sum(&rows) > sum(&one) * 2.0, "three rows cover more than one");
    // the outer rows sit 15 px (30 px at 512) above and below the path
    assert!((0..256).any(|x| rows.at(x, 113)[0] > 0.5) && (0..256).any(|x| rows.at(x, 143)[0] > 0.5));
    assert!((0..256).all(|x| one.at(x, 113)[0] < 0.01), "a single row stays on the path");
    // a mask that is black on the left half removes the copies there
    let masked = eval(&graph_with(&format!("size 256\nnode 9 pixel_processor expr=\"step(0.5,u)\"\nnode 1 path_splatter file=\"a.spl\" rows=3 row_spacing=30 spacing=16 scale=12 seed=3 mask=@9\noutput o 1\n"), &[("a.spl", LINE)]), 256);
    let half = |t: &materialcraft::tex::Tex, x0: i64| (x0..x0 + 80).map(|x| (0..256).map(|y| t.at(x, y)[0]).sum::<f32>()).sum::<f32>();
    assert!(half(&masked, 20) < half(&rows, 20) * 0.1 && half(&masked, 160) > half(&rows, 160) * 0.5);
}

#[test]
fn a_picture_is_traced_into_a_spline() {
    use materialcraft::tex::Tex;
    // a bright disc on a dark ground
    let t = Tex::from_fn(128, 128, |u, v| {
        let d = ((u - 0.5).powi(2) + (v - 0.5).powi(2)).sqrt();
        let g = if d < 0.25 { 1.0 } else { 0.0 };
        [g, g, g, 1.0]
    });
    let sp = Spline::from_mask(&t, 0.5, 0.002).unwrap();
    assert_eq!(sp.paths.len(), 1);
    assert!(sp.paths[0].closed && sp.paths[0].pts.len() >= 8);
    for q in &sp.paths[0].pts {
        let d = ((q.p[0] - 0.5).powi(2) + (q.p[1] - 0.5).powi(2)).sqrt();
        assert!((d - 0.25).abs() < 0.01, "point at distance {d}");
    }
    // the traced spline fills to about the area of the disc
    let g = graph_with("size 128\nnode 1 path_fill file=\"t.spl\"\noutput o 1\n", &[]);
    let mut g = g;
    g.resources.insert("t.spl".into(), std::sync::Arc::new(sp.to_text().into_bytes()));
    let area = sum(&eval(&g, 128));
    assert!((area - std::f32::consts::PI * 0.0625).abs() < 0.02, "area {area}");
    // a stripe across the seam comes out as an open path that ends one texture further
    let stripe = Tex::from_fn(64, 64, |u, _| {
        let g = if u < 0.1 || u > 0.9 { 1.0 } else { 0.0 };
        [g, g, g, 1.0]
    });
    let ss = Spline::from_mask(&stripe, 0.5, 0.002).unwrap();
    assert!(!ss.paths.is_empty() && ss.paths.iter().all(|p| p.pts.iter().all(|q| q.p[0].is_finite())));
    // flat pictures cannot be traced
    let flat = Tex::from_fn(16, 16, |_, _| [0.0, 0.0, 0.0, 1.0]);
    assert!(Spline::from_mask(&flat, 0.5, 0.002).is_err());
}
