// SPDX-License-Identifier: MIT OR Apache-2.0
//! GPU-versus-CPU parity. Needs the `gpu` feature and an adapter (a software one is accepted here); skips otherwise.
#![cfg(feature = "gpu")]
use materialcraft::{gpu, noise, ops, tex::Tex};

static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Serialises the tests (they flip global switches) and reports whether an adapter exists.
fn setup() -> Option<std::sync::MutexGuard<'static, ()>> {
    let g = LOCK.lock().unwrap_or_else(|e| e.into_inner());
    std::env::set_var("MC_GPU", "force");
    gpu::set_enabled(true);
    gpu::set_min_pixels(0);
    if gpu::adapter_name().is_some() { Some(g) } else { None }
}

fn max_diff(a: &Tex, b: &Tex) -> f32 {
    assert_eq!((a.w, a.h), (b.w, b.h));
    a.px.iter().zip(&b.px).flat_map(|(p, q)| (0..4).map(move |c| (p[c] - q[c]).abs())).fold(0.0, f32::max)
}

#[test]
fn perlin_matches_cpu() {
    let Some(_guard) = setup() else { return };
    let g = gpu::noise(96, 64, false, 5, 3, 1, 0.5, 7, 0).expect("gpu perlin");
    let c = Tex::gray_fn(96, 64, |u, v| noise::perlin(u, v, 5, 3, 7));
    assert!(max_diff(&g, &c) < 2e-3, "{}", max_diff(&g, &c));
}

#[test]
fn fractal_matches_cpu_in_every_mode() {
    let Some(_guard) = setup() else { return };
    for mode in 0..3 {
        let g = gpu::noise(64, 64, true, 4, 4, 5, 0.5, 3, mode).expect("gpu fbm");
        let k = noise::Fbm::new(4, 4, 5, 0.5, 3, mode);
        let c = Tex::gray_fn(64, 64, |u, v| k.sample(u, v));
        assert!(max_diff(&g, &c) < 5e-3, "mode {mode}: {}", max_diff(&g, &c));
    }
}

#[test]
fn blur_matches_cpu() {
    let Some(_guard) = setup() else { return };
    let src = Tex::from_fn(80, 48, |u, v| [u, v, (u * 7.0).sin() * 0.5 + 0.5, 1.0]);
    let g = gpu::blur(&src, 6.0).expect("gpu blur");
    assert!(max_diff(&g, &ops::blur(&src, 6.0)) < 1e-3);
}

#[test]
fn normal_matches_cpu() {
    let Some(_guard) = setup() else { return };
    let src = Tex::gray_fn(64, 64, |u, v| noise::perlin(u, v, 4, 4, 1));
    for dx in [false, true] {
        let g = gpu::normal_from_height(&src, 2.0, dx).expect("gpu normal");
        assert!(max_diff(&g, &ops::normal_from_height(&src, 2.0, dx)) < 1e-3);
    }
}

#[test]
fn small_textures_stay_on_the_cpu_by_default() {
    let _guard = LOCK.lock().unwrap_or_else(|e| e.into_inner());
    gpu::set_min_pixels(1 << 20);
    assert!(gpu::noise(8, 8, false, 4, 4, 1, 0.5, 0, 0).is_none());
    gpu::set_min_pixels(0);
}


#[test]
fn levels_matches_cpu() {
    let Some(_guard) = setup() else { return };
    let src = Tex::from_fn(64, 32, |u, v| [u, v, (u * 5.0).sin() * 0.5 + 0.5, 0.75]);
    for (lo, hi, g, ol, oh) in [(0.1, 0.9, 1.0, 0.0, 1.0), (0.2, 0.7, 2.2, 0.1, 0.8), (0.9, 0.1, 1.0, 0.0, 1.0)] {
        let a = gpu::levels(&src, lo, hi, g, ol, oh).expect("gpu levels");
        assert!(max_diff(&a, &ops::levels(&src, lo, hi, g, ol, oh)) < 2e-4);
    }
}

#[test]
fn directional_blur_matches_cpu() {
    let Some(_guard) = setup() else { return };
    let src = Tex::from_fn(64, 48, |u, v| [u, (v * 9.0).sin() * 0.5 + 0.5, 0.3, 1.0]);
    for (a, l) in [(0.0, 0.1), (0.125, 0.3), (0.7, 0.05)] {
        let g = gpu::directional_blur(&src, a, l).expect("gpu directional blur");
        assert!(max_diff(&g, &ops::directional_blur(&src, a, l)) < 1e-3, "angle {a} length {l}");
    }
}

#[test]
fn blend_matches_cpu_in_every_mode() {
    let Some(_guard) = setup() else { return };
    let fg = Tex::from_fn(40, 24, |u, v| [u, v, 1.0 - u, 0.8]);
    let bg = Tex::from_fn(40, 24, |u, v| [(v * 3.0).fract(), u * 0.5, 0.4, 1.0]);
    let mask = Tex::from_fn(40, 24, |u, _| [u, u, u, 1.0]);
    for mode in (0..=18).chain([20, 21]) {
        for am in 0..4 {
            for m in [None, Some(&mask)] {
                let g = gpu::blend(&fg, &bg, m, mode, 0.7, am).expect("gpu blend");
                let c = ops::blend_ex(&fg, &bg, m, mode, 0.7, am, [0.0, 0.0, 1.0, 1.0]);
                assert!(max_diff(&g, &c) < 2e-4, "mode {mode} alpha {am}: {}", max_diff(&g, &c));
            }
        }
    }
    // dissolve and cropped blends stay on the CPU
    assert!(gpu::blend(&fg, &bg, None, 19, 1.0, 1).is_none());
}

#[test]
fn whole_graph_matches_the_cpu() {
    use materialcraft::graph::Graph;
    let Some(_guard) = setup() else { return };
    let g = Graph::parse(concat!(
        "size 64 48\n",
        "node 1 fbm scale=5 octaves=4\nnode 2 levels in=@1 low=0.2 high=0.8 gamma=1.4\nnode 3 blur in=@2 radius=6\n",
        "node 4 directional_blur in=@3 angle=0.1 length=0.1\nnode 5 perlin scale=3 seed=4\n",
        "node 6 blend fg=@5 bg=@4 mode=5 opacity=0.8\nnode 7 normal in=@6 strength=3\n",
        "output h 6\noutput n 7\n",
    ))
    .unwrap();
    gpu::set_enabled(true);
    assert!(gpu::active());
    let on = g.evaluate(None).unwrap();
    gpu::set_enabled(false);
    assert!(!gpu::active());
    let off = g.evaluate(None).unwrap();
    gpu::set_enabled(true);
    assert!(max_diff(&on["h"], &off["h"]) < 3e-3, "height {}", max_diff(&on["h"], &off["h"]));
    assert!(max_diff(&on["n"], &off["n"]) < 2e-2, "normal {}", max_diff(&on["n"], &off["n"]));
}

#[test]
fn baker_matches_cpu() {
    use materialcraft::bake::{self, BakeSettings, Device, MapKind};
    let Some(_guard) = setup() else { return };
    let low = bake::parse_obj(&bake::uv_sphere_obj(12, 8, 1.0, 0.0)).unwrap();
    let high = bake::parse_obj(&bake::uv_sphere_obj(48, 32, 1.0, 0.06)).unwrap();
    let base = BakeSettings {
        width: 48,
        height: 40,
        aa: 2,
        padding: 2,
        front: 0.3,
        rear: 0.3,
        ao_rays: 16,
        thick_rays: 8,
        ao_dist: 0.5,
        ao_falloff: 1.0,
        maps: vec![MapKind::Normal, MapKind::WorldNormal, MapKind::Position, MapKind::AmbientOcclusion, MapKind::BentNormal, MapKind::Curvature, MapKind::Thickness, MapKind::Height, MapKind::ObjectId, MapKind::Opacity],
        high_normals: bake::NormalMode::Angle(80.0),
        ..Default::default()
    };
    let cpu = bake::bake_full(&low, Some(&high), None, &BakeSettings { device: Device::Cpu, ..base.clone() }, None).unwrap();
    let g = bake::bake_full(&low, Some(&high), None, &BakeSettings { device: Device::Gpu, ..base.clone() }, None).unwrap();
    assert_eq!(cpu.report.device, "CPU");
    assert!(g.report.device != "CPU", "{}", g.report.device);
    assert_eq!(cpu.report.hits, g.report.hits);
    assert_eq!(cpu.report.rays, g.report.rays);
    for (name, c) in &cpu.maps {
        let gm = &g.maps[name];
        let (mut mx, mut sum) = (0.0f32, 0.0f32);
        for (a, b) in c.px.iter().zip(&gm.px) {
            for k in 0..3 {
                let d = (a[k] - b[k]).abs();
                mx = mx.max(d);
                sum += d;
            }
        }
        let mean = sum / (c.px.len() * 3) as f32;
        let (tol_max, tol_mean) = match name.as_str() {
            "ambient_occlusion" | "bent_normal" | "thickness" => (0.35, 0.01),
            _ => (0.05, 0.002),
        };
        assert!(mx <= tol_max && mean <= tol_mean, "{name}: max {mx} mean {mean}");
    }
    // cancellation and the object filter go through the same kernels
    let p = bake::Progress::new();
    p.request_cancel();
    assert!(bake::bake_full(&low, Some(&high), None, &BakeSettings { device: Device::Gpu, ..base.clone() }, Some(&p)).is_err());
}

#[test]
fn baker_object_matching_on_the_gpu() {
    use materialcraft::bake::{self, BakeSettings, Device, MapKind, Mesh};
    let Some(_guard) = setup() else { return };
    let mk = |name: &str, z: f32, uv0: f32| format!("o {name}\nv 0 0 {z}\nv 0.5 0 {z}\nv 0.5 1 {z}\nv 0 1 {z}\nvt {uv0} 0\nvt {} 0\nvt {} 1\nvt {uv0} 1\nvn 0 0 1\nf 1/1/1 2/2/1 3/3/1\nf 1/1/1 3/3/1 4/4/1\n", uv0 + 0.5, uv0 + 0.5);
    let join = |p: &[String]| Mesh::merge(&p.iter().map(|t| bake::parse_obj(t).unwrap()).collect::<Vec<_>>());
    let low = join(&[mk("a_low", 0.0, 0.0), mk("b_low", 0.0, 0.5)]);
    let high = join(&[mk("a_high", 0.1, 0.0), mk("b_high", -0.1, 0.0)]);
    let s = BakeSettings { width: 32, height: 32, aa: 1, front: 0.3, rear: 0.3, maps: vec![MapKind::Height], device: Device::Gpu, ..Default::default() };
    let r = bake::bake_full(&low, Some(&high), None, &s, None).unwrap();
    let h = &r.maps["height"];
    let l = h.px[16 * 32 + 8][0];
    let rr = h.px[16 * 32 + 24][0];
    assert!(l > rr + 0.05, "{l} {rr}");
}

#[test]
fn baker_windows_give_the_same_result() {
    use materialcraft::bake::{self, BakeSettings, Device, MapKind};
    let Some(_guard) = setup() else { return };
    let low = bake::parse_obj(&bake::uv_sphere_obj(10, 6, 1.0, 0.0)).unwrap();
    let high = bake::parse_obj(&bake::uv_sphere_obj(30, 20, 1.0, 0.05)).unwrap();
    let s = BakeSettings { width: 40, height: 40, aa: 2, front: 0.3, rear: 0.3, ao_rays: 8, thick_rays: 8, maps: vec![MapKind::Normal, MapKind::AmbientOcclusion, MapKind::Thickness], device: Device::Gpu, high_normals: bake::NormalMode::Angle(80.0), ..Default::default() };
    std::env::remove_var("MC_BAKE_WINDOW");
    let one = bake::bake_full(&low, Some(&high), None, &s, None).unwrap();
    std::env::set_var("MC_BAKE_WINDOW", "300");
    let many = bake::bake_full(&low, Some(&high), None, &s, None).unwrap();
    std::env::remove_var("MC_BAKE_WINDOW");
    for (k, a) in &one.maps {
        assert_eq!(a.px, many.maps[k].px, "{k}");
    }
}

/// Evaluate `text` with the GPU on and off and return both result sets' named output.
fn both(text: &str, out: &str) -> (Tex, Tex) {
    use materialcraft::graph::Graph;
    let g = Graph::parse(text).unwrap();
    gpu::set_enabled(true);
    let on = g.evaluate(None).unwrap();
    gpu::set_enabled(false);
    let off = g.evaluate(None).unwrap();
    gpu::set_enabled(true);
    (on[out].clone(), off[out].clone())
}

#[test]
fn spatial_nodes_match_the_cpu() {
    let Some(_guard) = setup() else { return };
    let src = "size 64 48\nnode 1 fbm scale=4 octaves=4 seed=2\nnode 2 perlin scale=3 seed=9\n";
    let cases: &[(&str, &str, f32)] = &[
        ("node 3 radial_blur in=@1 mode=0 samples=32 angle=0.1 x=0.1 y=-0.05", "radial spin", 4e-3),
        ("node 3 radial_blur in=@1 mode=1 amount=0.3 radius=0.4", "radial zoom", 4e-3),
        ("node 3 bilateral_blur in=@1 radius=3 range=0.1", "bilateral", 4e-3),
        ("node 3 slope_blur in=@1 slope=@2 samples=12 intensity=0.1 mode=0", "slope avg", 6e-3),
        ("node 3 slope_blur in=@1 slope=@2 samples=12 intensity=0.1 mode=1", "slope min", 6e-3),
        ("node 3 slope_blur in=@1 slope=@2 samples=12 intensity=0.1 mode=2", "slope max", 6e-3),
        ("node 3 bent_normal in=@1 radius=12 relief=0.5 directions=12 steps=6", "bent normal", 2e-2),
    ];
    for (node, name, tol) in cases {
        let (on, off) = both(&format!("{src}{node}\noutput o 3\n"), "o");
        let d = max_diff(&on, &off);
        assert!(d < *tol, "{name}: {d}");
    }
    // the occlusion output of the bent normal node
    let (on, off) = both(&format!("{src}node 3 bent_normal in=@1 radius=12 relief=0.5\noutput o 3.ao\n"), "o");
    assert!(max_diff(&on, &off) < 1e-2, "bent ao {}", max_diff(&on, &off));
}

#[test]
fn spatial_kernels_really_run_on_the_gpu() {
    let Some(_guard) = setup() else { return };
    let t = Tex::gray_fn(32, 32, |u, v| noise::perlin(u, v, 3, 3, 1));
    assert!(gpu::radial_blur(&t, 0, 16, 0.5, 0.5, 0.5, 0.0).is_some());
    assert!(gpu::radial_blur(&t, 1, 24, 0.5, 0.5, 0.2, 0.4).is_some());
    assert!(gpu::bilateral_blur(&t, &t, 3, 0.1).is_some());
    assert!(gpu::slope_blur(&t, &t, 8, 0.1, 0).is_some());
    assert!(gpu::bent_normal(&t, 8, 4, 0.05, 0.5, false, false).is_some());
    // mismatched sizes stay on the CPU
    let small = Tex::gray_fn(16, 16, |u, _| u);
    assert!(gpu::slope_blur(&t, &small, 8, 0.1, 0).is_none());
}

#[test]
fn extrude_and_anisotropic_match_the_cpu() {
    let Some(_guard) = setup() else { return };
    let src = "size 64 64\nnode 1 fbm scale=3 octaves=3 seed=5\nnode 2 levels in=@1 low=0.5 high=0.52\n";
    for (node, out, tol) in [
        ("node 3 shape_extrude in=@2 length=30 angle=0.2", "o", 6e-3f32),
        ("node 3 shape_extrude in=@2 length=30 angle=0.2", "o2", 6e-3),
        ("node 3 anisotropic_blur in=@1 intensity=10 anisotropy=0.7 angle=0.3", "o", 6e-3),
    ] {
        let pin = if out == "o2" { "3.walls" } else { "3" };
        let (on, off) = both(&format!("{src}{node}\noutput {out} {pin}\n"), out);
        let d = max_diff(&on, &off);
        assert!(d < tol, "{node} {out}: {d}");
    }
    let t = Tex::gray_fn(32, 32, |u, v| if (u - 0.5).abs() < 0.2 && (v - 0.5).abs() < 0.2 { 1.0 } else { 0.0 });
    assert!(gpu::extrude(&t, 10, 0.01, 0.0, 0.05).is_some());
}

#[test]
fn warps_match_the_cpu() {
    let Some(_guard) = setup() else { return };
    let src = "size 64 48\nnode 1 fbm scale=6 octaves=4 seed=1\nnode 2 perlin scale=3 seed=7\nnode 4 normal in=@2 strength=2\n";
    let mut cases = vec![];
    for f in 0..3 {
        cases.push((format!("node 3 warp in=@1 warp=@2 intensity=0.08 filtering={f}"), format!("warp f{f}")));
        cases.push((format!("node 3 directional_warp in=@1 warp=@2 angle=0.2 intensity=0.2 offset=0.4 filtering={f}"), format!("dir f{f}")));
    }
    cases.push(("node 3 vector_warp in=@1 vector=@4 intensity=0.06".into(), "vector".into()));
    cases.push(("node 3 vector_warp in=@1 vector=@4 intensity=0.06 directx=1".into(), "vector dx".into()));
    for (node, name) in cases {
        let (on, off) = both(&format!("{src}{node}\noutput o 3\n"), "o");
        let d = max_diff(&on, &off);
        // nearest sampling may flip a texel where a coordinate lands on an edge; compare the mean for it
        if name.ends_with("f1") {
            let mean: f32 = on.px.iter().zip(&off.px).map(|(a, b)| (a[0] - b[0]).abs()).sum::<f32>() / on.px.len() as f32;
            assert!(mean < 5e-3, "{name}: mean {mean}");
        } else {
            assert!(d < 8e-3, "{name}: {d}");
        }
    }
}

#[test]
fn directional_noise_matches_the_cpu() {
    let Some(_guard) = setup() else { return };
    for (sc, st, ang) in [(6, 8.0, 0.0), (4, 20.0, 0.3), (10, 3.0, 0.7)] {
        let (on, off) = both(&format!("size 64 48\nnode 1 directional_noise scale={sc} stretch={st} angle={ang} seed=3\noutput o 1\n"), "o");
        assert!(max_diff(&on, &off) < 1e-2, "{sc} {st} {ang}: {}", max_diff(&on, &off));
    }
}
