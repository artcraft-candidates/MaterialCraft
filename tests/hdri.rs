// SPDX-License-Identifier: MIT OR Apache-2.0
//! Environments: file readers, spherical harmonics, GGX prefiltering, importance sampling.
use materialcraft::hdri::{self, Env};
use materialcraft::tex::Tex;

fn close(a: [f32; 3], b: [f32; 3], tol: f32) -> bool {
    (0..3).all(|k| (a[k] - b[k]).abs() <= tol)
}

#[test]
fn radiance_file_round_trip() {
    let t = Tex::from_fn(40, 20, |u, v| [u * 4.0, v * 0.5, 0.25 + u * v, 1.0]);
    let bytes = materialcraft::exr::encode_hdr(&t);
    let d = hdri::decode_hdr(&bytes).unwrap();
    assert_eq!((d.w, d.h), (40, 20));
    for (a, b) in t.px.iter().zip(&d.px) {
        for k in 0..3 {
            // RGBE keeps about 1 % relative precision
            assert!((a[k] - b[k]).abs() <= 0.012 * a[0].max(a[1]).max(a[2]).max(0.01), "{a:?} {b:?}");
        }
    }
}

#[test]
fn radiance_run_length_scanlines() {
    // hand-built new-style RLE file: 8 pixels wide, one scanline
    let mut b = b"#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n-Y 1 +X 8\n".to_vec();
    b.extend_from_slice(&[2, 2, 0, 8]);
    for ch in [128u8, 64, 32, 129] {
        b.push(128 + 8); // a run of eight equal bytes
        b.push(ch);
    }
    let t = hdri::decode_hdr(&b).unwrap();
    assert_eq!((t.w, t.h), (8, 1));
    let f = 2f32.powi(129 - 136);
    assert!((t.px[3][0] - 128.0 * f).abs() < 1e-6 && (t.px[7][2] - 32.0 * f).abs() < 1e-6);
}

#[test]
fn openexr_round_trip_half_and_float() {
    let t = Tex::from_fn(33, 17, |u, v| [u * 100.0, v, 0.5, 1.0]);
    for float in [false, true] {
        let bytes = materialcraft::exr::encode_exr(&t, float);
        let d = hdri::decode_exr(&bytes).unwrap();
        assert_eq!((d.w, d.h), (33, 17));
        for (a, b) in t.px.iter().zip(&d.px) {
            assert!((a[0] - b[0]).abs() <= if float { 1e-5 } else { a[0] * 1e-3 + 1e-3 }, "{a:?} {b:?}");
            assert!((a[1] - b[1]).abs() <= if float { 1e-5 } else { 1e-3 });
        }
    }
}

#[test]
fn bad_files_are_rejected_with_a_message() {
    assert!(hdri::decode_hdr(b"hello").is_err());
    assert!(hdri::decode_exr(b"hello world").is_err());
    let mut b = materialcraft::exr::encode_exr(&Tex::from_fn(4, 4, |_, _| [1.0; 4]), false);
    b.truncate(b.len() / 2);
    assert!(hdri::decode_exr(&b).is_err());
    assert!(hdri::load_bytes("x.tga", b"nope").is_err());
}

#[test]
fn direction_mapping_round_trips() {
    for (u, v) in [(0.5, 0.5), (0.1, 0.3), (0.9, 0.8), (0.25, 0.5)] {
        let d = hdri::uv_to_dir(u, v);
        let (a, b) = hdri::dir_to_uv(d);
        assert!((a - u).abs() < 1e-4 && (b - v).abs() < 1e-4, "{u} {v} -> {a} {b}");
    }
    let f = hdri::uv_to_dir(0.5, 0.5);
    assert!((f[2] + 1.0).abs() < 1e-5, "{f:?}"); // the middle of the image looks down -Z
    assert!((hdri::uv_to_dir(0.3, 0.0)[1] - 1.0).abs() < 1e-5); // the top row is +Y
}

#[test]
fn white_furnace_keeps_its_colour_at_every_roughness() {
    let env = hdri::uniform([0.8, 0.5, 0.2]);
    for n in [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.3, -0.5, 0.8]] {
        let n = materialcraft::v3::norm(n);
        assert!(close(env.irradiance(n), [0.8, 0.5, 0.2], 0.01), "{:?}", env.irradiance(n));
        for r in [0.0, 0.3, 0.7, 1.0] {
            assert!(close(env.specular(n, r), [0.8, 0.5, 0.2], 0.01), "r {r}: {:?}", env.specular(n, r));
        }
    }
}

#[test]
fn sh_irradiance_follows_the_bright_side() {
    // bright above the horizon, dark below
    let img = Tex::from_fn(128, 64, |_, v| if v < 0.5 { [2.0, 2.0, 2.0, 1.0] } else { [0.0, 0.0, 0.0, 1.0] });
    let env = Env::from_image("half", &img, 128);
    let up = env.irradiance([0.0, 1.0, 0.0]);
    let down = env.irradiance([0.0, -1.0, 0.0]);
    let side = env.irradiance([1.0, 0.0, 0.0]);
    // exact values for a hemisphere light: up = 2 * (SH truncation of 1) ~ 1.9, down ~ 0.1, side = 1
    assert!(up[0] > 1.7 && up[0] < 2.1, "{up:?}");
    assert!(down[0] < 0.3, "{down:?}");
    assert!((side[0] - 1.0).abs() < 0.12, "{side:?}");
}

#[test]
fn prefiltered_levels_blur_and_keep_energy() {
    let env = hdri::preset("sky");
    assert!(env.n_levels() >= 6);
    // the sun is a hot spot at level 0 and spreads out at high roughness, but the mean stays
    let sun = materialcraft::v3::norm([0.45, 0.62, -0.45]);
    let sharp = env.specular(sun, 0.0)[0];
    let rough = env.specular(sun, 0.8)[0];
    assert!(sharp > 20.0 && rough < sharp * 0.2, "{sharp} {rough}");
    let mean_of = |lvl: usize| {
        let l = &env.levels[lvl];
        let mut s = 0.0;
        let mut ws = 0.0;
        for y in 0..l.h {
            let w = ((y as f32 + 0.5) / l.h as f32 * std::f32::consts::PI).sin();
            for x in 0..l.w {
                s += l.px[y * l.w + x][1] * w;
                ws += w;
            }
        }
        s / ws
    };
    let (m0, m3) = (mean_of(0), mean_of(3));
    assert!((m0 - m3).abs() / m0 < 0.25, "mean {m0} vs {m3}");
}

#[test]
fn importance_sampler_prefers_bright_directions() {
    let env = hdri::preset("sky");
    let s = env.sampler();
    let sun = materialcraft::v3::norm([0.45, 0.62, -0.45]);
    let mut near_sun = 0;
    let n = 2000;
    for i in 0..n {
        let (d, pdf) = s.sample((i as f32 + 0.5) / n as f32, 0.5);
        assert!(pdf > 0.0 && pdf.is_finite());
        if materialcraft::v3::dot(d, sun) > 0.9 {
            near_sun += 1;
        }
    }
    // the sun covers about 2 % of the sky area but carries most of the light
    assert!(near_sun as f32 / n as f32 > 0.2, "{near_sun}");
    // the density integrates to one over the sphere (coarse check on a grid)
    let mut total = 0.0;
    let (gw, gh) = (256, 128);
    for y in 0..gh {
        let v = (y as f32 + 0.5) / gh as f32;
        for x in 0..gw {
            let d = hdri::uv_to_dir((x as f32 + 0.5) / gw as f32, v);
            total += s.pdf(d) * (2.0 * std::f32::consts::PI / gw as f32) * (std::f32::consts::PI / gh as f32) * (v * std::f32::consts::PI).sin();
        }
    }
    assert!((total - 1.0).abs() < 0.08, "{total}");
}

#[test]
fn every_preset_builds() {
    for (name, _) in hdri::PRESETS {
        let e = hdri::preset(name);
        assert_eq!(&e.name, name);
        assert!(e.peak > 0.0 && e.mean > 0.0 && e.mean.is_finite(), "{name}");
    }
}
