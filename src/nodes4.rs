// SPDX-License-Identifier: MIT OR Apache-2.0
//! Fourth set of nodes: material (PBR) utilities - workflow converters, material blending, height combining,
//! curvature, leak and dust masks, palette tools and a mirror tiler. `compute` returns Ok(None) for kinds it does not
//! know. Everything tiles and works on non-square graphs; pixel sizes are given for 512 px and scaled by `size / 512`.
use crate::graph::{Error, Node};
use crate::ops;
use crate::tex::{lum, Tex};
use std::collections::BTreeMap;
use std::sync::Arc as Rc;

fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

fn smooth(a: f32, b: f32, x: f32) -> f32 {
    let d = b - a;
    let t = if d.abs() < 1e-9 { if x >= b { 1.0 } else { 0.0 } } else { ((x - a) / d).clamp(0.0, 1.0) };
    t * t * (3.0 - 2.0 * t)
}

/// Reflectance of a non-metal at normal incidence.
const DIELECTRIC_F0: f32 = 0.04;

fn brightness(c: [f32; 4]) -> f32 {
    (0.299 * c[0] * c[0] + 0.587 * c[1] * c[1] + 0.114 * c[2] * c[2]).sqrt()
}

/// Metalness that best explains a diffuse / specular pair (quadratic from the metal-roughness model).
fn solve_metallic(diffuse: f32, specular: f32, one_minus_spec: f32) -> f32 {
    if specular < DIELECTRIC_F0 {
        return 0.0;
    }
    let a = DIELECTRIC_F0;
    let b = diffuse * one_minus_spec / (1.0 - DIELECTRIC_F0) + specular - 2.0 * DIELECTRIC_F0;
    let c = DIELECTRIC_F0 - specular;
    let d = (b * b - 4.0 * a * c).max(0.0);
    ((-b + d.sqrt()) / (2.0 * a)).clamp(0.0, 1.0)
}

fn decode(p: [f32; 4]) -> [f32; 3] {
    [(p[0] - 0.5) * 2.0, (p[1] - 0.5) * 2.0, (p[2] - 0.5) * 2.0]
}

fn encode(v: [f32; 3]) -> [f32; 4] {
    let l = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if l < 1e-8 {
        return [0.5, 0.5, 1.0, 1.0];
    }
    [v[0] / l * 0.5 + 0.5, v[1] / l * 0.5 + 0.5, v[2] / l * 0.5 + 0.5, 1.0]
}

fn grey(v: f32) -> [f32; 4] {
    [v, v, v, 1.0]
}

/// k-means over RGB samples; returns `k` centres sorted by luminance.
fn kmeans(samples: &[[f32; 3]], k: usize, iters: usize) -> Vec<[f32; 3]> {
    if samples.is_empty() {
        return vec![[0.0; 3]; k];
    }
    let mut sorted: Vec<[f32; 3]> = samples.to_vec();
    sorted.sort_by(|a, b| (0.2126 * a[0] + 0.7152 * a[1] + 0.0722 * a[2]).partial_cmp(&(0.2126 * b[0] + 0.7152 * b[1] + 0.0722 * b[2])).unwrap_or(std::cmp::Ordering::Equal));
    let mut centres: Vec<[f32; 3]> = (0..k).map(|i| sorted[((i as f32 + 0.5) / k as f32 * sorted.len() as f32) as usize % sorted.len()]).collect();
    for _ in 0..iters {
        let mut sum = vec![[0.0f32; 3]; k];
        let mut cnt = vec![0usize; k];
        for s in samples {
            let mut best = (0usize, f32::MAX);
            for (i, c) in centres.iter().enumerate() {
                let d = (0..3).map(|j| (s[j] - c[j]) * (s[j] - c[j])).sum::<f32>();
                if d < best.1 {
                    best = (i, d);
                }
            }
            for j in 0..3 {
                sum[best.0][j] += s[j];
            }
            cnt[best.0] += 1;
        }
        for i in 0..k {
            if cnt[i] > 0 {
                for j in 0..3 {
                    centres[i][j] = sum[i][j] / cnt[i] as f32;
                }
            }
        }
    }
    centres.sort_by(|a, b| (0.2126 * a[0] + 0.7152 * a[1] + 0.0722 * a[2]).partial_cmp(&(0.2126 * b[0] + 0.7152 * b[1] + 0.0722 * b[2])).unwrap_or(std::cmp::Ordering::Equal));
    centres
}

/// Run a node of this module. Ok(None) = not one of ours.
pub fn compute(kind: &str, id: u32, n: &Node, ins: &BTreeMap<String, Rc<Tex>>, s: usize, hs: usize, _seed: u32) -> Result<Option<Tex>, Error> {
    let f = |k: &str, d: f32| n.nums.get(k).copied().filter(|v| v.is_finite()).unwrap_or(d);
    let need = |slot: &str| -> Result<&Tex, Error> { ins.get(slot).map(|r| r.as_ref()).ok_or_else(|| Error::MissingInput(id, slot.to_string())) };
    let opt = |slot: &str| -> Option<&Tex> { ins.get(slot).map(|r| r.as_ref()) };
    let rs = s as f32 / 512.0;
    let out_idx = f("_out", 0.0).round() as usize;
    let out = match kind {
        "metal_rough_to_spec_gloss" => {
            let (base, metal, rough) = (need("base_color")?, opt("metallic"), opt("roughness"));
            Tex::from_fn(base.w, base.h, |u, v| {
                let b = base.sample(u, v);
                let m = metal.map(|t| lum(t.sample(u, v))).unwrap_or(0.0).clamp(0.0, 1.0);
                let r = rough.map(|t| lum(t.sample(u, v))).unwrap_or(0.5).clamp(0.0, 1.0);
                match out_idx {
                    0 => [b[0] * (1.0 - m), b[1] * (1.0 - m), b[2] * (1.0 - m), b[3]],
                    1 => [lerp(DIELECTRIC_F0, b[0], m), lerp(DIELECTRIC_F0, b[1], m), lerp(DIELECTRIC_F0, b[2], m), 1.0],
                    _ => grey(1.0 - r),
                }
            })
        }
        "spec_gloss_to_metal_rough" => {
            let (diff, spec, gloss) = (need("diffuse")?, need("specular")?, opt("glossiness"));
            Tex::from_fn(diff.w, diff.h, |u, v| {
                let (d, sp) = (diff.sample(u, v), spec.sample(u, v));
                let one_minus = 1.0 - sp[0].max(sp[1]).max(sp[2]).clamp(0.0, 1.0);
                let m = solve_metallic(brightness(d), brightness(sp), one_minus);
                match out_idx {
                    0 => {
                        let mut o = [0.0f32; 4];
                        for c in 0..3 {
                            let from_diffuse = d[c] * (one_minus / (1.0 - DIELECTRIC_F0)) / (1.0 - m).max(1e-4);
                            let from_spec = (sp[c] - DIELECTRIC_F0 * (1.0 - m)) / m.max(1e-4);
                            o[c] = lerp(from_diffuse, from_spec, m * m).clamp(0.0, 1.0);
                        }
                        o[3] = d[3];
                        o
                    }
                    1 => grey(m),
                    _ => grey(1.0 - gloss.map(|t| lum(t.sample(u, v))).unwrap_or(0.5).clamp(0.0, 1.0)),
                }
            })
        }
        "material_blend" => {
            let (opacity, influence, sharp) = (f("opacity", 1.0).clamp(0.0, 1.0), f("height_influence", 1.0).clamp(0.0, 8.0), f("sharpness", 1.0).clamp(0.1, 32.0));
            let mask = opt("mask");
            let (b_base, b_nrm, b_h, b_r, b_m, b_ao) = (opt("bg_base"), opt("bg_normal"), opt("bg_height"), opt("bg_rough"), opt("bg_metal"), opt("bg_ao"));
            let (f_base, f_nrm, f_h, f_r, f_m, f_ao) = (opt("fg_base"), opt("fg_normal"), opt("fg_height"), opt("fg_rough"), opt("fg_metal"), opt("fg_ao"));
            let g = |t: Option<&Tex>, u: f32, v: f32, d: f32| t.map(|t| lum(t.sample(u, v))).unwrap_or(d);
            Tex::from_fn(s, hs, |u, v| {
                let k = (mask.map(|m| lum(m.sample(u, v))).unwrap_or(1.0) * opacity).clamp(0.0, 1.0);
                let (ha, hf) = (g(b_h, u, v, 0.5), g(f_h, u, v, 0.5));
                // where the blend is half way, the higher surface wins; at 0 and 1 the mask decides alone
                let w = (k + (hf - ha) * influence * 4.0 * k * (1.0 - k)).clamp(0.0, 1.0);
                let w = ((w - 0.5) * sharp + 0.5).clamp(0.0, 1.0);
                match out_idx {
                    0 => {
                        let a = b_base.map(|t| t.sample(u, v)).unwrap_or([0.5, 0.5, 0.5, 1.0]);
                        let b = f_base.map(|t| t.sample(u, v)).unwrap_or([0.5, 0.5, 0.5, 1.0]);
                        [lerp(a[0], b[0], w), lerp(a[1], b[1], w), lerp(a[2], b[2], w), lerp(a[3], b[3], w)]
                    }
                    1 => {
                        let a = decode(b_nrm.map(|t| t.sample(u, v)).unwrap_or([0.5, 0.5, 1.0, 1.0]));
                        let b = decode(f_nrm.map(|t| t.sample(u, v)).unwrap_or([0.5, 0.5, 1.0, 1.0]));
                        encode([lerp(a[0], b[0], w), lerp(a[1], b[1], w), lerp(a[2], b[2], w)])
                    }
                    2 => grey(lerp(ha, hf, w)),
                    3 => grey(lerp(g(b_r, u, v, 0.5), g(f_r, u, v, 0.5), w)),
                    4 => grey(lerp(g(b_m, u, v, 0.0), g(f_m, u, v, 0.0), w)),
                    _ => grey(lerp(g(b_ao, u, v, 1.0), g(f_ao, u, v, 1.0), w)),
                }
            })
        }
        "height_combine" => {
            let (a, b) = (need("a")?, need("b")?);
            let mask = opt("mask");
            let (mode, amount) = (f("mode", 0.0).round() as i32, f("amount", 1.0).clamp(0.0, 8.0));
            Tex::from_fn(a.w, a.h, |u, v| {
                let (x, y) = (lum(a.sample(u, v)), lum(b.sample(u, v)));
                let k = mask.map(|m| lum(m.sample(u, v))).unwrap_or(1.0).clamp(0.0, 1.0);
                let r = match mode {
                    1 => x.min(y),
                    2 => x + (y - 0.5) * amount,
                    3 => (x - y).abs(),
                    _ => x.max(y),
                };
                grey(lerp(x, r, k))
            })
        }
        "convex_concave" => {
            let t = need("in")?;
            let low = ops::blur(t, f("radius", 4.0).clamp(0.0, 128.0) * rs);
            let gain = f("gain", 4.0).clamp(0.0, 64.0);
            let mut o = t.clone();
            for (i, p) in o.px.iter_mut().enumerate() {
                let d = lum(*p) - lum(low.px[i]);
                let v = if out_idx == 0 { d } else { -d };
                *p = grey((v * gain).clamp(0.0, 1.0));
            }
            o
        }
        "leak_streaks" => {
            let t = need("in")?;
            let noise = opt("noise");
            let len = (f("length", 120.0).clamp(0.0, 1024.0) * rs) / s.max(1) as f32;
            let (sn, cs) = (f("angle", 0.25) * std::f32::consts::TAU).sin_cos();
            let (intensity, breakup) = (f("intensity", 1.0).clamp(0.0, 8.0), f("breakup", 0.5).clamp(0.0, 1.0));
            let aspect = t.w as f32 / t.h as f32;
            let taps = 48;
            Tex::from_fn(t.w, t.h, |u, v| {
                let mut m = lum(t.sample(u, v));
                for i in 1..=taps {
                    let q = i as f32 / taps as f32;
                    let d = len * q * q;
                    let src = lum(t.sample(u - cs * d, v - sn * d * aspect)) * (1.0 - q);
                    m = m.max(src);
                }
                if let Some(nz) = noise {
                    m *= lerp(1.0, lum(nz.sample(u, v)), breakup);
                }
                grey((m * intensity).clamp(0.0, 1.0))
            })
        }
        "settled_dust" => {
            let t = need("in")?;
            let noise = opt("noise");
            let (amount, slope, cavity) = (f("amount", 0.6).clamp(0.0, 1.0), f("slope", 0.3).clamp(0.0, 1.0), f("cavity", 0.5).clamp(0.0, 1.0));
            let nm = ops::normal_from_height(t, f("relief", 3.0).clamp(0.0, 32.0), false);
            let low = ops::blur(t, f("radius", 8.0).clamp(0.0, 128.0) * rs);
            let mut o = t.clone();
            for (i, p) in o.px.iter_mut().enumerate() {
                let up = (nm.px[i][2] - 0.5) * 2.0;
                let settle = smooth(1.0 - slope, 1.0, up);
                let cav = ((lum(low.px[i]) - lum(t.px[i])) * 6.0).clamp(0.0, 1.0);
                let mut m = (settle * amount * (1.0 - cavity * 0.5) + cav * cavity).clamp(0.0, 1.0);
                if let Some(nz) = noise {
                    m *= lum(nz.px[i.min(nz.px.len() - 1)]);
                }
                *p = grey(m);
            }
            o
        }
        "palette_extract" => {
            let t = need("in")?;
            let k = f("count", 5.0).round().clamp(2.0, 16.0) as usize;
            let stride = (t.px.len() / 16384).max(1);
            let samples: Vec<[f32; 3]> = t.px.iter().step_by(stride).map(|p| [p[0], p[1], p[2]]).collect();
            let c = kmeans(&samples, k, 12);
            Tex::from_fn(s, hs, |u, _| {
                let i = ((u * k as f32) as usize).min(k - 1);
                [c[i][0], c[i][1], c[i][2], 1.0]
            })
        }
        "palette_apply" => {
            let (t, pal) = (need("in")?, need("palette")?);
            let k = f("count", 5.0).round().clamp(2.0, 16.0) as usize;
            let amount = f("amount", 1.0).clamp(0.0, 1.0);
            let cols: Vec<[f32; 4]> = (0..k).map(|i| pal.sample((i as f32 + 0.5) / k as f32, 0.5)).collect();
            t.map(|p| {
                let mut best = (0usize, f32::MAX);
                for (i, c) in cols.iter().enumerate() {
                    let d = (0..3).map(|j| (p[j] - c[j]) * (p[j] - c[j])).sum::<f32>();
                    if d < best.1 {
                        best = (i, d);
                    }
                }
                let c = cols[best.0];
                [lerp(p[0], c[0], amount), lerp(p[1], c[1], amount), lerp(p[2], c[2], amount), p[3]]
            })
        }
        "make_it_tile_mirror" => {
            let t = need("in")?;
            let mode = f("mode", 0.0).round() as i32;
            let fold = |x: f32| (1.0 - 2.0 * x).abs();
            Tex::from_fn(t.w, t.h, |u, v| {
                let su = if mode == 2 { u } else { fold(u) };
                let sv = if mode == 1 { v } else { fold(v) };
                t.sample(su, sv)
            })
        }
        _ => return Ok(None),
    };
    Ok(Some(out))
}
