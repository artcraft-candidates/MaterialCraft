// SPDX-License-Identifier: MIT OR Apache-2.0
//! Post-processing of a rendered frame: ambient occlusion from the depth buffer, depth of field, bloom, white balance,
//! contrast / saturation, vignette and tone mapping. The GLSL post shader of the GPU viewport follows the same steps in
//! the same order, so both renderers give the same picture.
use crate::scene::{Post, ToneMap};
use crate::tex::{lum, par_map, Tex};
use crate::v3::*;

/// Camera facts the depth based effects need.
#[derive(Clone, Copy, Debug)]
pub struct PostCtx {
    /// tan of half the vertical field of view
    pub tan_half: f32,
    pub aspect: f32,
    /// distance from the camera to the middle of the scene (the default focus distance)
    pub centre_dist: f32,
}

fn smooth(a: f32, b: f32, x: f32) -> f32 {
    let t = ((x - a) / (b - a)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

fn aces(x: f32) -> f32 {
    ((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14)).clamp(0.0, 1.0)
}

fn hable(x: f32) -> f32 {
    const A: f32 = 0.15;
    const B: f32 = 0.50;
    const C: f32 = 0.10;
    const D: f32 = 0.20;
    const E: f32 = 0.02;
    const F: f32 = 0.30;
    ((x * (A * x + C * B) + D * E) / (x * (A * x + B) + D * F)) - E / F
}

/// Scene-referred linear colour to display-referred linear colour (0..1), before the display gamma.
pub fn tonemap(c: V, t: ToneMap) -> V {
    let c = [c[0].max(0.0), c[1].max(0.0), c[2].max(0.0)];
    match t {
        ToneMap::Linear => [c[0].min(1.0), c[1].min(1.0), c[2].min(1.0)],
        ToneMap::Reinhard => [c[0] / (1.0 + c[0]), c[1] / (1.0 + c[1]), c[2] / (1.0 + c[2])],
        ToneMap::Aces => [aces(c[0]), aces(c[1]), aces(c[2])],
        ToneMap::Hable => {
            let w = 1.0 / hable(11.2);
            [(hable(2.0 * c[0]) * w).clamp(0.0, 1.0), (hable(2.0 * c[1]) * w).clamp(0.0, 1.0), (hable(2.0 * c[2]) * w).clamp(0.0, 1.0)]
        }
        ToneMap::Filmic => {
            // tone map the luminance so saturated colours keep their hue, then let only the brightest highlights
            // roll off per channel
            let l = 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2];
            let k = aces(l) / l.max(1e-4);
            let hue = [(c[0] * k).min(1.0), (c[1] * k).min(1.0), (c[2] * k).min(1.0)];
            let a = [aces(c[0]), aces(c[1]), aces(c[2])];
            let m = smooth(2.0, 8.0, l) * 0.7;
            [hue[0] + (a[0] - hue[0]) * m, hue[1] + (a[1] - hue[1]) * m, hue[2] + (a[2] - hue[2]) * m]
        }
    }
}

/// Display encoding used by both renderers (a plain 2.2 power, like the classic view).
pub fn encode(c: V) -> V {
    [c[0].clamp(0.0, 1.0).powf(1.0 / 2.2), c[1].clamp(0.0, 1.0).powf(1.0 / 2.2), c[2].clamp(0.0, 1.0).powf(1.0 / 2.2)]
}

/// Interleaved gradient noise in 0..1 (cheap per-pixel dither used by the ambient occlusion and the depth of field).
pub fn ign(x: f32, y: f32) -> f32 {
    let t = 52.982_92 * ((x * 0.067_110_56 + y * 0.005_837_15).fract());
    t.fract()
}

fn view_pos(ctx: &PostCtx, w: usize, h: usize, x: f32, y: f32, z: f32) -> V {
    let nx = (x + 0.5) / w as f32 * 2.0 - 1.0;
    let ny = 1.0 - (y + 0.5) / h as f32 * 2.0;
    [nx * ctx.aspect * ctx.tan_half * z, ny * ctx.tan_half * z, z]
}

fn depth_at(depth: &[f32], w: usize, h: usize, x: i64, y: i64) -> f32 {
    depth[(y.clamp(0, h as i64 - 1) as usize) * w + x.clamp(0, w as i64 - 1) as usize]
}

/// Darkening factor 0..1 per pixel from the depth buffer (1 = open).
fn ssao_factor(depth: &[f32], w: usize, h: usize, ctx: &PostCtx, radius: f32, x: usize, y: usize) -> f32 {
    let z = depth[y * w + x];
    if !z.is_finite() {
        return 1.0;
    }
    let p = view_pos(ctx, w, h, x as f32, y as f32, z);
    // normal from the smaller of the two depth steps in each direction
    let dz = |dx: i64, dy: i64| depth_at(depth, w, h, x as i64 + dx, y as i64 + dy);
    let pick = |a: f32, b: f32| if (a - z).abs() < (b - z).abs() { (a, 1.0) } else { (b, -1.0) };
    let (zx, sx) = pick(dz(1, 0), dz(-1, 0));
    let (zy, sy) = pick(dz(0, 1), dz(0, -1));
    let px = view_pos(ctx, w, h, x as f32 + sx, y as f32, if zx.is_finite() { zx } else { z });
    let py = view_pos(ctx, w, h, x as f32, y as f32 + sy, if zy.is_finite() { zy } else { z });
    let mut n = norm(cross(sub(px, p), sub(py, p)));
    // pixel y runs down the screen, view y up: orient the normal towards the camera
    if dot(n, p) > 0.0 {
        n = mul(n, -1.0);
    }
    let (t, b) = crate::hdri::basis(n);
    let rot = ign(x as f32, y as f32) * std::f32::consts::TAU;
    const N: usize = 14;
    let mut occ = 0.0;
    for i in 0..N {
        // spiral over the hemisphere, samples closer to the surface are more important
        let f = (i as f32 + 0.5) / N as f32;
        let a = f * 7.0 * std::f32::consts::PI + rot;
        let hz = (1.0 - f * 0.95).max(0.05);
        let hr = (1.0 - hz * hz).max(0.0).sqrt();
        let (s, c) = a.sin_cos();
        let v = [c * hr, s * hr, hz];
        let scale = radius * (0.2 + 0.8 * f * f);
        let sp = add(p, mul(add(add(mul(t, v[0]), mul(b, v[1])), mul(n, v[2])), scale));
        if sp[2] <= 0.05 {
            continue;
        }
        let sx_ = ((sp[0] / (sp[2] * ctx.aspect * ctx.tan_half)) * 0.5 + 0.5) * w as f32 - 0.5;
        let sy_ = (0.5 - sp[1] / (sp[2] * ctx.tan_half) * 0.5) * h as f32 - 0.5;
        let sz = depth_at(depth, w, h, sx_.round() as i64, sy_.round() as i64);
        if sz.is_finite() && sz < sp[2] - 0.01 * radius {
            let range = smooth(0.0, 1.0, radius / (p[2] - sz).abs().max(1e-4));
            occ += range;
        }
    }
    1.0 - occ / N as f32
}

/// Bilinear lookup that clamps at the border (no wrap-around glow).
fn at_clamped(t: &Tex, u: f32, v: f32) -> [f32; 3] {
    let x = (u * t.w as f32 - 0.5).clamp(0.0, t.w as f32 - 1.0);
    let y = (v * t.h as f32 - 0.5).clamp(0.0, t.h as f32 - 1.0);
    let (x0, y0) = (x.floor() as usize, y.floor() as usize);
    let (x1, y1) = ((x0 + 1).min(t.w - 1), (y0 + 1).min(t.h - 1));
    let (fx, fy) = (x - x0 as f32, y - y0 as f32);
    let p = |x: usize, y: usize| t.px[y * t.w + x];
    let mut o = [0.0; 3];
    for k in 0..3 {
        let a = p(x0, y0)[k] + (p(x1, y0)[k] - p(x0, y0)[k]) * fx;
        let b = p(x0, y1)[k] + (p(x1, y1)[k] - p(x0, y1)[k]) * fx;
        o[k] = a + (b - a) * fy;
    }
    o
}

fn box_half(t: &Tex) -> Tex {
    let (w, h) = ((t.w / 2).max(1), (t.h / 2).max(1));
    let px = par_map(w * h, |i| {
        let (x, y) = (i % w, i / w);
        let mut a = [0.0f32; 4];
        for (dx, dy) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
            let p = t.px[((y * 2 + dy).min(t.h - 1)) * t.w + (x * 2 + dx).min(t.w - 1)];
            for k in 0..3 {
                a[k] += p[k] * 0.25;
            }
        }
        a[3] = 1.0;
        a
    });
    Tex { w, h, px }
}

/// Glow of the bright parts: a mix of the box-filtered levels 2..5 of the thresholded image.
fn bloom_image(hdr: &Tex, p: &Post) -> Tex {
    let thr = p.bloom_threshold;
    let bright = Tex { w: hdr.w, h: hdr.h, px: hdr.px.iter().map(|c| {
        let l = lum(*c);
        let k = smooth(thr, thr + 0.5 + thr * 0.5, l);
        [c[0] * k, c[1] * k, c[2] * k, 1.0]
    }).collect() };
    let mut levels = vec![bright];
    for _ in 0..6 {
        if levels.last().unwrap().w <= 2 || levels.last().unwrap().h <= 2 {
            break;
        }
        let n = box_half(levels.last().unwrap());
        levels.push(n);
    }
    // the radius shifts the weights towards the coarser levels
    let r = p.bloom_radius.clamp(0.0, 1.0);
    let wts: Vec<f32> = (0..levels.len()).map(|k| {
        let centre = 1.5 + r * 3.0;
        (-((k as f32 - centre) / 1.1).powi(2)).exp()
    }).collect();
    let wsum: f32 = wts.iter().sum::<f32>().max(1e-6);
    let (w, h) = (hdr.w, hdr.h);
    let px = par_map(w * h, |i| {
        let (u, v) = (((i % w) as f32 + 0.5) / w as f32, ((i / w) as f32 + 0.5) / h as f32);
        let mut a = [0.0f32; 4];
        for (k, lv) in levels.iter().enumerate() {
            let c = at_clamped(lv, u, v);
            for q in 0..3 {
                a[q] += c[q] * wts[k] / wsum;
            }
        }
        a
    });
    Tex { w, h, px }
}

/// Apply the whole chain. `hdr` holds exposed linear radiance, `depth` the view-space distance along the camera axis
/// per pixel (infinity for the background). Returns display-encoded RGB.
pub fn apply(hdr: &Tex, depth: &[f32], ctx: &PostCtx, p: &Post) -> Tex {
    let (w, h) = (hdr.w, hdr.h);
    // 1. ambient occlusion
    let mut cur: Vec<[f32; 4]> = if p.ssao > 0.0 && depth.len() == w * h {
        par_map(w * h, |i| {
            let f = ssao_factor(depth, w, h, ctx, p.ssao_radius, i % w, i / w);
            let k = 1.0 - p.ssao * (1.0 - f);
            let c = hdr.px[i];
            [c[0] * k, c[1] * k, c[2] * k, 1.0]
        })
    } else {
        hdr.px.clone()
    };
    // 2. depth of field
    if p.dof > 0.0 && depth.len() == w * h {
        let focus = if p.focus > 0.0 { p.focus } else { ctx.centre_dist };
        let max_r = 0.014 * h as f32;
        let coc = |z: f32| -> f32 {
            if !z.is_finite() {
                max_r
            } else {
                (p.dof * (z - focus).abs() / z.max(0.05) * 0.045 * h as f32).min(max_r)
            }
        };
        let src = Tex { w, h, px: cur.clone() };
        cur = par_map(w * h, |i| {
            let (x, y) = ((i % w) as f32, (i / w) as f32);
            let r = coc(depth[i]);
            if r < 0.5 {
                return src.px[i];
            }
            let rot = ign(x, y) * std::f32::consts::TAU;
            let mut acc = [0.0f32; 3];
            let mut ws = 0.0;
            const N: usize = 28;
            for k in 0..N {
                let f = (k as f32 + 0.5) / N as f32;
                let a = k as f32 * 2.399_963 + rot;
                let rr = f.sqrt() * r;
                let (sx, sy) = ((x + a.cos() * rr).round().clamp(0.0, w as f32 - 1.0) as usize, (y + a.sin() * rr).round().clamp(0.0, h as f32 - 1.0) as usize);
                let j = sy * w + sx;
                // a sample only blurs into this pixel when its own circle of confusion reaches it
                let wgt = if coc(depth[j]) + 0.5 >= rr { 1.0 } else { 0.0 };
                let c = src.px[j];
                for q in 0..3 {
                    acc[q] += c[q] * wgt;
                }
                ws += wgt;
            }
            if ws < 1.0 {
                return src.px[i];
            }
            [acc[0] / ws, acc[1] / ws, acc[2] / ws, 1.0]
        });
    }
    // 3. bloom
    let hdr2 = Tex { w, h, px: cur };
    let bloom = if p.bloom > 0.0 { Some(bloom_image(&hdr2, p)) } else { None };
    // 4..7 per pixel: bloom add, white balance, contrast, saturation, vignette, tone map, encode
    let wb = [1.0 + 0.25 * p.temperature - 0.1 * p.tint, 1.0 + 0.2 * p.tint, 1.0 - 0.25 * p.temperature - 0.1 * p.tint];
    let px = par_map(w * h, |i| {
        let mut c = [hdr2.px[i][0], hdr2.px[i][1], hdr2.px[i][2]];
        if let Some(b) = &bloom {
            for q in 0..3 {
                c[q] += b.px[i][q] * p.bloom;
            }
        }
        for q in 0..3 {
            c[q] *= wb[q];
        }
        if (p.contrast - 1.0).abs() > 1e-4 {
            for q in 0..3 {
                c[q] = 0.18 * (c[q].max(0.0) / 0.18).powf(p.contrast);
            }
        }
        if (p.saturation - 1.0).abs() > 1e-4 {
            let l = 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2];
            for q in 0..3 {
                c[q] = l + (c[q] - l) * p.saturation;
            }
        }
        if p.vignette > 0.0 {
            let nx = ((i % w) as f32 + 0.5) / w as f32 * 2.0 - 1.0;
            let ny = ((i / w) as f32 + 0.5) / h as f32 * 2.0 - 1.0;
            let r = (nx * nx * ctx.aspect.min(1.5).max(0.5) + ny * ny).sqrt() / 1.5;
            let k = 1.0 - p.vignette * smooth(0.35, 1.0, r);
            for q in 0..3 {
                c[q] *= k;
            }
        }
        let t = encode(tonemap(c, p.tonemap));
        [t[0], t[1], t[2], 1.0]
    });
    Tex { w, h, px }
}
