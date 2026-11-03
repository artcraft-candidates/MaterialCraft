// SPDX-License-Identifier: MIT OR Apache-2.0
//! Path nodes: stroke, fill, distance, border, shapes along a path, railway track, cable and seam stitches. They all
//! read a spline (an embedded `.spl` resource named by `file`, or a built-in `preset`), measure every pixel against
//! it with a wrap-around distance grid, and shade from the distance, the position along the path and the side.
//! Every node tiles seamlessly (the path wraps with the texture) and works on non-square graphs. Pixel sizes are
//! given for a 512 px texture and scaled by `size / 512`. `compute` returns Ok(None) for kinds it does not know.
use crate::graph::{Error, Node};
use crate::noise;
use crate::ops;
use crate::spline::{Polyline, Spline};
use crate::tex::{lum, par_rows, Tex};
use std::collections::BTreeMap;
use std::f32::consts::TAU;
use std::sync::Arc as Rc;

fn grey(v: f32) -> [f32; 4] {
    [v, v, v, 1.0]
}

fn smooth(a: f32, b: f32, x: f32) -> f32 {
    let d = b - a;
    let t = if d.abs() < 1e-9 { if x >= b { 1.0 } else { 0.0 } } else { ((x - a) / d).clamp(0.0, 1.0) };
    t * t * (3.0 - 2.0 * t)
}

/// Make `period` divide `len` evenly, so a pattern along a path that wraps around the texture meets itself.
fn fit_period(len: f32, period: f32, on: bool) -> f32 {
    let period = period.max(0.5);
    if !on || len < period {
        return period;
    }
    len / (len / period).round().max(1.0)
}

// ====================================================================== distance field of a set of paths
struct Seg {
    a: [f32; 2],
    d: [f32; 2],
    len: f32,
    s0: f32,
    wa: f32,
    wb: f32,
    za: f32,
    zb: f32,
    path: u32,
}

struct PInfo {
    len: f32,
    closed: bool,
}

struct Field {
    cs: f32,
    gw: usize,
    gh: usize,
    w: f32,
    h: f32,
    bins: Vec<Vec<(u32, i8, i8)>>,
    segs: Vec<Seg>,
    paths: Vec<PInfo>,
    reach: f32,
}

#[derive(Clone, Copy)]
struct Hit {
    /// distance to the path in pixels
    d: f32,
    /// distance along its path in pixels
    s: f32,
    /// +1 on the left of the direction of travel, -1 on the right
    side: f32,
    /// width factor at the nearest point
    wf: f32,
    plen: f32,
    closed: bool,
    /// height value at the nearest point
    z: f32,
    /// unit vector from the nearest point to the pixel (zero on the path itself)
    away: [f32; 2],
    /// unit direction of the path at the nearest point
    dir: [f32; 2],
}

impl Field {
    fn build(polys: &[Polyline], w: usize, h: usize, reach: f32) -> Field {
        let (wf, hf) = (w as f32, h as f32);
        let reach = reach.min(wf.min(hf) * 0.5).max(1.0);
        let mut segs: Vec<Seg> = Vec::new();
        let mut paths: Vec<PInfo> = Vec::new();
        for (pi, poly) in polys.iter().enumerate() {
            let n = poly.pts.len();
            let closed = poly.closed && n > 2;
            let count = if closed { n } else { n.saturating_sub(1) };
            let mut s0 = 0.0;
            for i in 0..count {
                let (a, b) = (poly.pts[i], poly.pts[(i + 1) % n]);
                let d = [b[0] - a[0], b[1] - a[1]];
                let len = (d[0] * d[0] + d[1] * d[1]).sqrt();
                if len < 1e-6 || !len.is_finite() {
                    continue;
                }
                segs.push(Seg { a, d, len, s0, wa: poly.w[i], wb: poly.w[(i + 1) % n], za: poly.z.get(i).copied().unwrap_or(1.0), zb: poly.z.get((i + 1) % n).copied().unwrap_or(1.0), path: pi as u32 });
                s0 += len;
            }
            paths.push(PInfo { len: s0.max(1e-3), closed });
        }
        let cs = reach.clamp(8.0, 64.0);
        let (gw, gh) = (((wf / cs).ceil() as usize).max(1), ((hf / cs).ceil() as usize).max(1));
        let mut bins: Vec<Vec<(u32, i8, i8)>> = vec![Vec::new(); gw * gh];
        for (si, sg) in segs.iter().enumerate() {
            for sy in -1i8..=1 {
                for sx in -1i8..=1 {
                    let (ox, oy) = (sx as f32 * wf, sy as f32 * hf);
                    let (ax, ay) = (sg.a[0] + ox, sg.a[1] + oy);
                    let (bx, by) = (ax + sg.d[0], ay + sg.d[1]);
                    let (x0, x1) = (ax.min(bx) - reach, ax.max(bx) + reach);
                    let (y0, y1) = (ay.min(by) - reach, ay.max(by) + reach);
                    if x1 < 0.0 || y1 < 0.0 || x0 >= wf || y0 >= hf {
                        continue;
                    }
                    let (cx0, cx1) = (((x0.max(0.0)) / cs) as usize, (((x1.min(wf - 1e-3)) / cs) as usize).min(gw - 1));
                    let (cy0, cy1) = (((y0.max(0.0)) / cs) as usize, (((y1.min(hf - 1e-3)) / cs) as usize).min(gh - 1));
                    for cy in cy0..=cy1 {
                        for cx in cx0..=cx1 {
                            bins[cy * gw + cx].push((si as u32, sx, sy));
                        }
                    }
                }
            }
        }
        Field { cs, gw, gh, w: wf, h: hf, bins, segs, paths, reach }
    }

    fn query(&self, x: f32, y: f32) -> Option<Hit> {
        let cx = ((x / self.cs) as usize).min(self.gw - 1);
        let cy = ((y / self.cs) as usize).min(self.gh - 1);
        let mut best = self.reach * self.reach;
        let mut found: Option<(usize, f32, f32, f32, [f32; 2])> = None;
        for &(si, sx, sy) in &self.bins[cy * self.gw + cx] {
            let sg = &self.segs[si as usize];
            let (ax, ay) = (sg.a[0] + sx as f32 * self.w, sg.a[1] + sy as f32 * self.h);
            let (px, py) = (x - ax, y - ay);
            let t = ((px * sg.d[0] + py * sg.d[1]) / (sg.len * sg.len)).clamp(0.0, 1.0);
            let (qx, qy) = (px - sg.d[0] * t, py - sg.d[1] * t);
            let d2 = qx * qx + qy * qy;
            if d2 < best {
                best = d2;
                let cross = sg.d[0] * py - sg.d[1] * px;
                found = Some((si as usize, t, cross, d2, [qx, qy]));
            }
        }
        found.map(|(si, t, cross, d2, q)| {
            let sg = &self.segs[si];
            let p = &self.paths[sg.path as usize];
            Hit { d: d2.sqrt(), s: sg.s0 + t * sg.len, side: if cross >= 0.0 { 1.0 } else { -1.0 }, wf: sg.wa + (sg.wb - sg.wa) * t, plen: p.len, closed: p.closed, z: sg.za + (sg.zb - sg.za) * t, away: if d2 > 1e-10 { [q[0] / d2.sqrt(), q[1] / d2.sqrt()] } else { [0.0, 0.0] }, dir: [sg.d[0] / sg.len, sg.d[1] / sg.len] }
        })
    }
}

// ====================================================================== shared parameter handling
struct Ctx6<'a> {
    kind: &'a str,
    n: &'a Node,
    s: usize,
    hs: usize,
}

impl Ctx6<'_> {
    fn f(&self, k: &str, d: f32) -> f32 {
        self.n.nums.get(k).copied().filter(|v| v.is_finite()).unwrap_or(d)
    }

    /// Closed shapes by default for the nodes that need them, a wave for the others.
    fn default_preset(&self) -> f32 {
        if self.kind.starts_with("path_fill") || self.kind == "path_border" {
            3.0
        } else {
            2.0
        }
    }

    fn spline(&self, id: u32, load: &dyn Fn(&str) -> Result<Vec<u8>, String>) -> Result<Spline, Error> {
        let mut sp = match self.n.strs.get("file").filter(|f| !f.is_empty()) {
            Some(file) => {
                let bytes = load(file).map_err(|m| Error::BadParam(id, m))?;
                Spline::from_bytes(&bytes).map_err(|m| Error::BadParam(id, format!("'{file}': {m}")))?
            }
            None => Spline::preset(self.f("preset", self.default_preset()).round().clamp(0.0, 6.0) as u32 + 1, self.f("amplitude", 0.12), self.f("frequency", 2.0)),
        };
        // up to three more splines are merged in (their paths come after the first file's)
        for key in ["file2", "file3", "file4"] {
            if let Some(file) = self.n.strs.get(key).filter(|f| !f.is_empty()) {
                let bytes = load(file).map_err(|m| Error::BadParam(id, m))?;
                sp.merge(Spline::from_bytes(&bytes).map_err(|m| Error::BadParam(id, format!("'{file}': {m}")))?);
            }
        }
        let (scale, rot, ox, oy) = (self.f("path_scale", 1.0).clamp(0.05, 8.0), self.f("path_rotate", 0.0), self.f("path_x", 0.0), self.f("path_y", 0.0));
        let mirror = self.f("path_mirror", 0.0).round() as u32;
        let reverse = self.f("path_reverse", 0.0) > 0.5;
        if (scale - 1.0).abs() > 1e-6 || rot.abs() > 1e-6 || ox.abs() > 1e-6 || oy.abs() > 1e-6 || mirror != 0 || reverse {
            sp = sp.transformed([scale, scale], rot, [ox, oy], (mirror & 1 != 0, mirror & 2 != 0), reverse);
        }
        Ok(sp)
    }

    /// Flattened paths for this texture size; `path` picks one (-1 = all).
    fn polys(&self, id: u32, load: &dyn Fn(&str) -> Result<Vec<u8>, String>) -> Result<Vec<Polyline>, Error> {
        let sp = self.spline(id, load)?;
        let all = sp.flatten(self.s as f32, self.hs as f32);
        let pick = self.f("path", -1.0).round();
        let count = self.f("path_count", if pick >= 0.0 { 1.0 } else { -1.0 }).round();
        Ok(if pick >= 0.0 {
            let rest = all.into_iter().skip(pick as usize);
            if count > 0.0 { rest.take(count as usize).collect() } else { rest.collect() }
        } else {
            all
        })
    }
}

fn max_wf(polys: &[Polyline]) -> f32 {
    polys.iter().flat_map(|p| p.w.iter().copied()).fold(1.0f32, f32::max).clamp(0.0, 16.0)
}

fn shade(c: &Ctx6, field: &Field, f: impl Fn(Option<Hit>, usize, usize) -> f32 + Sync) -> Tex {
    let (w, h) = (c.s, c.hs);
    let mut px = vec![[0.0f32; 4]; w * h];
    par_rows(&mut px, w, |start, piece| {
        for (i, o) in piece.iter_mut().enumerate() {
            let (x, y) = ((start + i) % w, (start + i) / w);
            *o = grey(f(field.query(x as f32 + 0.5, y as f32 + 0.5), x, y));
        }
    });
    Tex { w, h, px }
}

/// Like [`shade`] for a node with several outputs: the path is looked up once per pixel and `f(hit, x, y, k)` gives
/// output `k` for every output whose bit is set in `want`. Returns the outputs in index order (None = not wanted).
fn shade_multi(c: &Ctx6, field: &Field, want: u32, nout: usize, f: impl Fn(Option<Hit>, usize, usize, usize) -> f32 + Sync) -> Vec<Option<Tex>> {
    let (w, h) = (c.s, c.hs);
    let ks: Vec<usize> = (0..nout).filter(|k| want & (1 << k) != 0).collect();
    let m = ks.len().max(1);
    let mut vals = vec![0.0f32; w * h * m];
    let rows_per = crate::pool::rows_per_piece(w, h);
    let my = crate::tex::my_job();
    crate::pool::for_chunks(&mut vals, rows_per * w * m, &|ci, piece| {
        for (ry, row) in piece.chunks_mut(w * m).enumerate() {
            if crate::tex::stale(my) {
                return;
            }
            let y = ci * rows_per + ry;
            for x in 0..w {
                let hit = field.query(x as f32 + 0.5, y as f32 + 0.5);
                for (j, &k) in ks.iter().enumerate() {
                    row[x * m + j] = f(hit, x, y, k);
                }
            }
        }
    });
    let mut out: Vec<Option<Tex>> = (0..nout).map(|_| None).collect();
    for (j, &k) in ks.iter().enumerate() {
        out[k] = Some(Tex { w, h, px: vals.iter().skip(j).step_by(m).map(|v| grey(*v)).collect() });
    }
    out
}

/// Output `out_idx` of a multi-output node; the other wanted outputs go to the graph, which hands them to their own
/// requests in this pass instead of running the node again.
fn take_outputs(outs: Vec<Option<Tex>>, out_idx: usize, fin: &dyn Fn(Tex) -> Tex) -> Tex {
    let mut mine = None;
    for (k, t) in outs.into_iter().enumerate() {
        let Some(t) = t else { continue };
        if k == out_idx {
            mine = Some(fin(t));
        } else {
            crate::graph::emit_sibling(k, fin(t));
        }
    }
    mine.unwrap_or_else(|| Tex::solid(1, 1, [0.0, 0.0, 0.0, 1.0]))
}

/// Inside mask (coverage) of the closed paths, drawn with wrapped copies so it tiles.
fn inside_mask(polys: &[Polyline], w: usize, h: usize, even_odd: bool) -> Vec<f32> {
    let mut vp: Vec<crate::vector::Poly> = Vec::new();
    for sy in -1i32..=1 {
        for sx in -1i32..=1 {
            for p in polys {
                if p.pts.len() < 3 {
                    continue;
                }
                let pts = p.pts.iter().map(|q| [q[0] + sx as f32 * w as f32, q[1] + sy as f32 * h as f32]).collect();
                vp.push(crate::vector::Poly { pts, closed: true });
            }
        }
    }
    crate::vector::fill(&vp, w, h, even_odd).a
}

fn aa(x: f32) -> f32 {
    (x + 0.5).clamp(0.0, 1.0)
}

// ====================================================================== nodes
pub fn compute(
    kind: &str,
    id: u32,
    n: &Node,
    ins: &BTreeMap<String, Rc<Tex>>,
    s: usize,
    hs: usize,
    seed: u32,
    load: &dyn Fn(&str) -> Result<Vec<u8>, String>,
) -> Result<Option<Tex>, Error> {
    let rs = s as f32 / 512.0;
    let c = Ctx6 { kind, n, s, hs };
    let f = |k: &str, d: f32| c.f(k, d);
    let out_idx = f("_out", 0.0).round() as usize;
    // the outputs this evaluation reads: a multi-output node computes them all in one go (bit k = output k)
    let want = (f("_want", 0.0).max(0.0) as u32) | (1u32 << out_idx.min(31));
    let (con, inv) = (f("contrast", 1.0).clamp(0.0, 16.0), f("invert", 0.0) > 0.5);
    let fin = |t: Tex| -> Tex {
        if (con - 1.0).abs() > 1e-4 || inv {
            ops::contrast_invert(&t, con, inv)
        } else {
            t
        }
    };
    let fit = f("fit", 1.0) > 0.5;
    let out = match kind {
        "path_stroke" => {
            let polys = c.polys(id, load)?;
            let width = f("width", 12.0).clamp(0.1, 1024.0) * rs;
            let soft = f("softness", 0.0).clamp(0.0, 1.0);
            let profile = f("profile", 0.0).round() as u32;
            let (tin, tout) = (f("taper_start", 0.0).clamp(0.0, 1.0), f("taper_end", 0.0).clamp(0.0, 1.0));
            let (dash, gap) = (f("dash_length", 0.0).clamp(0.0, 4096.0) * rs, f("dash_gap", 8.0).clamp(0.0, 4096.0) * rs);
            let reach = width * 0.5 * max_wf(&polys) * (1.0 + soft) + 3.0;
            let field = Field::build(&polys, s, hs, reach);
            fin(shade(&c, &field, |hit, _, _| {
                let Some(h) = hit else { return 0.0 };
                let mut half = width * 0.5 * h.wf;
                if !h.closed {
                    let (a, b) = (if tin > 0.0 { (h.s / (tin * h.plen)).min(1.0) } else { 1.0 }, if tout > 0.0 { ((h.plen - h.s) / (tout * h.plen)).min(1.0) } else { 1.0 });
                    half *= smooth(0.0, 1.0, a.min(b));
                }
                if half < 0.05 {
                    return 0.0;
                }
                let edge = (soft * half).max(0.0);
                let mut cov = if edge > 1.0 { 1.0 - smooth(half - edge, half, h.d) } else { aa(half - h.d) };
                if dash > 0.5 {
                    let period = fit_period(h.plen, dash + gap, fit);
                    let dl = (period - gap).max(1.0);
                    let ds = h.s % period;
                    cov *= aa(ds.min(dl - ds) + 0.0);
                    if ds > dl {
                        cov = 0.0;
                    }
                }
                let q = (h.d / half).clamp(0.0, 1.0);
                match profile {
                    1 => cov * (1.0 - q * q).sqrt(),
                    2 => cov * ((1.0 - q) / 0.4).clamp(0.0, 1.0),
                    _ => cov,
                }
            }))
        }
        "path_fill" => {
            let polys = c.polys(id, load)?;
            let m = inside_mask(&polys, s, hs, f("even_odd", 0.0) > 0.5);
            let mut t = Tex { w: s, h: hs, px: m.into_iter().map(grey).collect() };
            let soft = f("softness", 0.0).clamp(0.0, 1.0);
            if soft > 0.0 {
                t = ops::blur(&t, soft * 24.0 * rs);
            }
            fin(t)
        }
        "path_distance" => {
            let polys = c.polys(id, load)?;
            let range = f("range", 64.0).clamp(1.0, 2048.0) * rs;
            let mode = f("mode", 0.0).round() as u32;
            let field = Field::build(&polys, s, hs, range);
            let inside = if polys.iter().any(|p| p.closed) && mode != 0 { Some(inside_mask(&polys, s, hs, false)) } else { None };
            fin(shade(&c, &field, |hit, x, y| {
                let ins_v = inside.as_ref().map(|m| m[y * s + x] > 0.5);
                match mode {
                    0 => hit.map_or(0.0, |h| (1.0 - h.d / range).clamp(0.0, 1.0)),
                    1 => match (hit, ins_v) {
                        (Some(h), Some(i)) => 0.5 + 0.5 * if i { 1.0 } else { -1.0 } * (h.d / range).min(1.0),
                        (Some(h), None) => 0.5 + 0.5 * h.side * (h.d / range).min(1.0),
                        (None, Some(i)) => {
                            if i {
                                1.0
                            } else {
                                0.0
                            }
                        }
                        (None, None) => 0.5,
                    },
                    2 => match (hit, ins_v) {
                        (_, Some(false)) | (_, None) => 0.0,
                        (Some(h), Some(true)) => (h.d / range).clamp(0.0, 1.0),
                        (None, Some(true)) => 1.0,
                    },
                    _ => match (hit, ins_v) {
                        (_, Some(true)) | (_, None) => 0.0,
                        (Some(h), Some(false)) => (h.d / range).clamp(0.0, 1.0),
                        (None, Some(false)) => 1.0,
                    },
                }
            }))
        }
        "path_border" => {
            let polys: Vec<Polyline> = c.polys(id, load)?.into_iter().filter(|p| p.closed).collect();
            let width = f("width", 32.0).clamp(0.5, 1024.0) * rs;
            let side = f("side", 0.0).round() as u32;
            let profile = f("profile", 0.0).round() as u32;
            let flat = f("flat", 0.0).clamp(0.0, 0.95);
            let field = Field::build(&polys, s, hs, width);
            let inside = inside_mask(&polys, s, hs, false);
            let outs = shade_multi(&c, &field, want, 2, |hit, x, y, k| {
                let mask_out = k == 1;
                let Some(h) = hit else { return 0.0 };
                let ins_v = inside[y * s + x] > 0.5;
                let ok = match side {
                    0 => ins_v,
                    1 => !ins_v,
                    _ => true,
                };
                if !ok {
                    return 0.0;
                }
                let q = (h.d / width).clamp(0.0, 1.0);
                if mask_out {
                    return aa(width - h.d);
                }
                let r = ((q - flat) / (1.0 - flat)).clamp(0.0, 1.0);
                // the path line is the top of the border and the far edge is level with the surface
                let hgt = match profile {
                    1 => (1.0 - r * r).sqrt(),
                    2 => 1.0 - smooth(0.0, 1.0, r),
                    _ => 1.0 - r,
                };
                hgt * aa(width - h.d)
            });
            take_outputs(outs, out_idx, &fin)
        }
        "path_scatter" | "path_splatter" => {
            let polys = c.polys(id, load)?;
            let pat = ins.get("in").map(|r| r.as_ref());
            let mask = ins.get("mask").map(|r| r.as_ref());
            let rows = if kind == "path_splatter" { f("rows", 3.0).round().clamp(1.0, 32.0) as usize } else { 1 };
            let row_gap = f("row_spacing", 24.0).clamp(0.0, 1024.0) * rs;
            let spacing = f("spacing", 24.0).clamp(1.0, 2048.0) * rs;
            let size = f("scale", 24.0).clamp(1.0, 2048.0) * rs;
            let (var, jal, jac) = (f("scale_variation", 0.2).clamp(0.0, 1.0), f("jitter_along", 0.0).clamp(0.0, 1.0), f("jitter_across", 0.0).clamp(0.0, 512.0) * rs);
            let rot = f("rotation", 0.0).round() as u32; // 0 follow the path, 1 fixed, 2 random
            let (ang0, ang_var) = (f("angle", 0.0) * TAU, f("angle_variation", 0.0).clamp(0.0, 1.0) * TAU);
            let (density, op_var) = (f("density", 1.0).clamp(0.0, 1.0), f("opacity_variation", 0.0).clamp(0.0, 1.0));
            let blend = f("blend", 0.0).round() as u32; // 0 max, 1 add
            let mut dest = vec![0.0f32; s * hs];
            let mut budget: i64 = 400_000_000;
            'all: for (pi, poly) in polys.iter().enumerate() {
                let step = fit_period(poly.length(), spacing, fit);
                for (i0, sm) in poly.resample(step, f("offset", 0.0) * rs).into_iter().enumerate() {
                  for row in 0..rows {
                    let i = i0 * rows + row;
                    let key = (pi as i32) * 100_003 + i as i32;
                    let r = |k: i32| noise::rnd(key, k, seed);
                    let row_off = (row as f32 - (rows as f32 - 1.0) * 0.5) * row_gap;
                    let here = mask.map(|m| lum(m.sample((sm.p[0] - sm.t[1] * row_off) / s as f32, (sm.p[1] + sm.t[0] * row_off) / hs as f32))).unwrap_or(1.0);
                    if r(1) > density * here {
                        continue;
                    }
                    let sz = (size * sm.w * (1.0 + var * (r(2) - 0.5) * 2.0)).max(0.5);
                    let nrm = [-sm.t[1], sm.t[0]];
                    let (ja, jc) = (jal * step * (r(3) - 0.5), jac * (r(4) - 0.5) * 2.0 + row_off);
                    let (cx, cy) = (sm.p[0] + sm.t[0] * ja + nrm[0] * jc, sm.p[1] + sm.t[1] * ja + nrm[1] * jc);
                    let ang = match rot {
                        0 => sm.t[1].atan2(sm.t[0]) + ang0 + ang_var * (r(5) - 0.5),
                        1 => ang0 + ang_var * (r(5) - 0.5),
                        _ => r(5) * TAU,
                    };
                    let amp = 1.0 - op_var * r(6);
                    let (sa, ca) = ang.sin_cos();
                    let rad = sz * 0.7072 + 1.0;
                    let (x0, x1, y0, y1) = ((cx - rad).floor() as i64, (cx + rad).ceil() as i64, (cy - rad).floor() as i64, (cy + rad).ceil() as i64);
                    budget -= (x1 - x0 + 1) * (y1 - y0 + 1);
                    if budget < 0 {
                        break 'all;
                    }
                    for py in y0..=y1 {
                        for px in x0..=x1 {
                            let (dx, dy) = (px as f32 + 0.5 - cx, py as f32 + 0.5 - cy);
                            let (lx, ly) = (dx * ca + dy * sa, -dx * sa + dy * ca);
                            let (u, v) = (lx / sz + 0.5, ly / sz + 0.5);
                            if u < 0.0 || u >= 1.0 || v < 0.0 || v >= 1.0 {
                                continue;
                            }
                            let val = match pat {
                                Some(t) => lum(t.sample(u, v)) * t.sample(u, v)[3],
                                None => 1.0 - smooth(0.44, 0.5, (lx * lx + ly * ly).sqrt() / sz),
                            } * amp;
                            if val <= 0.0 {
                                continue;
                            }
                            let idx = (py.rem_euclid(hs as i64) as usize) * s + px.rem_euclid(s as i64) as usize;
                            dest[idx] = if blend == 1 { (dest[idx] + val).min(1.0) } else { dest[idx].max(val) };
                        }
                    }
                  }
                }
            }
            fin(Tex { w: s, h: hs, px: dest.into_iter().map(grey).collect() })
        }
        "path_sample" => {
            // maps that carry what the path knows: position along it, its width, its height values and the side
            let polys = c.polys(id, load)?;
            let width = f("width", 40.0).clamp(0.5, 1024.0) * rs;
            let wmax = max_wf(&polys).max(1e-3);
            let field = Field::build(&polys, s, hs, width * 0.5 * wmax + 3.0);
            let outs = shade_multi(&c, &field, want, 4, |hit, _, _, k| {
                let Some(h) = hit else { return if k == 3 { 0.5 } else { 0.0 } };
                let half = (width * 0.5 * h.wf).max(0.05);
                let cov = aa(half - h.d);
                match k {
                    1 => cov * (h.wf / wmax).clamp(0.0, 1.0),
                    2 => cov * h.z.clamp(0.0, 1.0),
                    3 => 0.5 + 0.5 * cov * h.side * (h.d / half).min(1.0),
                    _ => cov * (h.s / h.plen).clamp(0.0, 1.0),
                }
            });
            take_outputs(outs, out_idx, &fin)
        }
        "path_warp" => {
            // pushes, pulls or slides the picture near the path (strongest on the line, fading out over `width`)
            let src = ins.get("in").map(|r| r.as_ref()).ok_or_else(|| Error::MissingInput(id, "in".to_string()))?;
            let polys = c.polys(id, load)?;
            let width = f("width", 80.0).clamp(1.0, 1024.0) * rs;
            let amount = f("amount", 24.0).clamp(-1024.0, 1024.0) * rs;
            let mode = f("mode", 0.0).round() as u32; // 0 push away, 1 slide along, 2 swirl around the line
            let field = Field::build(&polys, s, hs, width);
            let (sf, hf) = (s as f32, hs as f32);
            Tex::from_fn(s, hs, |u, v| {
                let (x, y) = (u * sf, v * hf);
                match field.query(x, y) {
                    Some(h) if h.d < width => {
                        let fall = 1.0 - smooth(0.0, 1.0, h.d / width);
                        let d = match mode {
                            1 => h.dir,
                            2 => [-h.away[1] * h.side, h.away[0] * h.side],
                            _ => h.away,
                        };
                        src.sample((x - d[0] * amount * fall) / sf, (y - d[1] * amount * fall) / hf)
                    }
                    _ => src.sample(u, v),
                }
            })
        }
        "path_track" => {
            let polys = c.polys(id, load)?;
            let gauge = f("gauge", 60.0).clamp(2.0, 1024.0) * rs;
            let rw = f("rail_width", 8.0).clamp(0.5, 256.0) * rs;
            let (spacing, sl, sw) = (f("sleeper_spacing", 28.0).clamp(2.0, 1024.0) * rs, f("sleeper_length", 110.0).clamp(2.0, 2048.0) * rs, f("sleeper_width", 14.0).clamp(1.0, 512.0) * rs);
            let (rh, sh) = (f("rail_height", 0.9).clamp(0.0, 1.0), f("sleeper_height", 0.55).clamp(0.0, 1.0));
            let field = Field::build(&polys, s, hs, (sl * 0.5).max(gauge * 0.5 + rw) + 3.0);
            let outs = shade_multi(&c, &field, want, 3, |hit, _, _, k| {
                let Some(h) = hit else { return 0.0 };
                let (rail, sleeper) = track_parts(&h, gauge, rw, spacing, sl, sw, fit);
                match k {
                    1 => rail,
                    2 => sleeper,
                    _ => (rh * rail.max(0.0) * track_dome(&h, gauge, rw)).max(sh * sleeper),
                }
            });
            take_outputs(outs, out_idx, &fin)
        }
        "path_cable" => {
            let polys = c.polys(id, load)?;
            let width = f("width", 40.0).clamp(1.0, 1024.0) * rs;
            let strands = f("strands", 3.0).round().clamp(2.0, 8.0) as usize;
            let size = f("strand_size", 0.5).clamp(0.15, 0.95);
            let twist = f("twist", 3.0).clamp(0.0, 64.0);
            let field = Field::build(&polys, s, hs, width * 0.5 * max_wf(&polys) + 3.0);
            let outs = shade_multi(&c, &field, want, 3, |hit, _, _, k| {
                let Some(h) = hit else { return 0.0 };
                let half = width * 0.5 * h.wf;
                if half < 0.05 {
                    return 0.0;
                }
                let u = h.side * h.d / half;
                let turns = if h.closed || twist > 0.0 { (twist * h.plen / (512.0 * rs)).round().max(1.0) } else { 0.0 };
                let theta = TAU * turns * h.s / h.plen;
                let rr = 1.0 - size;
                let mut best = (0.0f32, 0usize);
                for k in 0..strands {
                    let ph = theta + TAU * k as f32 / strands as f32;
                    let q = (u - rr * ph.sin()) / size;
                    if q.abs() < 1.0 {
                        let v = 0.3 * (0.5 + 0.5 * ph.cos()) + 0.7 * (1.0 - q * q).sqrt();
                        if v > best.0 {
                            best = (v, k);
                        }
                    }
                }
                let edge = aa((1.0 - u.abs()) * half);
                match k {
                    1 => if best.0 > 0.0 { edge } else { 0.0 },
                    2 => if best.0 > 0.0 { (best.1 + 1) as f32 / strands as f32 * edge } else { 0.0 },
                    _ => best.0 * edge,
                }
            });
            take_outputs(outs, out_idx, &fin)
        }
        "path_seam" => {
            let polys = c.polys(id, load)?;
            let (len, gap) = (f("stitch_length", 18.0).clamp(1.0, 1024.0) * rs, f("stitch_gap", 10.0).clamp(0.0, 1024.0) * rs);
            let (sw, gw) = (f("stitch_width", 5.0).clamp(0.5, 256.0) * rs, f("groove_width", 12.0).clamp(0.0, 512.0) * rs);
            let (depth, sh) = (f("groove_depth", 0.5).clamp(0.0, 1.0), f("stitch_height", 0.7).clamp(0.0, 1.0));
            let field = Field::build(&polys, s, hs, (gw * 0.5).max(len * 0.5 + sw) + 3.0);
            let outs = shade_multi(&c, &field, want, 2, |hit, _, _, k| {
                let Some(h) = hit else { return if k == 1 { 0.0 } else { 0.5 } };
                let period = fit_period(h.plen, len + gap, fit);
                let l = (period - gap).min(period).max(0.5);
                let a = ((h.s % period) - period * 0.5).abs();
                let excess = (a - l * 0.5).max(0.0);
                let dd = (excess * excess + h.d * h.d).sqrt();
                let cov = aa(sw * 0.5 - dd);
                if k == 1 {
                    return cov;
                }
                let dome = (1.0 - (dd / (sw * 0.5)).min(1.0).powi(2)).sqrt() * cov;
                let groove = if gw > 0.0 { (1.0 - (h.d / (gw * 0.5)).min(1.0).powi(2)).max(0.0) } else { 0.0 };
                (0.5 + 0.5 * sh * dome - 0.5 * depth * groove).clamp(0.0, 1.0)
            });
            take_outputs(outs, out_idx, &fin)
        }
        _ => return Ok(None),
    };
    Ok(Some(out))
}

/// (rail coverage, sleeper coverage) of a track at a hit.
fn track_parts(h: &Hit, gauge: f32, rw: f32, spacing: f32, sl: f32, sw: f32, fit: bool) -> (f32, f32) {
    let rail = aa(rw * 0.5 - (h.d - gauge * 0.5).abs());
    let period = fit_period(h.plen, spacing, fit);
    let a = ((h.s % period) - period * 0.5).abs();
    let sleeper = aa(sw * 0.5 - a).min(aa(sl * 0.5 - h.d));
    (rail, sleeper)
}

fn track_dome(h: &Hit, gauge: f32, rw: f32) -> f32 {
    let x = ((h.d - gauge * 0.5).abs() / (rw * 0.5)).min(1.0);
    (1.0 - x * x).sqrt().max(0.35)
}
