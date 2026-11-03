// SPDX-License-Identifier: MIT OR Apache-2.0
//! Float RGBA texture with wrapped (tileable) sampling.

use std::sync::atomic::{AtomicU64, Ordering};

/// Id of the newest editor evaluation. A running evaluation whose own id differs is stale: the parallel loops below
/// stop at the next row, so a superseded job gives the CPU back almost at once instead of finishing a heavy node.
pub static CURRENT_JOB: AtomicU64 = AtomicU64::new(0);

thread_local! {
    static MY_JOB: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

/// Mark the calling thread as running evaluation `id` (0 = not an editor job, never stale).
pub fn set_job(id: u64) {
    MY_JOB.with(|c| c.set(id));
}

pub fn my_job() -> u64 {
    MY_JOB.with(|c| c.get())
}

pub fn stale(my: u64) -> bool {
    my != 0 && CURRENT_JOB.load(Ordering::Relaxed) != my
}

pub fn job_stale() -> bool {
    stale(my_job())
}

#[derive(Clone, Debug)]
pub struct Tex {
    pub w: usize,
    pub h: usize,
    pub px: Vec<[f32; 4]>,
}

/// floor() without a libm call: the baseline x86-64 target has no SSE4.1 rounding instruction, so `f32::floor` is a
/// function call there. Exact for |x| < 2^31 (callers use texture coordinates).
#[inline(always)]
pub fn ffloor(x: f32) -> f32 {
    let i = x as i32 as f32;
    if i > x {
        i - 1.0
    } else {
        i
    }
}

pub fn lum(p: [f32; 4]) -> f32 {
    0.2126 * p[0] + 0.7152 * p[1] + 0.0722 * p[2]
}

fn san(v: f32) -> f32 {
    if v.is_finite() {
        v
    } else {
        0.0
    }
}

impl Tex {
    pub fn solid(w: usize, h: usize, c: [f32; 4]) -> Tex {
        Tex { w, h, px: vec![c; w * h] }
    }

    /// Build a texture from a per-pixel function of (u, v) in [0,1). Runs multi-threaded.
    pub fn from_fn<F: Fn(f32, f32) -> [f32; 4] + Sync>(w: usize, h: usize, f: F) -> Tex {
        let w = w.max(1);
        let h = h.max(1);
        let mut px = vec![[0.0f32; 4]; w * h];
        let rows_per = crate::pool::rows_per_piece(w, h);
        let my = my_job();
        // whole rows per piece: the pixel position comes from two counters, not a division per pixel
        crate::pool::for_chunks(&mut px, rows_per * w, &|ci, chunk| {
            for (ry, row) in chunk.chunks_mut(w).enumerate() {
                if stale(my) {
                    return;
                }
                let v = ((ci * rows_per + ry) as f32 + 0.5) / h as f32;
                for (x, p) in row.iter_mut().enumerate() {
                    let o = f((x as f32 + 0.5) / w as f32, v);
                    *p = [san(o[0]), san(o[1]), san(o[2]), san(o[3])];
                }
            }
        });
        Tex { w, h, px }
    }

    pub fn gray_fn<F: Fn(f32, f32) -> f32 + Sync>(w: usize, h: usize, f: F) -> Tex {
        Tex::from_fn(w, h, |u, v| {
            let g = f(u, v);
            [g, g, g, 1.0]
        })
    }

    pub fn at(&self, x: i64, y: i64) -> [f32; 4] {
        let xi = x.rem_euclid(self.w as i64) as usize;
        let yi = y.rem_euclid(self.h as i64) as usize;
        self.px[yi * self.w + xi]
    }

    /// Bilinear, wrapped sampling in uv space.
    #[inline]
    pub fn sample(&self, u: f32, v: f32) -> [f32; 4] {
        let u = if u.is_finite() { u } else { 0.0 };
        let v = if v.is_finite() { v } else { 0.0 };
        let x = u * self.w as f32 - 0.5;
        let y = v * self.h as f32 - 0.5;
        let x0 = ffloor(x);
        let y0 = ffloor(y);
        let fx = x - x0;
        let fy = y - y0;
        let (w, h) = (self.w as i64, self.h as i64);
        // wrap the two columns and the two rows once (a branch instead of a division in the common case)
        let wrap = |i: i64, n: i64| if i >= 0 && i < n { i } else { i.rem_euclid(n) };
        let (xa, ya) = (wrap(x0 as i64, w), wrap(y0 as i64, h));
        let xb = if xa + 1 == w { 0 } else { xa + 1 };
        let yb = if ya + 1 == h { 0 } else { ya + 1 };
        let (r0, r1) = ((ya * w) as usize, (yb * w) as usize);
        let a = self.px[r0 + xa as usize];
        let b = self.px[r0 + xb as usize];
        let c = self.px[r1 + xa as usize];
        let d = self.px[r1 + xb as usize];
        let mut o = [0.0; 4];
        for i in 0..4 {
            let top = a[i] + (b[i] - a[i]) * fx;
            let bot = c[i] + (d[i] - c[i]) * fx;
            o[i] = top + (bot - top) * fy;
        }
        o
    }

    /// Catmull-Rom (bicubic) wrapped sampling in uv space: sharper than bilinear for rotations, scaling and warps. The
    /// result stays inside the range of the four nearest texels, so it never overshoots.
    pub fn sample_cubic(&self, u: f32, v: f32) -> [f32; 4] {
        let u = if u.is_finite() { u } else { 0.0 };
        let v = if v.is_finite() { v } else { 0.0 };
        let x = u * self.w as f32 - 0.5;
        let y = v * self.h as f32 - 0.5;
        let (x0, y0) = (ffloor(x), ffloor(y));
        let (fx, fy) = (x - x0, y - y0);
        let (xi, yi) = (x0 as i64, y0 as i64);
        let wts = |t: f32| {
            [
                0.5 * (-t * t * t + 2.0 * t * t - t),
                0.5 * (3.0 * t * t * t - 5.0 * t * t + 2.0),
                0.5 * (-3.0 * t * t * t + 4.0 * t * t + t),
                0.5 * (t * t * t - t * t),
            ]
        };
        let (wx, wy) = (wts(fx), wts(fy));
        let mut o = [0.0f32; 4];
        let inside = xi >= 1 && yi >= 1 && xi + 2 < self.w as i64 && yi + 2 < self.h as i64;
        for j in 0..4 {
            for i in 0..4 {
                let p = if inside {
                    self.px[(yi as usize + j - 1) * self.w + xi as usize + i - 1]
                } else {
                    self.at(xi + i as i64 - 1, yi + j as i64 - 1)
                };
                let w = wx[i] * wy[j];
                for c in 0..4 {
                    o[c] += p[c] * w;
                }
            }
        }
        let n = [self.at(xi, yi), self.at(xi + 1, yi), self.at(xi, yi + 1), self.at(xi + 1, yi + 1)];
        for c in 0..4 {
            let lo = n.iter().map(|p| p[c]).fold(f32::MAX, f32::min);
            let hi = n.iter().map(|p| p[c]).fold(f32::MIN, f32::max);
            o[c] = o[c].clamp(lo, hi);
        }
        o
    }

    pub fn map<F: Fn([f32; 4]) -> [f32; 4] + Sync>(&self, f: F) -> Tex {
        let mut out = self.clone();
        par_rows(&mut out.px, self.w.max(1), |_, chunk| {
            for p in chunk.iter_mut() {
                *p = f(*p);
            }
        });
        out
    }

    pub fn gray_at(&self, x: i64, y: i64) -> f32 {
        lum(self.at(x, y))
    }
}

/// One channel of a texture, for kernels that only read gray values. Sampling it gives exactly channel `c` of
/// [`Tex::sample`] (the same operations in the same order) while touching a quarter of the memory.
pub struct Plane {
    pub w: usize,
    pub h: usize,
    pub v: Vec<f32>,
}

impl Plane {
    pub fn of(t: &Tex, c: usize) -> Plane {
        let c = c.min(3);
        Plane { w: t.w.max(1), h: t.h.max(1), v: t.px.iter().map(|p| p[c]).collect() }
    }

    #[inline]
    pub fn sample(&self, u: f32, v: f32) -> f32 {
        let u = if u.is_finite() { u } else { 0.0 };
        let v = if v.is_finite() { v } else { 0.0 };
        let x = u * self.w as f32 - 0.5;
        let y = v * self.h as f32 - 0.5;
        let x0 = ffloor(x);
        let y0 = ffloor(y);
        let fx = x - x0;
        let fy = y - y0;
        let (w, h) = (self.w as i64, self.h as i64);
        let wrap = |i: i64, n: i64| if i >= 0 && i < n { i } else { i.rem_euclid(n) };
        let (xa, ya) = (wrap(x0 as i64, w), wrap(y0 as i64, h));
        let xb = if xa + 1 == w { 0 } else { xa + 1 };
        let yb = if ya + 1 == h { 0 } else { ya + 1 };
        let (r0, r1) = ((ya * w) as usize, (yb * w) as usize);
        let (a, b, c, d) = (self.v[r0 + xa as usize], self.v[r0 + xb as usize], self.v[r1 + xa as usize], self.v[r1 + xb as usize]);
        let top = a + (b - a) * fx;
        let bot = c + (d - c) * fx;
        top + (bot - top) * fy
    }
}

/// Run `f(first_index, chunk)` on consecutive pieces of `px` in parallel. Pieces are whole rows of width `w`, and a
/// superseded editor job stops between rows of work, so a stale evaluation gives the CPU back quickly.
pub fn par_rows<F: Fn(usize, &mut [[f32; 4]]) + Sync>(px: &mut [[f32; 4]], w: usize, f: F) {
    let w = w.max(1);
    let rows = (px.len() / w).max(1);
    // pieces of at most 16 rows, so the stale check runs regularly
    let per = crate::pool::rows_per_piece(w, rows).min(16) * w;
    let my = my_job();
    crate::pool::for_chunks(px, per, &|ci, piece| {
        if !stale(my) {
            f(ci * per, piece);
        }
    });
}

/// Parallel map over 0..n, preserving order.
pub fn par_map<T: Send + Default + Clone, F: Fn(usize) -> T + Sync>(n: usize, f: F) -> Vec<T> {
    let mut out = vec![T::default(); n];
    if n == 0 {
        return out;
    }
    // several pieces per core so uneven items balance out
    let per = n.div_ceil(crate::pool::threads() * 6).max(1);
    let my = my_job();
    crate::pool::for_chunks(&mut out, per, &|ci, chunk| {
        for (i, slot) in chunk.iter_mut().enumerate() {
            if i % 4096 == 0 && stale(my) {
                return;
            }
            *slot = f(ci * per + i);
        }
    });
    out
}
