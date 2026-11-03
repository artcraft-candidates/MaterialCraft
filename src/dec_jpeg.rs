// SPDX-License-Identifier: MIT OR Apache-2.0
//! JPEG reader: baseline, extended-sequential and progressive Huffman files, 8 bit, 1 / 3 / 4 components,
//! any sampling factors, restart intervals, JFIF and Adobe markers (YCbCr, RGB, CMYK, YCCK).
//! Not supported: arithmetic coding, lossless JPEG, 12 bit samples. The EXIF orientation tag is ignored.
use crate::tex::Tex;

const ZIGZAG: [usize; 64] = [
    0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27, 20, 13, 6, 7, 14, 21, 28, 35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51, 58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63,
];

#[derive(Clone)]
struct Huff {
    fast: Vec<u16>, // 9 bit lookup: (length << 8) | symbol, 0 = not found
    maxcode: [i32; 18],
    valptr: [i32; 17],
    mincode: [i32; 17],
    vals: Vec<u8>,
}

impl Huff {
    fn new(counts: &[u8; 16], vals: &[u8]) -> Result<Huff, String> {
        let mut h = Huff { fast: vec![0; 512], maxcode: [-1; 18], valptr: [0; 17], mincode: [0; 17], vals: vals.to_vec() };
        let mut code = 0i32;
        let mut k = 0i32;
        for len in 1..=16usize {
            h.valptr[len] = k;
            h.mincode[len] = code;
            let n = counts[len - 1] as i32;
            if n > 0 {
                if code + n > (1 << len) {
                    return Err("bad Huffman table".into());
                }
                if len <= 9 {
                    for i in 0..n {
                        let c = (code + i) as usize;
                        let sym = *vals.get((k + i) as usize).ok_or("bad Huffman table")?;
                        let base = c << (9 - len);
                        for j in 0..(1usize << (9 - len)) {
                            h.fast[base + j] = ((len as u16) << 8) | sym as u16;
                        }
                    }
                }
                code += n;
                k += n;
                h.maxcode[len] = code - 1;
            } else {
                h.maxcode[len] = -1;
            }
            code <<= 1;
        }
        if k as usize > vals.len() {
            return Err("bad Huffman table".into());
        }
        h.maxcode[17] = i32::MAX;
        Ok(h)
    }
}

struct Bits<'a> {
    d: &'a [u8],
    pos: usize,
    buf: u64,
    n: u32,
    marker: Option<u8>,
}

impl<'a> Bits<'a> {
    fn new(d: &'a [u8], pos: usize) -> Self {
        Bits { d, pos, buf: 0, n: 0, marker: None }
    }

    fn fill(&mut self) {
        while self.n <= 48 {
            let byte = if self.marker.is_some() || self.pos >= self.d.len() {
                0
            } else {
                let b = self.d[self.pos];
                if b == 0xff {
                    let nx = self.d.get(self.pos + 1).copied().unwrap_or(0xd9);
                    if nx == 0 {
                        self.pos += 2;
                        0xff
                    } else if nx == 0xff {
                        // fill bytes
                        self.pos += 1;
                        continue;
                    } else {
                        self.marker = Some(nx);
                        0
                    }
                } else {
                    self.pos += 1;
                    b
                }
            };
            self.buf |= (byte as u64) << (56 - self.n);
            self.n += 8;
        }
    }

    fn peek(&mut self, k: u32) -> u32 {
        if self.n < k {
            self.fill();
        }
        (self.buf >> (64 - k)) as u32
    }

    fn skip(&mut self, k: u32) {
        self.buf <<= k;
        self.n -= k;
    }

    fn get(&mut self, k: u32) -> i32 {
        if k == 0 {
            return 0;
        }
        let v = self.peek(k);
        self.skip(k);
        v as i32
    }

    fn bit(&mut self) -> bool {
        self.get(1) != 0
    }

    fn decode(&mut self, h: &Huff) -> Result<u8, String> {
        let p = self.peek(16);
        let e = h.fast[(p >> 7) as usize];
        if e != 0 {
            self.skip((e >> 8) as u32);
            return Ok((e & 255) as u8);
        }
        for len in 10..=16usize {
            let code = (p >> (16 - len)) as i32;
            if h.maxcode[len] >= 0 && code <= h.maxcode[len] && code >= h.mincode[len] {
                let idx = h.valptr[len] + code - h.mincode[len];
                self.skip(len as u32);
                return h.vals.get(idx as usize).copied().ok_or_else(|| "bad Huffman code".to_string());
            }
        }
        Err("bad Huffman code in JPEG data".into())
    }

    /// Discard buffered bits and step over the next RSTn marker.
    fn restart(&mut self) {
        self.buf = 0;
        self.n = 0;
        // a marker may already have been seen while filling
        if self.marker.is_none() {
            while self.pos + 1 < self.d.len() {
                if self.d[self.pos] == 0xff && self.d[self.pos + 1] != 0 && self.d[self.pos + 1] != 0xff {
                    self.marker = Some(self.d[self.pos + 1]);
                    break;
                }
                self.pos += 1;
            }
        }
        if let Some(m) = self.marker {
            if (0xd0..=0xd7).contains(&m) {
                self.pos += 2;
                self.marker = None;
            }
        }
    }
}

#[inline]
fn extend(v: i32, t: u32) -> i32 {
    if t == 0 {
        0
    } else if v < (1 << (t - 1)) {
        v - (1 << t) + 1
    } else {
        v
    }
}

#[derive(Clone)]
struct Comp {
    id: u8,
    h: usize,
    v: usize,
    tq: usize,
    bw: usize, // blocks per row, padded to whole MCUs
    bh: usize,
    coef: Vec<i16>,
    dc_tab: usize,
    ac_tab: usize,
    pred: i32,
}

pub fn decode(d: &[u8]) -> Result<Tex, String> {
    if d.len() < 4 || d[0] != 0xff || d[1] != 0xd8 {
        return Err("not a JPEG file".into());
    }
    let mut pos = 2usize;
    let mut qt = [[0u16; 64]; 4];
    let mut dc: Vec<Option<Huff>> = vec![None; 4];
    let mut ac: Vec<Option<Huff>> = vec![None; 4];
    let mut comps: Vec<Comp> = Vec::new();
    let (mut w, mut h) = (0usize, 0usize);
    let (mut hmax, mut vmax) = (1usize, 1usize);
    let mut progressive = false;
    let mut restart = 0usize;
    let mut adobe: Option<u8> = None;
    let mut jfif = false;
    let mut got_scan = false;
    let (mut mcux, mut mcuy) = (0usize, 0usize);
    loop {
        // find the next marker
        while pos < d.len() && d[pos] != 0xff {
            pos += 1;
        }
        while pos < d.len() && d[pos] == 0xff {
            pos += 1;
        }
        if pos >= d.len() {
            break;
        }
        let m = d[pos];
        pos += 1;
        if m == 0xd9 {
            break;
        }
        if m == 0x01 || (0xd0..=0xd8).contains(&m) || m == 0 {
            continue;
        }
        if pos + 2 > d.len() {
            return Err("truncated JPEG".into());
        }
        let len = u16::from_be_bytes([d[pos], d[pos + 1]]) as usize;
        if len < 2 || pos + len > d.len() {
            return Err("truncated JPEG segment".into());
        }
        let seg = &d[pos + 2..pos + len];
        let next = pos + len;
        match m {
            0xe0 => {
                if seg.starts_with(b"JFIF") {
                    jfif = true;
                }
            }
            0xee => {
                if seg.len() >= 12 && seg.starts_with(b"Adobe") {
                    adobe = Some(seg[11]);
                }
            }
            0xdb => {
                let mut p = 0;
                while p < seg.len() {
                    let pq = seg[p] >> 4;
                    let tq = (seg[p] & 15) as usize;
                    p += 1;
                    if tq > 3 {
                        return Err("bad quantisation table".into());
                    }
                    for i in 0..64 {
                        let v = if pq == 0 {
                            *seg.get(p + i).ok_or("truncated quantisation table")? as u16
                        } else {
                            if p + i * 2 + 1 >= seg.len() {
                                return Err("truncated quantisation table".into());
                            }
                            u16::from_be_bytes([seg[p + i * 2], seg[p + i * 2 + 1]])
                        };
                        qt[tq][ZIGZAG[i]] = v;
                    }
                    p += if pq == 0 { 64 } else { 128 };
                }
            }
            0xc4 => {
                let mut p = 0;
                while p + 17 <= seg.len() {
                    let tc = seg[p] >> 4;
                    let th = (seg[p] & 15) as usize;
                    let mut counts = [0u8; 16];
                    counts.copy_from_slice(&seg[p + 1..p + 17]);
                    let n: usize = counts.iter().map(|c| *c as usize).sum();
                    if th > 3 || p + 17 + n > seg.len() {
                        return Err("bad Huffman table".into());
                    }
                    let t = Huff::new(&counts, &seg[p + 17..p + 17 + n])?;
                    if tc == 0 {
                        dc[th] = Some(t);
                    } else {
                        ac[th] = Some(t);
                    }
                    p += 17 + n;
                }
            }
            0xc0 | 0xc1 | 0xc2 => {
                if !comps.is_empty() {
                    return Err("JPEG has several frames".into());
                }
                progressive = m == 0xc2;
                if seg.len() < 6 {
                    return Err("bad JPEG frame header".into());
                }
                if seg[0] != 8 {
                    return Err(format!("unsupported JPEG precision ({} bit)", seg[0]));
                }
                h = u16::from_be_bytes([seg[1], seg[2]]) as usize;
                w = u16::from_be_bytes([seg[3], seg[4]]) as usize;
                let nc = seg[5] as usize;
                if w == 0 || h == 0 {
                    return Err("JPEG size is zero (DNL not supported)".into());
                }
                if !matches!(nc, 1 | 3 | 4) || seg.len() < 6 + nc * 3 {
                    return Err(format!("unsupported JPEG component count {nc}"));
                }
                for i in 0..nc {
                    let o = 6 + i * 3;
                    let (hs, vs) = ((seg[o + 1] >> 4) as usize, (seg[o + 1] & 15) as usize);
                    if !(1..=4).contains(&hs) || !(1..=4).contains(&vs) || seg[o + 2] > 3 {
                        return Err("bad JPEG sampling factors".into());
                    }
                    comps.push(Comp { id: seg[o], h: hs, v: vs, tq: seg[o + 2] as usize, bw: 0, bh: 0, coef: Vec::new(), dc_tab: 0, ac_tab: 0, pred: 0 });
                }
                hmax = comps.iter().map(|c| c.h).max().unwrap();
                vmax = comps.iter().map(|c| c.v).max().unwrap();
                mcux = (w + 8 * hmax - 1) / (8 * hmax);
                mcuy = (h + 8 * vmax - 1) / (8 * vmax);
                if w.checked_mul(h).map_or(true, |n| n > 100_000_000) {
                    return Err("JPEG image too large".into());
                }
                for c in comps.iter_mut() {
                    c.bw = mcux * c.h;
                    c.bh = mcuy * c.v;
                    c.coef = vec![0i16; c.bw * c.bh * 64];
                }
            }
            0xc3 | 0xc5..=0xc7 | 0xc9..=0xcb | 0xcd..=0xcf => {
                return Err("unsupported JPEG type (lossless, hierarchical or arithmetic coding)".into());
            }
            0xdd => {
                if seg.len() >= 2 {
                    restart = u16::from_be_bytes([seg[0], seg[1]]) as usize;
                }
            }
            0xda => {
                if comps.is_empty() {
                    return Err("JPEG scan before frame header".into());
                }
                if seg.is_empty() {
                    return Err("bad JPEG scan header".into());
                }
                let ns = seg[0] as usize;
                if ns == 0 || ns > 4 || seg.len() < 1 + ns * 2 + 3 {
                    return Err("bad JPEG scan header".into());
                }
                let mut scan: Vec<usize> = Vec::new();
                for i in 0..ns {
                    let cid = seg[1 + i * 2];
                    let tabs = seg[2 + i * 2];
                    let ci = comps.iter().position(|c| c.id == cid).ok_or("JPEG scan names an unknown component")?;
                    comps[ci].dc_tab = (tabs >> 4) as usize & 3;
                    comps[ci].ac_tab = (tabs & 15) as usize & 3;
                    scan.push(ci);
                }
                let o = 1 + ns * 2;
                let (ss, se, ahl) = (seg[o] as usize, seg[o + 1] as usize, seg[o + 2]);
                let (ah, al) = ((ahl >> 4) as u32, (ahl & 15) as u32);
                if !progressive && (ss != 0 || se != 63) && !(ss == 0 && se == 63) {
                    // sequential scans always cover the whole block; tolerate odd headers
                }
                let end = decode_scan(d, next, &mut comps, &scan, &dc, &ac, progressive, ss, se.min(63), ah, al, restart, w, h, hmax, vmax, mcux, mcuy)?;
                got_scan = true;
                pos = end;
                continue;
            }
            _ => {}
        }
        pos = next;
    }
    if comps.is_empty() || !got_scan {
        return Err("JPEG has no image data".into());
    }
    // dequantise + inverse DCT into planes
    let mut planes: Vec<Vec<f32>> = Vec::new();
    for c in &comps {
        let pw = c.bw * 8;
        let ph = c.bh * 8;
        let mut plane = vec![0.0f32; pw * ph];
        let q = &qt[c.tq];
        let mut blk = [0f32; 64];
        for by in 0..c.bh {
            for bx in 0..c.bw {
                let co = &c.coef[(by * c.bw + bx) * 64..(by * c.bw + bx + 1) * 64];
                for i in 0..64 {
                    blk[i] = co[i] as f32 * q[i] as f32;
                }
                idct(&mut blk);
                for y in 0..8 {
                    let row = (by * 8 + y) * pw + bx * 8;
                    for x in 0..8 {
                        plane[row + x] = (blk[y * 8 + x] + 128.0).clamp(0.0, 255.0);
                    }
                }
            }
        }
        planes.push(plane);
    }
    // upsample each plane to full size
    let mut full: Vec<Vec<f32>> = Vec::new();
    for (ci, c) in comps.iter().enumerate() {
        let pw = c.bw * 8;
        let cw = (w * c.h + hmax - 1) / hmax;
        let ch = (h * c.v + vmax - 1) / vmax;
        full.push(upsample(&planes[ci], pw, cw, ch, hmax / c.h.max(1), vmax / c.v.max(1), hmax % c.h == 0 && vmax % c.v == 0, w, h, c.h, c.v, hmax, vmax));
    }
    let n = comps.len();
    let mut px = vec![[0.0f32, 0.0, 0.0, 1.0]; w * h];
    let ids: Vec<u8> = comps.iter().map(|c| c.id).collect();
    let is_rgb = n == 3 && (adobe == Some(0) || (!jfif && adobe.is_none() && ids == [b'R', b'G', b'B']));
    let inv = 1.0 / 255.0;
    match n {
        1 => {
            for i in 0..w * h {
                let v = full[0][i] * inv;
                px[i] = [v, v, v, 1.0];
            }
        }
        3 => {
            for i in 0..w * h {
                let (a, b, c) = (full[0][i], full[1][i], full[2][i]);
                let (r, g, bl) = if is_rgb { (a, b, c) } else { ycc(a, b, c) };
                px[i] = [r * inv, g * inv, bl * inv, 1.0];
            }
        }
        _ => {
            let ycck = adobe == Some(2);
            for i in 0..w * h {
                let (mut c, mut m, mut y, k) = (full[0][i], full[1][i], full[2][i], full[3][i]);
                if ycck {
                    let (r, g, b) = ycc(c, m, y);
                    c = 255.0 - r;
                    m = 255.0 - g;
                    y = 255.0 - b;
                }
                // Adobe files store inverted CMYK, so the value multiplies directly with K
                let (r, g, b) = if adobe.is_some() { (c * k / 255.0, m * k / 255.0, y * k / 255.0) } else { ((255.0 - c) * (255.0 - k) / 255.0, (255.0 - m) * (255.0 - k) / 255.0, (255.0 - y) * (255.0 - k) / 255.0) };
                px[i] = [r * inv, g * inv, b * inv, 1.0];
            }
        }
    }
    Ok(Tex { w, h, px })
}

#[inline]
fn ycc(y: f32, cb: f32, cr: f32) -> (f32, f32, f32) {
    let (cb, cr) = (cb - 128.0, cr - 128.0);
    ((y + 1.402 * cr).clamp(0.0, 255.0), (y - 0.344136 * cb - 0.714136 * cr).clamp(0.0, 255.0), (y + 1.772 * cb).clamp(0.0, 255.0))
}

fn idct_table() -> &'static [[f32; 8]; 8] {
    use std::sync::OnceLock;
    static T: OnceLock<[[f32; 8]; 8]> = OnceLock::new();
    T.get_or_init(|| {
        let mut t = [[0.0f32; 8]; 8];
        for u in 0..8 {
            let cu = if u == 0 { (0.5f64).sqrt() } else { 1.0 };
            for x in 0..8 {
                t[u][x] = (0.5 * cu * (((2 * x + 1) as f64) * u as f64 * std::f64::consts::PI / 16.0).cos()) as f32;
            }
        }
        t
    })
}

/// In-place 8x8 inverse DCT (coefficients in natural order).
fn idct(b: &mut [f32; 64]) {
    if b[1..].iter().all(|v| *v == 0.0) {
        let v = b[0] / 8.0;
        b.iter_mut().for_each(|x| *x = v);
        return;
    }
    let t = idct_table();
    let mut tmp = [0f32; 64];
    for y in 0..8 {
        // rows: tmp[y][x] = sum_u t[u][x] * b[y][u]
        for x in 0..8 {
            let mut s = 0.0;
            for u in 0..8 {
                s += t[u][x] * b[y * 8 + u];
            }
            tmp[y * 8 + x] = s;
        }
    }
    for x in 0..8 {
        for y in 0..8 {
            let mut s = 0.0;
            for v in 0..8 {
                s += t[v][y] * tmp[v * 8 + x];
            }
            b[y * 8 + x] = s;
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn upsample(plane: &[f32], pw: usize, cw: usize, ch: usize, fx: usize, fy: usize, exact: bool, w: usize, h: usize, hs: usize, vs: usize, hmax: usize, vmax: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; w * h];
    if hs == hmax && vs == vmax {
        for y in 0..h {
            out[y * w..(y + 1) * w].copy_from_slice(&plane[y * pw..y * pw + w]);
        }
        return out;
    }
    let at = |x: isize, y: isize| -> f32 { plane[(y.clamp(0, ch as isize - 1) as usize) * pw + x.clamp(0, cw as isize - 1) as usize] };
    for y in 0..h {
        for x in 0..w {
            out[y * w + x] = if exact && fx <= 2 && fy <= 2 {
                // triangle filter: samples sit at the centre of each fx * fy cell
                let (sx, sy) = (x / fx, y / fy);
                let (nx, ny): (isize, isize) = (if fx == 2 { if x % 2 == 0 { -1 } else { 1 } } else { 0 }, if fy == 2 { if y % 2 == 0 { -1 } else { 1 } } else { 0 });
                let (sx, sy) = (sx as isize, sy as isize);
                let a = at(sx, sy);
                let bx = if fx == 2 { at(sx + nx, sy) } else { a };
                let by = if fy == 2 { at(sx, sy + ny) } else { a };
                let bxy = if fx == 2 && fy == 2 { at(sx + nx, sy + ny) } else if fx == 2 { bx } else { by };
                (9.0 * a + 3.0 * bx + 3.0 * by + bxy) / 16.0
            } else {
                let sx = (x * hs / hmax).min(cw - 1);
                let sy = (y * vs / vmax).min(ch - 1);
                plane[sy * pw + sx]
            };
        }
    }
    out
}

#[allow(clippy::too_many_arguments)]
fn decode_scan(
    d: &[u8],
    start: usize,
    comps: &mut [Comp],
    scan: &[usize],
    dc: &[Option<Huff>],
    ac: &[Option<Huff>],
    progressive: bool,
    ss: usize,
    se: usize,
    ah: u32,
    al: u32,
    restart: usize,
    w: usize,
    h: usize,
    hmax: usize,
    vmax: usize,
    mcux: usize,
    mcuy: usize,
) -> Result<usize, String> {
    let mut br = Bits::new(d, start);
    for c in comps.iter_mut() {
        c.pred = 0;
    }
    let mut eobrun = 0u32;
    let single = scan.len() == 1;
    // block grid walked by this scan
    let (units_x, units_y) = if single {
        let c = &comps[scan[0]];
        let cw = (w * c.h + hmax - 1) / hmax;
        let ch = (h * c.v + vmax - 1) / vmax;
        ((cw + 7) / 8, (ch + 7) / 8)
    } else {
        (mcux, mcuy)
    };
    let total = units_x * units_y;
    let mut count = 0usize;
    for unit in 0..total {
        if restart > 0 && count == restart {
            br.restart();
            for c in comps.iter_mut() {
                c.pred = 0;
            }
            eobrun = 0;
            count = 0;
        }
        count += 1;
        let (ux, uy) = (unit % units_x, unit / units_x);
        for &ci in scan {
            let (hs, vs) = if single { (1, 1) } else { (comps[ci].h, comps[ci].v) };
            for by in 0..vs {
                for bx in 0..hs {
                    let (bxx, byy) = (ux * hs + bx, uy * vs + by);
                    let c = &mut comps[ci];
                    if bxx >= c.bw || byy >= c.bh {
                        continue;
                    }
                    let off = (byy * c.bw + bxx) * 64;
                    let (dct, act) = (dc[c.dc_tab].as_ref(), ac[c.ac_tab].as_ref());
                    if !progressive {
                        let dct = dct.ok_or("JPEG scan uses a missing DC table")?;
                        let act = act.ok_or("JPEG scan uses a missing AC table")?;
                        let blk = &mut c.coef[off..off + 64];
                        let t = br.decode(dct)? as u32;
                        if t > 15 {
                            return Err("bad JPEG DC code".into());
                        }
                        let diff = extend(br.get(t), t);
                        c.pred += diff;
                        blk[0] = c.pred as i16;
                        let mut k = 1;
                        while k < 64 {
                            let rs = br.decode(act)?;
                            let (r, s) = ((rs >> 4) as usize, (rs & 15) as u32);
                            if s == 0 {
                                if r == 15 {
                                    k += 16;
                                    continue;
                                }
                                break;
                            }
                            k += r;
                            if k > 63 {
                                return Err("bad JPEG AC run".into());
                            }
                            blk[ZIGZAG[k]] = extend(br.get(s), s) as i16;
                            k += 1;
                        }
                    } else if ss == 0 {
                        let blk = &mut c.coef[off..off + 64];
                        if ah == 0 {
                            let dct = dct.ok_or("JPEG scan uses a missing DC table")?;
                            let t = br.decode(dct)? as u32;
                            if t > 15 {
                                return Err("bad JPEG DC code".into());
                            }
                            let diff = extend(br.get(t), t);
                            c.pred += diff;
                            blk[0] = (c.pred << al) as i16;
                        } else if br.bit() {
                            blk[0] |= (1 << al) as i16;
                        }
                    } else {
                        let act = act.ok_or("JPEG scan uses a missing AC table")?;
                        let blk = &mut c.coef[off..off + 64];
                        if ah == 0 {
                            if eobrun > 0 {
                                eobrun -= 1;
                                continue;
                            }
                            let mut k = ss;
                            while k <= se {
                                let rs = br.decode(act)?;
                                let (r, s) = ((rs >> 4) as u32, (rs & 15) as u32);
                                if s == 0 {
                                    if r < 15 {
                                        eobrun = (1u32 << r) - 1;
                                        if r > 0 {
                                            eobrun += br.get(r) as u32;
                                        }
                                        break;
                                    }
                                    k += 16;
                                    continue;
                                }
                                k += r as usize;
                                if k > 63 {
                                    return Err("bad JPEG AC run".into());
                                }
                                blk[ZIGZAG[k]] = (extend(br.get(s), s) << al) as i16;
                                k += 1;
                            }
                        } else {
                            let p1 = 1i16 << al;
                            let m1 = -(1i16 << al);
                            let mut k = ss;
                            if eobrun == 0 {
                                while k <= se {
                                    let rs = br.decode(act)?;
                                    let (mut r, s) = ((rs >> 4) as i32, (rs & 15) as u32);
                                    let mut val = 0i16;
                                    if s != 0 {
                                        val = if br.bit() { p1 } else { m1 };
                                    } else if r != 15 {
                                        eobrun = 1u32 << r;
                                        if r > 0 {
                                            eobrun += br.get(r as u32) as u32;
                                        }
                                        break;
                                    }
                                    while k <= se {
                                        let cf = &mut blk[ZIGZAG[k]];
                                        if *cf != 0 {
                                            if br.bit() && (*cf & p1) == 0 {
                                                *cf += if *cf >= 0 { p1 } else { m1 };
                                            }
                                        } else {
                                            r -= 1;
                                            if r < 0 {
                                                break;
                                            }
                                        }
                                        k += 1;
                                    }
                                    if val != 0 && k <= se {
                                        blk[ZIGZAG[k]] = val;
                                    }
                                    k += 1;
                                }
                            }
                            if eobrun > 0 {
                                while k <= se {
                                    let cf = &mut blk[ZIGZAG[k]];
                                    if *cf != 0 && br.bit() && (*cf & p1) == 0 {
                                        *cf += if *cf >= 0 { p1 } else { m1 };
                                    }
                                    k += 1;
                                }
                                eobrun -= 1;
                            }
                        }
                    }
                }
            }
        }
    }
    // position after the entropy coded data: the next marker that is not a restart
    let mut p = br.pos;
    if let Some(_m) = br.marker {
        // `pos` points at the 0xff of the marker already
    }
    while p + 1 < d.len() {
        if d[p] == 0xff && d[p + 1] != 0 && d[p + 1] != 0xff && !(0xd0..=0xd7).contains(&d[p + 1]) {
            break;
        }
        p += 1;
    }
    Ok(p)
}
