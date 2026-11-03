// SPDX-License-Identifier: MIT OR Apache-2.0
//! WebP reader: lossless (VP8L), lossy (VP8 key frames, with the optional alpha plane) and the extended file
//! layout. For animated files the first frame is returned. Chroma is upsampled with the same 9-3-3-1 filter the
//! reference decoder uses, so the pictures match it value for value.
use crate::tex::Tex;
use crate::webp_tables::*;

pub fn decode(b: &[u8]) -> Result<Tex, String> {
    if b.len() < 20 || &b[..4] != b"RIFF" || &b[8..12] != b"WEBP" {
        return Err("not a WebP file".into());
    }
    let riff_end = (u32::from_le_bytes([b[4], b[5], b[6], b[7]]) as usize + 8).min(b.len());
    let mut pos = 12usize;
    let mut alph: Option<&[u8]> = None;
    let mut canvas: Option<(usize, usize)> = None;
    let mut animated = false;
    while pos + 8 <= riff_end {
        let tag = &b[pos..pos + 4];
        let len = u32::from_le_bytes([b[pos + 4], b[pos + 5], b[pos + 6], b[pos + 7]]) as usize;
        let start = pos + 8;
        let end = start.checked_add(len).filter(|e| *e <= b.len()).ok_or("truncated WebP chunk")?;
        let data = &b[start..end];
        match tag {
            b"VP8X" => {
                if data.len() < 10 {
                    return Err("bad VP8X chunk".into());
                }
                animated = data[0] & 2 != 0;
                let w = 1 + (data[4] as usize | (data[5] as usize) << 8 | (data[6] as usize) << 16);
                let h = 1 + (data[7] as usize | (data[8] as usize) << 8 | (data[9] as usize) << 16);
                crate::imgio::check_size(w, h)?;
                canvas = Some((w, h));
            }
            b"ALPH" => alph = Some(data),
            b"VP8L" => return lossless(data),
            b"VP8 " => return lossy(data, alph),
            b"ANMF" if animated => {
                // first frame only, placed on a transparent canvas
                if data.len() < 16 {
                    return Err("bad ANMF chunk".into());
                }
                let fx = 2 * (data[0] as usize | (data[1] as usize) << 8 | (data[2] as usize) << 16);
                let fy = 2 * (data[3] as usize | (data[4] as usize) << 8 | (data[5] as usize) << 16);
                let frame = decode_frame(&data[16..])?;
                let (cw, ch) = canvas.unwrap_or((frame.w + fx, frame.h + fy));
                let mut px = vec![[0.0f32; 4]; cw * ch];
                for y in 0..frame.h.min(ch.saturating_sub(fy)) {
                    for x in 0..frame.w.min(cw.saturating_sub(fx)) {
                        px[(fy + y) * cw + fx + x] = frame.px[y * frame.w + x];
                    }
                }
                return Ok(Tex { w: cw, h: ch, px });
            }
            _ => {}
        }
        pos = end + (len & 1);
    }
    Err("WebP file has no image data".into())
}

fn decode_frame(b: &[u8]) -> Result<Tex, String> {
    let mut pos = 0;
    let mut alph: Option<&[u8]> = None;
    while pos + 8 <= b.len() {
        let tag = &b[pos..pos + 4];
        let len = u32::from_le_bytes([b[pos + 4], b[pos + 5], b[pos + 6], b[pos + 7]]) as usize;
        let end = (pos + 8 + len).min(b.len());
        let data = &b[pos + 8..end];
        match tag {
            b"ALPH" => alph = Some(data),
            b"VP8L" => return lossless(data),
            b"VP8 " => return lossy(data, alph),
            _ => {}
        }
        pos = end + (len & 1);
    }
    Err("WebP frame has no image data".into())
}

// ================================================================================================ lossless (VP8L)
struct LBits<'a> {
    d: &'a [u8],
    pos: usize,
    buf: u64,
    n: u32,
    over: bool,
}

impl<'a> LBits<'a> {
    fn new(d: &'a [u8]) -> Self {
        LBits { d, pos: 0, buf: 0, n: 0, over: false }
    }
    fn fill(&mut self) {
        while self.n <= 56 {
            let byte = if self.pos < self.d.len() {
                self.d[self.pos]
            } else {
                if self.pos >= self.d.len() + 8 {
                    self.over = true;
                }
                0
            };
            self.pos += 1;
            self.buf |= (byte as u64) << self.n;
            self.n += 8;
        }
    }
    fn peek(&mut self, k: u32) -> u32 {
        if self.n < k {
            self.fill();
        }
        (self.buf & ((1u64 << k) - 1)) as u32
    }
    fn skip(&mut self, k: u32) {
        self.buf >>= k;
        self.n -= k;
    }
    fn read(&mut self, k: u32) -> u32 {
        if k == 0 {
            return 0;
        }
        let v = self.peek(k);
        self.skip(k);
        v
    }
    fn at_end(&self) -> bool {
        // consumed more bits than the data holds
        (self.pos as i64 * 8 - self.n as i64) > self.d.len() as i64 * 8
    }
}

struct Huff {
    single: Option<u32>,
    fast: Vec<u32>, // 8 bit lookup: (length << 16) | symbol; 0 = longer code
    first_code: [u32; 16],
    first_index: [u32; 16],
    count: [u32; 16],
    sorted: Vec<u16>,
}

impl Huff {
    fn build(lengths: &[u8]) -> Result<Huff, String> {
        let mut count = [0u32; 16];
        let mut nz = 0;
        let mut last = 0usize;
        for (i, l) in lengths.iter().enumerate() {
            if *l > 0 {
                count[*l as usize] += 1;
                nz += 1;
                last = i;
            }
        }
        if nz == 0 {
            return Err("empty prefix code".into());
        }
        if nz == 1 {
            return Ok(Huff { single: Some(last as u32), fast: Vec::new(), first_code: [0; 16], first_index: [0; 16], count, sorted: Vec::new() });
        }
        let mut first_code = [0u32; 16];
        let mut first_index = [0u32; 16];
        let mut code = 0u32;
        let mut idx = 0u32;
        for l in 1..16 {
            code = (code + count[l - 1]) << 1;
            if l == 1 {
                code = 0;
            }
            first_code[l] = code;
            first_index[l] = idx;
            idx += count[l];
            if count[l] > 0 && code + count[l] > (1 << l) {
                return Err("over-subscribed prefix code".into());
            }
        }
        // canonical codes are assigned in order of (length, symbol)
        let mut sorted = vec![0u16; nz];
        let mut next = first_index;
        for (s, l) in lengths.iter().enumerate() {
            if *l > 0 {
                sorted[next[*l as usize] as usize] = s as u16;
                next[*l as usize] += 1;
            }
        }
        let mut fast = vec![0u32; 256];
        for l in 1..=8usize {
            for k in 0..count[l] {
                let c = first_code[l] + k;
                let sym = sorted[(first_index[l] + k) as usize] as u32;
                // reverse the code bits: the stream delivers the first code bit first
                let mut rev = 0u32;
                for bit in 0..l {
                    if c & (1 << bit) != 0 {
                        rev |= 1 << (l - 1 - bit);
                    }
                }
                let mut j = rev as usize;
                while j < 256 {
                    fast[j] = ((l as u32) << 16) | sym;
                    j += 1 << l;
                }
            }
        }
        Ok(Huff { single: None, fast, first_code, first_index, count, sorted })
    }

    fn read(&self, br: &mut LBits) -> Result<u32, String> {
        if let Some(s) = self.single {
            return Ok(s);
        }
        let p = br.peek(15);
        let e = self.fast[(p & 255) as usize];
        if e != 0 {
            br.skip(e >> 16);
            return Ok(e & 0xffff);
        }
        let mut code = 0u32;
        for l in 1..16usize {
            code = (code << 1) | ((p >> (l - 1)) & 1);
            if self.count[l] > 0 && code >= self.first_code[l] && code - self.first_code[l] < self.count[l] {
                br.skip(l as u32);
                return Ok(self.sorted[(self.first_index[l] + code - self.first_code[l]) as usize] as u32);
            }
        }
        Err("bad prefix code in WebP data".into())
    }
}

const CODE_LENGTH_ORDER: [usize; 19] = [17, 18, 0, 1, 2, 3, 4, 5, 16, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];

fn read_prefix_code(br: &mut LBits, alphabet: usize) -> Result<Huff, String> {
    let mut lengths = vec![0u8; alphabet.max(256)];
    if br.read(1) == 1 {
        let n = br.read(1) + 1;
        let first8 = br.read(1) == 1;
        let s = br.read(if first8 { 8 } else { 1 }) as usize;
        if s >= alphabet {
            return Err("bad WebP prefix symbol".into());
        }
        lengths[s] = 1;
        if n == 2 {
            let s2 = br.read(8) as usize;
            if s2 >= alphabet {
                return Err("bad WebP prefix symbol".into());
            }
            lengths[s2] = 1;
        }
    } else {
        let num = br.read(4) as usize + 4;
        let mut cl = [0u8; 19];
        for i in 0..num {
            cl[CODE_LENGTH_ORDER[i]] = br.read(3) as u8;
        }
        let lh = Huff::build(&cl)?;
        let max_symbol = if br.read(1) == 1 {
            let nb = 2 + 2 * br.read(3);
            let m = 2 + br.read(nb) as usize;
            if m > alphabet {
                return Err("bad prefix code length limit".into());
            }
            m
        } else {
            alphabet
        };
        let mut sym = 0usize;
        let mut left = max_symbol;
        let mut prev = 8u8;
        while sym < alphabet {
            if left == 0 {
                break;
            }
            left -= 1;
            let c = lh.read(br)? as u8;
            if c < 16 {
                lengths[sym] = c;
                sym += 1;
                if c != 0 {
                    prev = c;
                }
            } else {
                let (extra, off) = match c {
                    16 => (2, 3),
                    17 => (3, 3),
                    _ => (7, 11),
                };
                let rep = br.read(extra) as usize + off;
                if sym + rep > alphabet {
                    return Err("bad prefix code repeat".into());
                }
                let v = if c == 16 { prev } else { 0 };
                for _ in 0..rep {
                    lengths[sym] = v;
                    sym += 1;
                }
            }
        }
    }
    if br.at_end() {
        return Err("truncated WebP data".into());
    }
    Huff::build(&lengths[..alphabet.max(256).min(lengths.len())])
}

struct Group {
    h: [Huff; 5],
}

fn subsample(size: usize, bits: u32) -> usize {
    (size + (1 << bits) - 1) >> bits
}

struct Transform {
    kind: u32,
    bits: u32,
    xsize: usize,
    ysize: usize,
    data: Vec<u32>,
}

fn decode_image_stream(br: &mut LBits, xsize: usize, ysize: usize, level0: bool, transforms: &mut Vec<Transform>) -> Result<(Vec<u32>, usize), String> {
    let mut xs = xsize;
    if level0 {
        let mut used = [false; 4];
        while br.read(1) == 1 {
            let kind = br.read(2);
            if used[kind as usize] {
                return Err("WebP transform used twice".into());
            }
            used[kind as usize] = true;
            let mut t = Transform { kind, bits: 0, xsize: xs, ysize, data: Vec::new() };
            match kind {
                0 | 1 => {
                    t.bits = br.read(3) + 2;
                    let (sx, sy) = (subsample(xs, t.bits), subsample(ysize, t.bits));
                    t.data = decode_image_stream(br, sx, sy, false, transforms)?.0;
                }
                3 => {
                    let ncol = br.read(8) as usize + 1;
                    t.bits = if ncol > 16 {
                        0
                    } else if ncol > 4 {
                        1
                    } else if ncol > 2 {
                        2
                    } else {
                        3
                    };
                    let mut table = decode_image_stream(br, ncol, 1, false, transforms)?.0;
                    // the colour table is delta coded
                    for i in 1..table.len() {
                        let (a, c) = (table[i - 1], table[i]);
                        let mut o = 0u32;
                        for sh in [0, 8, 16, 24] {
                            o |= ((((a >> sh) & 255) + ((c >> sh) & 255)) & 255) << sh;
                        }
                        table[i] = o;
                    }
                    table.resize(1usize << (8 >> t.bits), 0);
                    t.data = table;
                    xs = subsample(xs, t.bits);
                }
                _ => {}
            }
            transforms.push(t);
        }
    }
    // colour cache
    let cache_bits = if br.read(1) == 1 {
        let b = br.read(4);
        if !(1..=11).contains(&b) {
            return Err("bad WebP colour cache size".into());
        }
        b
    } else {
        0
    };
    // prefix codes
    let mut meta: Option<(Vec<u32>, u32, usize)> = None;
    let mut ngroups = 1usize;
    if level0 && br.read(1) == 1 {
        let bits = br.read(3) + 2;
        let (mx, my) = (subsample(xs, bits), subsample(ysize, bits));
        let mut dummy = Vec::new();
        let img = decode_image_stream(br, mx, my, false, &mut dummy)?.0;
        let idx: Vec<u32> = img.iter().map(|p| (p >> 8) & 0xffff).collect();
        ngroups = idx.iter().copied().max().unwrap_or(0) as usize + 1;
        meta = Some((idx, bits, mx));
    }
    let mut groups: Vec<Group> = Vec::with_capacity(ngroups);
    for _ in 0..ngroups {
        let g = read_prefix_code(br, 256 + 24 + if cache_bits > 0 { 1 << cache_bits } else { 0 })?;
        let r = read_prefix_code(br, 256)?;
        let bl = read_prefix_code(br, 256)?;
        let a = read_prefix_code(br, 256)?;
        let d = read_prefix_code(br, 40)?;
        groups.push(Group { h: [g, r, bl, a, d] });
    }
    // pixels
    let total = xs.checked_mul(ysize).ok_or("WebP image too large")?;
    let mut px = vec![0u32; total];
    let mut cache = vec![0u32; if cache_bits > 0 { 1 << cache_bits } else { 0 }];
    let (mut x, mut y) = (0usize, 0usize);
    let mut p = 0usize;
    let mut gi = 0usize;
    let insert = |cache: &mut Vec<u32>, v: u32| {
        if !cache.is_empty() {
            let k = (v.wrapping_mul(0x1e35a7bd) >> (32 - cache_bits)) as usize;
            cache[k] = v;
        }
    };
    let prefix_value = |br: &mut LBits, sym: u32| -> u32 {
        if sym < 4 {
            sym + 1
        } else {
            let extra = (sym - 2) >> 1;
            let off = (2 + (sym & 1)) << extra;
            off + br.read(extra) + 1
        }
    };
    while p < total {
        if let Some((idx, bits, mx)) = &meta {
            if x & ((1 << bits) - 1) == 0 || p == 0 {
                gi = idx[(y >> bits) * mx + (x >> bits)] as usize;
            }
        }
        let g = &groups[gi];
        let s = g.h[0].read(br)?;
        if s < 256 {
            let r = g.h[1].read(br)?;
            let bl = g.h[2].read(br)?;
            let a = g.h[3].read(br)?;
            let v = (a << 24) | (r << 16) | (s << 8) | bl;
            px[p] = v;
            insert(&mut cache, v);
            p += 1;
            x += 1;
            if x == xs {
                x = 0;
                y += 1;
            }
        } else if s < 280 {
            let len = prefix_value(br, s - 256) as usize;
            let dsym = g.h[4].read(br)?;
            let dcode = prefix_value(br, dsym) as usize;
            let dist = if dcode > 120 {
                dcode - 120
            } else {
                let c = CODE_TO_PLANE[dcode - 1] as i64;
                let (yo, xo) = (c >> 4, 8 - (c & 15));
                let d = yo * xs as i64 + xo;
                if d >= 1 {
                    d as usize
                } else {
                    1
                }
            };
            if p < dist || total - p < len {
                return Err("bad WebP backward reference".into());
            }
            for _ in 0..len {
                let v = px[p - dist];
                px[p] = v;
                insert(&mut cache, v);
                p += 1;
            }
            x += len;
            while x >= xs {
                x -= xs;
                y += 1;
            }
            // the group is chosen at the start of the next symbol
            if let Some((idx, bits, mx)) = &meta {
                if y < ysize {
                    gi = idx[(y >> bits) * mx + (x >> bits)] as usize;
                }
            }
        } else {
            let k = (s - 280) as usize;
            if k >= cache.len() {
                return Err("bad WebP colour cache index".into());
            }
            let v = cache[k];
            px[p] = v;
            insert(&mut cache, v);
            p += 1;
            x += 1;
            if x == xs {
                x = 0;
                y += 1;
            }
        }
        if br.over {
            return Err("truncated WebP data".into());
        }
        // refresh the group for the next pixel (needed when a literal ends a block row)
        if let Some((idx, bits, mx)) = &meta {
            if y < ysize {
                gi = idx[(y >> bits) * mx + (x >> bits)] as usize;
            }
        }
    }
    Ok((px, xs))
}

#[inline]
fn add_px(a: u32, b: u32) -> u32 {
    let ag = (a & 0xff00ff00).wrapping_add(b & 0xff00ff00) & 0xff00ff00;
    let rb = (a & 0x00ff00ff).wrapping_add(b & 0x00ff00ff) & 0x00ff00ff;
    ag | rb
}

#[inline]
fn avg2(a: u32, b: u32) -> u32 {
    (((a ^ b) & 0xfefefefe) >> 1).wrapping_add(a & b)
}

#[inline]
fn chan(p: u32, sh: u32) -> i32 {
    ((p >> sh) & 255) as i32
}

fn select(l: u32, t: u32, tl: u32) -> u32 {
    let mut pl = 0;
    let mut pt = 0;
    for sh in [0, 8, 16, 24] {
        pl += (chan(t, sh) - chan(tl, sh)).abs();
        pt += (chan(l, sh) - chan(tl, sh)).abs();
    }
    if pl < pt {
        l
    } else {
        t
    }
}

fn clamp_add_sub_full(a: u32, b: u32, c: u32) -> u32 {
    let mut o = 0u32;
    for sh in [0, 8, 16, 24] {
        o |= ((chan(a, sh) + chan(b, sh) - chan(c, sh)).clamp(0, 255) as u32) << sh;
    }
    o
}

fn clamp_add_sub_half(a: u32, b: u32) -> u32 {
    let mut o = 0u32;
    for sh in [0, 8, 16, 24] {
        let (x, y) = (chan(a, sh), chan(b, sh));
        o |= ((x + (x - y) / 2).clamp(0, 255) as u32) << sh;
    }
    o
}

fn predict(mode: u32, l: u32, t: u32, tr: u32, tl: u32) -> u32 {
    match mode {
        0 => 0xff000000,
        1 => l,
        2 => t,
        3 => tr,
        4 => tl,
        5 => avg2(avg2(l, tr), t),
        6 => avg2(l, tl),
        7 => avg2(l, t),
        8 => avg2(tl, t),
        9 => avg2(t, tr),
        10 => avg2(avg2(l, tl), avg2(t, tr)),
        11 => select(l, t, tl),
        12 => clamp_add_sub_full(l, t, tl),
        13 => clamp_add_sub_half(avg2(l, t), tl),
        _ => 0xff000000,
    }
}

fn inverse_transform(t: &Transform, px: &mut Vec<u32>) {
    let (w, h) = (t.xsize, t.ysize);
    match t.kind {
        0 => {
            let bits = t.bits;
            let bw = subsample(w, bits);
            // first pixel: black; rest of the top row: left; left column: top
            px[0] = add_px(px[0], 0xff000000);
            for x in 1..w {
                px[x] = add_px(px[x], px[x - 1]);
            }
            for y in 1..h {
                px[y * w] = add_px(px[y * w], px[(y - 1) * w]);
                for x in 1..w {
                    let mode = (t.data[(y >> bits) * bw + (x >> bits)] >> 8) & 15;
                    let i = y * w + x;
                    let pred = predict(mode, px[i - 1], px[i - w], px[i - w + 1], px[i - w - 1]);
                    px[i] = add_px(px[i], pred);
                }
            }
        }
        1 => {
            let bits = t.bits;
            let bw = subsample(w, bits);
            for y in 0..h {
                for x in 0..w {
                    let m = t.data[(y >> bits) * bw + (x >> bits)];
                    let (g2r, g2b, r2b) = ((m & 255) as i8 as i32, ((m >> 8) & 255) as i8 as i32, ((m >> 16) & 255) as i8 as i32);
                    let p = px[y * w + x];
                    let green = ((p >> 8) & 255) as i8 as i32;
                    let mut red = ((p >> 16) & 255) as i32;
                    let mut blue = (p & 255) as i32;
                    red = (red + ((g2r * green) >> 5)) & 255;
                    blue = (blue + ((g2b * green) >> 5) + ((r2b * (red as i8 as i32)) >> 5)) & 255;
                    px[y * w + x] = (p & 0xff00ff00) | ((red as u32) << 16) | blue as u32;
                }
            }
        }
        2 => {
            for p in px.iter_mut() {
                let g = (*p >> 8) & 255;
                let r = (((*p >> 16) & 255) + g) & 255;
                let b = ((*p & 255) + g) & 255;
                *p = (*p & 0xff00ff00) | (r << 16) | b;
            }
        }
        _ => {
            // colour indexing: `w` is the full width, the packed image is narrower
            let bits = t.bits;
            let pw = subsample(w, bits);
            let per = 1usize << bits;
            let depth = 8 >> bits;
            let mask = (1u32 << depth) - 1;
            let mut out = vec![0u32; w * h];
            for y in 0..h {
                for x in 0..w {
                    let src = px[y * pw + x / per];
                    let idx = if bits == 0 { (src >> 8) & 255 } else { (((src >> 8) & 255) >> (depth * (x % per) as u32)) & mask };
                    out[y * w + x] = t.data.get(idx as usize).copied().unwrap_or(0);
                }
            }
            *px = out;
        }
    }
}

/// Decode the transforms and the entropy coded image of a VP8L bit stream (after any header).
fn lossless_image(br: &mut LBits, w: usize, h: usize) -> Result<Vec<u32>, String> {
    let mut transforms = Vec::new();
    let (mut px, _) = decode_image_stream(br, w, h, true, &mut transforms)?;
    for t in transforms.iter().rev() {
        inverse_transform(t, &mut px);
    }
    Ok(px)
}

fn lossless(d: &[u8]) -> Result<Tex, String> {
    if d.len() < 5 || d[0] != 0x2f {
        return Err("bad WebP lossless header".into());
    }
    let bits = u32::from_le_bytes([d[1], d[2], d[3], d[4]]);
    let w = (bits & 0x3fff) as usize + 1;
    let h = ((bits >> 14) & 0x3fff) as usize + 1;
    crate::imgio::check_size(w, h)?;
    let has_alpha = (bits >> 28) & 1 == 1;
    if bits >> 29 != 0 {
        return Err("unsupported WebP lossless version".into());
    }
    let mut br = LBits::new(&d[5..]);
    let px = lossless_image(&mut br, w, h)?;
    let out: Vec<[f32; 4]> = px
        .iter()
        .map(|p| [((p >> 16) & 255) as f32 / 255.0, ((p >> 8) & 255) as f32 / 255.0, (p & 255) as f32 / 255.0, if has_alpha { (p >> 24) as f32 / 255.0 } else { 1.0 }])
        .collect();
    Ok(Tex { w, h, px: out })
}

// ================================================================================================ lossy (VP8)
struct Bool<'a> {
    d: &'a [u8],
    pos: usize,
    value: u32,
    range: u32,
    bits: i32,
}

impl<'a> Bool<'a> {
    fn new(d: &'a [u8]) -> Self {
        let b0 = d.first().copied().unwrap_or(0) as u32;
        let b1 = d.get(1).copied().unwrap_or(0) as u32;
        Bool { d, pos: 2, value: (b0 << 8) | b1, range: 255, bits: 0 }
    }
    #[inline]
    fn bit(&mut self, prob: u32) -> u32 {
        let split = 1 + (((self.range - 1) * prob) >> 8);
        let big = split << 8;
        let r = if self.value >= big {
            self.range -= split;
            self.value -= big;
            1
        } else {
            self.range = split;
            0
        };
        while self.range < 128 {
            self.value <<= 1;
            self.range <<= 1;
            self.bits += 1;
            if self.bits == 8 {
                self.bits = 0;
                self.value |= self.d.get(self.pos).copied().unwrap_or(0) as u32;
                self.pos += 1;
            }
        }
        r
    }
    fn flag(&mut self) -> bool {
        self.bit(128) == 1
    }
    fn value_n(&mut self, n: u32) -> i32 {
        let mut v = 0;
        for _ in 0..n {
            v = (v << 1) | self.bit(128) as i32;
        }
        v
    }
    fn signed_n(&mut self, n: u32) -> i32 {
        let v = self.value_n(n);
        if self.flag() {
            -v
        } else {
            v
        }
    }
    fn eof(&self) -> bool {
        self.pos > self.d.len() + 2
    }
}

const ZIGZAG: [usize; 16] = [0, 1, 4, 8, 5, 2, 3, 6, 9, 12, 13, 10, 7, 11, 14, 15];
const BANDS: [usize; 17] = [0, 1, 2, 3, 6, 4, 5, 6, 6, 6, 6, 6, 6, 6, 6, 7, 0];
const CAT3: [u32; 3] = [173, 148, 140];
const CAT4: [u32; 4] = [176, 155, 140, 135];
const CAT5: [u32; 5] = [180, 157, 141, 134, 130];
const CAT6: [u32; 11] = [254, 254, 243, 230, 196, 177, 153, 140, 133, 130, 129];

struct Quant {
    y1: [i32; 2],
    y2: [i32; 2],
    uv: [i32; 2],
}

#[derive(Clone, Copy, Default)]
struct MbInfo {
    segment: u8,
    skip: bool,
    i4: bool,
    ymode: u8,
    uvmode: u8,
    modes: [u8; 16],
    // filter info
    limit: u8,
    ilevel: u8,
    hev: u8,
    inner: bool,
}

fn get_large(br: &mut Bool, p: &[u8; 11]) -> i32 {
    if br.bit(p[3] as u32) == 0 {
        if br.bit(p[4] as u32) == 0 {
            2
        } else {
            3 + br.bit(p[5] as u32) as i32
        }
    } else if br.bit(p[6] as u32) == 0 {
        if br.bit(p[7] as u32) == 0 {
            5 + br.bit(159) as i32
        } else {
            let mut v = 7 + 2 * br.bit(165) as i32;
            v += br.bit(145) as i32;
            v
        }
    } else {
        let b1 = br.bit(p[8] as u32);
        let b0 = br.bit(p[(9 + b1) as usize] as u32);
        let cat = (2 * b1 + b0) as usize;
        let tab: &[u32] = match cat {
            0 => &CAT3,
            1 => &CAT4,
            2 => &CAT5,
            _ => &CAT6,
        };
        let mut v = 0i32;
        for t in tab {
            v += v + br.bit(*t) as i32;
        }
        v + 3 + (8 << cat)
    }
}

/// Returns the position after the last non-zero coefficient (like the reference decoder).
fn get_coeffs(br: &mut Bool, probs: &[u8], ty: usize, ctx: usize, dq: &[i32; 2], first: usize, out: &mut [i16]) -> usize {
    let pr = |n: usize, c: usize| -> &[u8; 11] {
        let o = ((ty * 8 + BANDS[n]) * 3 + c) * 11;
        probs[o..o + 11].try_into().unwrap()
    };
    let mut n = first;
    let mut p = pr(n, ctx);
    while n < 16 {
        if br.bit(p[0] as u32) == 0 {
            return n;
        }
        while br.bit(p[1] as u32) == 0 {
            n += 1;
            p = pr(n, 0);
            if n == 16 {
                return 16;
            }
        }
        let v;
        if br.bit(p[2] as u32) == 0 {
            v = 1;
            p = pr(n + 1, 1);
        } else {
            v = get_large(br, p);
            p = pr(n + 1, 2);
        }
        let neg = br.bit(128) == 1;
        let val = if neg { -v } else { v };
        out[ZIGZAG[n]] = (val * dq[(n > 0) as usize]) as i16;
        n += 1;
    }
    16
}

#[inline]
fn clip8(v: i32) -> u8 {
    v.clamp(0, 255) as u8
}

fn idct_add(c: &[i16], dst: &mut [u8], stride: usize, off: usize) {
    let mul1 = |a: i32| ((a * 20091) >> 16) + a;
    let mul2 = |a: i32| (a * 35468) >> 16;
    let mut tmp = [0i32; 16];
    for i in 0..4 {
        let a = c[i] as i32 + c[8 + i] as i32;
        let b = c[i] as i32 - c[8 + i] as i32;
        let cc = mul2(c[4 + i] as i32) - mul1(c[12 + i] as i32);
        let d = mul1(c[4 + i] as i32) + mul2(c[12 + i] as i32);
        tmp[i * 4] = a + d;
        tmp[i * 4 + 1] = b + cc;
        tmp[i * 4 + 2] = b - cc;
        tmp[i * 4 + 3] = a - d;
    }
    for i in 0..4 {
        let dc = tmp[i] + 4;
        let a = dc + tmp[8 + i];
        let b = dc - tmp[8 + i];
        let cc = mul2(tmp[4 + i]) - mul1(tmp[12 + i]);
        let d = mul1(tmp[4 + i]) + mul2(tmp[12 + i]);
        let row = off + i * stride;
        dst[row] = clip8(dst[row] as i32 + ((a + d) >> 3));
        dst[row + 1] = clip8(dst[row + 1] as i32 + ((b + cc) >> 3));
        dst[row + 2] = clip8(dst[row + 2] as i32 + ((b - cc) >> 3));
        dst[row + 3] = clip8(dst[row + 3] as i32 + ((a - d) >> 3));
    }
}

fn wht(input: &[i16; 16], out: &mut [i16; 384]) {
    let mut tmp = [0i32; 16];
    for i in 0..4 {
        let a0 = input[i] as i32 + input[12 + i] as i32;
        let a1 = input[4 + i] as i32 + input[8 + i] as i32;
        let a2 = input[4 + i] as i32 - input[8 + i] as i32;
        let a3 = input[i] as i32 - input[12 + i] as i32;
        tmp[i] = a0 + a1;
        tmp[8 + i] = a0 - a1;
        tmp[4 + i] = a3 + a2;
        tmp[12 + i] = a3 - a2;
    }
    for i in 0..4 {
        let dc = tmp[i * 4] + 3;
        let a0 = dc + tmp[3 + i * 4];
        let a1 = tmp[1 + i * 4] + tmp[2 + i * 4];
        let a2 = tmp[1 + i * 4] - tmp[2 + i * 4];
        let a3 = dc - tmp[3 + i * 4];
        out[(i * 4) * 16] = ((a0 + a1) >> 3) as i16;
        out[(i * 4 + 1) * 16] = ((a3 + a2) >> 3) as i16;
        out[(i * 4 + 2) * 16] = ((a0 - a1) >> 3) as i16;
        out[(i * 4 + 3) * 16] = ((a3 - a2) >> 3) as i16;
    }
}

// mode ids: 0 DC, 1 TM, 2 V (VE), 3 H (HE), 4 RD, 5 VR, 6 LD, 7 VL, 8 HD, 9 HU
fn predict4(mode: u8, buf: &mut [u8], stride: usize, off: usize, topright: [u8; 4]) {
    // neighbours: X = top-left, A..D = top, E..H = top-right, I..L = left
    let x = buf[off - stride - 1] as i32;
    let t = |i: usize| buf[off - stride + i] as i32;
    let (a, b, c, d) = (t(0), t(1), t(2), t(3));
    let (e, f, g, h) = (topright[0] as i32, topright[1] as i32, topright[2] as i32, topright[3] as i32);
    let l = |i: usize| buf[off + i * stride - 1] as i32;
    let (i, j, k, ll) = (l(0), l(1), l(2), l(3));
    let avg3 = |p: i32, q: i32, r: i32| ((p + 2 * q + r + 2) >> 2) as u8;
    let avg2 = |p: i32, q: i32| ((p + q + 1) >> 1) as u8;
    let mut px = [[0u8; 4]; 4]; // [y][x]
    match mode {
        0 => {
            let dc = ((a + b + c + d + i + j + k + ll + 4) >> 3) as u8;
            px = [[dc; 4]; 4];
        }
        1 => {
            for yy in 0..4 {
                for xx in 0..4 {
                    px[yy][xx] = clip8(l(yy) + t(xx) - x);
                }
            }
        }
        2 => {
            let v = [avg3(x, a, b), avg3(a, b, c), avg3(b, c, d), avg3(c, d, e)];
            px = [v; 4];
        }
        3 => {
            px[0] = [avg3(x, i, j); 4];
            px[1] = [avg3(i, j, k); 4];
            px[2] = [avg3(j, k, ll); 4];
            px[3] = [avg3(k, ll, ll); 4];
        }
        4 => {
            px[3][0] = avg3(j, k, ll);
            let v = avg3(i, j, k);
            px[3][1] = v;
            px[2][0] = v;
            let v = avg3(x, i, j);
            px[3][2] = v;
            px[2][1] = v;
            px[1][0] = v;
            let v = avg3(a, x, i);
            px[3][3] = v;
            px[2][2] = v;
            px[1][1] = v;
            px[0][0] = v;
            let v = avg3(b, a, x);
            px[2][3] = v;
            px[1][2] = v;
            px[0][1] = v;
            let v = avg3(c, b, a);
            px[1][3] = v;
            px[0][2] = v;
            px[0][3] = avg3(d, c, b);
        }
        5 => {
            let v = avg2(x, a);
            px[0][0] = v;
            px[2][1] = v;
            let v = avg2(a, b);
            px[0][1] = v;
            px[2][2] = v;
            let v = avg2(b, c);
            px[0][2] = v;
            px[2][3] = v;
            px[0][3] = avg2(c, d);
            px[3][0] = avg3(k, j, i);
            px[2][0] = avg3(j, i, x);
            let v = avg3(i, x, a);
            px[1][0] = v;
            px[3][1] = v;
            let v = avg3(x, a, b);
            px[1][1] = v;
            px[3][2] = v;
            let v = avg3(a, b, c);
            px[1][2] = v;
            px[3][3] = v;
            px[1][3] = avg3(b, c, d);
        }
        6 => {
            px[0][0] = avg3(a, b, c);
            let v = avg3(b, c, d);
            px[0][1] = v;
            px[1][0] = v;
            let v = avg3(c, d, e);
            px[0][2] = v;
            px[1][1] = v;
            px[2][0] = v;
            let v = avg3(d, e, f);
            px[0][3] = v;
            px[1][2] = v;
            px[2][1] = v;
            px[3][0] = v;
            let v = avg3(e, f, g);
            px[1][3] = v;
            px[2][2] = v;
            px[3][1] = v;
            let v = avg3(f, g, h);
            px[2][3] = v;
            px[3][2] = v;
            px[3][3] = avg3(g, h, h);
        }
        7 => {
            px[0][0] = avg2(a, b);
            let v = avg2(b, c);
            px[0][1] = v;
            px[2][0] = v;
            let v = avg2(c, d);
            px[0][2] = v;
            px[2][1] = v;
            let v = avg2(d, e);
            px[0][3] = v;
            px[2][2] = v;
            px[1][0] = avg3(a, b, c);
            let v = avg3(b, c, d);
            px[1][1] = v;
            px[3][0] = v;
            let v = avg3(c, d, e);
            px[1][2] = v;
            px[3][1] = v;
            let v = avg3(d, e, f);
            px[1][3] = v;
            px[3][2] = v;
            px[2][3] = avg3(e, f, g);
            px[3][3] = avg3(f, g, h);
        }
        8 => {
            let v = avg2(i, x);
            px[0][0] = v;
            px[1][2] = v;
            let v = avg2(j, i);
            px[1][0] = v;
            px[2][2] = v;
            let v = avg2(k, j);
            px[2][0] = v;
            px[3][2] = v;
            px[3][0] = avg2(ll, k);
            px[0][3] = avg3(a, b, c);
            px[0][2] = avg3(x, a, b);
            let v = avg3(i, x, a);
            px[0][1] = v;
            px[1][3] = v;
            let v = avg3(j, i, x);
            px[1][1] = v;
            px[2][3] = v;
            let v = avg3(k, j, i);
            px[2][1] = v;
            px[3][3] = v;
            px[3][1] = avg3(ll, k, j);
        }
        _ => {
            px[0][0] = avg2(i, j);
            let v = avg2(j, k);
            px[0][2] = v;
            px[1][0] = v;
            let v = avg2(k, ll);
            px[1][2] = v;
            px[2][0] = v;
            px[0][1] = avg3(i, j, k);
            let v = avg3(j, k, ll);
            px[0][3] = v;
            px[1][1] = v;
            let v = avg3(k, ll, ll);
            px[1][3] = v;
            px[2][1] = v;
            let v = ll as u8;
            px[2][2] = v;
            px[2][3] = v;
            px[3] = [v; 4];
        }
    }
    for yy in 0..4 {
        for xx in 0..4 {
            buf[off + yy * stride + xx] = px[yy][xx];
        }
    }
}

/// 16x16 luma and 8x8 chroma prediction on a bordered block buffer (`stride` bytes per row, border row and column first).
fn predict_block(mode: u8, size: usize, buf: &mut [u8], stride: usize, has_top: bool, has_left: bool) {
    let base = stride + 1;
    let top = |x: usize| buf[base - stride + x] as i32;
    let left = |y: usize| buf[base + y * stride - 1] as i32;
    let tl = buf[base - stride - 1] as i32;
    let mut vals = vec![0u8; size * size];
    match mode {
        0 => {
            let (mut s, shift) = (0i32, if size == 16 { 4 } else { 3 });
            let dc = match (has_top, has_left) {
                (true, true) => {
                    for k in 0..size {
                        s += top(k) + left(k);
                    }
                    (s + size as i32) >> (shift + 1)
                }
                (true, false) => {
                    for k in 0..size {
                        s += top(k);
                    }
                    (s + (size as i32 >> 1)) >> shift
                }
                (false, true) => {
                    for k in 0..size {
                        s += left(k);
                    }
                    (s + (size as i32 >> 1)) >> shift
                }
                _ => 128,
            };
            vals.iter_mut().for_each(|v| *v = dc as u8);
        }
        1 => {
            for y in 0..size {
                for x in 0..size {
                    vals[y * size + x] = clip8(left(y) + top(x) - tl);
                }
            }
        }
        2 => {
            for y in 0..size {
                for x in 0..size {
                    vals[y * size + x] = top(x) as u8;
                }
            }
        }
        _ => {
            for y in 0..size {
                for x in 0..size {
                    vals[y * size + x] = left(y) as u8;
                }
            }
        }
    }
    for y in 0..size {
        for x in 0..size {
            buf[base + y * stride + x] = vals[y * size + x];
        }
    }
}

fn simple_filter_edge(p: &mut [u8], pos: usize, step: usize, count: usize, along: usize, thresh: i32) {
    let thresh2 = 2 * thresh + 1;
    for i in 0..count {
        let o = pos + i * along;
        let (p1, p0, q0, q1) = (p[o - 2 * step] as i32, p[o - step] as i32, p[o] as i32, p[o + step] as i32);
        if 4 * (p0 - q0).abs() + (p1 - q1).abs() <= thresh2 {
            filter2(p, o, step, p1, p0, q0, q1);
        }
    }
}

#[inline]
fn sclip1(v: i32) -> i32 {
    v.clamp(-128, 127)
}
#[inline]
fn sclip2(v: i32) -> i32 {
    v.clamp(-16, 15)
}

fn filter2(p: &mut [u8], o: usize, step: usize, p1: i32, p0: i32, q0: i32, q1: i32) {
    let a = 3 * (q0 - p0) + sclip1(p1 - q1);
    let a1 = sclip2((a + 4) >> 3);
    let a2 = sclip2((a + 3) >> 3);
    p[o - step] = clip8(p0 + a2);
    p[o] = clip8(q0 - a1);
}

fn filter4(p: &mut [u8], o: usize, step: usize) {
    let (p1, p0, q0, q1) = (p[o - 2 * step] as i32, p[o - step] as i32, p[o] as i32, p[o + step] as i32);
    let a = 3 * (q0 - p0);
    let a1 = sclip2((a + 4) >> 3);
    let a2 = sclip2((a + 3) >> 3);
    let a3 = (a1 + 1) >> 1;
    p[o - 2 * step] = clip8(p1 + a3);
    p[o - step] = clip8(p0 + a2);
    p[o] = clip8(q0 - a1);
    p[o + step] = clip8(q1 - a3);
}

fn filter6(p: &mut [u8], o: usize, step: usize) {
    let (p2, p1, p0) = (p[o - 3 * step] as i32, p[o - 2 * step] as i32, p[o - step] as i32);
    let (q0, q1, q2) = (p[o] as i32, p[o + step] as i32, p[o + 2 * step] as i32);
    let a = sclip1(3 * (q0 - p0) + sclip1(p1 - q1));
    let a1 = (27 * a + 63) >> 7;
    let a2 = (18 * a + 63) >> 7;
    let a3 = (9 * a + 63) >> 7;
    p[o - 3 * step] = clip8(p2 + a3);
    p[o - 2 * step] = clip8(p1 + a2);
    p[o - step] = clip8(p0 + a1);
    p[o] = clip8(q0 - a1);
    p[o + step] = clip8(q1 - a2);
    p[o + 2 * step] = clip8(q2 - a3);
}

#[allow(clippy::too_many_arguments)]
fn complex_filter_edge(p: &mut [u8], pos: usize, step: usize, count: usize, along: usize, thresh: i32, ithresh: i32, hev_thresh: i32, mb_edge: bool) {
    let thresh2 = 2 * thresh + 1;
    for i in 0..count {
        let o = pos + i * along;
        let g = |k: isize| p[(o as isize + k * step as isize) as usize] as i32;
        let (p3, p2, p1, p0, q0, q1, q2, q3) = (g(-4), g(-3), g(-2), g(-1), g(0), g(1), g(2), g(3));
        if 4 * (p0 - q0).abs() + (p1 - q1).abs() > thresh2 {
            continue;
        }
        if (p3 - p2).abs() > ithresh || (p2 - p1).abs() > ithresh || (p1 - p0).abs() > ithresh || (q3 - q2).abs() > ithresh || (q2 - q1).abs() > ithresh || (q1 - q0).abs() > ithresh {
            continue;
        }
        let hev = (p1 - p0).abs() > hev_thresh || (q1 - q0).abs() > hev_thresh;
        if hev {
            filter2(p, o, step, p1, p0, q0, q1);
        } else if mb_edge {
            filter6(p, o, step);
        } else {
            filter4(p, o, step);
        }
    }
}

fn lossy(d: &[u8], alph: Option<&[u8]>) -> Result<Tex, String> {
    if d.len() < 10 {
        return Err("truncated WebP data".into());
    }
    let tag = d[0] as u32 | (d[1] as u32) << 8 | (d[2] as u32) << 16;
    if tag & 1 != 0 {
        return Err("WebP frame is not a key frame".into());
    }
    if (tag >> 1) & 7 > 3 {
        return Err("bad WebP profile".into());
    }
    let part0 = (tag >> 5) as usize;
    if d[3] != 0x9d || d[4] != 0x01 || d[5] != 0x2a {
        return Err("bad WebP start code".into());
    }
    let w = (d[6] as usize | (d[7] as usize) << 8) & 0x3fff;
    let h = (d[8] as usize | (d[9] as usize) << 8) & 0x3fff;
    crate::imgio::check_size(w, h)?;
    let body = &d[10..];
    if part0 > body.len() {
        return Err("truncated WebP data".into());
    }
    let mut br = Bool::new(&body[..part0]);
    let (mbw, mbh) = ((w + 15) >> 4, (h + 15) >> 4);
    br.flag(); // colour space
    br.flag(); // clamping type
    // segments
    let use_segment = br.flag();
    let mut update_map = false;
    let mut abs_delta = true;
    let mut seg_quant = [0i32; 4];
    let mut seg_filter = [0i32; 4];
    let mut seg_probs = [255u32; 3];
    if use_segment {
        update_map = br.flag();
        if br.flag() {
            abs_delta = br.flag();
            for s in 0..4 {
                seg_quant[s] = if br.flag() { br.signed_n(7) } else { 0 };
            }
            for s in 0..4 {
                seg_filter[s] = if br.flag() { br.signed_n(6) } else { 0 };
            }
        }
        if update_map {
            for s in 0..3 {
                seg_probs[s] = if br.flag() { br.value_n(8) as u32 } else { 255 };
            }
        }
    }
    // loop filter
    let simple = br.flag();
    let level = br.value_n(6);
    let sharpness = br.value_n(3);
    let use_lf_delta = br.flag();
    let mut ref_delta = [0i32; 4];
    let mut mode_delta = [0i32; 4];
    if use_lf_delta && br.flag() {
        for i in 0..4 {
            if br.flag() {
                ref_delta[i] = br.signed_n(6);
            }
        }
        for i in 0..4 {
            if br.flag() {
                mode_delta[i] = br.signed_n(6);
            }
        }
    }
    let filter_type = if level == 0 { 0 } else if simple { 1 } else { 2 };
    // partitions
    let nparts = 1usize << br.value_n(2);
    let rest = &body[part0..];
    if rest.len() < 3 * (nparts - 1) {
        return Err("truncated WebP partitions".into());
    }
    let mut parts: Vec<Bool> = Vec::new();
    {
        let mut start = 3 * (nparts - 1);
        let mut left = rest.len() - start;
        for p in 0..nparts {
            let size = if p + 1 < nparts { (rest[3 * p] as usize | (rest[3 * p + 1] as usize) << 8 | (rest[3 * p + 2] as usize) << 16).min(left) } else { left };
            parts.push(Bool::new(&rest[start..start + size]));
            start += size;
            left -= size;
        }
    }
    // quantisers
    let base_q = br.value_n(7);
    let mut dq = [0i32; 5];
    for v in dq.iter_mut() {
        *v = if br.flag() { br.signed_n(4) } else { 0 };
    }
    let mut quants: Vec<Quant> = Vec::new();
    for s in 0..4 {
        let q = if use_segment {
            if abs_delta {
                seg_quant[s]
            } else {
                seg_quant[s] + base_q
            }
        } else {
            base_q
        };
        let cl = |v: i32, m: i32| v.clamp(0, m) as usize;
        quants.push(Quant {
            y1: [DC_TABLE[cl(q + dq[0], 127)] as i32, AC_TABLE[cl(q, 127)] as i32],
            y2: [DC_TABLE[cl(q + dq[1], 127)] as i32 * 2, ((AC_TABLE[cl(q + dq[2], 127)] as i32 * 101581) >> 16).max(8)],
            uv: [DC_TABLE[cl(q + dq[3], 117)] as i32, AC_TABLE[cl(q + dq[4], 127)] as i32],
        });
    }
    br.flag(); // refresh entropy probs
    let mut probs = COEFF_PROBA.to_vec();
    for i in 0..probs.len() {
        if br.bit(COEFF_UPDATE[i] as u32) == 1 {
            probs[i] = br.value_n(8) as u8;
        }
    }
    let use_skip = br.flag();
    let skip_p = if use_skip { br.value_n(8) as u32 } else { 0 };
    // per macroblock modes
    let mut mbs = vec![MbInfo::default(); mbw * mbh];
    let mut intra_t = vec![0u8; mbw * 4];
    for my in 0..mbh {
        let mut intra_l = [0u8; 4];
        for mx in 0..mbw {
            let mut mb = MbInfo::default();
            if update_map {
                mb.segment = if br.bit(seg_probs[0]) == 0 { br.bit(seg_probs[1]) as u8 } else { 2 + br.bit(seg_probs[2]) as u8 };
            }
            if use_skip {
                mb.skip = br.bit(skip_p) == 1;
            }
            mb.i4 = br.bit(145) == 0;
            let top = &mut intra_t[mx * 4..mx * 4 + 4];
            if !mb.i4 {
                let ym = if br.bit(156) == 1 {
                    if br.bit(128) == 1 {
                        1
                    } else {
                        3
                    }
                } else if br.bit(163) == 1 {
                    2
                } else {
                    0
                };
                mb.ymode = ym;
                top.fill(ym);
                intra_l.fill(ym);
            } else {
                for y in 0..4 {
                    let mut ym = intra_l[y] as usize;
                    for x in 0..4 {
                        let o = (top[x] as usize * 10 + ym) * 9;
                        let pr = &BMODE_PROBA[o..o + 9];
                        let b = |i: usize, br: &mut Bool| br.bit(pr[i] as u32);
                        ym = if b(0, &mut br) == 0 {
                            0
                        } else if b(1, &mut br) == 0 {
                            1
                        } else if b(2, &mut br) == 0 {
                            2
                        } else if b(3, &mut br) == 0 {
                            if b(4, &mut br) == 0 {
                                3
                            } else if b(5, &mut br) == 0 {
                                4
                            } else {
                                5
                            }
                        } else if b(6, &mut br) == 0 {
                            6
                        } else if b(7, &mut br) == 0 {
                            7
                        } else if b(8, &mut br) == 0 {
                            8
                        } else {
                            9
                        };
                        top[x] = ym as u8;
                        mb.modes[y * 4 + x] = ym as u8;
                    }
                    intra_l[y] = ym as u8;
                }
            }
            mb.uvmode = if br.bit(142) == 0 {
                0
            } else if br.bit(114) == 0 {
                2
            } else if br.bit(183) == 1 {
                1
            } else {
                3
            };
            mbs[my * mbw + mx] = mb;
        }
        if br.eof() {
            return Err("truncated WebP data".into());
        }
    }
    // filter strengths per (segment, i4)
    let strength = |seg: usize, i4: bool| -> (u8, u8, u8, bool) {
        let mut base = if use_segment {
            let mut b = seg_filter[seg];
            if !abs_delta {
                b += level;
            }
            b
        } else {
            level
        };
        if use_lf_delta {
            base += ref_delta[0];
            if i4 {
                base += mode_delta[0];
            }
        }
        let lv = base.clamp(0, 63);
        if lv == 0 {
            return (0, 0, 0, i4);
        }
        let mut il = lv;
        if sharpness > 0 {
            il >>= if sharpness > 4 { 2 } else { 1 };
            if il > 9 - sharpness {
                il = 9 - sharpness;
            }
        }
        if il < 1 {
            il = 1;
        }
        (( 2 * lv + il) as u8, il as u8, if lv >= 40 { 2 } else if lv >= 15 { 1 } else { 0 }, i4)
    };
    // reconstruction
    let (yw, yh) = (mbw * 16, mbh * 16);
    let (cw, ch) = (mbw * 8, mbh * 8);
    let mut yp = vec![0u8; yw * yh];
    let mut up = vec![0u8; cw * ch];
    let mut vp = vec![0u8; cw * ch];
    let mut above_nz = vec![[false; 9]; mbw]; // 4 Y, 2 U, 2 V, Y2
    for my in 0..mbh {
        let mut left_nz = [false; 9];
        for mx in 0..mbw {
            let mb = &mut mbs[my * mbw + mx];
            let tbr = &mut parts[my & (nparts - 1)];
            let mut coef = [0i16; 384];
            let q = &quants[mb.segment as usize];
            let mut any = false;
            let skip = mb.skip && use_skip;
            if !skip {
                let an = &mut above_nz[mx];
                let first;
                let ac_type;
                if !mb.i4 {
                    let mut dc = [0i16; 16];
                    let ctx = an[8] as usize + left_nz[8] as usize;
                    let nz = get_coeffs(tbr, &probs, 1, ctx, &q.y2, 0, &mut dc);
                    an[8] = nz > 0;
                    left_nz[8] = nz > 0;
                    if nz > 1 {
                        wht(&dc, &mut coef);
                    } else {
                        let dc0 = ((dc[0] as i32 + 3) >> 3) as i16;
                        for i in 0..16 {
                            coef[i * 16] = dc0;
                        }
                    }
                    first = 1;
                    ac_type = 0;
                } else {
                    first = 0;
                    ac_type = 3;
                }
                for y in 0..4 {
                    for x in 0..4 {
                        let ctx = an[x] as usize + left_nz[y] as usize;
                        let blk = &mut coef[(y * 4 + x) * 16..(y * 4 + x + 1) * 16];
                        let nz = get_coeffs(tbr, &probs, ac_type, ctx, &q.y1, first, blk);
                        let l = nz > first;
                        an[x] = l;
                        left_nz[y] = l;
                        if nz > 1 || blk[0] != 0 {
                            any = true;
                        }
                    }
                }
                for ch_i in 0..2 {
                    for y in 0..2 {
                        for x in 0..2 {
                            let (ai, li) = (4 + ch_i * 2 + x, 4 + ch_i * 2 + y);
                            let ctx = an[ai] as usize + left_nz[li] as usize;
                            let bi = 16 + ch_i * 4 + y * 2 + x;
                            let blk = &mut coef[bi * 16..(bi + 1) * 16];
                            let nz = get_coeffs(tbr, &probs, 2, ctx, &q.uv, 0, blk);
                            let l = nz > 0;
                            an[ai] = l;
                            left_nz[li] = l;
                            if nz > 1 || blk[0] != 0 {
                                any = true;
                            }
                        }
                    }
                }
            } else {
                for i in 0..8 {
                    above_nz[mx][i] = false;
                    left_nz[i] = false;
                }
                if !mb.i4 {
                    above_nz[mx][8] = false;
                    left_nz[8] = false;
                }
            }
            if filter_type > 0 {
                let (lim, il, hev, inner) = strength(mb.segment as usize, mb.i4);
                mb.limit = lim;
                mb.ilevel = il;
                mb.hev = hev;
                mb.inner = inner || any;
            }
            // luma: bordered buffer 17 rows x 21 columns
            const YS: usize = 21;
            let mut yb = [127u8; 17 * YS];
            // top row (including top-left and top-right)
            if my > 0 {
                let row = (my * 16 - 1) * yw;
                for x in 0..16 {
                    yb[1 + x] = yp[row + mx * 16 + x];
                }
                if mx + 1 < mbw {
                    for x in 0..4 {
                        yb[17 + x] = yp[row + mx * 16 + 16 + x];
                    }
                } else {
                    for x in 0..4 {
                        yb[17 + x] = yp[row + mx * 16 + 15];
                    }
                }
                yb[0] = if mx > 0 { yp[row + mx * 16 - 1] } else { 129 };
            }
            for y in 0..16 {
                yb[(y + 1) * YS] = if mx > 0 { yp[(my * 16 + y) * yw + mx * 16 - 1] } else { 129 };
            }
            let has_top = my > 0;
            let has_left = mx > 0;
            if !mb.i4 {
                predict_block(mb.ymode, 16, &mut yb, YS, has_top, has_left);
                for by in 0..4 {
                    for bx in 0..4 {
                        let c = &coef[(by * 4 + bx) * 16..(by * 4 + bx + 1) * 16];
                        if c.iter().any(|v| *v != 0) {
                            idct_add(c, &mut yb, YS, (by * 4 + 1) * YS + bx * 4 + 1);
                        }
                    }
                }
            } else {
                let tr: [u8; 4] = [yb[17], yb[18], yb[19], yb[20]];
                for by in 0..4 {
                    for bx in 0..4 {
                        let off = (by * 4 + 1) * YS + bx * 4 + 1;
                        let topright = if bx == 3 {
                            tr
                        } else {
                            let o = off - YS + 4;
                            [yb[o], yb[o + 1], yb[o + 2], yb[o + 3]]
                        };
                        predict4(mb.modes[by * 4 + bx], &mut yb, YS, off, topright);
                        let c = &coef[(by * 4 + bx) * 16..(by * 4 + bx + 1) * 16];
                        if c.iter().any(|v| *v != 0) {
                            idct_add(c, &mut yb, YS, off);
                        }
                    }
                }
            }
            for y in 0..16 {
                let dst = (my * 16 + y) * yw + mx * 16;
                yp[dst..dst + 16].copy_from_slice(&yb[(y + 1) * YS + 1..(y + 1) * YS + 17]);
            }
            // chroma
            const CS: usize = 9;
            for (pi, plane) in [&mut up, &mut vp].into_iter().enumerate() {
                let mut cb = [127u8; 9 * CS];
                if my > 0 {
                    let row = (my * 8 - 1) * cw;
                    for x in 0..8 {
                        cb[1 + x] = plane[row + mx * 8 + x];
                    }
                    cb[0] = if mx > 0 { plane[row + mx * 8 - 1] } else { 129 };
                }
                for y in 0..8 {
                    cb[(y + 1) * CS] = if mx > 0 { plane[(my * 8 + y) * cw + mx * 8 - 1] } else { 129 };
                }
                predict_block(mb.uvmode, 8, &mut cb, CS, has_top, has_left);
                for by in 0..2 {
                    for bx in 0..2 {
                        let bi = 16 + pi * 4 + by * 2 + bx;
                        let c = &coef[bi * 16..(bi + 1) * 16];
                        if c.iter().any(|v| *v != 0) {
                            idct_add(c, &mut cb, CS, (by * 4 + 1) * CS + bx * 4 + 1);
                        }
                    }
                }
                for y in 0..8 {
                    let dst = (my * 8 + y) * cw + mx * 8;
                    plane[dst..dst + 8].copy_from_slice(&cb[(y + 1) * CS + 1..(y + 1) * CS + 9]);
                }
            }
        }
        if parts[my & (nparts - 1)].eof() {
            return Err("truncated WebP data".into());
        }
    }
    // loop filter, macroblock by macroblock
    if filter_type > 0 {
        for my in 0..mbh {
            for mx in 0..mbw {
                let mb = mbs[my * mbw + mx];
                if mb.limit == 0 {
                    continue;
                }
                let (limit, il, hev) = (mb.limit as i32, mb.ilevel as i32, mb.hev as i32);
                let yo = my * 16 * yw + mx * 16;
                if filter_type == 1 {
                    if mx > 0 {
                        simple_filter_edge(&mut yp, yo, 1, 16, yw, limit + 4);
                    }
                    if mb.inner {
                        for k in 1..4 {
                            simple_filter_edge(&mut yp, yo + 4 * k, 1, 16, yw, limit);
                        }
                    }
                    if my > 0 {
                        simple_filter_edge(&mut yp, yo, yw, 16, 1, limit + 4);
                    }
                    if mb.inner {
                        for k in 1..4 {
                            simple_filter_edge(&mut yp, yo + 4 * k * yw, yw, 16, 1, limit);
                        }
                    }
                } else {
                    let co = my * 8 * cw + mx * 8;
                    if mx > 0 {
                        complex_filter_edge(&mut yp, yo, 1, 16, yw, limit + 4, il, hev, true);
                        complex_filter_edge(&mut up, co, 1, 8, cw, limit + 4, il, hev, true);
                        complex_filter_edge(&mut vp, co, 1, 8, cw, limit + 4, il, hev, true);
                    }
                    if mb.inner {
                        for k in 1..4 {
                            complex_filter_edge(&mut yp, yo + 4 * k, 1, 16, yw, limit, il, hev, false);
                        }
                        complex_filter_edge(&mut up, co + 4, 1, 8, cw, limit, il, hev, false);
                        complex_filter_edge(&mut vp, co + 4, 1, 8, cw, limit, il, hev, false);
                    }
                    if my > 0 {
                        complex_filter_edge(&mut yp, yo, yw, 16, 1, limit + 4, il, hev, true);
                        complex_filter_edge(&mut up, co, cw, 8, 1, limit + 4, il, hev, true);
                        complex_filter_edge(&mut vp, co, cw, 8, 1, limit + 4, il, hev, true);
                    }
                    if mb.inner {
                        for k in 1..4 {
                            complex_filter_edge(&mut yp, yo + 4 * k * yw, yw, 16, 1, limit, il, hev, false);
                        }
                        complex_filter_edge(&mut up, co + 4 * cw, cw, 8, 1, limit, il, hev, false);
                        complex_filter_edge(&mut vp, co + 4 * cw, cw, 8, 1, limit, il, hev, false);
                    }
                }
            }
        }
    }
    // colour conversion with fancy chroma upsampling
    let mut px = vec![[0.0f32, 0.0, 0.0, 1.0]; w * h];
    let (uw, uh) = ((w + 1) / 2, (h + 1) / 2);
    let mut uvrow_top = vec![[0i32; 2]; w];
    let mut uvrow_bot = vec![[0i32; 2]; w];
    let upsample_pair = |tu: usize, cu: usize, want_bottom: bool, top_out: &mut Vec<[i32; 2]>, bot_out: &mut Vec<[i32; 2]>| {
        // chroma rows `tu` (above) and `cu` (current); outputs the two luma rows between them
        let get = |row: usize, x: usize| -> [i32; 2] { [up[row * cw + x] as i32, vp[row * cw + x] as i32] };
        for k in 0..2 {
            let (mut tl, mut l) = (get(tu, 0)[k], get(cu, 0)[k]);
            top_out[0][k] = (3 * tl + l + 2) >> 2;
            if want_bottom {
                bot_out[0][k] = (3 * l + tl + 2) >> 2;
            }
            let last_pair = (w - 1) >> 1;
            for x in 1..=last_pair {
                let t = get(tu, x)[k];
                let c = get(cu, x)[k];
                let avg = tl + t + l + c + 8;
                let diag12 = (avg + 2 * (t + l)) >> 3;
                let diag03 = (avg + 2 * (tl + c)) >> 3;
                top_out[2 * x - 1][k] = (diag12 + tl) >> 1;
                top_out[2 * x][k] = (diag03 + t) >> 1;
                if want_bottom {
                    bot_out[2 * x - 1][k] = (diag03 + l) >> 1;
                    bot_out[2 * x][k] = (diag12 + c) >> 1;
                }
                tl = t;
                l = c;
            }
            if w & 1 == 0 {
                top_out[w - 1][k] = (3 * tl + l + 2) >> 2;
                if want_bottom {
                    bot_out[w - 1][k] = (3 * l + tl + 2) >> 2;
                }
            }
        }
    };
    let conv = |y: i32, u: i32, v: i32| -> [f32; 3] {
        let mh = |a: i32, c: i32| (a * c) >> 8;
        let clip = |v: i32| -> i32 {
            if v & !((256 << 6) - 1) == 0 {
                v >> 6
            } else if v < 0 {
                0
            } else {
                255
            }
        };
        let r = clip(mh(y, 19077) + mh(v, 26149) - 14234);
        let g = clip(mh(y, 19077) - mh(u, 6419) - mh(v, 13320) + 8708);
        let b = clip(mh(y, 19077) + mh(u, 33050) - 17685);
        [r as f32 / 255.0, g as f32 / 255.0, b as f32 / 255.0]
    };
    let mut emit = |row: usize, uv: &Vec<[i32; 2]>| {
        for x in 0..w {
            let c = conv(yp[row * yw + x] as i32, uv[x][0], uv[x][1]);
            px[row * w + x] = [c[0], c[1], c[2], 1.0];
        }
    };
    // row 0 uses chroma row 0 on its own
    upsample_pair(0, 0, false, &mut uvrow_top, &mut uvrow_bot);
    emit(0, &uvrow_top);
    let mut j = 1;
    while 2 * j < h {
        upsample_pair(j - 1, j, true, &mut uvrow_top, &mut uvrow_bot);
        emit(2 * j - 1, &uvrow_top);
        emit(2 * j, &uvrow_bot);
        j += 1;
    }
    if h > 1 && h % 2 == 0 {
        upsample_pair(uh - 1, uh - 1, false, &mut uvrow_top, &mut uvrow_bot);
        emit(h - 1, &uvrow_top);
    }
    let _ = uw;
    // alpha plane
    if let Some(a) = alph {
        let alpha = alpha_plane(a, w, h)?;
        for i in 0..w * h {
            px[i][3] = alpha[i] as f32 / 255.0;
        }
    }
    Ok(Tex { w, h, px })
}

fn alpha_plane(a: &[u8], w: usize, h: usize) -> Result<Vec<u8>, String> {
    if a.is_empty() {
        return Err("empty WebP alpha chunk".into());
    }
    let method = a[0] & 3;
    let filter = (a[0] >> 2) & 3;
    let data = &a[1..];
    let mut plane = vec![0u8; w * h];
    match method {
        0 => {
            if data.len() < w * h {
                return Err("truncated WebP alpha data".into());
            }
            plane.copy_from_slice(&data[..w * h]);
        }
        1 => {
            let mut br = LBits::new(data);
            let px = lossless_image(&mut br, w, h)?;
            for i in 0..w * h {
                plane[i] = ((px[i] >> 8) & 255) as u8;
            }
        }
        _ => return Err("unsupported WebP alpha compression".into()),
    }
    if filter != 0 {
        for y in 0..h {
            for x in 0..w {
                let i = y * w + x;
                let pred: i32 = if y == 0 {
                    if x == 0 {
                        0
                    } else {
                        plane[i - 1] as i32
                    }
                } else if x == 0 {
                    plane[i - w] as i32
                } else {
                    let (l, t, tl) = (plane[i - 1] as i32, plane[i - w] as i32, plane[i - w - 1] as i32);
                    match filter {
                        1 => l,
                        2 => t,
                        _ => (l + t - tl).clamp(0, 255),
                    }
                };
                plane[i] = (plane[i] as i32 + pred) as u8;
            }
        }
    }
    Ok(plane)
}
