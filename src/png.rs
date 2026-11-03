// SPDX-License-Identifier: MIT OR Apache-2.0
//! Dependency-free PNG encoder (8/16-bit, RGB or RGBA; adaptive row filters and deflate with dynamic Huffman codes)
//! and decoder.
use std::sync::OnceLock;

fn crc_table() -> &'static [u32; 256] {
    static T: OnceLock<[u32; 256]> = OnceLock::new();
    T.get_or_init(|| {
        let mut t = [0u32; 256];
        for (n, slot) in t.iter_mut().enumerate() {
            let mut c = n as u32;
            for _ in 0..8 {
                c = if c & 1 != 0 { 0xedb8_8320 ^ (c >> 1) } else { c >> 1 };
            }
            *slot = c;
        }
        t
    })
}

fn crc32(parts: &[&[u8]]) -> u32 {
    let t = crc_table();
    let mut c = 0xffff_ffffu32;
    for p in parts {
        for &b in *p {
            c = t[((c ^ b as u32) & 0xff) as usize] ^ (c >> 8);
        }
    }
    c ^ 0xffff_ffff
}

fn adler32(data: &[u8]) -> u32 {
    let (mut a, mut b) = (1u32, 0u32);
    for &d in data {
        a = (a + d as u32) % 65521;
        b = (b + a) % 65521;
    }
    (b << 16) | a
}

fn chunk(out: &mut Vec<u8>, kind: &[u8; 4], data: &[u8]) {
    out.extend_from_slice(&(data.len() as u32).to_be_bytes());
    out.extend_from_slice(kind);
    out.extend_from_slice(data);
    out.extend_from_slice(&crc32(&[kind, data]).to_be_bytes());
}

/// Pick the best of the five PNG row filters for every row (smallest sum of absolute residuals) and apply it.
fn filter_rows(plain: &[u8], h: usize, rb: usize, bpp: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(h * (rb + 1));
    let zero = vec![0u8; rb];
    let mut cand: Vec<Vec<u8>> = (0..5).map(|_| vec![0u8; rb]).collect();
    for y in 0..h {
        let row = &plain[y * rb..(y + 1) * rb];
        let prev = if y == 0 { &zero[..] } else { &plain[(y - 1) * rb..y * rb] };
        for i in 0..rb {
            let a = if i >= bpp { row[i - bpp] } else { 0 };
            let b = prev[i];
            let c = if i >= bpp { prev[i - bpp] } else { 0 };
            let x = row[i];
            cand[0][i] = x;
            cand[1][i] = x.wrapping_sub(a);
            cand[2][i] = x.wrapping_sub(b);
            cand[3][i] = x.wrapping_sub(((a as u16 + b as u16) / 2) as u8);
            let (pa, pb, pc) = ((b as i32 - c as i32).abs(), (a as i32 - c as i32).abs(), (a as i32 + b as i32 - 2 * c as i32).abs());
            let pred = if pa <= pb && pa <= pc { a } else if pb <= pc { b } else { c };
            cand[4][i] = x.wrapping_sub(pred);
        }
        let cost = |v: &Vec<u8>| -> u64 { v.iter().map(|&d| (d as i8).unsigned_abs() as u64).sum() };
        let best = (0..5).min_by_key(|&f| cost(&cand[f])).unwrap_or(0);
        out.push(best as u8);
        out.extend_from_slice(&cand[best]);
    }
    out
}

// ---- deflate (LZ77 with hash chains and lazy matching, dynamic Huffman blocks)
struct BitW {
    out: Vec<u8>,
    acc: u64,
    n: u32,
}

impl BitW {
    fn put(&mut self, v: u32, bits: u32) {
        self.acc |= (v as u64) << self.n;
        self.n += bits;
        while self.n >= 8 {
            self.out.push(self.acc as u8);
            self.acc >>= 8;
            self.n -= 8;
        }
    }
    /// Huffman codes are sent most significant bit first
    fn code(&mut self, code: u32, len: u32) {
        let mut r = 0u32;
        for i in 0..len {
            r |= ((code >> i) & 1) << (len - 1 - i);
        }
        self.put(r, len);
    }
    fn flush(&mut self) {
        if self.n > 0 {
            self.out.push(self.acc as u8);
            self.acc = 0;
            self.n = 0;
        }
    }
}

const LEN_BASE: [u16; 29] = [3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99, 115, 131, 163, 195, 227, 258];
const LEN_EXTRA: [u8; 29] = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0];
const DIST_BASE: [u16; 30] = [1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769, 1025, 1537, 2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577];
const DIST_EXTRA: [u8; 30] = [0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13];

fn len_code(l: usize) -> usize {
    LEN_BASE.iter().rposition(|&b| b as usize <= l).unwrap_or(0)
}

fn dist_code(d: usize) -> usize {
    DIST_BASE.iter().rposition(|&b| b as usize <= d).unwrap_or(0)
}

/// Huffman code lengths for the given symbol counts, none longer than `limit` (counts are flattened until they fit).
fn huff_lengths(freq: &[u32], limit: u8) -> Vec<u8> {
    let n = freq.len();
    let mut f: Vec<u64> = freq.iter().map(|&x| x as u64).collect();
    loop {
        let used: Vec<usize> = (0..n).filter(|&i| f[i] > 0).collect();
        let mut len = vec![0u8; n];
        if used.is_empty() {
            return len;
        }
        if used.len() == 1 {
            len[used[0]] = 1;
            return len;
        }
        // nodes: (weight, left, right) with leaves first
        let mut w: Vec<u64> = used.iter().map(|&i| f[i]).collect();
        let mut parent: Vec<usize> = vec![usize::MAX; used.len()];
        let mut heap: std::collections::BinaryHeap<std::cmp::Reverse<(u64, usize)>> = (0..used.len()).map(|i| std::cmp::Reverse((w[i], i))).collect();
        while heap.len() > 1 {
            let std::cmp::Reverse((wa, a)) = heap.pop().unwrap();
            let std::cmp::Reverse((wb, b)) = heap.pop().unwrap();
            let id = w.len();
            w.push(wa + wb);
            parent.push(usize::MAX);
            parent[a] = id;
            parent[b] = id;
            heap.push(std::cmp::Reverse((wa + wb, id)));
        }
        let mut ok = true;
        for (k, &sym) in used.iter().enumerate() {
            let (mut d, mut at) = (0u32, k);
            while parent[at] != usize::MAX {
                at = parent[at];
                d += 1;
            }
            if d > limit as u32 {
                ok = false;
            }
            len[sym] = d.min(255) as u8;
        }
        if ok {
            return len;
        }
        for x in f.iter_mut() {
            if *x > 0 {
                *x = (*x / 2).max(1);
            }
        }
    }
}

/// Canonical codes for the given lengths.
fn huff_codes(len: &[u8]) -> Vec<u32> {
    let mut count = [0u32; 16];
    for &l in len {
        count[l as usize] += 1;
    }
    count[0] = 0;
    let mut next = [0u32; 16];
    let mut code = 0u32;
    for b in 1..16 {
        code = (code + count[b - 1]) << 1;
        next[b] = code;
    }
    len.iter()
        .map(|&l| {
            if l == 0 {
                0
            } else {
                let c = next[l as usize];
                next[l as usize] += 1;
                c
            }
        })
        .collect()
}

#[derive(Clone, Copy)]
enum Tok {
    Lit(u8),
    Match(u16, u16),
}

fn lz77(data: &[u8]) -> Vec<Tok> {
    const WIN: usize = 32768;
    const HBITS: u32 = 15;
    const MAX_CHAIN: usize = 24;
    const NICE: usize = 128;
    let n = data.len();
    let mut toks = Vec::with_capacity(n / 2 + 16);
    let mut head = vec![u32::MAX; 1 << HBITS];
    let mut prev = vec![u32::MAX; WIN];
    let hash = |i: usize| -> usize { ((data[i] as u32) << 10 ^ (data[i + 1] as u32) << 5 ^ data[i + 2] as u32).wrapping_mul(2654435761) as usize >> (32 - HBITS) as usize & ((1 << HBITS) - 1) };
    let insert = |head: &mut Vec<u32>, prev: &mut Vec<u32>, i: usize| {
        if i + 2 < n {
            let h = hash(i);
            prev[i & (WIN - 1)] = head[h];
            head[h] = i as u32;
        }
    };
    let find = |head: &Vec<u32>, prev: &Vec<u32>, i: usize| -> (usize, usize) {
        if i + 2 >= n {
            return (0, 0);
        }
        let maxl = (n - i).min(258);
        let (mut best_l, mut best_d) = (0usize, 0usize);
        let mut cand = head[hash(i)];
        let mut chain = 0;
        while cand != u32::MAX && chain < MAX_CHAIN {
            let c = cand as usize;
            if c >= i || i - c > WIN {
                break;
            }
            if data[c + best_l.min(maxl - 1)] == data[i + best_l.min(maxl - 1)] {
                let mut l = 0;
                while l < maxl && data[c + l] == data[i + l] {
                    l += 1;
                }
                if l > best_l {
                    best_l = l;
                    best_d = i - c;
                    if l >= NICE || l == maxl {
                        break;
                    }
                }
            }
            let nx = prev[c & (WIN - 1)];
            if nx != u32::MAX && nx as usize >= c {
                break;
            }
            cand = nx;
            chain += 1;
        }
        if best_l >= 3 {
            (best_l, best_d)
        } else {
            (0, 0)
        }
    };
    let mut i = 0;
    while i < n {
        let (l, d) = find(&head, &prev, i);
        if l >= 3 {
            // lazy matching: a longer match one byte later wins
            insert(&mut head, &mut prev, i);
            let (l2, _) = if l < NICE { find(&head, &prev, i + 1) } else { (0, 0) };
            if l2 > l + 1 {
                toks.push(Tok::Lit(data[i]));
                i += 1;
                continue;
            }
            toks.push(Tok::Match(l as u16, d as u16));
            for k in 1..l {
                insert(&mut head, &mut prev, i + k);
            }
            i += l;
        } else {
            insert(&mut head, &mut prev, i);
            toks.push(Tok::Lit(data[i]));
            i += 1;
        }
    }
    toks
}

/// A raw deflate stream (RFC 1951) of `data`.
fn deflate(data: &[u8]) -> Vec<u8> {
    let toks = lz77(data);
    let mut bw = BitW { out: Vec::with_capacity(data.len() / 3 + 64), acc: 0, n: 0 };
    let blocks: Vec<&[Tok]> = if toks.is_empty() { vec![&toks[..]] } else { toks.chunks(1 << 16).collect() };
    let nb = blocks.len();
    for (bi, blk) in blocks.into_iter().enumerate() {
        let mut lf = vec![0u32; 286];
        let mut df = vec![0u32; 30];
        for t in blk {
            match *t {
                Tok::Lit(b) => lf[b as usize] += 1,
                Tok::Match(l, d) => {
                    lf[257 + len_code(l as usize)] += 1;
                    df[dist_code(d as usize)] += 1;
                }
            }
        }
        lf[256] = 1;
        if df.iter().all(|&x| x == 0) {
            df[0] = 1;
        }
        let ll = huff_lengths(&lf, 15);
        let dl = huff_lengths(&df, 15);
        let (lc, dc) = (huff_codes(&ll), huff_codes(&dl));
        let hlit = ll.iter().rposition(|&l| l > 0).map(|p| p + 1).unwrap_or(257).max(257);
        let hdist = dl.iter().rposition(|&l| l > 0).map(|p| p + 1).unwrap_or(1).max(1);
        // code length alphabet: the lengths are written without run-length symbols
        let mut cf = vec![0u32; 19];
        for &l in ll[..hlit].iter().chain(dl[..hdist].iter()) {
            cf[l as usize] += 1;
        }
        let cl = huff_lengths(&cf, 7);
        let cc = huff_codes(&cl);
        const ORDER: [usize; 19] = [16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15];
        let hclen = ORDER.iter().rposition(|&o| cl[o] > 0).map(|p| p + 1).unwrap_or(4).max(4);
        bw.put(if bi + 1 == nb { 1 } else { 0 }, 1);
        bw.put(2, 2);
        bw.put((hlit - 257) as u32, 5);
        bw.put((hdist - 1) as u32, 5);
        bw.put((hclen - 4) as u32, 4);
        for &o in &ORDER[..hclen] {
            bw.put(cl[o] as u32, 3);
        }
        for &l in ll[..hlit].iter().chain(dl[..hdist].iter()) {
            bw.code(cc[l as usize], cl[l as usize] as u32);
        }
        for t in blk {
            match *t {
                Tok::Lit(b) => bw.code(lc[b as usize], ll[b as usize] as u32),
                Tok::Match(l, d) => {
                    let (l, d) = (l as usize, d as usize);
                    let k = len_code(l);
                    bw.code(lc[257 + k], ll[257 + k] as u32);
                    bw.put((l - LEN_BASE[k] as usize) as u32, LEN_EXTRA[k] as u32);
                    let k = dist_code(d);
                    bw.code(dc[k], dl[k] as u32);
                    bw.put((d - DIST_BASE[k] as usize) as u32, DIST_EXTRA[k] as u32);
                }
            }
        }
        bw.code(lc[256], ll[256] as u32);
    }
    bw.flush();
    bw.out
}

/// Encode RGBA float pixels (0..1). `depth` must be 8 or 16 (anything else becomes 8).
pub fn encode(w: usize, h: usize, px: &[[f32; 4]], depth: u8) -> Vec<u8> {
    let depth = if depth == 16 { 16 } else { 8 };
    let opaque = px.iter().all(|p| p[3] >= 0.9999);
    let ch = if opaque { 3 } else { 4 };
    let bps = (depth / 8) as usize;
    let row_bytes = w * ch * bps;
    let mut plain = vec![0u8; h * row_bytes];
    for y in 0..h {
        let mut o = y * row_bytes;
        for x in 0..w {
            let p = px.get(y * w + x).copied().unwrap_or([0.0; 4]);
            for c in p.iter().take(ch) {
                let v = c.clamp(0.0, 1.0);
                if depth == 16 {
                    plain[o..o + 2].copy_from_slice(&((v * 65535.0 + 0.5) as u16).to_be_bytes());
                    o += 2;
                } else {
                    plain[o] = (v * 255.0 + 0.5) as u8;
                    o += 1;
                }
            }
        }
    }
    let raw = filter_rows(&plain, h, row_bytes, ch * bps);
    let mut z = vec![0x78, 0x9c];
    z.extend_from_slice(&deflate(&raw));
    z.extend_from_slice(&adler32(&raw).to_be_bytes());

    let mut out = vec![0x89, b'P', b'N', b'G', 0x0d, 0x0a, 0x1a, 0x0a];
    let mut ihdr = Vec::new();
    ihdr.extend_from_slice(&(w as u32).to_be_bytes());
    ihdr.extend_from_slice(&(h as u32).to_be_bytes());
    ihdr.extend_from_slice(&[depth, if opaque { 2 } else { 6 }, 0, 0, 0]);
    chunk(&mut out, b"IHDR", &ihdr);
    chunk(&mut out, b"IDAT", &z);
    chunk(&mut out, b"IEND", &[]);
    out
}

// ======================= decoder =======================
use crate::tex::Tex;

struct BitR<'a> {
    d: &'a [u8],
    pos: usize,
    bit: u32,
    cnt: u32,
}

impl<'a> BitR<'a> {
    fn bits(&mut self, n: u32) -> Result<u32, String> {
        while self.cnt < n {
            let b = *self.d.get(self.pos).ok_or_else(|| "unexpected end of compressed data".to_string())?;
            self.bit |= (b as u32) << self.cnt;
            self.pos += 1;
            self.cnt += 8;
        }
        let v = self.bit & ((1u32 << n) - 1);
        self.bit >>= n;
        self.cnt -= n;
        Ok(v)
    }
}

struct Huff {
    count: [u16; 16],
    symbol: Vec<u16>,
}

fn build(lengths: &[u8]) -> Huff {
    let mut count = [0u16; 16];
    for &l in lengths {
        count[l as usize & 15] += 1;
    }
    let mut offs = [0u16; 16];
    for i in 1..15 {
        offs[i + 1] = offs[i] + count[i];
    }
    let mut symbol = vec![0u16; lengths.len()];
    for (s, &l) in lengths.iter().enumerate() {
        if l != 0 {
            symbol[offs[l as usize & 15] as usize] = s as u16;
            offs[l as usize & 15] += 1;
        }
    }
    Huff { count, symbol }
}

fn decode_sym(br: &mut BitR, h: &Huff) -> Result<u16, String> {
    let (mut code, mut first, mut index) = (0i32, 0i32, 0i32);
    for len in 1..16 {
        code |= br.bits(1)? as i32;
        let count = h.count[len] as i32;
        if code - count < first {
            return h.symbol.get((index + (code - first)) as usize).copied().ok_or_else(|| "bad huffman code".to_string());
        }
        index += count;
        first += count;
        first <<= 1;
        code <<= 1;
    }
    Err("bad huffman code".into())
}

const LBASE: [u16; 29] = [3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99, 115, 131, 163, 195, 227, 258];
const LEXT: [u8; 29] = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0];
const DBASE: [u16; 30] = [
    1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769, 1025, 1537, 2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577,
];
const DEXT: [u8; 30] = [0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13];

fn codes(br: &mut BitR, out: &mut Vec<u8>, lh: &Huff, dh: &Huff) -> Result<(), String> {
    loop {
        let sym = decode_sym(br, lh)? as usize;
        if sym < 256 {
            out.push(sym as u8);
        } else if sym == 256 {
            return Ok(());
        } else {
            let s = sym - 257;
            if s >= 29 {
                return Err("bad length symbol".into());
            }
            let len = LBASE[s] as usize + br.bits(LEXT[s] as u32)? as usize;
            let ds = decode_sym(br, dh)? as usize;
            if ds >= 30 {
                return Err("bad distance symbol".into());
            }
            let dist = DBASE[ds] as usize + br.bits(DEXT[ds] as u32)? as usize;
            if dist > out.len() {
                return Err("distance too far back".into());
            }
            for _ in 0..len {
                let b = out[out.len() - dist];
                out.push(b);
            }
        }
        if out.len() > (1 << 30) {
            return Err("decompressed data too large".into());
        }
    }
}

pub fn inflate(data: &[u8]) -> Result<Vec<u8>, String> {
    let mut br = BitR { d: data, pos: 0, bit: 0, cnt: 0 };
    let mut out = Vec::new();
    loop {
        let last = br.bits(1)?;
        match br.bits(2)? {
            0 => {
                br.bit = 0;
                br.cnt = 0;
                let p = br.pos;
                if p + 4 > data.len() {
                    return Err("truncated stored block".into());
                }
                let len = u16::from_le_bytes([data[p], data[p + 1]]) as usize;
                let nlen = u16::from_le_bytes([data[p + 2], data[p + 3]]) as usize;
                if len != (!nlen & 0xffff) || p + 4 + len > data.len() {
                    return Err("bad stored block".into());
                }
                out.extend_from_slice(&data[p + 4..p + 4 + len]);
                br.pos = p + 4 + len;
            }
            1 => {
                let mut l = [0u8; 288];
                for (i, v) in l.iter_mut().enumerate() {
                    *v = if i < 144 { 8 } else if i < 256 { 9 } else if i < 280 { 7 } else { 8 };
                }
                codes(&mut br, &mut out, &build(&l), &build(&[5u8; 30]))?;
            }
            2 => {
                let nlen = br.bits(5)? as usize + 257;
                let ndist = br.bits(5)? as usize + 1;
                let ncode = br.bits(4)? as usize + 4;
                if nlen > 286 || ndist > 30 {
                    return Err("bad dynamic header".into());
                }
                const ORDER: [usize; 19] = [16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15];
                let mut lengths = [0u8; 320];
                for &o in ORDER.iter().take(ncode) {
                    lengths[o] = br.bits(3)? as u8;
                }
                let lc = build(&lengths[..19]);
                let mut idx = 0;
                let total = nlen + ndist;
                let mut ll = [0u8; 320];
                while idx < total {
                    let sym = decode_sym(&mut br, &lc)?;
                    if sym < 16 {
                        ll[idx] = sym as u8;
                        idx += 1;
                    } else {
                        let (prev, rep) = match sym {
                            16 => {
                                if idx == 0 {
                                    return Err("bad repeat".into());
                                }
                                (ll[idx - 1], 3 + br.bits(2)? as usize)
                            }
                            17 => (0, 3 + br.bits(3)? as usize),
                            _ => (0, 11 + br.bits(7)? as usize),
                        };
                        if idx + rep > total {
                            return Err("too many lengths".into());
                        }
                        for _ in 0..rep {
                            ll[idx] = prev;
                            idx += 1;
                        }
                    }
                }
                codes(&mut br, &mut out, &build(&ll[..nlen]), &build(&ll[nlen..total]))?;
            }
            _ => return Err("bad block type".into()),
        }
        if last == 1 {
            return Ok(out);
        }
    }
}

fn paeth(a: u8, b: u8, c: u8) -> u8 {
    let (ia, ib, ic) = (a as i32, b as i32, c as i32);
    let p = ia + ib - ic;
    let (pa, pb, pc) = ((p - ia).abs(), (p - ib).abs(), (p - ic).abs());
    if pa <= pb && pa <= pc {
        a
    } else if pb <= pc {
        b
    } else {
        c
    }
}

/// Decode a PNG: gray, gray+alpha, RGB, RGBA and palette images (1-16 bit, tRNS palette alpha, Adam7 interlacing).
pub fn decode(b: &[u8]) -> Result<Tex, String> {
    if b.len() < 8 || &b[..8] != b"\x89PNG\r\n\x1a\n" {
        return Err("not a PNG file".into());
    }
    let mut pos = 8;
    let (mut w, mut h, mut depth, mut ctype, mut inter) = (0usize, 0usize, 0u8, 0u8, 0u8);
    let mut plte: Vec<u8> = Vec::new();
    let mut trns: Vec<u8> = Vec::new();
    let mut idat: Vec<u8> = Vec::new();
    while pos + 8 <= b.len() {
        let len = u32::from_be_bytes([b[pos], b[pos + 1], b[pos + 2], b[pos + 3]]) as usize;
        let kind = &b[pos + 4..pos + 8];
        let end = pos + 8 + len;
        if end + 4 > b.len() {
            return Err("truncated chunk".into());
        }
        let data = &b[pos + 8..end];
        match kind {
            b"IHDR" if data.len() >= 13 => {
                w = u32::from_be_bytes([data[0], data[1], data[2], data[3]]) as usize;
                h = u32::from_be_bytes([data[4], data[5], data[6], data[7]]) as usize;
                depth = data[8];
                ctype = data[9];
                inter = data[12];
            }
            b"PLTE" => plte = data.to_vec(),
            b"tRNS" => trns = data.to_vec(),
            b"IDAT" => idat.extend_from_slice(data),
            b"IEND" => break,
            _ => {}
        }
        pos = end + 4;
    }
    if w == 0 || h == 0 || w > 16384 || h > 16384 {
        return Err("bad image size".into());
    }
    let ch = match (ctype, depth) {
        (0, 1 | 2 | 4 | 8 | 16) => 1,
        (2, 8 | 16) => 3,
        (3, 1 | 2 | 4 | 8) => 1,
        (4, 8 | 16) => 2,
        (6, 8 | 16) => 4,
        _ => return Err(format!("unsupported PNG format (colour type {ctype}, {depth}-bit)")),
    };
    if idat.len() < 6 {
        return Err("missing image data".into());
    }
    let raw = inflate(&idat[2..])?;
    let bits_pp = ch * depth as usize;
    let bpp = ((bits_pp + 7) / 8).max(1);
    let maxv = ((1u32 << depth) - 1) as f32;
    let passes: Vec<(usize, usize, usize, usize)> = if inter == 0 {
        vec![(0, 0, 1, 1)]
    } else {
        vec![(0, 0, 8, 8), (4, 0, 8, 8), (0, 4, 4, 8), (2, 0, 4, 4), (0, 2, 2, 4), (1, 0, 2, 2), (0, 1, 1, 2)]
    };
    let mut px = vec![[0.0f32; 4]; w * h];
    let mut off = 0usize;
    for (x0, y0, dx, dy) in passes {
        if x0 >= w || y0 >= h {
            continue;
        }
        let pw = (w - x0 + dx - 1) / dx;
        let ph = (h - y0 + dy - 1) / dy;
        let stride = (pw * bits_pp + 7) / 8;
        let need = ph * (stride + 1);
        if raw.len() < off + need {
            return Err("image data too short".into());
        }
        let mut img = vec![0u8; ph * stride];
        for y in 0..ph {
            let f = raw[off + y * (stride + 1)];
            let src = &raw[off + y * (stride + 1) + 1..off + (y + 1) * (stride + 1)];
            for i in 0..stride {
                let a = if i >= bpp { img[y * stride + i - bpp] } else { 0 };
                let up = if y > 0 { img[(y - 1) * stride + i] } else { 0 };
                let c = if y > 0 && i >= bpp { img[(y - 1) * stride + i - bpp] } else { 0 };
                let x = src[i];
                img[y * stride + i] = match f {
                    0 => x,
                    1 => x.wrapping_add(a),
                    2 => x.wrapping_add(up),
                    3 => x.wrapping_add(((a as u16 + up as u16) / 2) as u8),
                    4 => x.wrapping_add(paeth(a, up, c)),
                    _ => return Err("bad PNG filter".into()),
                };
            }
        }
        off += need;
        for py in 0..ph {
            let row = &img[py * stride..(py + 1) * stride];
            let sample = |idx: usize| -> u32 {
                match depth {
                    16 => u16::from_be_bytes([row[idx * 2], row[idx * 2 + 1]]) as u32,
                    8 => row[idx] as u32,
                    d => {
                        let bit = idx * d as usize;
                        let shift = 8 - d as usize - (bit % 8);
                        ((row[bit / 8] >> shift) & ((1u8 << d) - 1)) as u32
                    }
                }
            };
            for pxi in 0..pw {
                let v = |k: usize| sample(pxi * ch + k) as f32 / maxv;
                let p = match ctype {
                    0 => [v(0), v(0), v(0), 1.0],
                    4 => [v(0), v(0), v(0), v(1)],
                    2 => [v(0), v(1), v(2), 1.0],
                    6 => [v(0), v(1), v(2), v(3)],
                    _ => {
                        let i = sample(pxi) as usize;
                        if i * 3 + 2 >= plte.len() {
                            return Err("palette index out of range".into());
                        }
                        [plte[i * 3] as f32 / 255.0, plte[i * 3 + 1] as f32 / 255.0, plte[i * 3 + 2] as f32 / 255.0, trns.get(i).map(|a| *a as f32 / 255.0).unwrap_or(1.0)]
                    }
                };
                px[(y0 + py * dy) * w + x0 + pxi * dx] = p;
            }
        }
    }
    Ok(Tex { w, h, px })
}
