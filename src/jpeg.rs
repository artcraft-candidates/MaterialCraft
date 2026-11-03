// SPDX-License-Identifier: MIT OR Apache-2.0
//! Dependency-free baseline JPEG encoder (4:4:4, standard Huffman tables).
use crate::tex::Tex;

const ZIGZAG: [usize; 64] = [
    0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27, 20, 13, 6, 7, 14, 21, 28, 35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51, 58, 59, 52,
    45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63,
];

const QL: [u8; 64] = [
    16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55, 14, 13, 16, 24, 40, 57, 69, 56, 14, 17, 22, 29, 51, 87, 80, 62, 18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113, 92,
    49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99,
];
const QC: [u8; 64] = [
    17, 18, 24, 47, 99, 99, 99, 99, 18, 21, 26, 66, 99, 99, 99, 99, 24, 26, 56, 99, 99, 99, 99, 99, 47, 66, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99,
];

const DC_L_BITS: [u8; 16] = [0, 1, 5, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0];
const DC_C_BITS: [u8; 16] = [0, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0];
const DC_VALS: [u8; 12] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];
const AC_L_BITS: [u8; 16] = [0, 2, 1, 3, 3, 2, 4, 3, 5, 5, 4, 4, 0, 0, 1, 0x7d];
const AC_L_VALS: [u8; 162] = [
    0x01, 0x02, 0x03, 0x00, 0x04, 0x11, 0x05, 0x12, 0x21, 0x31, 0x41, 0x06, 0x13, 0x51, 0x61, 0x07, 0x22, 0x71, 0x14, 0x32, 0x81, 0x91, 0xa1, 0x08, 0x23, 0x42, 0xb1, 0xc1, 0x15, 0x52, 0xd1, 0xf0,
    0x24, 0x33, 0x62, 0x72, 0x82, 0x09, 0x0a, 0x16, 0x17, 0x18, 0x19, 0x1a, 0x25, 0x26, 0x27, 0x28, 0x29, 0x2a, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3a, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49,
    0x4a, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59, 0x5a, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69, 0x6a, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79, 0x7a, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89,
    0x8a, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99, 0x9a, 0xa2, 0xa3, 0xa4, 0xa5, 0xa6, 0xa7, 0xa8, 0xa9, 0xaa, 0xb2, 0xb3, 0xb4, 0xb5, 0xb6, 0xb7, 0xb8, 0xb9, 0xba, 0xc2, 0xc3, 0xc4, 0xc5,
    0xc6, 0xc7, 0xc8, 0xc9, 0xca, 0xd2, 0xd3, 0xd4, 0xd5, 0xd6, 0xd7, 0xd8, 0xd9, 0xda, 0xe1, 0xe2, 0xe3, 0xe4, 0xe5, 0xe6, 0xe7, 0xe8, 0xe9, 0xea, 0xf1, 0xf2, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7, 0xf8,
    0xf9, 0xfa,
];
const AC_C_BITS: [u8; 16] = [0, 2, 1, 2, 4, 4, 3, 4, 7, 5, 4, 4, 0, 1, 2, 0x77];
const AC_C_VALS: [u8; 162] = [
    0x00, 0x01, 0x02, 0x03, 0x11, 0x04, 0x05, 0x21, 0x31, 0x06, 0x12, 0x41, 0x51, 0x07, 0x61, 0x71, 0x13, 0x22, 0x32, 0x81, 0x08, 0x14, 0x42, 0x91, 0xa1, 0xb1, 0xc1, 0x09, 0x23, 0x33, 0x52, 0xf0,
    0x15, 0x62, 0x72, 0xd1, 0x0a, 0x16, 0x24, 0x34, 0xe1, 0x25, 0xf1, 0x17, 0x18, 0x19, 0x1a, 0x26, 0x27, 0x28, 0x29, 0x2a, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3a, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48,
    0x49, 0x4a, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59, 0x5a, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69, 0x6a, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79, 0x7a, 0x82, 0x83, 0x84, 0x85, 0x86, 0x87,
    0x88, 0x89, 0x8a, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99, 0x9a, 0xa2, 0xa3, 0xa4, 0xa5, 0xa6, 0xa7, 0xa8, 0xa9, 0xaa, 0xb2, 0xb3, 0xb4, 0xb5, 0xb6, 0xb7, 0xb8, 0xb9, 0xba, 0xc2, 0xc3,
    0xc4, 0xc5, 0xc6, 0xc7, 0xc8, 0xc9, 0xca, 0xd2, 0xd3, 0xd4, 0xd5, 0xd6, 0xd7, 0xd8, 0xd9, 0xda, 0xe2, 0xe3, 0xe4, 0xe5, 0xe6, 0xe7, 0xe8, 0xe9, 0xea, 0xf2, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7, 0xf8,
    0xf9, 0xfa,
];

/// Huffman table: DHT description + per-symbol (code, length).
struct Huff {
    bits: [u8; 16],
    vals: Vec<u8>,
    code: [(u16, u8); 256],
}

fn build(bits: &[u8; 16], vals: &[u8]) -> Huff {
    let mut code = [(0u16, 0u8); 256];
    let (mut c, mut k) = (0u16, 0usize);
    for len in 1..=16u8 {
        for _ in 0..bits[len as usize - 1] {
            code[vals[k] as usize] = (c, len);
            c += 1;
            k += 1;
        }
        c <<= 1;
    }
    Huff { bits: *bits, vals: vals.to_vec(), code }
}

/// True when the standard AC tables contain every symbol the encoder can emit.
pub fn tables_complete() -> bool {
    let need: Vec<u8> = std::iter::once(0x00).chain(std::iter::once(0xf0)).chain((0..16u8).flat_map(|r| (1..=10u8).map(move |s| r << 4 | s))).collect();
    for (bits, vals) in [(&AC_L_BITS, &AC_L_VALS[..]), (&AC_C_BITS, &AC_C_VALS[..])] {
        let total: usize = bits.iter().map(|b| *b as usize).sum();
        let h = build(bits, vals);
        if total != vals.len() || need.iter().any(|n| h.code[*n as usize].1 == 0) {
            return false;
        }
    }
    true
}

/// Fallback tables: every symbol gets an 8-bit code (valid, slightly larger files).
fn uniform_ac() -> Huff {
    let vals: Vec<u8> = std::iter::once(0x00).chain(std::iter::once(0xf0)).chain((0..16u8).flat_map(|r| (1..=10u8).map(move |s| r << 4 | s))).collect();
    let mut bits = [0u8; 16];
    bits[7] = vals.len() as u8;
    build(&bits, &vals)
}

struct Bits {
    out: Vec<u8>,
    acc: u32,
    n: u32,
}

impl Bits {
    fn put(&mut self, code: u32, len: u32) {
        if len == 0 {
            return;
        }
        self.acc = (self.acc << len) | (code & ((1 << len) - 1));
        self.n += len;
        while self.n >= 8 {
            let b = ((self.acc >> (self.n - 8)) & 0xff) as u8;
            self.out.push(b);
            if b == 0xff {
                self.out.push(0);
            }
            self.n -= 8;
        }
    }
    fn flush(&mut self) {
        if self.n > 0 {
            let pad = 8 - self.n;
            self.put((1 << pad) - 1, pad);
        }
    }
}

fn category(v: i32) -> (u32, u32) {
    let a = v.unsigned_abs();
    let size = 32 - a.leading_zeros();
    let bits = if v < 0 { (v - 1) as u32 & ((1u32 << size) - 1).max(0) } else { v as u32 };
    (size, bits)
}

fn quant_table(base: &[u8; 64], quality: u8) -> [u8; 64] {
    let q = quality.clamp(1, 100) as i32;
    let scale = if q < 50 { 5000 / q } else { 200 - 2 * q };
    let mut t = [1u8; 64];
    for i in 0..64 {
        t[i] = ((base[i] as i32 * scale + 50) / 100).clamp(1, 255) as u8;
    }
    t
}

fn dct_table() -> [[f32; 8]; 8] {
    let mut t = [[0.0f32; 8]; 8];
    for (u, row) in t.iter_mut().enumerate() {
        for (x, v) in row.iter_mut().enumerate() {
            let cu = if u == 0 { std::f32::consts::FRAC_1_SQRT_2 } else { 1.0 };
            *v = 0.5 * cu * ((2 * x + 1) as f32 * u as f32 * std::f32::consts::PI / 16.0).cos();
        }
    }
    t
}

fn seg(out: &mut Vec<u8>, marker: u8, body: &[u8]) {
    out.extend_from_slice(&[0xff, marker]);
    out.extend_from_slice(&((body.len() + 2) as u16).to_be_bytes());
    out.extend_from_slice(body);
}

/// Encode as baseline JPEG. `quality` 1..=100. Alpha is ignored.
pub fn encode(t: &Tex, quality: u8) -> Vec<u8> {
    let (w, h) = (t.w, t.h);
    let (ql, qc) = (quant_table(&QL, quality), quant_table(&QC, quality));
    let dct = dct_table();
    let (ac_l, ac_c) = if tables_complete() { (build(&AC_L_BITS, &AC_L_VALS), build(&AC_C_BITS, &AC_C_VALS)) } else { (uniform_ac(), uniform_ac()) };
    let (dc_l, dc_c) = (build(&DC_L_BITS, &DC_VALS), build(&DC_C_BITS, &DC_VALS));

    let mut out = vec![0xff, 0xd8];
    seg(&mut out, 0xe0, &[b'J', b'F', b'I', b'F', 0, 1, 1, 0, 0, 1, 0, 1, 0, 0]);
    for (id, q) in [(0u8, &ql), (1u8, &qc)] {
        let mut b = vec![id];
        b.extend(ZIGZAG.iter().map(|&z| q[z]));
        seg(&mut out, 0xdb, &b);
    }
    let mut sof = vec![8u8];
    sof.extend_from_slice(&(h as u16).to_be_bytes());
    sof.extend_from_slice(&(w as u16).to_be_bytes());
    sof.push(3);
    for (id, q) in [(1u8, 0u8), (2, 1), (3, 1)] {
        sof.extend_from_slice(&[id, 0x11, q]);
    }
    seg(&mut out, 0xc0, &sof);
    for (class_id, hf) in [(0x00u8, &dc_l), (0x10, &ac_l), (0x01, &dc_c), (0x11, &ac_c)] {
        let mut b = vec![class_id];
        b.extend_from_slice(&hf.bits);
        b.extend_from_slice(&hf.vals);
        seg(&mut out, 0xc4, &b);
    }
    seg(&mut out, 0xda, &[3, 1, 0x00, 2, 0x11, 3, 0x11, 0, 63, 0]);

    // colour conversion into planes
    let (bw, bh) = ((w + 7) / 8, (h + 7) / 8);
    let mut planes = vec![vec![0.0f32; bw * 8 * bh * 8]; 3];
    let stride = bw * 8;
    for y in 0..bh * 8 {
        for x in 0..bw * 8 {
            let p = t.px[(y.min(h - 1)) * w + x.min(w - 1)];
            let (r, g, b) = (p[0].clamp(0.0, 1.0) * 255.0, p[1].clamp(0.0, 1.0) * 255.0, p[2].clamp(0.0, 1.0) * 255.0);
            planes[0][y * stride + x] = 0.299 * r + 0.587 * g + 0.114 * b - 128.0;
            planes[1][y * stride + x] = -0.168_736 * r - 0.331_264 * g + 0.5 * b;
            planes[2][y * stride + x] = 0.5 * r - 0.418_688 * g - 0.081_312 * b;
        }
    }
    let mut bits = Bits { out: Vec::with_capacity(w * h / 2), acc: 0, n: 0 };
    let mut last_dc = [0i32; 3];
    for by in 0..bh {
        for bx in 0..bw {
            for c in 0..3 {
                let (q, dch, ach) = if c == 0 { (&ql, &dc_l, &ac_l) } else { (&qc, &dc_c, &ac_c) };
                let mut blk = [0.0f32; 64];
                for y in 0..8 {
                    for x in 0..8 {
                        blk[y * 8 + x] = planes[c][(by * 8 + y) * stride + bx * 8 + x];
                    }
                }
                // separable DCT
                let mut tmp = [0.0f32; 64];
                for y in 0..8 {
                    for u in 0..8 {
                        let mut s = 0.0;
                        for x in 0..8 {
                            s += blk[y * 8 + x] * dct[u][x];
                        }
                        tmp[y * 8 + u] = s;
                    }
                }
                let mut co = [0i32; 64];
                for u in 0..8 {
                    for v in 0..8 {
                        let mut s = 0.0;
                        for y in 0..8 {
                            s += tmp[y * 8 + u] * dct[v][y];
                        }
                        co[v * 8 + u] = (s / q[v * 8 + u] as f32).round() as i32;
                    }
                }
                let dc = co[0];
                let diff = dc - last_dc[c];
                last_dc[c] = dc;
                let (size, v) = category(diff);
                let (code, len) = dch.code[size as usize];
                bits.put(code as u32, len as u32);
                bits.put(v, size);
                let mut run = 0;
                for k in 1..64 {
                    let a = co[ZIGZAG[k]];
                    if a == 0 {
                        run += 1;
                        continue;
                    }
                    while run > 15 {
                        let (code, len) = ach.code[0xf0];
                        bits.put(code as u32, len as u32);
                        run -= 16;
                    }
                    let (size, v) = category(a);
                    let (code, len) = ach.code[(run << 4 | size) as usize];
                    bits.put(code as u32, len as u32);
                    bits.put(v, size);
                    run = 0;
                }
                if run > 0 {
                    let (code, len) = ach.code[0x00];
                    bits.put(code as u32, len as u32);
                }
            }
        }
    }
    bits.flush();
    out.extend_from_slice(&bits.out);
    out.extend_from_slice(&[0xff, 0xd9]);
    out
}
