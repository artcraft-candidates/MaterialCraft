// SPDX-License-Identifier: MIT OR Apache-2.0
//! DirectDraw Surface (`.dds`) reader. Only the first image (top mip level, first array slice or cube face) is read.
//!
//! Formats: uncompressed integer RGB / RGBA / luminance / alpha layouts (bit masks), DXGI 8, 10 (packed), 16 bit
//! integer and 16 / 32 bit float layouts, BC1 to BC5 (DXT1 to DXT5, ATI1 / ATI2), BC6H (unsigned and signed) and BC7.
//! Signed BC4 / BC5 and volume textures are not supported.
use crate::dds_tables::*;
use crate::tex::Tex;

pub struct Dds {
    pub tex: Tex,
    pub hdr: bool,
}

fn u32at(b: &[u8], o: usize) -> u32 {
    if o + 4 > b.len() {
        return 0;
    }
    u32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]])
}

#[derive(Clone, Copy, PartialEq, Debug)]
enum Fmt {
    Bc1,
    Bc2,
    Bc3,
    Bc4,
    Bc5,
    Bc6u,
    Bc6s,
    Bc7,
    /// R, G, B, A masks and bits per pixel
    Masked([u32; 4], u32, bool),
    /// channel count, bytes per channel, float
    Chan(usize, usize, bool),
    Rgb10A2,
    Bgra(bool),
    A8,
    B565,
    B5551,
}

pub fn decode(b: &[u8]) -> Result<Dds, String> {
    if b.len() < 128 || &b[..4] != b"DDS " || u32at(b, 4) != 124 {
        return Err("not a DDS file".into());
    }
    let flags = u32at(b, 8);
    let h = u32at(b, 12) as usize;
    let w = u32at(b, 16) as usize;
    crate::imgio::check_size(w, h)?;
    let pitch = u32at(b, 20) as usize;
    let depth = u32at(b, 24);
    let pf_flags = u32at(b, 80);
    let fourcc = &b[84..88];
    let bits = u32at(b, 88);
    let masks = [u32at(b, 92), u32at(b, 96), u32at(b, 100), u32at(b, 104)];
    if w == 0 || h == 0 || w > 16384 || h > 16384 {
        return Err("bad DDS image size".into());
    }
    if flags & 0x800000 != 0 && depth > 1 {
        return Err("volume DDS textures are not supported".into());
    }
    let mut data_off = 128usize;
    let fmt: Fmt;
    if pf_flags & 4 != 0 {
        fmt = match fourcc {
            b"DXT1" => Fmt::Bc1,
            b"DXT2" | b"DXT3" => Fmt::Bc2,
            b"DXT4" | b"DXT5" => Fmt::Bc3,
            b"ATI1" | b"BC4U" => Fmt::Bc4,
            b"ATI2" | b"BC5U" => Fmt::Bc5,
            b"DX10" => {
                if b.len() < 148 {
                    return Err("truncated DDS header".into());
                }
                data_off = 148;
                match u32at(b, 128) {
                    2 => Fmt::Chan(4, 4, true),
                    16 => Fmt::Chan(2, 4, true),
                    41 => Fmt::Chan(1, 4, true),
                    10 => Fmt::Chan(4, 2, true),
                    34 => Fmt::Chan(2, 2, true),
                    54 => Fmt::Chan(1, 2, true),
                    11 => Fmt::Chan(4, 2, false),
                    35 => Fmt::Chan(2, 2, false),
                    56 => Fmt::Chan(1, 2, false),
                    28 | 29 => Fmt::Chan(4, 1, false),
                    49 => Fmt::Chan(2, 1, false),
                    61 => Fmt::Chan(1, 1, false),
                    65 => Fmt::A8,
                    24 => Fmt::Rgb10A2,
                    85 => Fmt::B565,
                    86 => Fmt::B5551,
                    87 | 91 => Fmt::Bgra(true),
                    88 | 93 => Fmt::Bgra(false),
                    71 | 72 => Fmt::Bc1,
                    74 | 75 => Fmt::Bc2,
                    77 | 78 => Fmt::Bc3,
                    80 => Fmt::Bc4,
                    83 => Fmt::Bc5,
                    81 | 84 => return Err("signed BC4 / BC5 textures are not supported".into()),
                    95 => Fmt::Bc6u,
                    96 => Fmt::Bc6s,
                    98 | 99 => Fmt::Bc7,
                    f => return Err(format!("unsupported DXGI format {f}")),
                }
            }
            _ => {
                // legacy numeric D3DFORMAT codes stored in the FourCC field
                match u32::from_le_bytes([fourcc[0], fourcc[1], fourcc[2], fourcc[3]]) {
                    36 => Fmt::Chan(4, 2, false),
                    111 => Fmt::Chan(1, 2, true),
                    112 => Fmt::Chan(2, 2, true),
                    113 => Fmt::Chan(4, 2, true),
                    114 => Fmt::Chan(1, 4, true),
                    115 => Fmt::Chan(2, 4, true),
                    116 => Fmt::Chan(4, 4, true),
                    _ => return Err(format!("unsupported DDS FourCC '{}'", String::from_utf8_lossy(fourcc))),
                }
            }
        };
    } else if pf_flags & 0x40 != 0 {
        fmt = Fmt::Masked([masks[0], masks[1], masks[2], if pf_flags & 1 != 0 { masks[3] } else { 0 }], bits, false);
    } else if pf_flags & 0x20000 != 0 {
        fmt = Fmt::Masked([masks[0], masks[0], masks[0], if pf_flags & 1 != 0 { masks[3] } else { 0 }], bits, true);
    } else if pf_flags & 2 != 0 {
        fmt = Fmt::A8;
    } else {
        return Err("unsupported DDS pixel format".into());
    }
    let data = &b[data_off..];
    let mut px = vec![[0.0f32, 0.0, 0.0, 1.0]; w * h];
    let mut hdr = false;
    match fmt {
        Fmt::Bc1 | Fmt::Bc2 | Fmt::Bc3 | Fmt::Bc4 | Fmt::Bc5 | Fmt::Bc6u | Fmt::Bc6s | Fmt::Bc7 => {
            let bsize = if matches!(fmt, Fmt::Bc1 | Fmt::Bc4) { 8 } else { 16 };
            let (bw, bh) = ((w + 3) / 4, (h + 3) / 4);
            if data.len() < bw * bh * bsize {
                return Err("truncated DDS data".into());
            }
            let mut blk = [[0.0f32; 4]; 16];
            for by in 0..bh {
                for bx in 0..bw {
                    let src = &data[(by * bw + bx) * bsize..(by * bw + bx + 1) * bsize];
                    match fmt {
                        Fmt::Bc1 => bc1(src, true, &mut blk),
                        Fmt::Bc2 => {
                            bc1(&src[8..], false, &mut blk);
                            for i in 0..16 {
                                let byte = src[i / 2];
                                blk[i][3] = (if i % 2 == 0 { byte & 15 } else { byte >> 4 }) as f32 / 15.0;
                            }
                        }
                        Fmt::Bc3 => {
                            bc1(&src[8..], false, &mut blk);
                            let a = bc4_values(&src[..8]);
                            for i in 0..16 {
                                blk[i][3] = a[i];
                            }
                        }
                        Fmt::Bc4 => {
                            let r = bc4_values(src);
                            for i in 0..16 {
                                blk[i] = [r[i], r[i], r[i], 1.0];
                            }
                        }
                        Fmt::Bc5 => {
                            let r = bc4_values(&src[..8]);
                            let g = bc4_values(&src[8..]);
                            for i in 0..16 {
                                blk[i] = [r[i], g[i], 0.0, 1.0];
                            }
                        }
                        Fmt::Bc6u | Fmt::Bc6s => {
                            hdr = true;
                            bc6h(src, fmt == Fmt::Bc6s, &mut blk);
                        }
                        _ => bc7(src, &mut blk),
                    }
                    for i in 0..16 {
                        let (x, y) = (bx * 4 + i % 4, by * 4 + i / 4);
                        if x < w && y < h {
                            px[y * w + x] = blk[i];
                        }
                    }
                }
            }
        }
        _ => {
            let (bytes_pp, ch_bytes_ok): (usize, bool) = match fmt {
                Fmt::Masked(_, bits, _) => (((bits + 7) / 8) as usize, matches!(bits, 8 | 16 | 24 | 32)),
                Fmt::Chan(n, bps, _) => (n * bps, true),
                Fmt::Rgb10A2 | Fmt::Bgra(_) => (4, true),
                Fmt::A8 => (1, true),
                _ => (2, true),
            };
            if !ch_bytes_ok {
                return Err("unsupported DDS bit depth".into());
            }
            let row = if flags & 8 != 0 && pitch >= w * bytes_pp { pitch } else { w * bytes_pp };
            if data.len() < row * (h - 1) + w * bytes_pp {
                return Err("truncated DDS data".into());
            }
            for y in 0..h {
                for x in 0..w {
                    let s = &data[y * row + x * bytes_pp..y * row + (x + 1) * bytes_pp];
                    px[y * w + x] = match fmt {
                        Fmt::Masked(m, _, lum) => {
                            let mut v = 0u32;
                            for (i, byte) in s.iter().enumerate() {
                                v |= (*byte as u32) << (8 * i);
                            }
                            if lum {
                                let l = mask_val(v, m[0]);
                                [l, l, l, if m[3] != 0 { mask_val(v, m[3]) } else { 1.0 }]
                            } else {
                                [mask_val(v, m[0]), mask_val(v, m[1]), mask_val(v, m[2]), if m[3] != 0 { mask_val(v, m[3]) } else { 1.0 }]
                            }
                        }
                        Fmt::Chan(n, bps, float) => {
                            let mut c = [0.0f32, 0.0, 0.0, 1.0];
                            for k in 0..n {
                                let q = &s[k * bps..(k + 1) * bps];
                                c[k] = match (bps, float) {
                                    (1, _) => q[0] as f32 / 255.0,
                                    (2, true) => crate::exr::half_to_f32(u16::from_le_bytes([q[0], q[1]])),
                                    (2, false) => u16::from_le_bytes([q[0], q[1]]) as f32 / 65535.0,
                                    _ => f32::from_le_bytes([q[0], q[1], q[2], q[3]]),
                                };
                                if !c[k].is_finite() {
                                    c[k] = 0.0;
                                }
                            }
                            if n == 1 {
                                c = [c[0], c[0], c[0], 1.0];
                            }
                            if float {
                                hdr = true;
                            }
                            c
                        }
                        Fmt::Rgb10A2 => {
                            let v = u32::from_le_bytes([s[0], s[1], s[2], s[3]]);
                            [(v & 1023) as f32 / 1023.0, ((v >> 10) & 1023) as f32 / 1023.0, ((v >> 20) & 1023) as f32 / 1023.0, (v >> 30) as f32 / 3.0]
                        }
                        Fmt::Bgra(alpha) => [s[2] as f32 / 255.0, s[1] as f32 / 255.0, s[0] as f32 / 255.0, if alpha { s[3] as f32 / 255.0 } else { 1.0 }],
                        Fmt::A8 => [0.0, 0.0, 0.0, s[0] as f32 / 255.0],
                        Fmt::B565 => {
                            let v = u16::from_le_bytes([s[0], s[1]]) as u32;
                            [mask_val(v, 0xf800), mask_val(v, 0x07e0), mask_val(v, 0x001f), 1.0]
                        }
                        _ => {
                            let v = u16::from_le_bytes([s[0], s[1]]) as u32;
                            [mask_val(v, 0x7c00), mask_val(v, 0x03e0), mask_val(v, 0x001f), mask_val(v, 0x8000)]
                        }
                    };
                }
            }
        }
    }
    Ok(Dds { tex: Tex { w, h, px }, hdr })
}

fn mask_val(v: u32, mask: u32) -> f32 {
    if mask == 0 {
        return 0.0;
    }
    let sh = mask.trailing_zeros();
    let n = (mask >> sh).trailing_ones();
    let m = if n >= 32 { u32::MAX } else { (1u32 << n) - 1 };
    ((v & mask) >> sh) as f32 / m as f32
}

fn c565(v: u16) -> [u32; 3] {
    let (r, g, b) = (((v >> 11) & 31) as u32, ((v >> 5) & 63) as u32, (v & 31) as u32);
    [(r << 3) | (r >> 2), (g << 2) | (g >> 4), (b << 3) | (b >> 2)]
}

/// BC1 colour block. With `punch` set, `c0 <= c1` selects the three colour mode with a transparent fourth entry.
fn bc1(s: &[u8], punch: bool, out: &mut [[f32; 4]; 16]) {
    let c0 = u16::from_le_bytes([s[0], s[1]]);
    let c1 = u16::from_le_bytes([s[2], s[3]]);
    let (a, b) = (c565(c0), c565(c1));
    let mut pal = [[0u32; 4]; 4];
    pal[0] = [a[0], a[1], a[2], 255];
    pal[1] = [b[0], b[1], b[2], 255];
    if c0 > c1 || !punch {
        for k in 0..3 {
            pal[2][k] = (2 * a[k] + b[k]) / 3;
            pal[3][k] = (a[k] + 2 * b[k]) / 3;
        }
        pal[2][3] = 255;
        pal[3][3] = 255;
    } else {
        for k in 0..3 {
            pal[2][k] = (a[k] + b[k]) / 2;
        }
        pal[2][3] = 255;
        pal[3] = [0, 0, 0, 0];
    }
    let bits = u32::from_le_bytes([s[4], s[5], s[6], s[7]]);
    for i in 0..16 {
        let p = pal[((bits >> (2 * i)) & 3) as usize];
        out[i] = [p[0] as f32 / 255.0, p[1] as f32 / 255.0, p[2] as f32 / 255.0, p[3] as f32 / 255.0];
    }
}

/// Sixteen unsigned values of a BC4 / BC3-alpha block.
fn bc4_values(s: &[u8]) -> [f32; 16] {
    let (a0, a1) = (s[0] as u32, s[1] as u32);
    let mut pal = [0u32; 8];
    pal[0] = a0;
    pal[1] = a1;
    if a0 > a1 {
        for i in 1..7 {
            pal[i + 1] = ((7 - i as u32) * a0 + i as u32 * a1) / 7;
        }
    } else {
        for i in 1..5 {
            pal[i + 1] = ((5 - i as u32) * a0 + i as u32 * a1) / 5;
        }
        pal[6] = 0;
        pal[7] = 255;
    }
    let mut bits = 0u64;
    for i in 0..6 {
        bits |= (s[2 + i] as u64) << (8 * i);
    }
    let mut o = [0.0f32; 16];
    for i in 0..16 {
        o[i] = pal[((bits >> (3 * i)) & 7) as usize] as f32 / 255.0;
    }
    o
}

struct BitR {
    v: u128,
    pos: u32,
}

impl BitR {
    fn new(s: &[u8]) -> Self {
        let mut a = [0u8; 16];
        a.copy_from_slice(&s[..16]);
        BitR { v: u128::from_le_bytes(a), pos: 0 }
    }
    fn get(&mut self, n: u32) -> u32 {
        if n == 0 {
            return 0;
        }
        let r = ((self.v >> self.pos) & ((1u128 << n) - 1)) as u32;
        self.pos += n;
        r
    }
}

const W2: [u32; 4] = [0, 21, 43, 64];
const W3: [u32; 8] = [0, 9, 18, 27, 37, 46, 55, 64];
const W4: [u32; 16] = [0, 4, 9, 13, 17, 21, 26, 30, 34, 38, 43, 47, 51, 55, 60, 64];

fn weight(prec: u32, i: u32) -> u32 {
    match prec {
        2 => W2[i as usize],
        3 => W3[i as usize],
        _ => W4[i as usize],
    }
}

fn lerp(a: u32, b: u32, w: u32) -> u32 {
    (a * (64 - w) + b * w + 32) >> 6
}

fn bc7(s: &[u8], out: &mut [[f32; 4]; 16]) {
    let mut br = BitR::new(s);
    let mode = (br.v.trailing_zeros()).min(8);
    if mode >= 8 {
        *out = [[0.0; 4]; 16];
        return;
    }
    br.pos = mode + 1;
    // (subsets, partition bits, rotation bits, index selection bit, colour bits, alpha bits, p-bit kind, index bits, index bits 2)
    // p-bit kind: 0 none, 1 per endpoint, 2 shared per subset
    let (ns, pbits, rotbits, selbit, cb, ab, pk, ib, ib2): (usize, u32, u32, u32, u32, u32, u8, u32, u32) = match mode {
        0 => (3, 4, 0, 0, 4, 0, 1, 3, 0),
        1 => (2, 6, 0, 0, 6, 0, 2, 3, 0),
        2 => (3, 6, 0, 0, 5, 0, 0, 2, 0),
        3 => (2, 6, 0, 0, 7, 0, 1, 2, 0),
        4 => (1, 0, 2, 1, 5, 6, 0, 2, 3),
        5 => (1, 0, 2, 0, 7, 8, 0, 2, 2),
        6 => (1, 0, 0, 0, 7, 7, 1, 4, 0),
        _ => (2, 6, 0, 0, 5, 5, 1, 2, 0),
    };
    let shape = br.get(pbits) as usize;
    let rot = br.get(rotbits);
    let sel = br.get(selbit);
    let ne = ns * 2;
    let mut ep = [[0u32; 4]; 6];
    for ch in 0..3 {
        for e in 0..ne {
            ep[e][ch] = br.get(cb);
        }
    }
    if ab > 0 {
        for e in 0..ne {
            ep[e][3] = br.get(ab);
        }
    }
    let mut pb = [0u32; 6];
    match pk {
        1 => {
            for e in 0..ne {
                pb[e] = br.get(1);
            }
        }
        2 => {
            for sset in 0..ns {
                let p = br.get(1);
                pb[sset * 2] = p;
                pb[sset * 2 + 1] = p;
            }
        }
        _ => {}
    }
    // expand endpoints to 8 bit
    let has_a = ab > 0;
    for e in 0..ne {
        for ch in 0..4 {
            let (bits, v) = if ch == 3 {
                if !has_a {
                    ep[e][3] = 255;
                    continue;
                }
                (ab, ep[e][3])
            } else {
                (cb, ep[e][ch])
            };
            let (bits, v) = if pk != 0 { (bits + 1, (v << 1) | pb[e]) } else { (bits, v) };
            // alpha shares the p-bit only in modes with a p-bit and an alpha channel (6 and 7)
            ep[e][ch] = (v << (8 - bits)) | (v >> (2 * bits - 8));
        }
    }
    let _ = has_a;
    // anchors
    let part = &PARTITION[ns - 1][shape];
    let fix = &FIXUP[ns - 1][shape];
    let anchor = |i: usize| -> bool { i == 0 || (ns > 1 && (0..ns).any(|sset| sset > 0 && fix[sset] as usize == i)) };
    let mut idx = [0u32; 16];
    for i in 0..16 {
        let n = if anchor(i) { ib - 1 } else { ib };
        idx[i] = br.get(n);
    }
    let mut idx2 = [0u32; 16];
    if ib2 > 0 {
        for i in 0..16 {
            let n = if i == 0 { ib2 - 1 } else { ib2 };
            idx2[i] = br.get(n);
        }
    }
    for i in 0..16 {
        let sset = part[i] as usize;
        let (e0, e1) = (ep[sset * 2], ep[sset * 2 + 1]);
        let mut c = [0u32; 4];
        if ib2 > 0 {
            // separate colour and alpha indices; the selector swaps which index set drives which
            let (ci, cprec, ai, aprec) = if sel == 0 { (idx[i], ib, idx2[i], ib2) } else { (idx2[i], ib2, idx[i], ib) };
            let (wc, wa) = (weight(cprec, ci), weight(aprec, ai));
            for ch in 0..3 {
                c[ch] = lerp(e0[ch], e1[ch], wc);
            }
            c[3] = lerp(e0[3], e1[3], wa);
        } else {
            let w = weight(ib, idx[i]);
            for ch in 0..4 {
                c[ch] = lerp(e0[ch], e1[ch], w);
            }
        }
        if rot > 0 {
            c.swap(3, rot as usize - 1);
        }
        out[i] = [c[0] as f32 / 255.0, c[1] as f32 / 255.0, c[2] as f32 / 255.0, c[3] as f32 / 255.0];
    }
}

fn sign_extend(v: i32, bits: u32) -> i32 {
    if bits == 0 || bits >= 32 {
        return v;
    }
    let sh = 32 - bits;
    (v << sh) >> sh
}

/// (mode id, subsets - 1, transformed, index precision, [region][endpoint] -> (r, g, b) precision)
type Bc6Mode = (u8, usize, bool, u32, [[[u32; 3]; 2]; 2]);

const BC6_MODES: [Bc6Mode; 14] = [
    (0x00, 1, true, 3, [[[10, 10, 10], [5, 5, 5]], [[5, 5, 5], [5, 5, 5]]]),
    (0x01, 1, true, 3, [[[7, 7, 7], [6, 6, 6]], [[6, 6, 6], [6, 6, 6]]]),
    (0x02, 1, true, 3, [[[11, 11, 11], [5, 4, 4]], [[5, 4, 4], [5, 4, 4]]]),
    (0x06, 1, true, 3, [[[11, 11, 11], [4, 5, 4]], [[4, 5, 4], [4, 5, 4]]]),
    (0x0a, 1, true, 3, [[[11, 11, 11], [4, 4, 5]], [[4, 4, 5], [4, 4, 5]]]),
    (0x0e, 1, true, 3, [[[9, 9, 9], [5, 5, 5]], [[5, 5, 5], [5, 5, 5]]]),
    (0x12, 1, true, 3, [[[8, 8, 8], [6, 5, 5]], [[6, 5, 5], [6, 5, 5]]]),
    (0x16, 1, true, 3, [[[8, 8, 8], [5, 6, 5]], [[5, 6, 5], [5, 6, 5]]]),
    (0x1a, 1, true, 3, [[[8, 8, 8], [5, 5, 6]], [[5, 5, 6], [5, 5, 6]]]),
    (0x1e, 1, false, 3, [[[6, 6, 6], [6, 6, 6]], [[6, 6, 6], [6, 6, 6]]]),
    (0x03, 0, false, 4, [[[10, 10, 10], [10, 10, 10]], [[0, 0, 0], [0, 0, 0]]]),
    (0x07, 0, true, 4, [[[11, 11, 11], [9, 9, 9]], [[0, 0, 0], [0, 0, 0]]]),
    (0x0b, 0, true, 4, [[[12, 12, 12], [8, 8, 8]], [[0, 0, 0], [0, 0, 0]]]),
    (0x0f, 0, true, 4, [[[16, 16, 16], [4, 4, 4]], [[0, 0, 0], [0, 0, 0]]]),
];

fn bc6_unquantize(comp: i32, bits: u32, signed: bool) -> i32 {
    if signed {
        if bits >= 16 {
            return comp;
        }
        let (neg, c) = if comp < 0 { (true, -comp) } else { (false, comp) };
        let unq = if c == 0 {
            0
        } else if c >= (1 << (bits - 1)) - 1 {
            0x7fff
        } else {
            ((c << 15) + 0x4000) >> (bits - 1)
        };
        if neg {
            -unq
        } else {
            unq
        }
    } else if bits >= 15 {
        comp
    } else if comp == 0 {
        0
    } else if comp == (1 << bits) - 1 {
        0xffff
    } else {
        ((comp << 16) + 0x8000) >> bits
    }
}

fn bc6h(s: &[u8], signed: bool, out: &mut [[f32; 4]; 16]) {
    let bad = |out: &mut [[f32; 4]; 16]| *out = [[0.0, 0.0, 0.0, 1.0]; 16];
    let first5 = s[0] & 31;
    let mode_bits = if first5 & 3 < 2 { (first5 & 3) as usize } else { first5 as usize };
    let Some(mi) = BC6_MODES.iter().position(|m| m.0 as usize == mode_bits) else {
        bad(out);
        return;
    };
    let (_, sub, transformed, iprec, prec) = BC6_MODES[mi];
    let desc = &BC6H_DESC[mi];
    let br = BitR::new(s);
    let header_bits = if sub > 0 { 82 } else { 65 };
    // endpoint storage: [region][0 = A / 1 = B][channel]
    let mut ep = [[[0i32; 3]; 2]; 2];
    let mut shape = 0usize;
    for bit in 0..header_bits {
        if (br.v >> bit) & 1 == 0 {
            continue;
        }
        let (f, p) = desc[bit];
        let p = p as u32;
        match f {
            2 => shape |= 1 << p,
            3..=14 => {
                // 3.. = RW RX RY RZ GW GX GY GZ BW BX BY BZ
                let k = (f - 3) as usize;
                let (ch, which) = (k / 4, k % 4);
                let (region, ab) = match which {
                    0 => (0, 0),
                    1 => (0, 1),
                    2 => (1, 0),
                    _ => (1, 1),
                };
                ep[region][ab][ch] |= 1 << p;
            }
            _ => {}
        }
    }
    if shape >= 32 && sub > 0 {
        bad(out);
        return;
    }
    let regions = sub + 1;
    // sign extension
    if signed {
        for ch in 0..3 {
            ep[0][0][ch] = sign_extend(ep[0][0][ch], prec[0][0][ch]);
        }
    }
    if signed || transformed {
        for r in 0..regions {
            for ab in 0..2 {
                if r == 0 && ab == 0 {
                    continue;
                }
                for ch in 0..3 {
                    ep[r][ab][ch] = sign_extend(ep[r][ab][ch], prec[r][ab][ch]);
                }
            }
        }
    }
    if transformed {
        for ch in 0..3 {
            let mask = (1i32 << prec[0][0][ch]) - 1;
            let base = ep[0][0][ch];
            ep[0][1][ch] = (ep[0][1][ch] + base) & mask;
            if sub > 0 {
                ep[1][0][ch] = (ep[1][0][ch] + base) & mask;
                ep[1][1][ch] = (ep[1][1][ch] + base) & mask;
            }
            if signed {
                ep[0][1][ch] = sign_extend(ep[0][1][ch], prec[0][0][ch]);
                if sub > 0 {
                    ep[1][0][ch] = sign_extend(ep[1][0][ch], prec[0][0][ch]);
                    ep[1][1][ch] = sign_extend(ep[1][1][ch], prec[0][0][ch]);
                }
            }
        }
    }
    let mut br = BitR { v: br.v, pos: header_bits as u32 };
    let part = &PARTITION[sub][shape];
    let fix = &FIXUP[sub][shape];
    for i in 0..16 {
        let anchor = i == 0 || (sub > 0 && fix[1] as usize == i);
        let n = if anchor { iprec - 1 } else { iprec };
        let index = br.get(n);
        if index >= if sub > 0 { 8 } else { 16 } {
            bad(out);
            return;
        }
        let r = part[i] as usize;
        let w = if sub > 0 { W3[index as usize] } else { W4[index as usize] } as i32;
        let mut c = [0.0f32; 4];
        for ch in 0..3 {
            let bits = prec[0][0][ch];
            let a = bc6_unquantize(ep[r][0][ch], bits, signed);
            let b = bc6_unquantize(ep[r][1][ch], bits, signed);
            let v = (a * (64 - w) + b * w + 32) >> 6;
            let fin = if signed {
                if v < 0 {
                    -(((-v) * 31) >> 5)
                } else {
                    (v * 31) >> 5
                }
            } else {
                (v * 31) >> 6
            };
            // `fin` holds the bit pattern of a half float (sign and magnitude when signed)
            let half = if signed {
                if fin < 0 {
                    0x8000u16 | (-fin) as u16
                } else {
                    fin as u16
                }
            } else {
                fin.clamp(0, 0xffff) as u16
            };
            c[ch] = crate::exr::half_to_f32(half);
        }
        c[3] = 1.0;
        out[i] = c;
    }
}
