// SPDX-License-Identifier: MIT OR Apache-2.0
//! Simple bitmap readers: Targa (`.tga`) and Windows bitmap (`.bmp`).
//!
//! Both return the stored values as 0..1 numbers (no colour conversion), top row first, like the PNG reader.
use crate::tex::Tex;

fn rd16(b: &[u8], o: usize) -> usize {
    u16::from_le_bytes([b[o], b[o + 1]]) as usize
}

fn rd32(b: &[u8], o: usize) -> u32 {
    u32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]])
}

fn rd32s(b: &[u8], o: usize) -> i32 {
    rd32(b, o) as i32
}

const MAX_SIDE: usize = 16384;

/// True when `b` looks like a Targa file (the format has no magic number, so the header is validated).
pub fn looks_like_tga(b: &[u8]) -> bool {
    if b.len() < 18 {
        return false;
    }
    let (cmap, kind, depth) = (b[1], b[2], b[16]);
    let (w, h) = (rd16(b, 12), rd16(b, 14));
    matches!(cmap, 0 | 1) && matches!(kind, 1 | 2 | 3 | 9 | 10 | 11) && matches!(depth, 8 | 15 | 16 | 24 | 32) && w > 0 && h > 0
}

pub fn decode_tga(b: &[u8]) -> Result<Tex, String> {
    if b.len() < 18 {
        return Err("TGA file too short".into());
    }
    let idlen = b[0] as usize;
    let cmap_type = b[1];
    let kind = b[2];
    let cmap_first = rd16(b, 3);
    let cmap_len = rd16(b, 5);
    let cmap_depth = b[7] as usize;
    let (w, h) = (rd16(b, 12), rd16(b, 14));
    let depth = b[16] as usize;
    let desc = b[17];
    if w == 0 || h == 0 || w > MAX_SIDE || h > MAX_SIDE {
        return Err("bad TGA image size".into());
    }
    crate::imgio::check_size(w, h)?;
    if !matches!(kind, 1 | 2 | 3 | 9 | 10 | 11) {
        return Err(format!("unsupported TGA image type {kind}"));
    }
    let rle = kind >= 9;
    let base = kind & 7;
    let valid_depth = match base {
        1 => depth == 8,
        2 => matches!(depth, 15 | 16 | 24 | 32),
        _ => matches!(depth, 8 | 16),
    };
    if !valid_depth {
        return Err(format!("unsupported TGA pixel depth {depth}"));
    }
    let mut pos = 18 + idlen;
    // colour map
    let mut palette: Vec<[f32; 4]> = Vec::new();
    if cmap_type == 1 {
        let eb = (cmap_depth + 7) / 8;
        if !matches!(cmap_depth, 15 | 16 | 24 | 32) {
            return Err("unsupported TGA colour map depth".into());
        }
        if b.len() < pos + cmap_len * eb {
            return Err("truncated TGA colour map".into());
        }
        for i in 0..cmap_len {
            palette.push(tga_pixel(&b[pos + i * eb..pos + (i + 1) * eb], cmap_depth, true));
        }
        pos += cmap_len * eb;
    }
    let bpp = (depth + 7) / 8;
    let total = w * h;
    // decode the (possibly run-length coded) pixel stream into raw pixels
    let mut raw: Vec<u8> = Vec::with_capacity(total * bpp);
    if rle {
        while raw.len() < total * bpp {
            if pos >= b.len() {
                return Err("truncated TGA data".into());
            }
            let hd = b[pos];
            pos += 1;
            let n = (hd & 0x7f) as usize + 1;
            if hd & 0x80 != 0 {
                if pos + bpp > b.len() {
                    return Err("truncated TGA data".into());
                }
                for _ in 0..n {
                    raw.extend_from_slice(&b[pos..pos + bpp]);
                }
                pos += bpp;
            } else {
                if pos + n * bpp > b.len() {
                    return Err("truncated TGA data".into());
                }
                raw.extend_from_slice(&b[pos..pos + n * bpp]);
                pos += n * bpp;
            }
        }
        raw.truncate(total * bpp);
    } else {
        if b.len() < pos + total * bpp {
            return Err("truncated TGA data".into());
        }
        raw.extend_from_slice(&b[pos..pos + total * bpp]);
    }
    let mut px = vec![[0.0f32; 4]; total];
    for i in 0..total {
        let s = &raw[i * bpp..(i + 1) * bpp];
        px[i] = match base {
            1 => {
                let idx = s[0] as usize;
                if idx < cmap_first || idx - cmap_first >= palette.len() {
                    return Err("TGA palette index out of range".into());
                }
                palette[idx - cmap_first]
            }
            3 => {
                if bpp == 1 {
                    let v = s[0] as f32 / 255.0;
                    [v, v, v, 1.0]
                } else {
                    let v = s[0] as f32 / 255.0;
                    [v, v, v, s[1] as f32 / 255.0]
                }
            }
            _ => tga_pixel(s, depth, desc & 0x0f != 0 || depth == 32),
        };
    }
    // 32-bit files whose alpha channel is entirely zero are treated as opaque
    if base == 2 && depth == 32 && px.iter().all(|p| p[3] == 0.0) {
        for p in px.iter_mut() {
            p[3] = 1.0;
        }
    }
    // origin: bit 5 set = top row first; bit 4 set = right to left
    let top_first = desc & 0x20 != 0;
    let right_to_left = desc & 0x10 != 0;
    if !top_first || right_to_left {
        let src = px.clone();
        for y in 0..h {
            let sy = if top_first { y } else { h - 1 - y };
            for x in 0..w {
                let sx = if right_to_left { w - 1 - x } else { x };
                px[y * w + x] = src[sy * w + sx];
            }
        }
    }
    Ok(Tex { w, h, px })
}

fn tga_pixel(s: &[u8], depth: usize, use_alpha: bool) -> [f32; 4] {
    match depth {
        15 | 16 => {
            let v = u16::from_le_bytes([s[0], s[1]]);
            let c = |sh: u16| ((v >> sh) & 31) as f32 / 31.0;
            let a = if depth == 16 && use_alpha { (v >> 15) as f32 } else { 1.0 };
            [c(10), c(5), c(0), if depth == 16 && !use_alpha { 1.0 } else { a }]
        }
        24 => [s[2] as f32 / 255.0, s[1] as f32 / 255.0, s[0] as f32 / 255.0, 1.0],
        _ => [s[2] as f32 / 255.0, s[1] as f32 / 255.0, s[0] as f32 / 255.0, if use_alpha { s[3] as f32 / 255.0 } else { 1.0 }],
    }
}

// ------------------------------------------------------------------------------------------------ BMP
fn mask_shift_bits(mask: u32) -> (u32, u32) {
    if mask == 0 {
        return (0, 0);
    }
    let sh = mask.trailing_zeros();
    let bits = (mask >> sh).trailing_ones();
    (sh, bits)
}

fn extract(v: u32, mask: u32) -> f32 {
    let (sh, bits) = mask_shift_bits(mask);
    if bits == 0 {
        return 0.0;
    }
    let m = if bits >= 32 { u32::MAX } else { (1u32 << bits) - 1 };
    ((v & mask) >> sh) as f32 / m as f32
}

pub fn decode_bmp(b: &[u8]) -> Result<Tex, String> {
    if b.len() < 26 || &b[..2] != b"BM" {
        return Err("not a BMP file".into());
    }
    let data_off = rd32(b, 10) as usize;
    let hsize = rd32(b, 14) as usize;
    if hsize < 12 || b.len() < 14 + hsize {
        return Err("bad BMP header".into());
    }
    let (w, h_signed, bpp, comp, ncolors);
    let mut masks = [0u32; 4];
    if hsize == 12 {
        w = rd16(b, 18) as i32;
        h_signed = rd16(b, 20) as i32;
        bpp = rd16(b, 24);
        comp = 0;
        ncolors = 0usize;
    } else {
        if hsize < 40 {
            return Err("unsupported BMP header".into());
        }
        w = rd32s(b, 18);
        h_signed = rd32s(b, 22);
        bpp = rd16(b, 28);
        comp = rd32(b, 30);
        ncolors = rd32(b, 46) as usize;
    }
    if w <= 0 || h_signed == 0 {
        return Err("bad BMP image size".into());
    }
    let top_down = h_signed < 0;
    let (w, h) = (w as usize, h_signed.unsigned_abs() as usize);
    crate::imgio::check_size(w, h)?;
    if w > MAX_SIDE || h > MAX_SIDE {
        return Err("bad BMP image size".into());
    }
    let mut has_alpha_mask = false;
    let mut pal_off = 14 + hsize;
    if comp == 3 || comp == 6 {
        if hsize >= 52 {
            for i in 0..3 {
                masks[i] = rd32(b, 54 + i * 4);
            }
            if hsize >= 56 {
                masks[3] = rd32(b, 66);
            }
        } else {
            if b.len() < 14 + hsize + 12 {
                return Err("truncated BMP".into());
            }
            for i in 0..3 {
                masks[i] = rd32(b, 14 + hsize + i * 4);
            }
            pal_off += 12;
            if comp == 6 && b.len() >= 14 + hsize + 16 {
                masks[3] = rd32(b, 14 + hsize + 12);
                pal_off += 4;
            }
        }
        has_alpha_mask = masks[3] != 0;
    } else if comp == 0 && hsize >= 108 && bpp >= 16 {
        // V4 / V5 headers may still carry masks for 16 / 32 bit data
        masks = [rd32(b, 54), rd32(b, 58), rd32(b, 62), rd32(b, 66)];
        has_alpha_mask = masks[3] != 0;
        if masks[0] | masks[1] | masks[2] == 0 {
            masks = [0; 4];
        }
    }
    if hsize >= 108 && (comp == 3 || comp == 0) && bpp >= 16 && masks[0] | masks[1] | masks[2] == 0 {
        masks = [0; 4];
    }
    // palette
    let mut palette: Vec<[f32; 3]> = Vec::new();
    if bpp <= 8 {
        let n = if ncolors == 0 { 1usize << bpp } else { ncolors.min(256) };
        let es = if hsize == 12 { 3 } else { 4 };
        for i in 0..n {
            let o = pal_off + i * es;
            if o + 3 > b.len() {
                break;
            }
            palette.push([b[o + 2] as f32 / 255.0, b[o + 1] as f32 / 255.0, b[o] as f32 / 255.0]);
        }
    }
    if data_off >= b.len() {
        return Err("BMP has no pixel data".into());
    }
    let data = &b[data_off..];
    let mut px = vec![[0.0f32, 0.0, 0.0, 1.0]; w * h];
    let row_of = |y: usize| if top_down { y } else { h - 1 - y };
    match comp {
        0 | 3 | 6 => {
            let stride = ((w * bpp + 31) / 32) * 4;
            if data.len() < stride * h && data.len() < stride * (h - 1) + (w * bpp + 7) / 8 {
                return Err("truncated BMP pixel data".into());
            }
            let (default_masks, default_bits) = match bpp {
                16 => ([0x7c00u32, 0x03e0, 0x001f, 0], 5),
                32 => ([0x00ff0000u32, 0x0000ff00, 0x000000ff, 0], 8),
                _ => ([0; 4], 0),
            };
            let _ = default_bits;
            let m = if masks[0] | masks[1] | masks[2] != 0 { masks } else { default_masks };
            let mut any_alpha = false;
            for y in 0..h {
                let row = &data[y * stride..];
                let out = row_of(y) * w;
                for x in 0..w {
                    px[out + x] = match bpp {
                        1 | 2 | 4 | 8 => {
                            let bit = x * bpp;
                            let v = ((row[bit / 8] >> (8 - bpp - (bit % 8))) & ((1u16 << bpp) - 1) as u8) as usize;
                            let c = palette.get(v).copied().unwrap_or([0.0; 3]);
                            [c[0], c[1], c[2], 1.0]
                        }
                        16 => {
                            let v = u16::from_le_bytes([row[x * 2], row[x * 2 + 1]]) as u32;
                            [extract(v, m[0]), extract(v, m[1]), extract(v, m[2]), if has_alpha_mask { extract(v, m[3]) } else { 1.0 }]
                        }
                        24 => [row[x * 3 + 2] as f32 / 255.0, row[x * 3 + 1] as f32 / 255.0, row[x * 3] as f32 / 255.0, 1.0],
                        32 => {
                            let v = rd32(row, x * 4);
                            let a = if has_alpha_mask {
                                extract(v, m[3])
                            } else {
                                // plain BI_RGB: the fourth byte is padding unless some of it is used
                                let a = (v >> 24) as f32 / 255.0;
                                if a > 0.0 {
                                    any_alpha = true;
                                }
                                a
                            };
                            [extract(v, m[0]), extract(v, m[1]), extract(v, m[2]), a]
                        }
                        _ => return Err(format!("unsupported BMP bit depth {bpp}")),
                    };
                }
            }
            if bpp == 32 && !has_alpha_mask && !any_alpha {
                for p in px.iter_mut() {
                    p[3] = 1.0;
                }
            }
        }
        1 | 2 => {
            if !(comp == 1 && bpp == 8 || comp == 2 && bpp == 4) {
                return Err("bad BMP run-length format".into());
            }
            let mut idx = vec![0u8; w * h];
            let mut covered = vec![false; w * h];
            let (mut x, mut y) = (0usize, 0usize);
            let mut p = 0usize;
            let put = |x: usize, y: usize, v: u8, idx: &mut Vec<u8>, covered: &mut Vec<bool>| {
                if x < w && y < h {
                    idx[y * w + x] = v;
                    covered[y * w + x] = true;
                }
            };
            'rle: while p + 1 < data.len() {
                let (n, v) = (data[p] as usize, data[p + 1]);
                p += 2;
                if n > 0 {
                    for i in 0..n {
                        let val = if comp == 1 { v } else if i % 2 == 0 { v >> 4 } else { v & 15 };
                        put(x, y, val, &mut idx, &mut covered);
                        x += 1;
                    }
                } else {
                    match v {
                        0 => {
                            x = 0;
                            y += 1;
                        }
                        1 => break 'rle,
                        2 => {
                            if p + 1 >= data.len() {
                                break 'rle;
                            }
                            x += data[p] as usize;
                            y += data[p + 1] as usize;
                            p += 2;
                        }
                        cnt => {
                            let cnt = cnt as usize;
                            let nbytes = if comp == 1 { cnt } else { (cnt + 1) / 2 };
                            if p + nbytes > data.len() {
                                break 'rle;
                            }
                            for i in 0..cnt {
                                let val = if comp == 1 {
                                    data[p + i]
                                } else if i % 2 == 0 {
                                    data[p + i / 2] >> 4
                                } else {
                                    data[p + i / 2] & 15
                                };
                                put(x, y, val, &mut idx, &mut covered);
                                x += 1;
                            }
                            p += nbytes + (nbytes & 1);
                        }
                    }
                }
                if y >= h {
                    break;
                }
            }
            for yy in 0..h {
                for xx in 0..w {
                    let c = palette.get(idx[yy * w + xx] as usize).copied().unwrap_or([0.0; 3]);
                    // pixels the stream skipped are transparent
                    px[row_of(yy) * w + xx] = [c[0], c[1], c[2], if covered[yy * w + xx] { 1.0 } else { 0.0 }];
                }
            }
        }
        _ => return Err(format!("unsupported BMP compression {comp}")),
    }
    Ok(Tex { w, h, px })
}
