// SPDX-License-Identifier: MIT OR Apache-2.0
//! TIFF reader: little and big endian classic TIFF, first image only, strips or tiles, chunky or planar.
//! Compression none, LZW, Deflate (both codes) and PackBits; horizontal and floating-point predictors.
//! Samples: 1 to 32 bit unsigned integers, 16 / 32 / 64 bit floating point. Photometric: white / black is zero,
//! RGB, palette, CMYK (alpha from the ExtraSamples tag). Not supported: JPEG-in-TIFF, BigTIFF, YCbCr, Lab.
use crate::tex::Tex;

struct Rd<'a> {
    b: &'a [u8],
    le: bool,
}

impl<'a> Rd<'a> {
    fn u16(&self, o: usize) -> Result<u16, String> {
        let s = self.b.get(o..o + 2).ok_or("truncated TIFF")?;
        Ok(if self.le { u16::from_le_bytes([s[0], s[1]]) } else { u16::from_be_bytes([s[0], s[1]]) })
    }
    fn u32(&self, o: usize) -> Result<u32, String> {
        let s = self.b.get(o..o + 4).ok_or("truncated TIFF")?;
        Ok(if self.le { u32::from_le_bytes([s[0], s[1], s[2], s[3]]) } else { u32::from_be_bytes([s[0], s[1], s[2], s[3]]) })
    }
}

struct Entry {
    kind: u16,
    count: usize,
    off: usize, // where the value bytes start
}

fn type_size(k: u16) -> usize {
    match k {
        1 | 2 | 6 | 7 => 1,
        3 | 8 => 2,
        4 | 9 | 11 => 4,
        5 | 10 | 12 => 8,
        _ => 1,
    }
}

pub fn decode(b: &[u8]) -> Result<Tex, String> {
    if b.len() < 8 {
        return Err("not a TIFF file".into());
    }
    let le = match &b[..4] {
        [0x49, 0x49, 0x2a, 0] => true,
        [0x4d, 0x4d, 0, 0x2a] => false,
        [0x49, 0x49, 0x2b, 0] | [0x4d, 0x4d, 0, 0x2b] => return Err("BigTIFF is not supported".into()),
        _ => return Err("not a TIFF file".into()),
    };
    let r = Rd { b, le };
    let ifd = r.u32(4)? as usize;
    let n = r.u16(ifd)? as usize;
    let mut tags: std::collections::HashMap<u16, Entry> = std::collections::HashMap::new();
    for i in 0..n {
        let e = ifd + 2 + i * 12;
        let tag = r.u16(e)?;
        let kind = r.u16(e + 2)?;
        let count = r.u32(e + 4)? as usize;
        let size = type_size(kind).checked_mul(count).ok_or("bad TIFF tag")?;
        let off = if size <= 4 { e + 8 } else { r.u32(e + 8)? as usize };
        if off.checked_add(size).map_or(true, |end| end > b.len()) {
            continue;
        }
        tags.insert(tag, Entry { kind, count, off });
    }
    let val = |tag: u16, idx: usize| -> Result<Option<u32>, String> {
        let Some(e) = tags.get(&tag) else { return Ok(None) };
        if idx >= e.count {
            return Ok(None);
        }
        let sz = type_size(e.kind);
        let o = e.off + idx * sz;
        Ok(Some(match sz {
            1 => *b.get(o).ok_or("truncated TIFF")? as u32,
            2 => r.u16(o)? as u32,
            _ => r.u32(o)?,
        }))
    };
    let one = |tag: u16, d: u32| -> Result<u32, String> { Ok(val(tag, 0)?.unwrap_or(d)) };
    let w = one(256, 0)? as usize;
    let h = one(257, 0)? as usize;
    if w == 0 || h == 0 || w > 16384 || h > 16384 || w * h > 100_000_000 {
        return Err("bad TIFF image size".into());
    }
    let spp = one(277, 1)? as usize;
    if spp == 0 || spp > 8 {
        return Err("unsupported TIFF sample count".into());
    }
    let bits: Vec<u32> = (0..spp).map(|i| val(258, i).ok().flatten().unwrap_or(1)).collect();
    let bps = bits[0];
    if bits.iter().any(|v| *v != bps) {
        return Err("TIFF with mixed sample sizes is not supported".into());
    }
    let sfmt = one(339, 1)?;
    let comp = one(259, 1)?;
    let photo = one(262, 1)?;
    let planar = one(284, 1)?;
    let predictor = one(317, 1)?;
    let float = sfmt == 3;
    if float && !matches!(bps, 16 | 32 | 64) {
        return Err("unsupported TIFF float sample size".into());
    }
    if !float && !matches!(bps, 1 | 2 | 4 | 8 | 16 | 32) {
        return Err(format!("unsupported TIFF sample size ({bps} bit)"));
    }
    if !matches!(comp, 1 | 5 | 8 | 32946 | 32773) {
        return Err(format!("unsupported TIFF compression {comp}"));
    }
    if !matches!(photo, 0 | 1 | 2 | 3 | 5) {
        return Err(format!("unsupported TIFF photometric interpretation {photo}"));
    }
    let colour_samples = match photo {
        0 | 1 | 3 => 1,
        2 => 3,
        _ => 4,
    };
    if spp < colour_samples {
        return Err("TIFF has too few samples for its colour model".into());
    }
    let has_alpha = spp > colour_samples && matches!(val(338, 0)?, Some(1) | Some(2));
    // chunk layout
    let tiled = tags.contains_key(&322);
    let (cw, ch) = if tiled { (one(322, 0)? as usize, one(323, 0)? as usize) } else { (w, (one(278, h as u32)? as usize).clamp(1, h)) };
    if cw == 0 || ch == 0 {
        return Err("bad TIFF chunk size".into());
    }
    let (off_tag, cnt_tag) = if tiled { (324, 325) } else { (273, 279) };
    let across = (w + cw - 1) / cw;
    let down = (h + ch - 1) / ch;
    let per_plane = across * down;
    let planes = if planar == 2 { spp } else { 1 };
    let nchunks = per_plane * planes;
    let spc = if planar == 2 { 1 } else { spp }; // samples per chunk pixel
    let bytes_per_sample = ((bps + 7) / 8) as usize;
    // decoded sample planes
    let mut data = vec![0f32; w * h * spp];
    let maxv = if bps >= 32 { u32::MAX as f64 } else { ((1u64 << bps) - 1) as f64 };
    for ci in 0..nchunks {
        let off = val(off_tag, ci)?.ok_or("TIFF is missing strip offsets")? as usize;
        let cnt = val(cnt_tag, ci)?.map(|v| v as usize).unwrap_or(b.len().saturating_sub(off));
        let comp_data = b.get(off..off.saturating_add(cnt).min(b.len())).ok_or("TIFF strip lies outside the file")?;
        let plane = ci / per_plane;
        let idx = ci % per_plane;
        let (tx, ty) = (idx % across, idx / across);
        let rows_here = if tiled { ch } else { ch.min(h - ty * ch) };
        let row_bytes = ((cw * spc * bps as usize) + 7) / 8;
        let need = row_bytes * rows_here;
        let mut raw = match comp {
            1 => comp_data.to_vec(),
            5 => lzw(comp_data, need)?,
            8 | 32946 => {
                if comp_data.len() < 2 {
                    return Err("truncated TIFF Deflate data".into());
                }
                crate::png::inflate(&comp_data[2..])?
            }
            _ => packbits(comp_data, need),
        };
        if raw.len() < need {
            raw.resize(need, 0);
        }
        // predictors
        if predictor == 2 {
            for row in raw.chunks_mut(row_bytes).take(rows_here) {
                match bps {
                    8 => {
                        for i in spc..cw * spc {
                            row[i] = row[i].wrapping_add(row[i - spc]);
                        }
                    }
                    16 => {
                        for i in spc..cw * spc {
                            let a = u16r(row, i, le);
                            let p = u16r(row, i - spc, le);
                            u16w(row, i, a.wrapping_add(p), le);
                        }
                    }
                    32 => {
                        for i in spc..cw * spc {
                            let a = u32r(row, i, le);
                            let p = u32r(row, i - spc, le);
                            let v = a.wrapping_add(p);
                            let bytes = if le { v.to_le_bytes() } else { v.to_be_bytes() };
                            row[i * 4..i * 4 + 4].copy_from_slice(&bytes);
                        }
                    }
                    _ => {}
                }
            }
        } else if predictor == 3 {
            let bw = bytes_per_sample;
            for row in raw.chunks_mut(row_bytes).take(rows_here) {
                for i in spc..row.len() {
                    row[i] = row[i].wrapping_add(row[i - spc]);
                }
                let ns = cw * spc;
                let tmp = row[..ns * bw].to_vec();
                for s in 0..ns {
                    for j in 0..bw {
                        // plane j holds byte j counted from the most significant byte
                        let byte = tmp[j * ns + s];
                        if le {
                            row[s * bw + (bw - 1 - j)] = byte;
                        } else {
                            row[s * bw + j] = byte;
                        }
                    }
                }
            }
        }
        // store samples
        for ry in 0..rows_here {
            let y = ty * ch + ry;
            if y >= h {
                break;
            }
            let row = &raw[ry * row_bytes..(ry + 1) * row_bytes];
            for rx in 0..cw {
                let x = tx * cw + rx;
                if x >= w {
                    break;
                }
                for s in 0..spc {
                    let si = rx * spc + s;
                    let v: f32 = if float {
                        match bps {
                            16 => crate::exr::half_to_f32(u16r(row, si, le)),
                            32 => f32::from_bits(u32r(row, si, le)),
                            _ => {
                                let mut a = [0u8; 8];
                                a.copy_from_slice(&row[si * 8..si * 8 + 8]);
                                (if le { f64::from_le_bytes(a) } else { f64::from_be_bytes(a) }) as f32
                            }
                        }
                    } else {
                        let raw_v: u64 = match bps {
                            8 => row[si] as u64,
                            16 => u16r(row, si, le) as u64,
                            32 => u32r(row, si, le) as u64,
                            k => {
                                let bit = si * k as usize;
                                ((row[bit / 8] >> (8 - k as usize - (bit % 8))) & ((1u16 << k) - 1) as u8) as u64
                            }
                        };
                        if photo == 3 {
                            raw_v as f32 // palette index, converted below
                        } else {
                            (raw_v as f64 / maxv) as f32
                        }
                    };
                    let sidx = if planar == 2 { plane } else { s };
                    data[(y * w + x) * spp + sidx] = if v.is_finite() { v } else { 0.0 };
                }
            }
        }
    }
    // colour model
    let palette: Option<Vec<[f32; 3]>> = if photo == 3 {
        let e = tags.get(&320).ok_or("palette TIFF without a colour map")?;
        let n = 1usize << bps;
        if e.count < n * 3 {
            return Err("TIFF colour map too short".into());
        }
        Some((0..n).map(|i| [r.u16(e.off + i * 2).unwrap_or(0) as f32 / 65535.0, r.u16(e.off + (n + i) * 2).unwrap_or(0) as f32 / 65535.0, r.u16(e.off + (2 * n + i) * 2).unwrap_or(0) as f32 / 65535.0]).collect())
    } else {
        None
    };
    let mut px = vec![[0.0f32, 0.0, 0.0, 1.0]; w * h];
    for i in 0..w * h {
        let s = &data[i * spp..(i + 1) * spp];
        let a = if has_alpha { s[colour_samples] } else { 1.0 };
        px[i] = match photo {
            0 => {
                let v = if float { s[0] } else { 1.0 - s[0] };
                [v, v, v, a]
            }
            1 => [s[0], s[0], s[0], a],
            2 => [s[0], s[1], s[2], a],
            3 => {
                let c = palette.as_ref().unwrap().get(s[0] as usize).copied().unwrap_or([0.0; 3]);
                [c[0], c[1], c[2], 1.0]
            }
            _ => [(1.0 - s[0]) * (1.0 - s[3]), (1.0 - s[1]) * (1.0 - s[3]), (1.0 - s[2]) * (1.0 - s[3]), a],
        };
    }
    Ok(Tex { w, h, px })
}

fn u16r(row: &[u8], i: usize, le: bool) -> u16 {
    if le {
        u16::from_le_bytes([row[i * 2], row[i * 2 + 1]])
    } else {
        u16::from_be_bytes([row[i * 2], row[i * 2 + 1]])
    }
}

fn u16w(row: &mut [u8], i: usize, v: u16, le: bool) {
    let b = if le { v.to_le_bytes() } else { v.to_be_bytes() };
    row[i * 2] = b[0];
    row[i * 2 + 1] = b[1];
}

fn u32r(row: &[u8], i: usize, le: bool) -> u32 {
    let s = [row[i * 4], row[i * 4 + 1], row[i * 4 + 2], row[i * 4 + 3]];
    if le {
        u32::from_le_bytes(s)
    } else {
        u32::from_be_bytes(s)
    }
}

fn packbits(d: &[u8], want: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(want);
    let mut i = 0;
    while i < d.len() && out.len() < want {
        let n = d[i] as i8;
        i += 1;
        if n >= 0 {
            let c = n as usize + 1;
            let end = (i + c).min(d.len());
            out.extend_from_slice(&d[i..end]);
            i = end;
        } else if n != -128 {
            if i >= d.len() {
                break;
            }
            out.extend(std::iter::repeat(d[i]).take((1 - n as isize) as usize));
            i += 1;
        }
    }
    out
}

/// TIFF flavour of LZW: most significant bit first, codes grow one code early.
fn lzw(d: &[u8], want: usize) -> Result<Vec<u8>, String> {
    let mut out: Vec<u8> = Vec::with_capacity(want);
    let mut prefix = vec![0u16; 4096];
    let mut suffix = vec![0u8; 4096];
    let mut length = vec![0u16; 4096];
    for i in 0..256 {
        suffix[i] = i as u8;
        length[i] = 1;
    }
    let mut next = 258usize;
    let mut bits = 9u32;
    let mut prev: Option<usize> = None;
    let (mut acc, mut nacc) = (0u32, 0u32);
    let mut pos = 0usize;
    let mut tmp: Vec<u8> = Vec::new();
    loop {
        while nacc < bits {
            if pos >= d.len() {
                return Ok(out); // ran out without an end code: keep what we have
            }
            acc = (acc << 8) | d[pos] as u32;
            pos += 1;
            nacc += 8;
        }
        let code = ((acc >> (nacc - bits)) & ((1 << bits) - 1)) as usize;
        nacc -= bits;
        if code == 257 {
            break;
        }
        if code == 256 {
            next = 258;
            bits = 9;
            prev = None;
            continue;
        }
        let emit = |c: usize, tmp: &mut Vec<u8>, prefix: &[u16], suffix: &[u8], length: &[u16]| {
            tmp.clear();
            tmp.resize(length[c] as usize, 0);
            let mut k = c;
            for j in (0..length[c] as usize).rev() {
                tmp[j] = suffix[k];
                k = prefix[k] as usize;
            }
        };
        match prev {
            None => {
                if code > 255 {
                    return Err("bad TIFF LZW data".into());
                }
                out.push(code as u8);
                prev = Some(code);
            }
            Some(p) => {
                if code < next {
                    emit(code, &mut tmp, &prefix, &suffix, &length);
                    if next < 4096 {
                        prefix[next] = p as u16;
                        suffix[next] = tmp[0];
                        length[next] = length[p] + 1;
                        next += 1;
                    }
                } else if code == next && next < 4096 {
                    emit(p, &mut tmp, &prefix, &suffix, &length);
                    let first = tmp[0];
                    prefix[next] = p as u16;
                    suffix[next] = first;
                    length[next] = length[p] + 1;
                    next += 1;
                    emit(code, &mut tmp, &prefix, &suffix, &length);
                } else {
                    return Err("bad TIFF LZW code".into());
                }
                out.extend_from_slice(&tmp);
                prev = Some(code);
                bits = match next {
                    0..=510 => 9,
                    511..=1022 => 10,
                    1023..=2046 => 11,
                    _ => 12,
                };
            }
        }
        if out.len() >= want {
            break;
        }
    }
    Ok(out)
}
