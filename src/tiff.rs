// SPDX-License-Identifier: MIT OR Apache-2.0
//! Dependency-free uncompressed TIFF writer (8 or 16 bit integer, or 32-bit float; RGB or RGBA).
use crate::tex::Tex;

pub fn encode(t: &Tex, depth: u8) -> Vec<u8> {
    let depth = match depth {
        32 => 32,
        16 => 16,
        _ => 8,
    };
    let float = depth == 32;
    let alpha = t.px.iter().any(|p| p[3] < 0.9999);
    let spp: usize = if alpha { 4 } else { 3 };
    let bytes = depth as usize / 8;
    let mut out = vec![b'I', b'I', 42, 0, 0, 0, 0, 0];
    for p in &t.px {
        for c in p.iter().take(spp) {
            if float {
                out.extend_from_slice(&c.to_le_bytes());
                continue;
            }
            let v = c.clamp(0.0, 1.0);
            if depth == 16 {
                out.extend_from_slice(&((v * 65535.0 + 0.5) as u16).to_le_bytes());
            } else {
                out.push((v * 255.0 + 0.5) as u8);
            }
        }
    }
    let data_len = out.len() - 8;
    if out.len() % 2 == 1 {
        out.push(0);
    }
    let bps_off = out.len() as u32;
    for _ in 0..spp {
        out.extend_from_slice(&(depth as u16).to_le_bytes());
    }
    let ifd_off = out.len() as u32;
    out[4..8].copy_from_slice(&ifd_off.to_le_bytes());
    let mut entries: Vec<(u16, u16, u32, u32)> = vec![
        (256, 4, 1, t.w as u32),
        (257, 4, 1, t.h as u32),
        (258, 3, spp as u32, bps_off),
        (259, 3, 1, 1),
        (262, 3, 1, 2),
        (273, 4, 1, 8),
        (277, 3, 1, spp as u32),
        (278, 4, 1, t.h as u32),
        (279, 4, 1, (data_len.min(u32::MAX as usize)) as u32),
        (284, 3, 1, 1),
    ];
    if alpha {
        entries.push((338, 3, 1, 2));
    }
    if float {
        // SampleFormat = IEEE floating point, for every sample
        let sf_off = out.len() as u32;
        let _ = sf_off;
        entries.push((339, 3, 1, 3));
    }
    let _ = bytes;
    out.extend_from_slice(&(entries.len() as u16).to_le_bytes());
    for (tag, ty, count, val) in entries {
        out.extend_from_slice(&tag.to_le_bytes());
        out.extend_from_slice(&ty.to_le_bytes());
        out.extend_from_slice(&count.to_le_bytes());
        // SHORT values with count 1 are left-justified in the 4-byte field
        if ty == 3 && count == 1 {
            out.extend_from_slice(&(val as u16).to_le_bytes());
            out.extend_from_slice(&[0, 0]);
        } else {
            out.extend_from_slice(&val.to_le_bytes());
        }
    }
    out.extend_from_slice(&0u32.to_le_bytes());
    out
}
