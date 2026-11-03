// SPDX-License-Identifier: MIT OR Apache-2.0
//! One entry point for reading bitmaps: the format is recognised from the file contents (the extension is only a
//! tie-breaker for formats without a signature, such as TGA).
use crate::tex::Tex;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Format {
    Png,
    Jpeg,
    Tiff,
    Exr,
    Hdr,
    WebP,
    Dds,
    Tga,
    Bmp,
}

impl Format {
    pub fn name(self) -> &'static str {
        match self {
            Format::Png => "PNG",
            Format::Jpeg => "JPEG",
            Format::Tiff => "TIFF",
            Format::Exr => "OpenEXR",
            Format::Hdr => "Radiance HDR",
            Format::WebP => "WebP",
            Format::Dds => "DDS",
            Format::Tga => "TGA",
            Format::Bmp => "BMP",
        }
    }
    /// Formats that normally hold linear, unbounded values.
    pub fn is_float(self) -> bool {
        matches!(self, Format::Exr | Format::Hdr)
    }
}

/// File extensions the importer accepts (lower case, without the dot).
pub const EXTENSIONS: &[&str] = &["png", "jpg", "jpeg", "jpe", "tif", "tiff", "exr", "hdr", "pic", "webp", "dds", "tga", "bmp"];

pub fn has_image_extension(name: &str) -> bool {
    let l = name.to_lowercase();
    EXTENSIONS.iter().any(|e| l.ends_with(&format!(".{e}")))
}

pub fn sniff(name: &str, b: &[u8]) -> Option<Format> {
    let l = name.to_lowercase();
    if b.starts_with(b"\x89PNG") {
        Some(Format::Png)
    } else if b.starts_with(&[0xff, 0xd8]) {
        Some(Format::Jpeg)
    } else if b.starts_with(b"II*\0") || b.starts_with(b"MM\0*") {
        Some(Format::Tiff)
    } else if b.starts_with(&[0x76, 0x2f, 0x31, 0x01]) {
        Some(Format::Exr)
    } else if b.starts_with(b"#?") {
        Some(Format::Hdr)
    } else if b.len() >= 12 && &b[..4] == b"RIFF" && &b[8..12] == b"WEBP" {
        Some(Format::WebP)
    } else if b.starts_with(b"DDS ") {
        Some(Format::Dds)
    } else if b.starts_with(b"BM") {
        Some(Format::Bmp)
    } else if l.ends_with(".tga") && crate::dec_basic::looks_like_tga(b) {
        Some(Format::Tga)
    } else if l.ends_with(".hdr") || l.ends_with(".pic") {
        Some(Format::Hdr)
    } else if crate::dec_basic::looks_like_tga(b) && (l.ends_with(".tga") || l.ends_with(".icb") || l.ends_with(".vda")) {
        Some(Format::Tga)
    } else {
        None
    }
}

/// Largest picture the readers accept (width x height). A damaged header must not make a reader ask for gigabytes.
pub const MAX_PIXELS: usize = 100_000_000;

pub fn check_size(w: usize, h: usize) -> Result<(), String> {
    if w == 0 || h == 0 {
        return Err("image has no pixels".into());
    }
    match w.checked_mul(h) {
        Some(n) if n <= MAX_PIXELS => Ok(()),
        _ => Err(format!("image of {w}x{h} pixels is too large (limit {} million pixels)", MAX_PIXELS / 1_000_000)),
    }
}

pub struct Decoded {
    pub format: Format,
    pub tex: Tex,
    /// true when the values are linear floating point data (EXR, HDR, floating point DDS)
    pub float: bool,
}

pub fn decode_full(name: &str, b: &[u8]) -> Result<Decoded, String> {
    let f = sniff(name, b).ok_or_else(|| format!("{name}: unrecognised image format (supported: PNG, JPEG, TIFF, OpenEXR, HDR, WebP, DDS, TGA, BMP)"))?;
    let (tex, float) = match f {
        Format::Png => (crate::png::decode(b)?, false),
        Format::Jpeg => (crate::dec_jpeg::decode(b)?, false),
        Format::Tiff => (crate::dec_tiff::decode(b)?, false),
        Format::Exr => (crate::hdri::decode_exr(b)?, true),
        Format::Hdr => (crate::hdri::decode_hdr(b)?, true),
        Format::WebP => (crate::dec_webp::decode(b)?, false),
        Format::Dds => {
            let d = crate::dec_dds::decode(b)?;
            (d.tex, d.hdr)
        }
        Format::Tga => (crate::dec_basic::decode_tga(b)?, false),
        Format::Bmp => (crate::dec_basic::decode_bmp(b)?, false),
    };
    Ok(Decoded { format: f, tex, float })
}

pub fn decode(name: &str, b: &[u8]) -> Result<Tex, String> {
    decode_full(name, b).map(|d| d.tex)
}

pub fn load(path: &std::path::Path) -> Result<Tex, String> {
    let b = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    decode(&path.to_string_lossy(), &b)
}
