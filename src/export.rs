// SPDX-License-Identifier: MIT OR Apache-2.0
//! Image export in PNG, JPEG, TIFF (8, 16 or 32-bit float), OpenEXR (half or float) and Radiance HDR.
use crate::tex::Tex;
use crate::{exr, jpeg, png, tiff};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Format {
    Png,
    Jpeg,
    Tiff,
    Exr,
    Hdr,
}

impl Format {
    pub fn ext(self) -> &'static str {
        match self {
            Format::Png => "png",
            Format::Jpeg => "jpg",
            Format::Tiff => "tif",
            Format::Exr => "exr",
            Format::Hdr => "hdr",
        }
    }
    pub fn name(self) -> &'static str {
        match self {
            Format::Png => "PNG",
            Format::Jpeg => "JPEG",
            Format::Tiff => "TIFF",
            Format::Exr => "EXR",
            Format::Hdr => "HDR",
        }
    }
    pub fn parse(s: &str) -> Option<Format> {
        match s.to_ascii_lowercase().as_str() {
            "png" => Some(Format::Png),
            "jpg" | "jpeg" => Some(Format::Jpeg),
            "tif" | "tiff" => Some(Format::Tiff),
            "exr" => Some(Format::Exr),
            "hdr" => Some(Format::Hdr),
            _ => None,
        }
    }
}

impl Format {
    /// Whether the format can store values outside 0..1 (float pixels).
    pub fn is_float(self) -> bool {
        matches!(self, Format::Exr | Format::Hdr)
    }
    /// Whether a 32-bit float depth is available.
    pub fn has_float32(self) -> bool {
        matches!(self, Format::Tiff | Format::Exr)
    }
}

/// `depth` is 8, 16 or 32 (32 = float: TIFF and EXR; EXR stores half floats below 32; ignored for JPEG and HDR);
/// `quality` 1..=100 (JPEG only).
pub fn encode(fmt: Format, t: &Tex, depth: u8, quality: u8) -> Vec<u8> {
    match fmt {
        Format::Png => png::encode(t.w, t.h, &t.px, if depth >= 16 { 16 } else { 8 }),
        Format::Jpeg => jpeg::encode(t, quality),
        Format::Tiff => tiff::encode(t, depth),
        Format::Exr => exr::encode_exr(t, depth >= 32),
        Format::Hdr => exr::encode_hdr(t),
    }
}
