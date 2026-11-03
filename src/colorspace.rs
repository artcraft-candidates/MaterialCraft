// SPDX-License-Identifier: MIT OR Apache-2.0
//! Output colour-space conversion for colour maps, built on the `craft-cms` ICC engine.
//! Graph values are sRGB-encoded; `convert` re-encodes them for another working space on export.

use crate::tex::Tex;
use craft_cms::{Builtin, Intent, Transform};

/// Output spaces offered on export: (id, label, built-in profile). The first entry means "no conversion".
pub const SPACES: [(&str, &str, Builtin); 6] = [
    ("srgb", "sRGB (no conversion)", Builtin::Srgb),
    ("p3", "P3 (D65)", Builtin::DisplayP3),
    ("wide", "Wide-gamut RGB (gamma 2.2)", Builtin::AdobeRgbCompat),
    ("romm", "ROMM RGB (D50)", Builtin::ProPhotoCompat),
    ("rec2020", "Rec. 2020", Builtin::Rec2020),
    ("linear", "Linear sRGB", Builtin::LinearSrgb),
];

/// Output identifiers that carry colour (everything else is data and is never converted).
pub fn is_color_output(name: &str) -> bool {
    matches!(name, "basecolor" | "emissive" | "diffuse" | "specular" | "output")
}

/// Position of an id in [`SPACES`].
pub fn find(id: &str) -> Option<usize> {
    SPACES.iter().position(|s| s.0.eq_ignore_ascii_case(id))
}

/// Convert an sRGB-encoded texture into the space `id`; alpha is kept. `srgb` returns a copy.
pub fn convert(t: &Tex, id: &str) -> Result<Tex, String> {
    let i = find(id).ok_or_else(|| format!("unknown colour space '{id}' (use {})", SPACES.iter().map(|s| s.0).collect::<Vec<_>>().join(", ")))?;
    if i == 0 {
        return Ok(Tex { w: t.w, h: t.h, px: t.px.clone() });
    }
    let tr = Transform::new(Builtin::Srgb.profile(), SPACES[i].2.profile(), Intent::RelativeColorimetric, true).map_err(|e| e.to_string())?;
    let src: Vec<f32> = t.px.iter().flat_map(|p| p.iter().copied()).collect();
    let mut dst = vec![0.0f32; src.len()];
    tr.convert_f32(&src, 4, &mut dst, 4, true).map_err(|e| e.to_string())?;
    let px = dst.chunks_exact(4).map(|c| [c[0], c[1], c[2], c[3]]).collect();
    Ok(Tex { w: t.w, h: t.h, px })
}
