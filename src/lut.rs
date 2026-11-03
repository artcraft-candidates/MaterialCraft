// SPDX-License-Identifier: MIT OR Apache-2.0
//! Colour lookup tables: `.cube` (1D and 3D), `.3dl`, and lookup images (a horizontal strip of N slices, or a
//! square "Hald" image). Applying one maps every pixel's colour through the table.
use crate::tex::Tex;

#[derive(Clone, Debug)]
pub enum Lut {
    /// `data` holds `n` entries
    D1 { n: usize, data: Vec<[f32; 3]>, min: [f32; 3], max: [f32; 3] },
    /// `data` holds `n`^3 entries, red varies fastest, then green, then blue
    D3 { n: usize, data: Vec<[f32; 3]>, min: [f32; 3], max: [f32; 3] },
}

impl Lut {
    pub fn size(&self) -> usize {
        match self {
            Lut::D1 { n, .. } | Lut::D3 { n, .. } => *n,
        }
    }
    pub fn is_3d(&self) -> bool {
        matches!(self, Lut::D3 { .. })
    }
    pub fn describe(&self) -> String {
        match self {
            Lut::D1 { n, .. } => format!("1D LUT, {n} entries"),
            Lut::D3 { n, .. } => format!("3D LUT, {n}x{n}x{n}"),
        }
    }

    /// Detects the format from the contents (and the file name for image formats) and reads it.
    pub fn load(name: &str, bytes: &[u8]) -> Result<Lut, String> {
        let lower = name.to_lowercase();
        if lower.ends_with(".cube") {
            return Lut::parse_cube(&String::from_utf8_lossy(bytes));
        }
        if lower.ends_with(".3dl") {
            return Lut::parse_3dl(&String::from_utf8_lossy(bytes));
        }
        if crate::imgio::sniff(name, bytes).is_some() {
            return Lut::from_image(&crate::imgio::decode(name, bytes)?);
        }
        let text = String::from_utf8_lossy(bytes);
        if text.contains("LUT_3D_SIZE") || text.contains("LUT_1D_SIZE") {
            Lut::parse_cube(&text)
        } else {
            Lut::parse_3dl(&text)
        }
    }

    pub fn parse_cube(text: &str) -> Result<Lut, String> {
        let (mut n1, mut n3) = (0usize, 0usize);
        let (mut min, mut max) = ([0.0f32; 3], [1.0f32; 3]);
        let mut data: Vec<[f32; 3]> = Vec::new();
        for raw in text.lines() {
            let line = raw.split('#').next().unwrap_or("").trim();
            if line.is_empty() {
                continue;
            }
            let mut it = line.split_whitespace();
            let first = it.next().unwrap_or("");
            let rest: Vec<&str> = it.collect();
            let nums = |v: &[&str]| -> Result<Vec<f32>, String> { v.iter().map(|s| s.parse::<f32>().map_err(|_| format!("bad number '{s}' in .cube file"))).collect() };
            match first {
                "TITLE" | "LUT_IN_VIDEO_RANGE" | "LUT_OUT_VIDEO_RANGE" => {}
                "LUT_1D_SIZE" => n1 = rest.first().and_then(|s| s.parse().ok()).ok_or("bad LUT_1D_SIZE")?,
                "LUT_3D_SIZE" => n3 = rest.first().and_then(|s| s.parse().ok()).ok_or("bad LUT_3D_SIZE")?,
                "LUT_1D_INPUT_RANGE" | "LUT_3D_INPUT_RANGE" => {
                    let v = nums(&rest)?;
                    if v.len() == 2 {
                        min = [v[0]; 3];
                        max = [v[1]; 3];
                    }
                }
                "DOMAIN_MIN" => {
                    let v = nums(&rest)?;
                    if v.len() == 3 {
                        min = [v[0], v[1], v[2]];
                    }
                }
                "DOMAIN_MAX" => {
                    let v = nums(&rest)?;
                    if v.len() == 3 {
                        max = [v[0], v[1], v[2]];
                    }
                }
                _ => {
                    let mut all = vec![first];
                    all.extend(rest);
                    let v = nums(&all)?;
                    if v.len() != 3 {
                        return Err(format!("unexpected line in .cube file: '{line}'"));
                    }
                    data.push([v[0], v[1], v[2]]);
                }
            }
        }
        if (0..3).any(|k| !(max[k] > min[k])) {
            return Err("the .cube domain is empty".into());
        }
        if n3 > 0 {
            if n3 < 2 || n3 > 256 {
                return Err(format!("unsupported 3D LUT size {n3}"));
            }
            if data.len() != n3 * n3 * n3 {
                return Err(format!("3D LUT of size {n3} needs {} entries, found {}", n3 * n3 * n3, data.len()));
            }
            Ok(Lut::D3 { n: n3, data, min, max })
        } else if n1 > 0 {
            if n1 < 2 || data.len() != n1 {
                return Err(format!("1D LUT of size {n1} needs {n1} entries, found {}", data.len()));
            }
            Ok(Lut::D1 { n: n1, data, min, max })
        } else {
            Err("the .cube file declares no LUT_1D_SIZE or LUT_3D_SIZE".into())
        }
    }

    /// 3D LUT text format used by several grading tools: an optional line of input values, then integer triples with
    /// blue varying fastest. The size follows from the number of triples and the bit depth from the largest value.
    pub fn parse_3dl(text: &str) -> Result<Lut, String> {
        let mut rows: Vec<Vec<f64>> = Vec::new();
        for raw in text.lines() {
            let line = raw.trim();
            if line.is_empty() || line.starts_with('#') || line.starts_with("Mesh") || line.starts_with("LUT") || line.starts_with("3DMESH") || line.starts_with("gamma") {
                continue;
            }
            let v: Result<Vec<f64>, _> = line.split_whitespace().map(|s| s.parse::<f64>()).collect();
            match v {
                Ok(v) if !v.is_empty() => rows.push(v),
                _ => return Err(format!("unexpected line in .3dl file: '{line}'")),
            }
        }
        // a first row with more than three numbers is the input shaper
        if rows.first().map(|r| r.len() > 3).unwrap_or(false) {
            rows.remove(0);
        }
        if rows.iter().any(|r| r.len() != 3) {
            return Err("the .3dl data rows must have three values".into());
        }
        let total = rows.len();
        let n = (total as f64).cbrt().round() as usize;
        if n < 2 || n * n * n != total {
            return Err(format!("{total} entries do not form a cube"));
        }
        let top = rows.iter().flatten().cloned().fold(0.0f64, f64::max);
        let scale = if top <= 1.0 {
            1.0
        } else if top <= 1023.0 {
            1023.0
        } else if top <= 4095.0 {
            4095.0
        } else {
            65535.0
        };
        // file order: red slowest, blue fastest -> convert to red fastest
        let mut data = vec![[0.0f32; 3]; total];
        for r in 0..n {
            for g in 0..n {
                for b in 0..n {
                    let src = &rows[(r * n + g) * n + b];
                    data[r + g * n + b * n * n] = [(src[0] / scale) as f32, (src[1] / scale) as f32, (src[2] / scale) as f32];
                }
            }
        }
        Ok(Lut::D3 { n, data, min: [0.0; 3], max: [1.0; 3] })
    }

    /// A strip (width = N*N, height = N: slices of blue left to right, red across, green down) or a square Hald image.
    pub fn from_image(t: &Tex) -> Result<Lut, String> {
        let at = |x: usize, y: usize| -> [f32; 3] {
            let p = t.px[y * t.w + x];
            [p[0], p[1], p[2]]
        };
        if t.w == t.h * t.h && t.h >= 2 {
            let n = t.h;
            let mut data = vec![[0.0f32; 3]; n * n * n];
            for b in 0..n {
                for g in 0..n {
                    for r in 0..n {
                        data[r + g * n + b * n * n] = at(b * n + r, g);
                    }
                }
            }
            return Ok(Lut::D3 { n, data, min: [0.0; 3], max: [1.0; 3] });
        }
        if t.w == t.h {
            // Hald level L: side L^3, cube size L^2
            let level = (t.w as f64).cbrt().round() as usize;
            if level >= 2 && level * level * level == t.w {
                let n = level * level;
                let mut data = vec![[0.0f32; 3]; n * n * n];
                for (i, d) in data.iter_mut().enumerate() {
                    *d = at(i % t.w, i / t.w);
                }
                return Ok(Lut::D3 { n, data, min: [0.0; 3], max: [1.0; 3] });
            }
        }
        // a single row or column is a 1D table
        if t.h == 1 && t.w >= 2 {
            return Ok(Lut::D1 { n: t.w, data: (0..t.w).map(|x| at(x, 0)).collect(), min: [0.0; 3], max: [1.0; 3] });
        }
        Err(format!("a lookup image must be N*N by N pixels (strip) or L^3 square (Hald); this one is {}x{}", t.w, t.h))
    }

    pub fn apply(&self, c: [f32; 3]) -> [f32; 3] {
        match self {
            Lut::D1 { n, data, min, max } => {
                let mut o = [0.0f32; 3];
                for k in 0..3 {
                    let x = ((c[k] - min[k]) / (max[k] - min[k])).clamp(0.0, 1.0) * (*n - 1) as f32;
                    let i = (x.floor() as usize).min(*n - 2);
                    let f = x - i as f32;
                    o[k] = data[i][k] * (1.0 - f) + data[i + 1][k] * f;
                }
                o
            }
            Lut::D3 { n, data, min, max } => {
                let m = (*n - 1) as f32;
                let mut idx = [0usize; 3];
                let mut fr = [0.0f32; 3];
                for k in 0..3 {
                    let x = ((c[k] - min[k]) / (max[k] - min[k])).clamp(0.0, 1.0) * m;
                    idx[k] = (x.floor() as usize).min(*n - 2);
                    fr[k] = x - idx[k] as f32;
                }
                let at = |r: usize, g: usize, b: usize| data[r + g * n + b * n * n];
                // tetrahedral interpolation
                let (r0, g0, b0) = (idx[0], idx[1], idx[2]);
                let (r1, g1, b1) = (r0 + 1, g0 + 1, b0 + 1);
                let (fr_, fg, fb) = (fr[0], fr[1], fr[2]);
                let c000 = at(r0, g0, b0);
                let c111 = at(r1, g1, b1);
                let mut o = [0.0f32; 3];
                for k in 0..3 {
                    o[k] = if fr_ > fg {
                        if fg > fb {
                            c000[k] + fr_ * (at(r1, g0, b0)[k] - c000[k]) + fg * (at(r1, g1, b0)[k] - at(r1, g0, b0)[k]) + fb * (c111[k] - at(r1, g1, b0)[k])
                        } else if fr_ > fb {
                            c000[k] + fr_ * (at(r1, g0, b0)[k] - c000[k]) + fb * (at(r1, g0, b1)[k] - at(r1, g0, b0)[k]) + fg * (c111[k] - at(r1, g0, b1)[k])
                        } else {
                            c000[k] + fb * (at(r0, g0, b1)[k] - c000[k]) + fr_ * (at(r1, g0, b1)[k] - at(r0, g0, b1)[k]) + fg * (c111[k] - at(r1, g0, b1)[k])
                        }
                    } else if fb > fg {
                        c000[k] + fb * (at(r0, g0, b1)[k] - c000[k]) + fg * (at(r0, g1, b1)[k] - at(r0, g0, b1)[k]) + fr_ * (c111[k] - at(r0, g1, b1)[k])
                    } else if fb > fr_ {
                        c000[k] + fg * (at(r0, g1, b0)[k] - c000[k]) + fb * (at(r0, g1, b1)[k] - at(r0, g1, b0)[k]) + fr_ * (c111[k] - at(r0, g1, b1)[k])
                    } else {
                        c000[k] + fg * (at(r0, g1, b0)[k] - c000[k]) + fr_ * (at(r1, g1, b0)[k] - at(r0, g1, b0)[k]) + fb * (c111[k] - at(r1, g1, b0)[k])
                    };
                }
                o
            }
        }
    }

    pub fn apply_tex(&self, t: &Tex, amount: f32) -> Tex {
        t.map(|p| {
            let o = self.apply([p[0], p[1], p[2]]);
            [p[0] + (o[0] - p[0]) * amount, p[1] + (o[1] - p[1]) * amount, p[2] + (o[2] - p[2]) * amount, p[3]]
        })
    }

    /// An identity 3D table of the given size (useful as a starting point and for tests).
    pub fn identity(n: usize) -> Lut {
        let mut data = Vec::with_capacity(n * n * n);
        for b in 0..n {
            for g in 0..n {
                for r in 0..n {
                    let m = (n - 1) as f32;
                    data.push([r as f32 / m, g as f32 / m, b as f32 / m]);
                }
            }
        }
        Lut::D3 { n, data, min: [0.0; 3], max: [1.0; 3] }
    }

    pub fn to_cube(&self, title: &str) -> String {
        let mut s = format!("TITLE \"{}\"\n", title.replace('"', ""));
        let (n, data, min, max, tag) = match self {
            Lut::D1 { n, data, min, max } => (n, data, min, max, "LUT_1D_SIZE"),
            Lut::D3 { n, data, min, max } => (n, data, min, max, "LUT_3D_SIZE"),
        };
        s.push_str(&format!("{tag} {n}\nDOMAIN_MIN {} {} {}\nDOMAIN_MAX {} {} {}\n", min[0], min[1], min[2], max[0], max[1], max[2]));
        for d in data {
            s.push_str(&format!("{:.6} {:.6} {:.6}\n", d[0], d[1], d[2]));
        }
        s
    }
}
