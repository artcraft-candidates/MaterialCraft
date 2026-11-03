// SPDX-License-Identifier: MIT OR Apache-2.0
//! Mesh import for the baker: Wavefront OBJ, PLY (ascii and binary), STL (ascii and binary) and glTF / GLB.
//! Every format ends up as the same triangle soup (`Mesh`) with per-corner normals, two UV sets, vertex colours and
//! object / material / group ids. UVs use the OBJ convention (v points up), so glTF files are flipped on import.
//! Written from the public file format descriptions; no third-party code.
use crate::v3::*;
use std::collections::HashMap;
use std::path::Path;

/// File name extensions the importer understands.
pub const FORMATS: &[&str] = &["obj", "ply", "stl", "gltf", "glb"];

/// Largest triangle count an import may produce (a safety net against corrupt files).
pub const MAX_TRIS: usize = 60_000_000;

#[derive(Clone, Debug)]
pub struct Tri {
    pub p: [V; 3],
    /// shading normal at each corner
    pub n: [V; 3],
    /// first UV set
    pub uv: [[f32; 2]; 3],
    /// second UV set (glTF TEXCOORD_1); zero when the file has none
    pub uv1: [[f32; 2]; 3],
    /// vertex colour at each corner (white when the file has none)
    pub col: [[f32; 3]; 3],
    /// index into `Mesh::objects`
    pub object: u32,
    pub material: u32,
    /// index into `Mesh::groups` (polygroups: OBJ `g`, glTF primitives)
    pub group: u32,
    /// smoothing group of the source face (0 = none)
    pub smooth: u32,
}

impl Tri {
    fn blank(p: [V; 3]) -> Tri {
        let n = norm(cross(sub(p[1], p[0]), sub(p[2], p[0])));
        Tri { p, n: [n; 3], uv: [[0.0; 2]; 3], uv1: [[0.0; 2]; 3], col: [[1.0; 3]; 3], object: 0, material: 0, group: 0, smooth: 0 }
    }
    /// Geometric (face) normal, unit length (zero-area triangles give +z).
    pub fn face_normal(&self) -> V {
        norm(cross(sub(self.p[1], self.p[0]), sub(self.p[2], self.p[0])))
    }
    pub fn area(&self) -> f32 {
        0.5 * len(cross(sub(self.p[1], self.p[0]), sub(self.p[2], self.p[0])))
    }
}

#[derive(Clone, Debug, Default)]
pub struct Mesh {
    pub tris: Vec<Tri>,
    pub objects: Vec<String>,
    pub materials: Vec<String>,
    pub groups: Vec<String>,
    pub has_uv: bool,
    pub has_uv1: bool,
    pub has_normals: bool,
    pub has_colors: bool,
    /// file name the mesh came from
    pub name: String,
}

/// How shading normals are made.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum NormalMode {
    /// as stored in the file
    Keep,
    /// one normal per face
    Flat,
    /// smooth across edges whose faces differ by less than this angle in degrees (180 = everywhere)
    Angle(f32),
    /// smooth inside each smoothing group of the source file
    Groups,
}

#[derive(Clone, Debug, Default)]
pub struct MeshStats {
    pub tris: usize,
    pub objects: usize,
    pub materials: usize,
    pub min: V,
    pub max: V,
    pub area: f32,
}

impl Mesh {
    pub fn bounds(&self) -> (V, V) {
        let (mut mn, mut mx) = ([f32::MAX; 3], [f32::MIN; 3]);
        for t in &self.tris {
            for p in &t.p {
                for k in 0..3 {
                    mn[k] = mn[k].min(p[k]);
                    mx[k] = mx[k].max(p[k]);
                }
            }
        }
        if self.tris.is_empty() {
            ([0.0; 3], [0.0; 3])
        } else {
            (mn, mx)
        }
    }

    pub fn stats(&self) -> MeshStats {
        let (min, max) = self.bounds();
        MeshStats { tris: self.tris.len(), objects: self.objects.len(), materials: self.materials.len(), min, max, area: self.tris.iter().map(|t| t.area()).sum() }
    }

    /// Diagonal of the bounding box (1 for an empty mesh).
    pub fn diagonal(&self) -> f32 {
        let (a, b) = self.bounds();
        len(sub(b, a)).max(1e-9)
    }

    /// Put several meshes into one. Objects, materials and groups with the same name are shared.
    pub fn merge(parts: &[Mesh]) -> Mesh {
        let mut out = Mesh::default();
        for m in parts {
            let map = |names: &[String], into: &mut Vec<String>| -> Vec<u32> {
                names
                    .iter()
                    .map(|n| match into.iter().position(|x| x == n) {
                        Some(i) => i as u32,
                        None => {
                            into.push(n.clone());
                            (into.len() - 1) as u32
                        }
                    })
                    .collect()
            };
            let (om, mm, gm) = (map(&m.objects, &mut out.objects), map(&m.materials, &mut out.materials), map(&m.groups, &mut out.groups));
            for t in &m.tris {
                let mut t = t.clone();
                t.object = om.get(t.object as usize).copied().unwrap_or(0);
                t.material = mm.get(t.material as usize).copied().unwrap_or(0);
                t.group = gm.get(t.group as usize).copied().unwrap_or(0);
                out.tris.push(t);
            }
            out.has_uv |= m.has_uv;
            out.has_uv1 |= m.has_uv1;
            out.has_normals |= m.has_normals;
            out.has_colors |= m.has_colors;
            if out.name.is_empty() {
                out.name = m.name.clone();
            }
        }
        out
    }

    /// Rebuild the shading normals. `Keep` leaves them alone.
    pub fn apply_normals(&mut self, mode: NormalMode) {
        match mode {
            NormalMode::Keep => {}
            NormalMode::Flat => {
                for t in &mut self.tris {
                    let n = t.face_normal();
                    t.n = [n; 3];
                }
            }
            NormalMode::Angle(deg) => self.smooth(Some(deg.clamp(0.0, 180.0).to_radians().cos()), false),
            NormalMode::Groups => self.smooth(None, true),
        }
    }

    fn smooth(&mut self, cos_limit: Option<f32>, by_group: bool) {
        let diag = self.diagonal();
        let q = (diag * 1e-6).max(1e-12);
        let key = |p: V| -> [i64; 3] { [(p[0] / q).round() as i64, (p[1] / q).round() as i64, (p[2] / q).round() as i64] };
        let fnrm: Vec<V> = self.tris.iter().map(|t| t.face_normal()).collect();
        // triangles without area (sphere poles) have no face normal: they borrow the normals of their neighbours
        let deg: Vec<bool> = self.tris.iter().map(|t| t.area() < 1e-10 * diag * diag).collect();
        let mut at: HashMap<[i64; 3], Vec<(u32, u8)>> = HashMap::new();
        for (i, t) in self.tris.iter().enumerate() {
            for c in 0..3 {
                at.entry(key(t.p[c])).or_default().push((i as u32, c as u8));
            }
        }
        let corner_angle = |t: &Tri, c: usize| -> f32 {
            let (a, b) = (norm(sub(t.p[(c + 1) % 3], t.p[c])), norm(sub(t.p[(c + 2) % 3], t.p[c])));
            dot(a, b).clamp(-1.0, 1.0).acos()
        };
        let mut out: Vec<[V; 3]> = Vec::with_capacity(self.tris.len());
        for (i, t) in self.tris.iter().enumerate() {
            let mut ns = [[0.0f32; 3]; 3];
            for c in 0..3 {
                let mut acc = [0.0f32; 3];
                if let Some(list) = at.get(&key(t.p[c])) {
                    for &(j, cj) in list {
                        let u = &self.tris[j as usize];
                        let ju = j as usize;
                        let ok = if deg[ju] {
                            false
                        } else if ju == i || deg[i] {
                            true
                        } else if by_group {
                            t.smooth != 0 && u.smooth == t.smooth
                        } else {
                            dot(fnrm[i], fnrm[ju]) >= cos_limit.unwrap_or(-2.0) - 1e-6
                        };
                        if ok {
                            acc = add(acc, mul(fnrm[j as usize], corner_angle(u, cj as usize).max(1e-4)));
                        }
                    }
                }
                ns[c] = if len(acc) < 1e-12 { fnrm[i] } else { norm(acc) };
            }
            out.push(ns);
        }
        for (t, n) in self.tris.iter_mut().zip(out) {
            t.n = n;
        }
    }
}

// ====================================================================== entry points
/// Read a mesh file; the format comes from the extension.
pub fn load_file(path: &Path) -> Result<Mesh, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let name = path.file_name().map(|s| s.to_string_lossy().to_string()).unwrap_or_default();
    load_bytes(&name, &bytes, path.parent()).map_err(|e| format!("{}: {e}", path.display()))
}

/// Read a mesh from memory. `dir` is where a glTF file finds external `.bin` buffers.
pub fn load_bytes(name: &str, bytes: &[u8], dir: Option<&Path>) -> Result<Mesh, String> {
    let ext = name.rsplit('.').next().unwrap_or("").to_ascii_lowercase();
    let mut m = match ext.as_str() {
        "obj" => parse_obj(std::str::from_utf8(bytes).map_err(|_| "the OBJ file is not text".to_string())?)?,
        "ply" => parse_ply(bytes)?,
        "stl" => parse_stl(bytes)?,
        "gltf" | "glb" => parse_gltf(bytes, dir)?,
        _ => return Err(format!("unknown mesh format '.{ext}' (supported: {})", FORMATS.join(", "))),
    };
    let stem = name.rsplit_once('.').map(|(a, _)| a).unwrap_or(name).to_string();
    if m.objects.is_empty() || (m.objects.len() == 1 && m.objects[0] == "default") {
        m.objects = vec![stem];
    }
    if m.tris.is_empty() {
        return Err("the mesh has no faces".into());
    }
    if m.tris.len() > MAX_TRIS {
        return Err(format!("the mesh has {} triangles, more than the {} limit", m.tris.len(), MAX_TRIS));
    }
    for t in &m.tris {
        if !t.p.iter().all(|p| p.iter().all(|x| x.is_finite())) {
            return Err("the mesh has coordinates that are not finite numbers".into());
        }
    }
    m.name = name.to_string();
    Ok(m)
}

// ====================================================================== polygon triangulation
/// Split a polygon into triangles (indices into `pts`). Convex polygons are fanned; others are ear-clipped.
fn triangulate(pts: &[V]) -> Vec<[usize; 3]> {
    let n = pts.len();
    if n < 3 {
        return vec![];
    }
    if n == 3 {
        return vec![[0, 1, 2]];
    }
    // Newell normal and the 2D projection that drops its largest axis
    let mut nn = [0.0f32; 3];
    for i in 0..n {
        let (a, b) = (pts[i], pts[(i + 1) % n]);
        nn[0] += (a[1] - b[1]) * (a[2] + b[2]);
        nn[1] += (a[2] - b[2]) * (a[0] + b[0]);
        nn[2] += (a[0] - b[0]) * (a[1] + b[1]);
    }
    let ax = if nn[0].abs() >= nn[1].abs() && nn[0].abs() >= nn[2].abs() { 0 } else if nn[1].abs() >= nn[2].abs() { 1 } else { 2 };
    let flip = nn[ax] < 0.0;
    let p2: Vec<[f32; 2]> = pts
        .iter()
        .map(|p| {
            let (a, b) = match ax {
                0 => (p[1], p[2]),
                1 => (p[2], p[0]),
                _ => (p[0], p[1]),
            };
            if flip {
                [b, a]
            } else {
                [a, b]
            }
        })
        .collect();
    let cross2 = |a: [f32; 2], b: [f32; 2], c: [f32; 2]| (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]);
    let convex = (0..n).all(|i| cross2(p2[i], p2[(i + 1) % n], p2[(i + 2) % n]) >= -1e-9);
    if convex {
        return (1..n - 1).map(|k| [0, k, k + 1]).collect();
    }
    let mut rest: Vec<usize> = (0..n).collect();
    let mut out = Vec::new();
    let mut guard = 0;
    while rest.len() > 3 && guard < n * n + 8 {
        guard += 1;
        let m = rest.len();
        let mut clipped = false;
        for i in 0..m {
            let (a, b, c) = (rest[(i + m - 1) % m], rest[i], rest[(i + 1) % m]);
            if cross2(p2[a], p2[b], p2[c]) <= 1e-12 {
                continue;
            }
            let inside = rest.iter().any(|&q| {
                if q == a || q == b || q == c {
                    return false;
                }
                let (d1, d2, d3) = (cross2(p2[a], p2[b], p2[q]), cross2(p2[b], p2[c], p2[q]), cross2(p2[c], p2[a], p2[q]));
                d1 >= -1e-12 && d2 >= -1e-12 && d3 >= -1e-12
            });
            if !inside {
                out.push([a, b, c]);
                rest.remove(i);
                clipped = true;
                break;
            }
        }
        if !clipped {
            // a degenerate polygon: fall back to a fan so no face is lost
            for k in 1..rest.len() - 1 {
                out.push([rest[0], rest[k], rest[k + 1]]);
            }
            return out;
        }
    }
    if rest.len() == 3 {
        out.push([rest[0], rest[1], rest[2]]);
    }
    out
}

fn name_index(list: &mut Vec<String>, name: &str) -> u32 {
    match list.iter().position(|x| x == name) {
        Some(i) => i as u32,
        None => {
            list.push(name.to_string());
            (list.len() - 1) as u32
        }
    }
}

// ====================================================================== OBJ
fn obj_index(tok: &str, len: usize) -> Result<Option<usize>, String> {
    if tok.is_empty() {
        return Ok(None);
    }
    let i: i64 = tok.parse().map_err(|_| format!("bad index '{tok}'"))?;
    let r = if i > 0 { i - 1 } else { len as i64 + i };
    if i == 0 || r < 0 || r as usize >= len {
        return Err(format!("index {i} is out of range"));
    }
    Ok(Some(r as usize))
}

pub fn parse_obj(text: &str) -> Result<Mesh, String> {
    let (mut vs, mut vcols, mut vts, mut vns): (Vec<V>, Vec<Option<[f32; 3]>>, Vec<[f32; 2]>, Vec<V>) = (vec![], vec![], vec![], vec![]);
    let mut m = Mesh::default();
    let (mut obj_name, mut grp_name, mut mat_name) = (String::new(), String::new(), String::new());
    let mut smooth = 0u32;
    let mut smooth_next = 1u32;
    let mut any_normals = false;
    let mut any_missing_normals = false;
    let mut pending = String::new();
    for (ln, raw) in text.lines().enumerate() {
        // a trailing backslash continues the line
        let line = raw.split('#').next().unwrap_or("").trim_end();
        if let Some(head) = line.strip_suffix('\\') {
            pending.push_str(head);
            pending.push(' ');
            continue;
        }
        let full: String;
        let l: &str = if pending.is_empty() {
            line
        } else {
            pending.push_str(line);
            full = std::mem::take(&mut pending);
            &full
        };
        obj_line(l, ln + 1, &mut vs, &mut vcols, &mut vts, &mut vns, &mut m, (&mut obj_name, &mut grp_name, &mut mat_name), &mut smooth, &mut smooth_next, &mut any_normals, &mut any_missing_normals)?;
    }
    if m.tris.is_empty() {
        return Err("the OBJ file has no faces".into());
    }
    m.has_normals = any_normals && !any_missing_normals;
    if any_missing_normals {
        // faces without normals: smooth inside smoothing groups, flat otherwise (the OBJ default)
        let flat: Vec<bool> = m.tris.iter().map(|t| t.n[0] == [0.0; 3]).collect();
        let keep: Vec<[V; 3]> = m.tris.iter().map(|t| t.n).collect();
        m.apply_normals(NormalMode::Groups);
        for (i, t) in m.tris.iter_mut().enumerate() {
            if !flat[i] {
                t.n = keep[i];
            }
        }
    }
    m.has_colors = vcols.iter().any(|c| c.is_some());
    if m.objects.is_empty() {
        m.objects.push("default".into());
    }
    Ok(m)
}

#[allow(clippy::too_many_arguments)]
fn obj_line(
    line: &str,
    ln: usize,
    vs: &mut Vec<V>,
    vcols: &mut Vec<Option<[f32; 3]>>,
    vts: &mut Vec<[f32; 2]>,
    vns: &mut Vec<V>,
    m: &mut Mesh,
    names: (&mut String, &mut String, &mut String),
    smooth: &mut u32,
    smooth_next: &mut u32,
    any_normals: &mut bool,
    any_missing: &mut bool,
) -> Result<Option<()>, String> {
    let (obj_name, grp_name, mat_name) = names;
    let mut it = line.split_whitespace();
    let Some(cmd) = it.next() else { return Ok(None) };
    let rest: Vec<&str> = it.collect();
    let num = |s: &str| s.parse::<f32>().map_err(|_| format!("line {ln}: bad number '{s}'"));
    match cmd {
        "v" if rest.len() >= 3 => {
            vs.push([num(rest[0])?, num(rest[1])?, num(rest[2])?]);
            vcols.push(if rest.len() >= 6 { Some([num(rest[3])?, num(rest[4])?, num(rest[5])?]) } else { None });
        }
        "vt" if rest.len() >= 2 => vts.push([num(rest[0])?, num(rest[1])?]),
        "vn" if rest.len() >= 3 => vns.push([num(rest[0])?, num(rest[1])?, num(rest[2])?]),
        "o" => *obj_name = rest.join(" "),
        "g" => *grp_name = rest.join(" "),
        "usemtl" => *mat_name = rest.join(" "),
        "s" => {
            *smooth = match rest.first().copied() {
                None | Some("off") | Some("0") => 0,
                Some(v) => v.parse::<u32>().unwrap_or_else(|_| {
                    *smooth_next += 1;
                    *smooth_next
                }),
            }
        }
        "f" if rest.len() >= 3 => {
            let mut corners: Vec<(usize, Option<usize>, Option<usize>)> = Vec::with_capacity(rest.len());
            for tok in &rest {
                let mut parts = tok.split('/');
                let p = obj_index(parts.next().unwrap_or(""), vs.len()).map_err(|e| format!("line {ln}: {e}"))?.ok_or_else(|| format!("line {ln}: a face corner has no vertex"))?;
                let uv = match parts.next() {
                    Some(s) => obj_index(s, vts.len()).map_err(|e| format!("line {ln}: {e}"))?,
                    None => None,
                };
                let n = match parts.next() {
                    Some(s) => obj_index(s, vns.len()).map_err(|e| format!("line {ln}: {e}"))?,
                    None => None,
                };
                corners.push((p, uv, n));
            }
            let object = {
                let nm = if !obj_name.is_empty() { obj_name.as_str() } else if !grp_name.is_empty() { grp_name.as_str() } else { "default" };
                name_index(&mut m.objects, nm)
            };
            let group = name_index(&mut m.groups, if grp_name.is_empty() { obj_name.as_str() } else { grp_name.as_str() });
            let material = name_index(&mut m.materials, mat_name.as_str());
            let pts: Vec<V> = corners.iter().map(|c| vs[c.0]).collect();
            for tri in triangulate(&pts) {
                let c = [corners[tri[0]], corners[tri[1]], corners[tri[2]]];
                let mut t = Tri::blank([vs[c[0].0], vs[c[1].0], vs[c[2].0]]);
                t.object = object;
                t.group = group;
                t.material = material;
                t.smooth = *smooth;
                let mut have_n = true;
                for j in 0..3 {
                    if let Some(u) = c[j].1 {
                        t.uv[j] = vts[u];
                        m.has_uv = true;
                    }
                    match c[j].2 {
                        Some(n) => {
                            t.n[j] = norm(vns[n]);
                            *any_normals = true;
                        }
                        None => have_n = false,
                    }
                    if let Some(col) = vcols[c[j].0] {
                        t.col[j] = col;
                    }
                }
                if !have_n {
                    *any_missing = true;
                    t.n = [[0.0; 3]; 3]; // marks "needs smoothing"
                }
                m.tris.push(t);
            }
        }
        _ => {}
    }
    Ok(Some(()))
}

// ====================================================================== PLY
#[derive(Clone, Copy, PartialEq, Debug)]
enum PType {
    I8,
    U8,
    I16,
    U16,
    I32,
    U32,
    F32,
    F64,
}

fn ptype(s: &str) -> Result<PType, String> {
    Ok(match s {
        "char" | "int8" => PType::I8,
        "uchar" | "uint8" => PType::U8,
        "short" | "int16" => PType::I16,
        "ushort" | "uint16" => PType::U16,
        "int" | "int32" => PType::I32,
        "uint" | "uint32" => PType::U32,
        "float" | "float32" => PType::F32,
        "double" | "float64" => PType::F64,
        o => return Err(format!("unknown PLY type '{o}'")),
    })
}

fn psize(t: PType) -> usize {
    match t {
        PType::I8 | PType::U8 => 1,
        PType::I16 | PType::U16 => 2,
        PType::I32 | PType::U32 | PType::F32 => 4,
        PType::F64 => 8,
    }
}

struct PProp {
    name: String,
    ty: PType,
    /// Some(count type) for list properties
    list: Option<PType>,
}

struct PElem {
    name: String,
    count: usize,
    props: Vec<PProp>,
}

struct PReader<'a> {
    b: &'a [u8],
    pos: usize,
    big: bool,
    ascii: bool,
}

impl PReader<'_> {
    fn read(&mut self, t: PType) -> Result<f64, String> {
        if self.ascii {
            // next whitespace separated token
            while self.pos < self.b.len() && self.b[self.pos].is_ascii_whitespace() {
                self.pos += 1;
            }
            let s = self.pos;
            while self.pos < self.b.len() && !self.b[self.pos].is_ascii_whitespace() {
                self.pos += 1;
            }
            if s == self.pos {
                return Err("the PLY file ends too early".into());
            }
            return std::str::from_utf8(&self.b[s..self.pos]).ok().and_then(|x| x.parse::<f64>().ok()).ok_or_else(|| "bad number in the PLY file".to_string());
        }
        let n = psize(t);
        if self.pos + n > self.b.len() {
            return Err("the PLY file ends too early".into());
        }
        let mut a = [0u8; 8];
        a[..n].copy_from_slice(&self.b[self.pos..self.pos + n]);
        if self.big {
            a[..n].reverse();
        }
        self.pos += n;
        Ok(match t {
            PType::I8 => a[0] as i8 as f64,
            PType::U8 => a[0] as f64,
            PType::I16 => i16::from_le_bytes([a[0], a[1]]) as f64,
            PType::U16 => u16::from_le_bytes([a[0], a[1]]) as f64,
            PType::I32 => i32::from_le_bytes([a[0], a[1], a[2], a[3]]) as f64,
            PType::U32 => u32::from_le_bytes([a[0], a[1], a[2], a[3]]) as f64,
            PType::F32 => f32::from_le_bytes([a[0], a[1], a[2], a[3]]) as f64,
            PType::F64 => f64::from_le_bytes(a),
        })
    }
}

pub fn parse_ply(b: &[u8]) -> Result<Mesh, String> {
    // header
    let end_tag = b"end_header";
    let hp = b.windows(end_tag.len()).position(|w| w == end_tag).ok_or("not a PLY file (no end_header)")?;
    let mut body = hp + end_tag.len();
    // skip the line break after end_header (\n or \r\n)
    if b.get(body) == Some(&b'\r') {
        body += 1;
    }
    if b.get(body) == Some(&b'\n') {
        body += 1;
    }
    let head = std::str::from_utf8(&b[..hp]).map_err(|_| "the PLY header is not text".to_string())?;
    let mut lines = head.lines();
    if lines.next().map(|l| l.trim()) != Some("ply") {
        return Err("not a PLY file (missing 'ply')".into());
    }
    let (mut ascii, mut big) = (false, false);
    let mut elems: Vec<PElem> = Vec::new();
    for l in lines {
        let t: Vec<&str> = l.split_whitespace().collect();
        match t.as_slice() {
            ["format", f, _] => match *f {
                "ascii" => ascii = true,
                "binary_little_endian" => {}
                "binary_big_endian" => big = true,
                o => return Err(format!("unknown PLY format '{o}'")),
            },
            ["element", name, count] => elems.push(PElem { name: (*name).into(), count: count.parse().map_err(|_| "bad element count")?, props: vec![] }),
            ["property", "list", ct, it, name] => {
                let e = elems.last_mut().ok_or("property before element")?;
                e.props.push(PProp { name: (*name).into(), ty: ptype(it)?, list: Some(ptype(ct)?) });
            }
            ["property", ty, name] => {
                let e = elems.last_mut().ok_or("property before element")?;
                e.props.push(PProp { name: (*name).into(), ty: ptype(ty)?, list: None });
            }
            _ => {}
        }
    }
    let mut r = PReader { b, pos: body, big, ascii };
    let mut verts: Vec<V> = Vec::new();
    let mut nrms: Vec<V> = Vec::new();
    let mut uvs: Vec<[f32; 2]> = Vec::new();
    let mut cols: Vec<[f32; 3]> = Vec::new();
    let mut faces: Vec<(Vec<usize>, Option<Vec<[f32; 2]>>)> = Vec::new();
    for e in &elems {
        if e.count > MAX_TRIS * 2 {
            return Err("the PLY file claims too many elements".into());
        }
        for _ in 0..e.count {
            let mut vals: HashMap<&str, f64> = HashMap::new();
            let mut list: Vec<usize> = Vec::new();
            let mut tex: Option<Vec<[f32; 2]>> = None;
            for p in &e.props {
                match p.list {
                    Some(ct) => {
                        let n = r.read(ct)? as usize;
                        if n > 1_000_000 {
                            return Err("a PLY list is too long".into());
                        }
                        let mut v = Vec::with_capacity(n);
                        for _ in 0..n {
                            v.push(r.read(p.ty)?);
                        }
                        if p.name == "vertex_indices" || p.name == "vertex_index" {
                            list = v.iter().map(|x| *x as usize).collect();
                        } else if p.name == "texcoord" {
                            tex = Some(v.chunks(2).filter(|c| c.len() == 2).map(|c| [c[0] as f32, c[1] as f32]).collect());
                        }
                    }
                    None => {
                        let v = r.read(p.ty)?;
                        vals.insert(p.name.as_str(), v);
                    }
                }
            }
            match e.name.as_str() {
                "vertex" => {
                    let g = |k: &str| vals.get(k).copied().unwrap_or(0.0) as f32;
                    verts.push([g("x"), g("y"), g("z")]);
                    if vals.contains_key("nx") {
                        nrms.push(norm([g("nx"), g("ny"), g("nz")]));
                    }
                    let uk = ["u", "s", "texture_u", "texture_s"].iter().find(|k| vals.contains_key(**k));
                    let vk = ["v", "t", "texture_v", "texture_t"].iter().find(|k| vals.contains_key(**k));
                    if let (Some(u), Some(v)) = (uk, vk) {
                        uvs.push([g(u), g(v)]);
                    }
                    if vals.contains_key("red") {
                        let sc = if e.props.iter().any(|p| p.name == "red" && matches!(p.ty, PType::U8)) { 1.0 / 255.0 } else { 1.0 };
                        cols.push([g("red") * sc, g("green") * sc, g("blue") * sc]);
                    }
                }
                "face" => faces.push((list, tex)),
                _ => {}
            }
        }
    }
    let mut m = Mesh { has_normals: nrms.len() == verts.len() && !verts.is_empty(), has_uv: uvs.len() == verts.len() && !verts.is_empty(), has_colors: cols.len() == verts.len() && !verts.is_empty(), ..Default::default() };
    m.objects.push("default".into());
    m.materials.push(String::new());
    m.groups.push("default".into());
    for (idx, tex) in faces {
        if idx.iter().any(|&i| i >= verts.len()) {
            return Err("a PLY face uses a vertex that does not exist".into());
        }
        let pts: Vec<V> = idx.iter().map(|&i| verts[i]).collect();
        for tri in triangulate(&pts) {
            let ids = [idx[tri[0]], idx[tri[1]], idx[tri[2]]];
            let mut t = Tri::blank([verts[ids[0]], verts[ids[1]], verts[ids[2]]]);
            for j in 0..3 {
                if m.has_normals {
                    t.n[j] = nrms[ids[j]];
                }
                if m.has_uv {
                    t.uv[j] = uvs[ids[j]];
                } else if let Some(tx) = &tex {
                    if let Some(u) = tx.get(tri[j]) {
                        t.uv[j] = *u;
                        m.has_uv = true;
                    }
                }
                if m.has_colors {
                    t.col[j] = cols[ids[j]];
                }
            }
            m.tris.push(t);
        }
    }
    if m.tris.is_empty() {
        return Err("the PLY file has no faces".into());
    }
    Ok(m)
}

// ====================================================================== STL
pub fn parse_stl(b: &[u8]) -> Result<Mesh, String> {
    let mut m = Mesh::default();
    m.objects.push("default".into());
    m.materials.push(String::new());
    m.groups.push("default".into());
    let binary = b.len() >= 84 && {
        let n = u32::from_le_bytes([b[80], b[81], b[82], b[83]]) as usize;
        b.len() == 84 + n * 50
    };
    let f = |s: &[u8]| f32::from_le_bytes([s[0], s[1], s[2], s[3]]);
    if binary {
        let n = u32::from_le_bytes([b[80], b[81], b[82], b[83]]) as usize;
        for i in 0..n {
            let o = 84 + i * 50;
            let nn = [f(&b[o..]), f(&b[o + 4..]), f(&b[o + 8..])];
            let v = |k: usize| [f(&b[o + 12 + k * 12..]), f(&b[o + 16 + k * 12..]), f(&b[o + 20 + k * 12..])];
            let mut t = Tri::blank([v(0), v(1), v(2)]);
            if nn.iter().all(|x| x.is_finite()) && len(nn) > 0.5 {
                t.n = [norm(nn); 3];
            }
            m.tris.push(t);
        }
    } else {
        let text = std::str::from_utf8(b).map_err(|_| "the STL file is neither valid binary nor text".to_string())?;
        let mut vs: Vec<V> = Vec::new();
        for l in text.lines() {
            let t: Vec<&str> = l.split_whitespace().collect();
            if t.len() == 4 && t[0] == "vertex" {
                let n = |s: &str| s.parse::<f32>().map_err(|_| format!("bad number '{s}' in the STL file"));
                vs.push([n(t[1])?, n(t[2])?, n(t[3])?]);
            }
        }
        if vs.len() % 3 != 0 || vs.is_empty() {
            return Err("the STL file has no complete facets".into());
        }
        for c in vs.chunks(3) {
            m.tris.push(Tri::blank([c[0], c[1], c[2]]));
        }
    }
    Ok(m)
}

// ====================================================================== JSON (for glTF)
#[derive(Clone, Debug)]
pub enum Json {
    Null,
    Bool(bool),
    Num(f64),
    Str(String),
    Arr(Vec<Json>),
    Obj(Vec<(String, Json)>),
}

impl Json {
    pub fn get(&self, k: &str) -> Option<&Json> {
        match self {
            Json::Obj(v) => v.iter().find(|(n, _)| n == k).map(|(_, j)| j),
            _ => None,
        }
    }
    pub fn arr(&self) -> &[Json] {
        match self {
            Json::Arr(v) => v,
            _ => &[],
        }
    }
    pub fn num(&self) -> Option<f64> {
        match self {
            Json::Num(n) => Some(*n),
            _ => None,
        }
    }
    pub fn usize(&self) -> Option<usize> {
        self.num().filter(|n| *n >= 0.0 && n.fract() == 0.0).map(|n| n as usize)
    }
    pub fn str(&self) -> Option<&str> {
        match self {
            Json::Str(s) => Some(s),
            _ => None,
        }
    }
}

struct JParser<'a> {
    s: &'a [u8],
    i: usize,
    depth: usize,
}

impl JParser<'_> {
    fn ws(&mut self) {
        while self.i < self.s.len() && self.s[self.i].is_ascii_whitespace() {
            self.i += 1;
        }
    }
    fn err<T>(&self, m: &str) -> Result<T, String> {
        Err(format!("JSON error at byte {}: {m}", self.i))
    }
    fn value(&mut self) -> Result<Json, String> {
        self.ws();
        self.depth += 1;
        if self.depth > 64 {
            return self.err("nesting is too deep");
        }
        let r = match self.s.get(self.i) {
            Some(b'{') => {
                self.i += 1;
                let mut v = Vec::new();
                let mut after_comma = false;
                loop {
                    self.ws();
                    if self.s.get(self.i) == Some(&b'}') && !after_comma {
                        self.i += 1;
                        break;
                    }
                    let k = match self.value()? {
                        Json::Str(s) => s,
                        _ => return self.err("object key must be a string"),
                    };
                    self.ws();
                    if self.s.get(self.i) != Some(&b':') {
                        return self.err("expected ':'");
                    }
                    self.i += 1;
                    v.push((k, self.value()?));
                    self.ws();
                    match self.s.get(self.i) {
                        Some(b',') => {
                            self.i += 1;
                            after_comma = true;
                        }
                        Some(b'}') => {
                            self.i += 1;
                            break;
                        }
                        _ => return self.err("expected ',' or '}'"),
                    }
                }
                Json::Obj(v)
            }
            Some(b'[') => {
                self.i += 1;
                let mut v = Vec::new();
                let mut after_comma = false;
                loop {
                    self.ws();
                    if self.s.get(self.i) == Some(&b']') && !after_comma {
                        self.i += 1;
                        break;
                    }
                    v.push(self.value()?);
                    self.ws();
                    match self.s.get(self.i) {
                        Some(b',') => {
                            self.i += 1;
                            after_comma = true;
                        }
                        Some(b']') => {
                            self.i += 1;
                            break;
                        }
                        _ => return self.err("expected ',' or ']'"),
                    }
                }
                Json::Arr(v)
            }
            Some(b'"') => {
                self.i += 1;
                let mut out = String::new();
                loop {
                    let Some(&c) = self.s.get(self.i) else { return self.err("unterminated string") };
                    self.i += 1;
                    match c {
                        b'"' => break,
                        b'\\' => {
                            let Some(&e) = self.s.get(self.i) else { return self.err("bad escape") };
                            self.i += 1;
                            match e {
                                b'n' => out.push('\n'),
                                b't' => out.push('\t'),
                                b'r' => out.push('\r'),
                                b'b' => out.push('\u{8}'),
                                b'f' => out.push('\u{c}'),
                                b'/' | b'\\' | b'"' => out.push(e as char),
                                b'u' => {
                                    let hex = |p: &mut Self| -> Result<u32, String> {
                                        let h = p.s.get(p.i..p.i + 4).and_then(|x| std::str::from_utf8(x).ok()).and_then(|x| u32::from_str_radix(x, 16).ok());
                                        p.i += 4;
                                        h.ok_or_else(|| "bad \\u escape".to_string())
                                    };
                                    let mut cp = hex(self)?;
                                    if (0xD800..0xDC00).contains(&cp) && self.s.get(self.i) == Some(&b'\\') && self.s.get(self.i + 1) == Some(&b'u') {
                                        self.i += 2;
                                        let lo = hex(self)?;
                                        cp = 0x10000 + ((cp - 0xD800) << 10) + (lo.wrapping_sub(0xDC00) & 0x3FF);
                                    }
                                    out.push(char::from_u32(cp).unwrap_or('\u{fffd}'));
                                }
                                _ => return self.err("bad escape"),
                            }
                        }
                        _ => {
                            // copy a whole UTF-8 sequence
                            let start = self.i - 1;
                            let mut end = self.i;
                            while end < self.s.len() && (self.s[end] & 0xC0) == 0x80 {
                                end += 1;
                            }
                            out.push_str(std::str::from_utf8(&self.s[start..end]).unwrap_or("\u{fffd}"));
                            self.i = end;
                        }
                    }
                }
                Json::Str(out)
            }
            Some(b't') if self.s[self.i..].starts_with(b"true") => {
                self.i += 4;
                Json::Bool(true)
            }
            Some(b'f') if self.s[self.i..].starts_with(b"false") => {
                self.i += 5;
                Json::Bool(false)
            }
            Some(b'n') if self.s[self.i..].starts_with(b"null") => {
                self.i += 4;
                Json::Null
            }
            Some(c) if *c == b'-' || c.is_ascii_digit() => {
                let st = self.i;
                while self.i < self.s.len() && matches!(self.s[self.i], b'-' | b'+' | b'.' | b'e' | b'E' | b'0'..=b'9') {
                    self.i += 1;
                }
                match std::str::from_utf8(&self.s[st..self.i]).ok().and_then(|x| x.parse::<f64>().ok()) {
                    Some(n) => Json::Num(n),
                    None => return self.err("bad number"),
                }
            }
            _ => return self.err("unexpected character"),
        };
        self.depth -= 1;
        Ok(r)
    }
}

pub fn parse_json(s: &str) -> Result<Json, String> {
    let mut p = JParser { s: s.as_bytes(), i: 0, depth: 0 };
    let v = p.value()?;
    p.ws();
    if p.i != s.len() {
        return Err(format!("JSON error at byte {}: extra text after the value", p.i));
    }
    Ok(v)
}

// ====================================================================== glTF / GLB
fn b64(s: &str) -> Result<Vec<u8>, String> {
    let mut out = Vec::with_capacity(s.len() * 3 / 4);
    let (mut acc, mut bits) = (0u32, 0);
    for c in s.bytes() {
        let v = match c {
            b'A'..=b'Z' => c - b'A',
            b'a'..=b'z' => c - b'a' + 26,
            b'0'..=b'9' => c - b'0' + 52,
            b'+' | b'-' => 62,
            b'/' | b'_' => 63,
            b'=' | b'\n' | b'\r' | b' ' => continue,
            _ => return Err("bad base64 in a glTF buffer".into()),
        };
        acc = (acc << 6) | v as u32;
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            out.push((acc >> bits) as u8);
            acc &= (1 << bits) - 1;
        }
    }
    Ok(out)
}

fn percent_decode(s: &str) -> String {
    let b = s.as_bytes();
    let mut out = Vec::new();
    let mut i = 0;
    while i < b.len() {
        if b[i] == b'%' && i + 2 < b.len() {
            if let Some(v) = std::str::from_utf8(&b[i + 1..i + 3]).ok().and_then(|h| u8::from_str_radix(h, 16).ok()) {
                out.push(v);
                i += 3;
                continue;
            }
        }
        out.push(b[i]);
        i += 1;
    }
    String::from_utf8_lossy(&out).to_string()
}

type M4 = [f32; 16]; // column-major

fn m4_identity() -> M4 {
    [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]
}

fn m4_mul(a: &M4, b: &M4) -> M4 {
    let mut o = [0.0f32; 16];
    for c in 0..4 {
        for r in 0..4 {
            o[c * 4 + r] = (0..4).map(|k| a[k * 4 + r] * b[c * 4 + k]).sum();
        }
    }
    o
}

fn m4_trs(t: [f32; 3], q: [f32; 4], s: [f32; 3]) -> M4 {
    let (x, y, z, w) = (q[0], q[1], q[2], q[3]);
    let n = (x * x + y * y + z * z + w * w).sqrt().max(1e-12);
    let (x, y, z, w) = (x / n, y / n, z / n, w / n);
    let r = [
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y + z * w), 2.0 * (x * z - y * w)],
        [2.0 * (x * y - z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z + x * w)],
        [2.0 * (x * z + y * w), 2.0 * (y * z - x * w), 1.0 - 2.0 * (x * x + y * y)],
    ];
    let mut m = m4_identity();
    for c in 0..3 {
        for rr in 0..3 {
            m[c * 4 + rr] = r[c][rr] * s[c];
        }
    }
    m[12] = t[0];
    m[13] = t[1];
    m[14] = t[2];
    m
}

fn m4_point(m: &M4, p: V) -> V {
    [m[0] * p[0] + m[4] * p[1] + m[8] * p[2] + m[12], m[1] * p[0] + m[5] * p[1] + m[9] * p[2] + m[13], m[2] * p[0] + m[6] * p[1] + m[10] * p[2] + m[14]]
}

/// Inverse transpose of the upper 3x3 (for normals).
fn m4_normal_matrix(m: &M4) -> [[f32; 3]; 3] {
    let a = [[m[0], m[4], m[8]], [m[1], m[5], m[9]], [m[2], m[6], m[10]]];
    let det = a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1]) - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0]) + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]);
    if det.abs() < 1e-20 {
        return [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    }
    let id = 1.0 / det;
    // cofactor matrix / det is the inverse transpose
    let c = |r: usize, c: usize| -> f32 {
        let (r1, r2) = ((r + 1) % 3, (r + 2) % 3);
        let (c1, c2) = ((c + 1) % 3, (c + 2) % 3);
        (a[r1][c1] * a[r2][c2] - a[r1][c2] * a[r2][c1]) * id
    };
    [[c(0, 0), c(0, 1), c(0, 2)], [c(1, 0), c(1, 1), c(1, 2)], [c(2, 0), c(2, 1), c(2, 2)]]
}

struct Gltf<'a> {
    j: &'a Json,
    bufs: Vec<Vec<u8>>,
}

impl Gltf<'_> {
    fn view(&self, bv: usize) -> Result<(&[u8], usize), String> {
        let v = self.j.get("bufferViews").map(|x| x.arr()).unwrap_or(&[]).get(bv).ok_or("a glTF accessor points at a missing bufferView")?;
        let buf = v.get("buffer").and_then(|x| x.usize()).ok_or("bufferView without buffer")?;
        let off = v.get("byteOffset").and_then(|x| x.usize()).unwrap_or(0);
        let len = v.get("byteLength").and_then(|x| x.usize()).ok_or("bufferView without byteLength")?;
        let stride = v.get("byteStride").and_then(|x| x.usize()).unwrap_or(0);
        let data = self.bufs.get(buf).ok_or("a glTF bufferView points at a missing buffer")?;
        if off + len > data.len() {
            return Err("a glTF bufferView reaches past the end of its buffer".into());
        }
        Ok((&data[off..off + len], stride))
    }

    /// Values of accessor `i` as floats (integer types are normalised when the accessor says so) and the component count.
    fn accessor(&self, i: usize) -> Result<(Vec<f32>, usize), String> {
        let a = self.j.get("accessors").map(|x| x.arr()).unwrap_or(&[]).get(i).ok_or("a glTF primitive points at a missing accessor")?;
        let count = a.get("count").and_then(|x| x.usize()).ok_or("accessor without count")?;
        let ct = a.get("componentType").and_then(|x| x.usize()).ok_or("accessor without componentType")?;
        let comps = match a.get("type").and_then(|x| x.str()) {
            Some("SCALAR") => 1,
            Some("VEC2") => 2,
            Some("VEC3") => 3,
            Some("VEC4") => 4,
            _ => return Err("unsupported glTF accessor type".into()),
        };
        let norm = matches!(a.get("normalized"), Some(Json::Bool(true)));
        let csize = match ct {
            5120 | 5121 => 1,
            5122 | 5123 => 2,
            5125 | 5126 => 4,
            _ => return Err("unsupported glTF component type".into()),
        };
        if a.get("sparse").is_some() {
            return Err("sparse glTF accessors are not supported".into());
        }
        let Some(bv) = a.get("bufferView").and_then(|x| x.usize()) else { return Ok((vec![0.0; count * comps], comps)) };
        let (data, stride) = self.view(bv)?;
        let aoff = a.get("byteOffset").and_then(|x| x.usize()).unwrap_or(0);
        let elem = csize * comps;
        let stride = if stride == 0 { elem } else { stride };
        let mut out = Vec::with_capacity(count * comps);
        for e in 0..count {
            for c in 0..comps {
                let o = aoff + e * stride + c * csize;
                if o + csize > data.len() {
                    return Err("a glTF accessor reaches past the end of its data".into());
                }
                let d = &data[o..];
                out.push(match ct {
                    5120 => {
                        let v = d[0] as i8 as f32;
                        if norm { (v / 127.0).max(-1.0) } else { v }
                    }
                    5121 => {
                        let v = d[0] as f32;
                        if norm { v / 255.0 } else { v }
                    }
                    5122 => {
                        let v = i16::from_le_bytes([d[0], d[1]]) as f32;
                        if norm { (v / 32767.0).max(-1.0) } else { v }
                    }
                    5123 => {
                        let v = u16::from_le_bytes([d[0], d[1]]) as f32;
                        if norm { v / 65535.0 } else { v }
                    }
                    5125 => u32::from_le_bytes([d[0], d[1], d[2], d[3]]) as f32,
                    _ => f32::from_le_bytes([d[0], d[1], d[2], d[3]]),
                });
            }
        }
        Ok((out, comps))
    }

    fn indices(&self, i: usize) -> Result<Vec<u32>, String> {
        let a = self.j.get("accessors").map(|x| x.arr()).unwrap_or(&[]).get(i).ok_or("missing index accessor")?;
        let count = a.get("count").and_then(|x| x.usize()).ok_or("accessor without count")?;
        let ct = a.get("componentType").and_then(|x| x.usize()).ok_or("accessor without componentType")?;
        let bv = a.get("bufferView").and_then(|x| x.usize()).ok_or("index accessor without bufferView")?;
        let (data, stride) = self.view(bv)?;
        let aoff = a.get("byteOffset").and_then(|x| x.usize()).unwrap_or(0);
        let cs = match ct {
            5121 => 1,
            5123 => 2,
            5125 => 4,
            _ => return Err("indices must be unsigned byte, short or int".into()),
        };
        let stride = if stride == 0 { cs } else { stride };
        let mut out = Vec::with_capacity(count);
        for e in 0..count {
            let o = aoff + e * stride;
            if o + cs > data.len() {
                return Err("an index accessor reaches past the end of its data".into());
            }
            out.push(match cs {
                1 => data[o] as u32,
                2 => u16::from_le_bytes([data[o], data[o + 1]]) as u32,
                _ => u32::from_le_bytes([data[o], data[o + 1], data[o + 2], data[o + 3]]),
            });
        }
        Ok(out)
    }
}

pub fn parse_gltf(bytes: &[u8], dir: Option<&Path>) -> Result<Mesh, String> {
    // GLB container: 12 byte header then chunks
    let (json_text, glb_bin): (String, Option<Vec<u8>>) = if bytes.len() >= 12 && &bytes[0..4] == b"glTF" {
        let total = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize;
        let end = total.min(bytes.len());
        let (mut json, mut bin) = (None, None);
        let mut p = 12;
        while p + 8 <= end {
            let l = u32::from_le_bytes([bytes[p], bytes[p + 1], bytes[p + 2], bytes[p + 3]]) as usize;
            let ty = &bytes[p + 4..p + 8];
            if p + 8 + l > end {
                return Err("a GLB chunk reaches past the end of the file".into());
            }
            let data = &bytes[p + 8..p + 8 + l];
            if ty == b"JSON" && json.is_none() {
                json = Some(String::from_utf8_lossy(data).to_string());
            } else if ty == b"BIN\0" && bin.is_none() {
                bin = Some(data.to_vec());
            }
            p += 8 + l + ((4 - l % 4) % 4);
        }
        (json.ok_or("the GLB file has no JSON chunk")?, bin)
    } else {
        (String::from_utf8(bytes.to_vec()).map_err(|_| "the glTF file is not text".to_string())?, None)
    };
    let j = parse_json(&json_text)?;
    if j.get("asset").is_none() {
        return Err("not a glTF file (no 'asset')".into());
    }
    // buffers
    let mut bufs: Vec<Vec<u8>> = Vec::new();
    for (i, b) in j.get("buffers").map(|x| x.arr()).unwrap_or(&[]).iter().enumerate() {
        match b.get("uri").and_then(|u| u.str()) {
            None => bufs.push(glb_bin.clone().filter(|_| i == 0).ok_or("a glTF buffer has no data")?),
            Some(u) if u.starts_with("data:") => {
                let (_, payload) = u.split_once(',').ok_or("bad data URI")?;
                bufs.push(b64(payload)?);
            }
            Some(u) => {
                let d = dir.ok_or_else(|| format!("the buffer '{u}' is an external file; open the .gltf from disk"))?;
                let rel = percent_decode(u);
                if rel.contains("..") || Path::new(&rel).is_absolute() {
                    return Err(format!("the buffer path '{rel}' is not allowed"));
                }
                bufs.push(std::fs::read(d.join(&rel)).map_err(|e| format!("{rel}: {e}"))?);
            }
        }
    }
    let g = Gltf { j: &j, bufs };
    let mut m = Mesh::default();
    m.groups.push("default".into());
    let mats: Vec<String> = j.get("materials").map(|x| x.arr()).unwrap_or(&[]).iter().enumerate().map(|(i, x)| x.get("name").and_then(|n| n.str()).map(String::from).unwrap_or_else(|| format!("material{i}"))).collect();
    m.materials = mats.clone();
    if m.materials.is_empty() {
        m.materials.push(String::new());
    }
    // node traversal
    let nodes = j.get("nodes").map(|x| x.arr()).unwrap_or(&[]);
    let mut roots: Vec<usize> = Vec::new();
    let scenes = j.get("scenes").map(|x| x.arr()).unwrap_or(&[]);
    let sc = j.get("scene").and_then(|x| x.usize()).unwrap_or(0);
    if let Some(s) = scenes.get(sc) {
        roots = s.get("nodes").map(|x| x.arr()).unwrap_or(&[]).iter().filter_map(|x| x.usize()).collect();
    } else if nodes.is_empty() {
        // no scene graph: every mesh with the identity transform
        for i in 0..j.get("meshes").map(|x| x.arr().len()).unwrap_or(0) {
            gltf_mesh(&g, &mut m, i, &m4_identity(), &format!("mesh{i}"), &mats)?;
        }
    } else {
        // nodes that nobody lists as a child
        let mut child = vec![false; nodes.len()];
        for n in nodes {
            for c in n.get("children").map(|x| x.arr()).unwrap_or(&[]) {
                if let Some(c) = c.usize() {
                    if c < child.len() {
                        child[c] = true;
                    }
                }
            }
        }
        roots = (0..nodes.len()).filter(|i| !child[*i]).collect();
    }
    let mut stack: Vec<(usize, M4, usize)> = roots.into_iter().rev().map(|r| (r, m4_identity(), 0)).collect();
    let mut visited = 0usize;
    while let Some((ni, parent, depth)) = stack.pop() {
        visited += 1;
        if visited > 100_000 || depth > 64 {
            return Err("the glTF scene graph is too large or loops".into());
        }
        let Some(n) = nodes.get(ni) else { return Err("a glTF scene points at a missing node".into()) };
        let local = if let Some(mx) = n.get("matrix") {
            let v: Vec<f32> = mx.arr().iter().filter_map(|x| x.num()).map(|x| x as f32).collect();
            if v.len() == 16 {
                let mut a = [0.0f32; 16];
                a.copy_from_slice(&v);
                a
            } else {
                m4_identity()
            }
        } else {
            let f3 = |k: &str, d: [f32; 3]| -> [f32; 3] {
                let v: Vec<f32> = n.get(k).map(|x| x.arr()).unwrap_or(&[]).iter().filter_map(|x| x.num()).map(|x| x as f32).collect();
                if v.len() == 3 { [v[0], v[1], v[2]] } else { d }
            };
            let q: Vec<f32> = n.get("rotation").map(|x| x.arr()).unwrap_or(&[]).iter().filter_map(|x| x.num()).map(|x| x as f32).collect();
            m4_trs(f3("translation", [0.0; 3]), if q.len() == 4 { [q[0], q[1], q[2], q[3]] } else { [0.0, 0.0, 0.0, 1.0] }, f3("scale", [1.0; 3]))
        };
        let world = m4_mul(&parent, &local);
        if let Some(mi) = n.get("mesh").and_then(|x| x.usize()) {
            let name = n.get("name").and_then(|x| x.str()).map(String::from).unwrap_or_else(|| format!("node{ni}"));
            gltf_mesh(&g, &mut m, mi, &world, &name, &mats)?;
        }
        for c in n.get("children").map(|x| x.arr()).unwrap_or(&[]).iter().rev() {
            if let Some(c) = c.usize() {
                stack.push((c, world, depth + 1));
            }
        }
    }
    if m.tris.is_empty() {
        return Err("the glTF file has no triangles".into());
    }
    Ok(m)
}

fn gltf_mesh(g: &Gltf, m: &mut Mesh, mi: usize, world: &M4, node_name: &str, _mats: &[String]) -> Result<(), String> {
    let mesh = g.j.get("meshes").map(|x| x.arr()).unwrap_or(&[]).get(mi).ok_or("a glTF node points at a missing mesh")?;
    let object = name_index(&mut m.objects, node_name);
    let nm = m4_normal_matrix(world);
    // a mirrored transform flips the winding
    let det = world[0] * (world[5] * world[10] - world[6] * world[9]) - world[4] * (world[1] * world[10] - world[2] * world[9]) + world[8] * (world[1] * world[6] - world[2] * world[5]);
    for (pi, prim) in mesh.get("primitives").map(|x| x.arr()).unwrap_or(&[]).iter().enumerate() {
        let mode = prim.get("mode").and_then(|x| x.usize()).unwrap_or(4);
        if !matches!(mode, 4..=6) {
            continue; // points and lines carry no surface
        }
        let attrs = prim.get("attributes").ok_or("a glTF primitive has no attributes")?;
        let acc = |k: &str| attrs.get(k).and_then(|x| x.usize());
        let (pos, _) = g.accessor(acc("POSITION").ok_or("a glTF primitive has no POSITION")?)?;
        let nvert = pos.len() / 3;
        let nrm = acc("NORMAL").map(|a| g.accessor(a)).transpose()?;
        let uv0 = acc("TEXCOORD_0").map(|a| g.accessor(a)).transpose()?;
        let uv1 = acc("TEXCOORD_1").map(|a| g.accessor(a)).transpose()?;
        let col = acc("COLOR_0").map(|a| g.accessor(a)).transpose()?;
        let idx: Vec<u32> = match prim.get("indices").and_then(|x| x.usize()) {
            Some(a) => g.indices(a)?,
            None => (0..nvert as u32).collect(),
        };
        if idx.iter().any(|&i| i as usize >= nvert) {
            return Err("a glTF index points past the vertex list".into());
        }
        let tris: Vec<[u32; 3]> = match mode {
            4 => idx.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect(),
            5 => (0..idx.len().saturating_sub(2)).map(|i| if i % 2 == 0 { [idx[i], idx[i + 1], idx[i + 2]] } else { [idx[i + 1], idx[i], idx[i + 2]] }).collect(),
            _ => (1..idx.len().saturating_sub(1)).map(|i| [idx[0], idx[i], idx[i + 1]]).collect(),
        };
        let material = prim.get("material").and_then(|x| x.usize()).unwrap_or(0).min(m.materials.len().saturating_sub(1)) as u32;
        let group = name_index(&mut m.groups, &format!("{node_name}.{pi}"));
        let p3 = |i: u32| -> V { m4_point(world, [pos[i as usize * 3], pos[i as usize * 3 + 1], pos[i as usize * 3 + 2]]) };
        for t in tris {
            let order = if det < 0.0 { [0usize, 2, 1] } else { [0, 1, 2] };
            let ids = [t[order[0]], t[order[1]], t[order[2]]];
            let mut tri = Tri::blank([p3(ids[0]), p3(ids[1]), p3(ids[2])]);
            tri.object = object;
            tri.group = group;
            tri.material = material;
            for c in 0..3 {
                let i = ids[c] as usize;
                if let Some((n, k)) = &nrm {
                    let v = [n[i * k], n[i * k + 1], n[i * k + 2]];
                    tri.n[c] = norm([nm[0][0] * v[0] + nm[0][1] * v[1] + nm[0][2] * v[2], nm[1][0] * v[0] + nm[1][1] * v[1] + nm[1][2] * v[2], nm[2][0] * v[0] + nm[2][1] * v[1] + nm[2][2] * v[2]]);
                }
                if let Some((u, k)) = &uv0 {
                    tri.uv[c] = [u[i * k], 1.0 - u[i * k + 1]];
                }
                if let Some((u, k)) = &uv1 {
                    tri.uv1[c] = [u[i * k], 1.0 - u[i * k + 1]];
                }
                if let Some((cl, k)) = &col {
                    tri.col[c] = [cl[i * k], cl[i * k + 1], cl[i * k + 2]];
                }
            }
            m.tris.push(tri);
        }
        m.has_normals |= nrm.is_some();
        m.has_uv |= uv0.is_some();
        m.has_uv1 |= uv1.is_some();
        m.has_colors |= col.is_some();
    }
    Ok(())
}
