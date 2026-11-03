// SPDX-License-Identifier: MIT OR Apache-2.0
//! Mesh map baking: transfers the detail of high-poly meshes onto the UV layout of low-poly meshes.
//!
//! * Rays start on a cage (the low-poly surface pushed out along averaged normals, or your own cage mesh), travel
//!   inwards through the front distance and on through the rear distance, and hit the high-poly meshes.
//! * Every texel is sampled with an n x n grid (anti-aliasing); samples that fall inside a UV triangle are shaded
//!   and averaged, so island borders do not blur into the background. Padding then pushes island colours outwards.
//! * Low and high objects can be matched by name (`rock_low` bakes only against `rock_high`).
//! * Tangent-space maps use a per-vertex basis computed from the UVs (angle weighted, split at UV seams and mirrored
//!   islands), in OpenGL (Y up) or DirectX (Y down) convention.
//! * Maps: tangent / world normal, position, ambient occlusion, bent normals, curvature, thickness, height,
//!   object / material / polygroup ids, vertex colour, opacity, world direction and the UV wireframe.
//!
//! The baker is multi-threaded, reports progress and can be cancelled.
use crate::bvh::{Bvh, Filter, Hit};
use crate::noise::rnd;
use crate::tex::Tex;
use crate::v3::*;
pub use crate::meshio::{Mesh, NormalMode, Tri};
use std::collections::{BTreeMap, HashMap};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::Mutex;
use std::time::Instant;

/// Compatibility wrapper: parse an OBJ file's text.
pub fn parse_obj(text: &str) -> Result<Mesh, String> {
    crate::meshio::parse_obj(text)
}

// ====================================================================== maps and settings
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum MapKind {
    Normal,
    WorldNormal,
    Position,
    AmbientOcclusion,
    BentNormal,
    WorldBentNormal,
    Curvature,
    Thickness,
    Height,
    ObjectId,
    MaterialId,
    GroupId,
    VertexColor,
    Opacity,
    WorldDirection,
    UvLayout,
}

impl MapKind {
    pub const ALL: [MapKind; 16] = [
        MapKind::Normal,
        MapKind::WorldNormal,
        MapKind::Position,
        MapKind::AmbientOcclusion,
        MapKind::BentNormal,
        MapKind::WorldBentNormal,
        MapKind::Curvature,
        MapKind::Thickness,
        MapKind::Height,
        MapKind::ObjectId,
        MapKind::MaterialId,
        MapKind::GroupId,
        MapKind::VertexColor,
        MapKind::Opacity,
        MapKind::WorldDirection,
        MapKind::UvLayout,
    ];

    /// The name used for files, package resources and Baked Map nodes.
    pub fn id(self) -> &'static str {
        match self {
            MapKind::Normal => "normal",
            MapKind::WorldNormal => "world_normal",
            MapKind::Position => "position",
            MapKind::AmbientOcclusion => "ambient_occlusion",
            MapKind::BentNormal => "bent_normal",
            MapKind::WorldBentNormal => "world_bent_normal",
            MapKind::Curvature => "curvature",
            MapKind::Thickness => "thickness",
            MapKind::Height => "height",
            MapKind::ObjectId => "object_id",
            MapKind::MaterialId => "material_id",
            MapKind::GroupId => "group_id",
            MapKind::VertexColor => "vertex_color",
            MapKind::Opacity => "opacity",
            MapKind::WorldDirection => "world_direction",
            MapKind::UvLayout => "uv_layout",
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            MapKind::Normal => "Normal (tangent space)",
            MapKind::WorldNormal => "World space normal",
            MapKind::Position => "Position",
            MapKind::AmbientOcclusion => "Ambient occlusion",
            MapKind::BentNormal => "Bent normal (tangent space)",
            MapKind::WorldBentNormal => "World space bent normal",
            MapKind::Curvature => "Curvature",
            MapKind::Thickness => "Thickness",
            MapKind::Height => "Height from cage",
            MapKind::ObjectId => "Object ID",
            MapKind::MaterialId => "Material ID",
            MapKind::GroupId => "Polygroup ID",
            MapKind::VertexColor => "Vertex colour",
            MapKind::Opacity => "Opacity (ray hits)",
            MapKind::WorldDirection => "World space direction",
            MapKind::UvLayout => "UV layout",
        }
    }

    pub fn from_id(s: &str) -> Option<MapKind> {
        MapKind::ALL.iter().copied().find(|m| m.id() == s || (s == "id" && *m == MapKind::ObjectId) || (s == "ao" && *m == MapKind::AmbientOcclusion))
    }

    /// Colour of texels that no island reaches.
    pub fn background(self) -> [f32; 4] {
        match self {
            MapKind::Normal | MapKind::WorldNormal | MapKind::BentNormal | MapKind::WorldBentNormal => [0.5, 0.5, 1.0, 1.0],
            MapKind::AmbientOcclusion => [1.0, 1.0, 1.0, 1.0],
            MapKind::Curvature => [0.5, 0.5, 0.5, 1.0],
            _ => [0.0, 0.0, 0.0, 1.0],
        }
    }

    /// Maps that are colours or vectors (stored as RGB) rather than grayscale values.
    pub fn is_colour(self) -> bool {
        matches!(self, MapKind::Normal | MapKind::WorldNormal | MapKind::Position | MapKind::BentNormal | MapKind::WorldBentNormal | MapKind::ObjectId | MapKind::MaterialId | MapKind::GroupId | MapKind::VertexColor)
    }
}

/// Tangent basis used for tangent-space maps.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Basis {
    /// per-vertex tangents from the UV derivatives, angle weighted, split at UV seams and mirrored UVs
    Smooth,
    /// one basis per triangle (flat)
    Face,
}

/// Normals of the cage used to push rays out.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CageNormals {
    /// the low-poly shading normals (hard edges open gaps in the cage)
    Shading,
    /// normals averaged over every face that meets at a position (a closed cage)
    Averaged,
}

/// Where the rays are traced.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Device {
    /// the GPU when one is usable, otherwise the CPU
    Auto,
    Cpu,
    /// the GPU, or an error when there is none
    Gpu,
}

impl Device {
    pub fn id(self) -> &'static str {
        match self {
            Device::Auto => "auto",
            Device::Cpu => "cpu",
            Device::Gpu => "gpu",
        }
    }
}

#[derive(Clone, Debug)]
pub struct BakeSettings {
    pub width: usize,
    pub height: usize,
    /// supersampling grid: n x n samples per texel (1, 2, 3, 4 or 8)
    pub aa: u32,
    /// texels of padding around islands
    pub padding: usize,
    /// ray start distance in front of the low-poly surface, as a fraction of the high-poly bounding box diagonal
    pub front: f32,
    /// how far behind the low-poly surface rays keep looking, same unit
    pub rear: f32,
    /// distance a ray start is nudged to avoid hitting itself, same unit
    pub bias: f32,
    pub cage_normals: CageNormals,
    /// match low and high objects by their names (suffix list below)
    pub match_by_name: bool,
    pub low_suffixes: Vec<String>,
    pub high_suffixes: Vec<String>,
    pub low_normals: NormalMode,
    pub high_normals: NormalMode,
    pub basis: Basis,
    /// DirectX convention (green channel points down)
    pub flip_y: bool,
    /// UV set of the low-poly mesh (0 or 1)
    pub uv_set: usize,
    /// bake only triangles of this material (empty = all)
    pub material: String,
    /// ignore high-poly triangles that face away from the ray
    pub ignore_backfaces: bool,
    pub maps: Vec<MapKind>,
    // ---- ambient occlusion and bent normals
    pub ao_rays: u32,
    /// occlusion distance as a fraction of the diagonal
    pub ao_dist: f32,
    /// 0 = a hit counts fully; higher values fade hits that are far away
    pub ao_falloff: f32,
    // ---- thickness
    pub thick_rays: u32,
    pub thick_dist: f32,
    // ---- curvature
    /// features with a radius below this (fraction of the diagonal) show as curved
    pub curv_radius: f32,
    pub curv_smooth: u32,
    // ---- position
    /// None = the bounding box of the high-poly meshes
    pub pos_range: Option<(V, V)>,
    // ---- world direction
    pub direction: V,
    /// where the rays are traced
    pub device: Device,
}

impl Default for BakeSettings {
    fn default() -> Self {
        BakeSettings {
            width: 1024,
            height: 1024,
            aa: 2,
            padding: 16,
            front: 0.05,
            rear: 0.05,
            bias: 0.0005,
            cage_normals: CageNormals::Averaged,
            match_by_name: true,
            low_suffixes: ["_low", "_lp", ".low", "-low", "_lowpoly"].iter().map(|s| s.to_string()).collect(),
            high_suffixes: ["_high", "_hp", ".high", "-high", "_highpoly"].iter().map(|s| s.to_string()).collect(),
            low_normals: NormalMode::Keep,
            high_normals: NormalMode::Keep,
            basis: Basis::Smooth,
            flip_y: false,
            uv_set: 0,
            material: String::new(),
            ignore_backfaces: false,
            maps: vec![MapKind::Normal, MapKind::WorldNormal, MapKind::Position, MapKind::AmbientOcclusion, MapKind::Curvature, MapKind::Thickness, MapKind::ObjectId],
            ao_rays: 64,
            ao_dist: 0.3,
            ao_falloff: 0.0,
            thick_rays: 32,
            thick_dist: 0.5,
            curv_radius: 0.05,
            curv_smooth: 2,
            pos_range: None,
            direction: [0.0, 1.0, 0.0],
            device: Device::Auto,
        }
    }
}

impl BakeSettings {
    /// Fast settings for looking at a result.
    pub fn preview() -> BakeSettings {
        BakeSettings { width: 512, height: 512, aa: 1, padding: 8, ao_rays: 16, thick_rays: 8, ..Default::default() }
    }
    /// Good quality for game assets.
    pub fn game() -> BakeSettings {
        BakeSettings { width: 2048, height: 2048, aa: 2, padding: 16, ao_rays: 64, thick_rays: 32, ..Default::default() }
    }
    /// Highest quality.
    pub fn final_quality() -> BakeSettings {
        BakeSettings { width: 4096, height: 4096, aa: 4, padding: 32, ao_rays: 256, thick_rays: 64, ..Default::default() }
    }

    pub fn wants(&self, m: MapKind) -> bool {
        self.maps.contains(&m)
    }

    /// Text form (one `key value` per line) used in packages and config files.
    pub fn to_text(&self) -> String {
        let mut s = String::new();
        let mut w = |k: &str, v: String| {
            s.push_str(k);
            s.push(' ');
            s.push_str(&v);
            s.push('\n');
        };
        w("size", format!("{} {}", self.width, self.height));
        w("aa", self.aa.to_string());
        w("padding", self.padding.to_string());
        w("front", self.front.to_string());
        w("rear", self.rear.to_string());
        w("bias", self.bias.to_string());
        w("cage_normals", (if self.cage_normals == CageNormals::Averaged { "averaged" } else { "shading" }).into());
        w("match_by_name", (self.match_by_name as u8).to_string());
        w("low_suffixes", self.low_suffixes.join(","));
        w("high_suffixes", self.high_suffixes.join(","));
        let nm = |m: NormalMode| match m {
            NormalMode::Keep => "keep".to_string(),
            NormalMode::Flat => "flat".to_string(),
            NormalMode::Angle(a) => format!("angle:{a}"),
            NormalMode::Groups => "groups".to_string(),
        };
        w("low_normals", nm(self.low_normals));
        w("high_normals", nm(self.high_normals));
        w("basis", (if self.basis == Basis::Smooth { "smooth" } else { "face" }).into());
        w("flip_y", (self.flip_y as u8).to_string());
        w("uv_set", self.uv_set.to_string());
        w("material", self.material.clone());
        w("ignore_backfaces", (self.ignore_backfaces as u8).to_string());
        w("maps", self.maps.iter().map(|m| m.id()).collect::<Vec<_>>().join(","));
        w("ao_rays", self.ao_rays.to_string());
        w("ao_dist", self.ao_dist.to_string());
        w("ao_falloff", self.ao_falloff.to_string());
        w("thick_rays", self.thick_rays.to_string());
        w("thick_dist", self.thick_dist.to_string());
        w("curv_radius", self.curv_radius.to_string());
        w("curv_smooth", self.curv_smooth.to_string());
        if let Some((a, b)) = self.pos_range {
            w("pos_range", format!("{} {} {} {} {} {}", a[0], a[1], a[2], b[0], b[1], b[2]));
        }
        w("direction", format!("{} {} {}", self.direction[0], self.direction[1], self.direction[2]));
        w("device", self.device.id().to_string());
        s
    }

    /// Set one option from its text form; used by `from_text` and the command line.
    pub fn set(&mut self, key: &str, val: &str) -> Result<(), String> {
        let f = |v: &str| v.trim().parse::<f32>().ok().filter(|x| x.is_finite()).ok_or_else(|| format!("{key}: '{v}' is not a number"));
        let u = |v: &str| v.trim().parse::<usize>().map_err(|_| format!("{key}: '{v}' is not a whole number"));
        let b = |v: &str| matches!(v.trim(), "1" | "true" | "on" | "yes");
        let list = |v: &str| -> Vec<String> { v.split(',').map(|x| x.trim().to_string()).filter(|x| !x.is_empty()).collect() };
        let nm = |v: &str| -> Result<NormalMode, String> {
            Ok(match v.trim() {
                "keep" => NormalMode::Keep,
                "flat" => NormalMode::Flat,
                "groups" => NormalMode::Groups,
                x if x.starts_with("angle") => NormalMode::Angle(x.trim_start_matches("angle").trim_start_matches(':').trim().parse().unwrap_or(60.0)),
                x => return Err(format!("{key}: unknown normal mode '{x}' (keep, flat, angle:60, groups)")),
            })
        };
        match key {
            "size" => {
                let p: Vec<&str> = val.split_whitespace().collect();
                self.width = u(p.first().ok_or("size needs a number")?)?;
                self.height = if p.len() > 1 { u(p[1])? } else { self.width };
            }
            "aa" => self.aa = u(val)? as u32,
            "padding" => self.padding = u(val)?,
            "front" => self.front = f(val)?,
            "rear" => self.rear = f(val)?,
            "bias" => self.bias = f(val)?,
            "cage_normals" => self.cage_normals = if val.trim() == "shading" { CageNormals::Shading } else { CageNormals::Averaged },
            "match_by_name" => self.match_by_name = b(val),
            "low_suffixes" => self.low_suffixes = list(val),
            "high_suffixes" => self.high_suffixes = list(val),
            "low_normals" => self.low_normals = nm(val)?,
            "high_normals" => self.high_normals = nm(val)?,
            "basis" => self.basis = if val.trim() == "face" { Basis::Face } else { Basis::Smooth },
            "flip_y" => self.flip_y = b(val),
            "uv_set" => self.uv_set = u(val)?.min(1),
            "material" => self.material = val.trim().to_string(),
            "ignore_backfaces" => self.ignore_backfaces = b(val),
            "maps" => {
                let mut v = Vec::new();
                for name in list(val) {
                    v.push(MapKind::from_id(&name).ok_or_else(|| format!("unknown map '{name}'"))?);
                }
                self.maps = v;
            }
            "ao_rays" => self.ao_rays = u(val)? as u32,
            "ao_dist" => self.ao_dist = f(val)?,
            "ao_falloff" => self.ao_falloff = f(val)?,
            "thick_rays" => self.thick_rays = u(val)? as u32,
            "thick_dist" => self.thick_dist = f(val)?,
            "curv_radius" => self.curv_radius = f(val)?,
            "curv_smooth" => self.curv_smooth = u(val)? as u32,
            "pos_range" => {
                let p: Vec<f32> = val.split_whitespace().filter_map(|x| x.parse().ok()).collect();
                if p.len() != 6 {
                    return Err("pos_range needs six numbers".into());
                }
                self.pos_range = Some(([p[0], p[1], p[2]], [p[3], p[4], p[5]]));
            }
            "device" => {
                self.device = match val.trim() {
                    "auto" => Device::Auto,
                    "cpu" => Device::Cpu,
                    "gpu" => Device::Gpu,
                    x => return Err(format!("{key}: unknown device '{x}' (auto, cpu, gpu)")),
                }
            }
            "direction" => {
                let p: Vec<f32> = val.split_whitespace().filter_map(|x| x.parse().ok()).collect();
                if p.len() != 3 {
                    return Err("direction needs three numbers".into());
                }
                self.direction = [p[0], p[1], p[2]];
            }
            _ => return Err(format!("unknown bake option '{key}'")),
        }
        Ok(())
    }

    pub fn from_text(text: &str) -> Result<BakeSettings, String> {
        let mut s = BakeSettings::default();
        for (i, line) in text.lines().enumerate() {
            let line = line.split('#').next().unwrap_or("").trim();
            if line.is_empty() {
                continue;
            }
            let (k, v) = line.split_once(' ').unwrap_or((line, ""));
            s.set(k, v).map_err(|e| format!("line {}: {e}", i + 1))?;
        }
        Ok(s)
    }
}

/// Meshes and settings of one bake, stored in a package so the maps can be rebuilt.
#[derive(Clone, Debug, Default)]
pub struct BakeSet {
    pub name: String,
    pub low: Vec<String>,
    pub high: Vec<String>,
    pub cage: Option<String>,
    pub settings: BakeSettings,
}

impl BakeSet {
    pub fn to_text(&self) -> String {
        let mut s = String::from("bakeset 1\n");
        for l in &self.low {
            s.push_str(&format!("low {l}\n"));
        }
        for h in &self.high {
            s.push_str(&format!("high {h}\n"));
        }
        if let Some(c) = &self.cage {
            s.push_str(&format!("cage {c}\n"));
        }
        s.push_str("settings\n");
        s.push_str(&self.settings.to_text());
        s
    }

    pub fn parse(name: &str, text: &str) -> Result<BakeSet, String> {
        let mut set = BakeSet { name: name.into(), ..Default::default() };
        let mut lines = text.lines();
        if lines.next().map(|l| l.trim()) != Some("bakeset 1") {
            return Err("not a bake set (expected 'bakeset 1')".into());
        }
        let mut rest = String::new();
        let mut in_settings = false;
        for l in lines {
            if in_settings {
                rest.push_str(l);
                rest.push('\n');
            } else if l.trim() == "settings" {
                in_settings = true;
            } else if let Some(v) = l.strip_prefix("low ") {
                set.low.push(v.trim().into());
            } else if let Some(v) = l.strip_prefix("high ") {
                set.high.push(v.trim().into());
            } else if let Some(v) = l.strip_prefix("cage ") {
                set.cage = Some(v.trim().into());
            }
        }
        set.settings = BakeSettings::from_text(&rest)?;
        Ok(set)
    }
}

// ====================================================================== baked maps inside a package
/// Resource name of a baked map in a package: `bake/<set>/<map>.png` (16-bit PNG).
pub fn resource_key(set: &str, map: MapKind) -> String {
    format!("bake/{set}/{}.png", map.id())
}

/// Resource name of a stored bake set (settings and mesh paths).
pub fn set_key(set: &str) -> String {
    format!("bake/{set}.cfg")
}

/// Put baked maps into a package's resources (16-bit PNG each) so `baked_map` nodes can read them. Older maps of
/// the same set are replaced; maps of the set that were not baked this time are kept.
pub fn store_maps(res: &mut BTreeMap<String, std::sync::Arc<Vec<u8>>>, set: &str, maps: &BTreeMap<String, Tex>) -> usize {
    let mut n = 0;
    for (id, t) in maps {
        if let Some(k) = MapKind::from_id(id) {
            res.insert(resource_key(set, k), std::sync::Arc::new(crate::png::encode(t.w, t.h, &t.px, 16)));
            n += 1;
        }
    }
    n
}

/// Names of the bake sets that have maps in a package, with the maps each has.
pub fn stored_sets(res: &BTreeMap<String, std::sync::Arc<Vec<u8>>>) -> BTreeMap<String, Vec<MapKind>> {
    let mut out: BTreeMap<String, Vec<MapKind>> = BTreeMap::new();
    for k in res.keys() {
        if let Some(rest) = k.strip_prefix("bake/") {
            if let Some((set, file)) = rest.split_once('/') {
                if let Some(m) = file.strip_suffix(".png").and_then(MapKind::from_id) {
                    out.entry(set.to_string()).or_default().push(m);
                }
            }
        }
    }
    out
}

// ====================================================================== progress and results
/// Shared progress of a running bake; another thread can read it and set `cancel`.
#[derive(Default)]
pub struct Progress {
    pub done: AtomicU64,
    pub total: AtomicU64,
    pub cancel: AtomicBool,
    stage: Mutex<String>,
}

impl Progress {
    pub fn new() -> Progress {
        Progress::default()
    }
    pub fn stage(&self) -> String {
        self.stage.lock().map(|s| s.clone()).unwrap_or_default()
    }
    fn set_stage(&self, s: &str, total: u64) {
        if let Ok(mut g) = self.stage.lock() {
            *g = s.to_string();
        }
        self.done.store(0, Ordering::Relaxed);
        self.total.store(total, Ordering::Relaxed);
    }
    pub fn fraction(&self) -> f32 {
        let t = self.total.load(Ordering::Relaxed).max(1);
        (self.done.load(Ordering::Relaxed) as f32 / t as f32).clamp(0.0, 1.0)
    }
    pub fn request_cancel(&self) {
        self.cancel.store(true, Ordering::Relaxed);
    }
}

#[derive(Clone, Debug, Default)]
pub struct BakeReport {
    pub low_tris: usize,
    pub high_tris: usize,
    pub texels: usize,
    pub covered: usize,
    /// texels where two triangles share a sample (overlapping UVs)
    pub overlapped: usize,
    pub samples: u64,
    pub hits: u64,
    pub misses: u64,
    pub rays: u64,
    pub seconds: f32,
    /// the high-poly bounding box diagonal (the unit of the relative distances)
    pub diagonal: f32,
    pub warnings: Vec<String>,
    /// what traced the rays: "CPU" or the GPU's name
    pub device: String,
}

#[derive(Debug)]
pub struct BakeResult {
    pub maps: BTreeMap<String, Tex>,
    pub report: BakeReport,
}

// ====================================================================== preparation
struct LowTri {
    p: [V; 3],
    n: [V; 3],
    t: [V; 3],
    /// bitangent handedness (+1 / -1) per corner
    s: [f32; 3],
    /// cage normals per corner
    cn: [V; 3],
    cage: Option<[V; 3]>,
    col: [[f32; 3]; 3],
    /// pixel coordinates of the corners
    px: [[f32; 2]; 3],
    /// altitude of each corner over the opposite edge, in pixels
    alt: [f32; 3],
    den: f32,
    obj: u32,
}

fn quant(p: V, q: f32) -> [i64; 3] {
    [(p[0] / q).round() as i64, (p[1] / q).round() as i64, (p[2] / q).round() as i64]
}

fn corner_angle(t: &Tri, c: usize) -> f32 {
    let (a, b) = (norm(sub(t.p[(c + 1) % 3], t.p[c])), norm(sub(t.p[(c + 2) % 3], t.p[c])));
    dot(a, b).clamp(-1.0, 1.0).acos()
}

/// Cage normals: the face normals of every triangle meeting at a position, angle weighted.
fn averaged_normals(tris: &[Tri], q: f32) -> Vec<[V; 3]> {
    let mut acc: HashMap<[i64; 3], V> = HashMap::new();
    for t in tris {
        let fnorm = t.face_normal();
        for c in 0..3 {
            let e = acc.entry(quant(t.p[c], q)).or_insert([0.0; 3]);
            *e = add(*e, mul(fnorm, corner_angle(t, c).max(1e-4)));
        }
    }
    tris.iter()
        .map(|t| {
            let mut o = [[0.0; 3]; 3];
            for c in 0..3 {
                o[c] = norm(acc.get(&quant(t.p[c], q)).copied().unwrap_or(t.face_normal()));
            }
            o
        })
        .collect()
}

/// Tangents and handedness per corner (see `Basis`).
fn tangent_basis(tris: &[Tri], uvset: usize, basis: Basis, q: f32) -> Vec<([V; 3], [f32; 3])> {
    let uv = |t: &Tri, c: usize| if uvset == 1 { t.uv1[c] } else { t.uv[c] };
    let mut face: Vec<(V, f32, bool)> = Vec::with_capacity(tris.len());
    for t in tris {
        let (e1, e2) = (sub(t.p[1], t.p[0]), sub(t.p[2], t.p[0]));
        let (u0, u1, u2) = (uv(t, 0), uv(t, 1), uv(t, 2));
        let (du1, dv1, du2, dv2) = (u1[0] - u0[0], u1[1] - u0[1], u2[0] - u0[0], u2[1] - u0[1]);
        let r = du1 * dv2 - du2 * dv1;
        if r.abs() < 1e-12 || len(cross(e1, e2)) < 1e-14 {
            face.push(([1.0, 0.0, 0.0], 1.0, false));
            continue;
        }
        let tan = mul(sub(mul(e1, dv2), mul(e2, dv1)), 1.0 / r);
        // mirrored UVs flip the handedness of the island
        face.push((norm(tan), if r < 0.0 { -1.0 } else { 1.0 }, true));
    }
    let mut acc: HashMap<([i64; 3], [i64; 2], i8), V> = HashMap::new();
    let ukey = |t: &Tri, c: usize| -> [i64; 2] {
        let u = uv(t, c);
        [(u[0] * 1e4).round() as i64, (u[1] * 1e4).round() as i64]
    };
    if basis == Basis::Smooth {
        for (i, t) in tris.iter().enumerate() {
            if !face[i].2 {
                continue;
            }
            for c in 0..3 {
                let e = acc.entry((quant(t.p[c], q), ukey(t, c), face[i].1 as i8)).or_insert([0.0; 3]);
                *e = add(*e, mul(face[i].0, corner_angle(t, c).max(1e-4)));
            }
        }
    }
    tris.iter()
        .enumerate()
        .map(|(i, t)| {
            let mut tg = [[1.0, 0.0, 0.0]; 3];
            let mut sg = [face[i].1; 3];
            for c in 0..3 {
                let base = if basis == Basis::Smooth { acc.get(&(quant(t.p[c], q), ukey(t, c), face[i].1 as i8)).copied().unwrap_or(face[i].0) } else { face[i].0 };
                let n = t.n[c];
                let mut v = sub(base, mul(n, dot(n, base)));
                if len(v) < 1e-9 {
                    // tangent parallel to the normal: any perpendicular
                    v = if n[0].abs() < 0.9 { cross(n, [1.0, 0.0, 0.0]) } else { cross(n, [0.0, 1.0, 0.0]) };
                }
                tg[c] = norm(v);
                sg[c] = face[i].1;
            }
            (tg, sg)
        })
        .collect()
}

/// Mean curvature at every corner of `mesh` (1 / length, positive on convex edges), from the dihedral angles.
fn mesh_curvature(mesh: &Mesh, smooth_passes: u32) -> Vec<[f32; 3]> {
    let q = (mesh.diagonal() * 1e-6).max(1e-12);
    let mut ids: HashMap<[i64; 3], u32> = HashMap::new();
    let mut vpos: Vec<V> = Vec::new();
    let mut corner: Vec<[u32; 3]> = Vec::with_capacity(mesh.tris.len());
    for t in &mesh.tris {
        let mut c = [0u32; 3];
        for k in 0..3 {
            let n = ids.len() as u32;
            let id = *ids.entry(quant(t.p[k], q)).or_insert(n);
            if id == n {
                vpos.push(t.p[k]);
            }
            c[k] = id;
        }
        corner.push(c);
    }
    let nv = vpos.len();
    let fnorm: Vec<V> = mesh.tris.iter().map(|t| t.face_normal()).collect();
    let mut area = vec![0.0f32; nv];
    for (i, t) in mesh.tris.iter().enumerate() {
        for k in 0..3 {
            area[corner[i][k] as usize] += t.area() / 3.0;
        }
    }
    // edge -> up to two faces with their opposite vertex
    let mut edges: HashMap<(u32, u32), Vec<(u32, u32)>> = HashMap::new();
    for (i, c) in corner.iter().enumerate() {
        for k in 0..3 {
            let (a, b, o) = (c[k], c[(k + 1) % 3], c[(k + 2) % 3]);
            if a == b {
                continue;
            }
            edges.entry((a.min(b), a.max(b))).or_default().push((i as u32, o));
        }
    }
    let mut h = vec![0.0f32; nv];
    let mut adj: Vec<Vec<u32>> = vec![Vec::new(); nv];
    for (&(a, b), faces) in &edges {
        adj[a as usize].push(b);
        adj[b as usize].push(a);
        if faces.len() != 2 {
            continue;
        }
        let (n1, n2) = (fnorm[faces[0].0 as usize], fnorm[faces[1].0 as usize]);
        let ang = len(cross(n1, n2)).atan2(dot(n1, n2));
        let l = len(sub(vpos[a as usize], vpos[b as usize]));
        // convex when the other face's far vertex lies behind the first face's plane
        let convex = dot(n1, sub(vpos[faces[1].1 as usize], vpos[a as usize])) < 0.0;
        let c = if convex { ang } else { -ang } * l;
        h[a as usize] += c;
        h[b as usize] += c;
    }
    for v in 0..nv {
        h[v] /= (4.0 * area[v]).max(1e-20);
    }
    for _ in 0..smooth_passes {
        let prev = h.clone();
        for v in 0..nv {
            if adj[v].is_empty() {
                continue;
            }
            let avg: f32 = adj[v].iter().map(|&n| prev[n as usize]).sum::<f32>() / adj[v].len() as f32;
            h[v] = 0.5 * prev[v] + 0.5 * avg;
        }
    }
    corner.iter().map(|c| [h[c[0] as usize], h[c[1] as usize], h[c[2] as usize]]).collect()
}

/// A distinct, pleasant colour for id maps.
pub fn id_color(i: u32) -> [f32; 3] {
    if i == 0 {
        return [0.85, 0.2, 0.2];
    }
    let h = (i as f32 * 0.618_034).fract();
    let (s, v) = (0.65 + 0.2 * rnd(i as i32, 1, 31), 0.85 + 0.1 * rnd(i as i32, 2, 31));
    let k = h * 6.0;
    let (f, p, q) = (k.fract(), v * (1.0 - s), 0.0);
    let (t, u) = (v * (1.0 - s * (1.0 - f)), v * (1.0 - s * f));
    let _ = q;
    match k as u32 % 6 {
        0 => [v, t, p],
        1 => [u, v, p],
        2 => [p, v, t],
        3 => [p, u, v],
        4 => [t, p, v],
        _ => [v, p, u],
    }
}

fn base_name(name: &str, suffixes: &[String]) -> String {
    let l = name.to_lowercase();
    for s in suffixes {
        let s = s.to_lowercase();
        if !s.is_empty() && l.ends_with(&s) {
            return l[..l.len() - s.len()].to_string();
        }
    }
    l
}

// ====================================================================== the bake
struct Ctx<'a> {
    s: &'a BakeSettings,
    w: usize,
    h: usize,
    spp: usize,
    lows: Vec<LowTri>,
    high: &'a Mesh,
    bvh: Bvh,
    rows: Vec<Vec<u32>>,
    /// per low object: which high objects may be hit (None = all)
    allowed: Vec<Option<Vec<bool>>>,
    curv: Vec<[f32; 3]>,
    diag: f32,
    front: f32,
    rear: f32,
    bias: f32,
    ao_dist: f32,
    thick_dist: f32,
    pos_min: V,
    pos_ext: V,
    dir: V,
    maps: Vec<MapKind>,
    slot: HashMap<MapKind, usize>,
    prog: Option<&'a Progress>,
}

struct RowOut {
    cover: Vec<bool>,
    data: Vec<Vec<[f32; 4]>>,
    samples: u64,
    hits: u64,
    misses: u64,
    rays: u64,
    overlapped: usize,
}

#[inline]
fn bary(t: &LowTri, x: f32, y: f32) -> [f32; 3] {
    let p = &t.px;
    let l0 = ((p[1][1] - p[2][1]) * (x - p[2][0]) + (p[2][0] - p[1][0]) * (y - p[2][1])) / t.den;
    let l1 = ((p[2][1] - p[0][1]) * (x - p[2][0]) + (p[0][0] - p[2][0]) * (y - p[2][1])) / t.den;
    [l0, l1, 1.0 - l0 - l1]
}

#[inline]
fn lerp3(a: &[V; 3], b: [f32; 3]) -> V {
    [a[0][0] * b[0] + a[1][0] * b[1] + a[2][0] * b[2], a[0][1] * b[0] + a[1][1] * b[1] + a[2][1] * b[2], a[0][2] * b[0] + a[1][2] * b[1] + a[2][2] * b[2]]
}

fn hammersley(i: u32, n: u32) -> (f32, f32) {
    let mut b = i;
    b = (b << 16) | (b >> 16);
    b = ((b & 0x5555_5555) << 1) | ((b & 0xAAAA_AAAA) >> 1);
    b = ((b & 0x3333_3333) << 2) | ((b & 0xCCCC_CCCC) >> 2);
    b = ((b & 0x0F0F_0F0F) << 4) | ((b & 0xF0F0_F0F0) >> 4);
    b = ((b & 0x00FF_00FF) << 8) | ((b & 0xFF00_FF00) >> 8);
    (i as f32 / n as f32, b as f32 * 2.328_306_4e-10)
}

/// A direction on the cosine-weighted hemisphere around `n`.
fn hemi(n: V, a: f32, b: f32) -> V {
    let up = if n[1].abs() < 0.99 { [0.0, 1.0, 0.0] } else { [1.0, 0.0, 0.0] };
    let tg = norm(cross(up, n));
    let bt = cross(n, tg);
    let phi = a * std::f32::consts::TAU;
    let (ct, st) = ((1.0 - b).sqrt(), b.sqrt());
    norm(add(add(mul(tg, st * phi.cos()), mul(bt, st * phi.sin())), mul(n, ct)))
}

/// What one sample contributes to every map (only the enabled ones are filled).
struct SampleOut {
    vals: Vec<[f32; 4]>,
    ids: [u32; 3],
    hit: bool,
    rays: u64,
}

/// Ray results computed ahead of time (on the GPU) for one sample.
#[derive(Clone, Copy, Default)]
struct Pre {
    hit: Option<Hit>,
    occ: f32,
    dsum: V,
    thick: f32,
}

/// One sample of a row: x, sub-sample index, low-poly triangle, barycentric coordinates.
type Sample = (u32, u16, u32, [f32; 3]);

impl Ctx<'_> {
    /// Origin, direction and length of the ray that looks for the high-poly surface of a low-poly point.
    fn primary(&self, lt: &LowTri, b: [f32; 3]) -> (V, V, f32) {
        let p = lerp3(&[lt.p[0], lt.p[1], lt.p[2]], b);
        let nc = norm(lerp3(&lt.cn, b));
        match &lt.cage {
            Some(cg) => {
                let oc = lerp3(cg, b);
                let v = sub(p, oc);
                let l = len(v).max(1e-9);
                (oc, mul(v, 1.0 / l), l + self.rear)
            }
            None => (add(p, mul(nc, self.front)), mul(nc, -1.0), self.front + self.rear),
        }
    }

    /// Point and normal of the surface the maps describe: the high-poly hit, or the low-poly surface for a miss.
    fn surface(&self, lt: &LowTri, b: [f32; 3], o: V, d: V, hit: &Option<Hit>) -> (V, V) {
        match hit {
            Some(h) => {
                let tr = &self.high.tris[h.tri as usize];
                let w0 = 1.0 - h.u - h.v;
                (add(o, mul(d, h.t)), norm(add(add(mul(tr.n[0], w0), mul(tr.n[1], h.u)), mul(tr.n[2], h.v))))
            }
            None => (lerp3(&[lt.p[0], lt.p[1], lt.p[2]], b), norm(lerp3(&lt.n, b))),
        }
    }

    /// Rays per sample for ambient occlusion / thickness (the per-texel budget is split over the sub-samples).
    fn hemi_rays(&self, total: u32) -> u32 {
        let spp = (self.spp as u32).max(1);
        ((total.max(1) + spp - 1) / spp).max(1)
    }

    fn filter(&self, obj: u32) -> Filter<'_> {
        Filter { objects: self.allowed.get(obj as usize).and_then(|a| a.as_deref()), cull_back: self.s.ignore_backfaces }
    }

    fn shade(&self, lt: &LowTri, b: [f32; 3], px: u32, py: u32, sub_i: u32, pre: Option<&Pre>) -> SampleOut {
        let s = self.s;
        let n = norm(lerp3(&lt.n, b));
        let t0 = lerp3(&lt.t, b);
        let t = norm(sub(t0, mul(n, dot(n, t0))));
        let sgn = if b[0] * lt.s[0] + b[1] * lt.s[1] + b[2] * lt.s[2] < 0.0 { -1.0 } else { 1.0 };
        let bt = mul(cross(n, t), sgn);
        let (o, d, tmax) = self.primary(lt, b);
        let hit = match pre {
            Some(pr) => pr.hit,
            None => {
                let filter = self.filter(lt.obj);
                self.bvh.closest(o, d, 1e-7, tmax, &filter)
            }
        };
        let mut rays = 1u64;
        let mut vals = vec![[0.0f32; 4]; self.maps.len()];
        let mut ids = [0u32; 3];
        // the surface the maps describe: the high-poly hit, or the low-poly surface when nothing is hit
        let (hp, hn) = self.surface(lt, b, o, d, &hit);
        let htri = hit.as_ref().map(|h| (&self.high.tris[h.tri as usize], *h));
        let slot = |m: MapKind| self.slot.get(&m).copied();
        if let Some(i) = slot(MapKind::Normal) {
            vals[i] = if htri.is_some() { [dot(hn, t), dot(hn, bt) * if s.flip_y { -1.0 } else { 1.0 }, dot(hn, n), 1.0] } else { [0.0, 0.0, 1.0, 1.0] };
        }
        if let Some(i) = slot(MapKind::WorldNormal) {
            vals[i] = [hn[0], hn[1], hn[2], 1.0];
        }
        if let Some(i) = slot(MapKind::Position) {
            let q = [(hp[0] - self.pos_min[0]) / self.pos_ext[0], (hp[1] - self.pos_min[1]) / self.pos_ext[1], (hp[2] - self.pos_min[2]) / self.pos_ext[2]];
            vals[i] = [q[0].clamp(0.0, 1.0), q[1].clamp(0.0, 1.0), q[2].clamp(0.0, 1.0), 1.0];
        }
        if let Some(h) = &hit {
            let tr = &self.high.tris[h.tri as usize];
            if let Some(i) = slot(MapKind::ObjectId) {
                ids[0] = tr.object + 1;
                vals[i] = [0.0; 4];
            }
            if let Some(i) = slot(MapKind::MaterialId) {
                ids[1] = tr.material + 1;
                vals[i] = [0.0; 4];
            }
            if let Some(i) = slot(MapKind::GroupId) {
                ids[2] = tr.group + 1;
                vals[i] = [0.0; 4];
            }
            if let Some(i) = slot(MapKind::VertexColor) {
                let w0 = 1.0 - h.u - h.v;
                let c = [tr.col[0][0] * w0 + tr.col[1][0] * h.u + tr.col[2][0] * h.v, tr.col[0][1] * w0 + tr.col[1][1] * h.u + tr.col[2][1] * h.v, tr.col[0][2] * w0 + tr.col[1][2] * h.u + tr.col[2][2] * h.v];
                vals[i] = [c[0], c[1], c[2], 1.0];
            }
            if let Some(i) = slot(MapKind::Height) {
                let v = 1.0 - h.t / (self.front + self.rear).max(1e-9);
                vals[i] = [v.clamp(0.0, 1.0), v.clamp(0.0, 1.0), v.clamp(0.0, 1.0), 1.0];
            }
        } else if let Some(i) = slot(MapKind::VertexColor) {
            let c = lerp3(&[lt.col[0], lt.col[1], lt.col[2]], b);
            vals[i] = [c[0], c[1], c[2], 1.0];
        }
        if let Some(i) = slot(MapKind::Opacity) {
            let v = if hit.is_some() { 1.0 } else { 0.0 };
            vals[i] = [v, v, v, 1.0];
        }
        if let Some(i) = slot(MapKind::WorldDirection) {
            let v = dot(hn, self.dir).max(0.0);
            vals[i] = [v, v, v, 1.0];
        }
        if let Some(i) = slot(MapKind::Curvature) {
            let c = match &htri {
                Some((tr, h)) => {
                    let _ = tr;
                    let k = &self.curv[h.tri as usize];
                    k[0] * (1.0 - h.u - h.v) + k[1] * h.u + k[2] * h.v
                }
                None => 0.0,
            };
            let x = c * self.s.curv_radius * self.diag;
            let v = 0.5 + 0.5 * (x * 2.0).tanh();
            vals[i] = [v, v, v, 1.0];
        }
        // ---- ambient occlusion and bent normals
        let want_ao = slot(MapKind::AmbientOcclusion);
        let want_bent = slot(MapKind::BentNormal);
        let want_wbent = slot(MapKind::WorldBentNormal);
        if want_ao.is_some() || want_bent.is_some() || want_wbent.is_some() {
            let m = self.hemi_rays(s.ao_rays);
            let org = add(hp, mul(hn, self.bias));
            let (r1, r2) = (rnd(px as i32, py as i32, 71 + sub_i * 3), rnd(py as i32, px as i32, 72 + sub_i * 3));
            let mut occ = 0.0f32;
            let mut dir_sum = [0.0f32; 3];
            let all = Filter { objects: None, cull_back: false };
            if let Some(pr) = pre {
                occ = pr.occ;
                dir_sum = pr.dsum;
                rays += m as u64;
            }
            for k in 0..if pre.is_some() { 0 } else { m } {
                let (a, bb) = hammersley(k, m);
                let dd = hemi(hn, (a + r1).fract(), (bb + r2).fract());
                rays += 1;
                match self.bvh.closest(org, dd, 1e-7, self.ao_dist, &all) {
                    Some(h) => {
                        let f = if s.ao_falloff > 0.0 { (1.0 - h.t / self.ao_dist).clamp(0.0, 1.0).powf(s.ao_falloff.max(0.01)) } else { 1.0 };
                        occ += f;
                        dir_sum = add(dir_sum, mul(dd, 1.0 - f));
                    }
                    None => dir_sum = add(dir_sum, dd),
                }
            }
            let ao = 1.0 - occ / m as f32;
            if let Some(i) = want_ao {
                vals[i] = [ao, ao, ao, 1.0];
            }
            let bent = if len(dir_sum) < 1e-6 { hn } else { norm(dir_sum) };
            if let Some(i) = want_bent {
                vals[i] = [dot(bent, t), dot(bent, bt) * if s.flip_y { -1.0 } else { 1.0 }, dot(bent, n), 1.0];
            }
            if let Some(i) = want_wbent {
                vals[i] = [bent[0], bent[1], bent[2], 1.0];
            }
        }
        // ---- thickness
        if let Some(i) = slot(MapKind::Thickness) {
            let m = self.hemi_rays(s.thick_rays);
            let inward = mul(hn, -1.0);
            let org = add(hp, mul(inward, self.bias));
            let (r1, r2) = (rnd(px as i32, py as i32, 91 + sub_i * 3), rnd(py as i32, px as i32, 92 + sub_i * 3));
            let all = Filter { objects: None, cull_back: false };
            let mut sum = 0.0;
            if let Some(pr) = pre {
                sum = pr.thick;
                rays += m as u64;
            }
            for k in 0..if pre.is_some() { 0 } else { m } {
                let (a, bb) = hammersley(k, m);
                let dd = hemi(inward, (a + r1).fract(), (bb + r2).fract());
                rays += 1;
                sum += match self.bvh.closest(org, dd, 1e-7, self.thick_dist, &all) {
                    Some(h) => (h.t / self.thick_dist).clamp(0.0, 1.0),
                    None => 1.0,
                };
            }
            let v = sum / m as f32;
            vals[i] = [v, v, v, 1.0];
        }
        SampleOut { vals, ids, hit: hit.is_some(), rays }
    }

    /// The samples of one row, sorted by texel and sub-sample.
    fn row_samples(&self, y: usize) -> Vec<Sample> {
        let (w, aa) = (self.w, self.s.aa.max(1) as usize);
        let bin = &self.rows[y];
        // collect the samples of every triangle crossing this row
        let mut samples: Vec<Sample> = Vec::new();
        for &ti in bin {
            let lt = &self.lows[ti as usize];
            let (mut x0, mut x1) = (f32::MAX, f32::MIN);
            for c in &lt.px {
                x0 = x0.min(c[0]);
                x1 = x1.max(c[0]);
            }
            let (xa, xb) = ((x0.floor().max(0.0)) as usize, (x1.ceil().min(w as f32)) as usize);
            for x in xa..xb.min(w) {
                for sj in 0..aa {
                    for si in 0..aa {
                        let (sx, sy) = (x as f32 + (si as f32 + 0.5) / aa as f32, y as f32 + (sj as f32 + 0.5) / aa as f32);
                        let b = bary(lt, sx, sy);
                        if b[0] >= -1e-5 && b[1] >= -1e-5 && b[2] >= -1e-5 {
                            samples.push((x as u32, (sj * aa + si) as u16, ti, b));
                        }
                    }
                }
            }
        }
        samples.sort_by_key(|s| (s.0, s.1));
        samples
    }

    fn bake_row(&self, y: usize, samples: &[Sample], pre: Option<&[Pre]>) -> RowOut {
        let (w, spp) = (self.w, self.spp);
        let n = self.spp.max(1);
        let mut out = RowOut { cover: vec![false; w], data: self.maps.iter().map(|m| vec![m.background(); w]).collect(), samples: 0, hits: 0, misses: 0, rays: 0, overlapped: 0 };
        if samples.is_empty() {
            return out;
        }
        let mut i = 0;
        while i < samples.len() {
            let x = samples[i].0;
            let mut j = i;
            while j < samples.len() && samples[j].0 == x {
                j += 1;
            }
            let group = &samples[i..j];
            // overlapping UVs: two triangles claim the same sub-sample
            let overlapped = group.windows(2).any(|p| p[0].1 == p[1].1);
            if overlapped {
                out.overlapped += 1;
            }
            let mut sum: Vec<[f32; 4]> = vec![[0.0; 4]; self.maps.len()];
            let mut idv: [Vec<(u32, u32)>; 3] = [Vec::new(), Vec::new(), Vec::new()];
            let mut count = 0.0f32;
            let mut wire = 0.0f32;
            for (gi, &(_, sub_i, ti, b)) in group.iter().enumerate() {
                let lt = &self.lows[ti as usize];
                let r = self.shade(lt, b, x, y as u32, sub_i as u32, pre.map(|p| &p[i + gi]));
                out.samples += 1;
                out.rays += r.rays;
                if r.hit {
                    out.hits += 1;
                } else {
                    out.misses += 1;
                }
                for (k, v) in r.vals.iter().enumerate() {
                    for c in 0..4 {
                        sum[k][c] += v[c];
                    }
                }
                for (k, id) in r.ids.iter().enumerate() {
                    if *id != 0 {
                        match idv[k].iter_mut().find(|e| e.0 == *id) {
                            Some(e) => e.1 += 1,
                            None => idv[k].push((*id, 1)),
                        }
                    }
                }
                if let Some(slot) = self.slot.get(&MapKind::UvLayout) {
                    let _ = slot;
                    let d = (0..3).map(|c| b[c] * lt.alt[c]).fold(f32::MAX, f32::min);
                    wire += (1.0 - (d - 0.5).clamp(0.0, 1.0)).clamp(0.0, 1.0);
                }
                count += 1.0;
            }
            let _ = (n, spp);
            for (k, m) in self.maps.iter().enumerate() {
                let a = sum[k];
                let v = match m {
                    MapKind::Normal | MapKind::WorldNormal | MapKind::BentNormal | MapKind::WorldBentNormal => {
                        let nrm = norm([a[0], a[1], a[2]]);
                        [nrm[0] * 0.5 + 0.5, nrm[1] * 0.5 + 0.5, nrm[2] * 0.5 + 0.5, 1.0]
                    }
                    MapKind::ObjectId | MapKind::MaterialId | MapKind::GroupId => {
                        let list = &idv[match m {
                            MapKind::ObjectId => 0,
                            MapKind::MaterialId => 1,
                            _ => 2,
                        }];
                        match list.iter().max_by_key(|e| e.1) {
                            Some(e) => {
                                let c = id_color(e.0 - 1);
                                [c[0], c[1], c[2], 1.0]
                            }
                            None => [0.0, 0.0, 0.0, 1.0],
                        }
                    }
                    MapKind::UvLayout => {
                        let v = (wire / count).clamp(0.0, 1.0);
                        [v, v, v, 1.0]
                    }
                    _ => [a[0] / count, a[1] / count, a[2] / count, 1.0],
                };
                out.data[k][x as usize] = v;
            }
            out.cover[x as usize] = true;
            i = j;
        }
        out
    }
}


/// `f(0..n)` on every core, results in order. (Unlike `tex::par_map` it ignores the editor's job cancellation.)
fn par_vec<T: Send, F: Fn(usize) -> T + Sync>(n: usize, f: F) -> Vec<T> {
    let next = AtomicUsize::new(0);
    let out: Mutex<Vec<Option<T>>> = Mutex::new((0..n).map(|_| None).collect());
    let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1).max(1).min(n.max(1));
    std::thread::scope(|sc| {
        for _ in 0..threads {
            sc.spawn(|| loop {
                let i = next.fetch_add(1, Ordering::Relaxed);
                if i >= n {
                    break;
                }
                let v = f(i);
                if let Ok(mut g) = out.lock() {
                    g[i] = Some(v);
                }
            });
        }
    });
    out.into_inner().unwrap_or_default().into_iter().flatten().collect()
}

enum GpuErr {
    Cancelled,
    /// the GPU could not do the work (out of memory, device lost, ...): the CPU takes over
    Failed,
}

impl Ctx<'_> {
    /// The rays on the GPU, the rest on the CPU, in windows of about two million samples: the CPU lists the samples
    /// and builds the rays, the GPU finds the high-poly hits and then traces the ambient occlusion and thickness
    /// rays from those hits, and the CPU turns everything into map values.
    fn bake_gpu(&self, scene: &crate::gpu::BakeScene, mask_idx: &[u32], prog: Option<&Progress>) -> Result<Vec<RowOut>, GpuErr> {
        use crate::gpu::HemiParams;
        let (w, h, s) = (self.w, self.h, self.s);
        // samples per window (MC_BAKE_WINDOW overrides it; the tests use a tiny window to cover the window seams)
        let window: usize = std::env::var("MC_BAKE_WINDOW").ok().and_then(|v| v.parse().ok()).unwrap_or(2_000_000);
        let rows_per = (window / (w * self.spp).max(1)).max(1);
        let slot = |m: MapKind| self.slot.get(&m).is_some();
        let want_ao = slot(MapKind::AmbientOcclusion) || slot(MapKind::BentNormal) || slot(MapKind::WorldBentNormal);
        let want_th = slot(MapKind::Thickness);
        let (ao_m, th_m) = (self.hemi_rays(s.ao_rays), self.hemi_rays(s.thick_rays));
        let mut outs: Vec<RowOut> = Vec::with_capacity(h);
        let mut y0 = 0;
        while y0 < h {
            if prog.map(|p| p.cancel.load(Ordering::Relaxed)).unwrap_or(false) {
                return Err(GpuErr::Cancelled);
            }
            let y1 = (y0 + rows_per).min(h);
            let n = y1 - y0;
            let samples: Vec<Vec<Sample>> = par_vec(n, |k| self.row_samples(y0 + k));
            let mut offs = vec![0usize; n + 1];
            for k in 0..n {
                offs[k + 1] = offs[k] + samples[k].len();
            }
            let total = offs[n];
            let mut pre: Vec<Pre> = vec![Pre::default(); total];
            if total > 0 {
                // primary rays
                let rays: Vec<Vec<[u32; 4]>> = par_vec(n, |k| {
                    let mut v = Vec::with_capacity(samples[k].len() * 3);
                    for &(_, _, ti, b) in &samples[k] {
                        let lt = &self.lows[ti as usize];
                        let (o, d, tmax) = self.primary(lt, b);
                        v.push([o[0].to_bits(), o[1].to_bits(), o[2].to_bits(), 1e-7f32.to_bits()]);
                        v.push([d[0].to_bits(), d[1].to_bits(), d[2].to_bits(), tmax.to_bits()]);
                        v.push([mask_idx.get(lt.obj as usize).copied().unwrap_or(u32::MAX), 0, 0, 0]);
                    }
                    v
                });
                let flat: Vec<[u32; 4]> = rays.into_iter().flatten().collect();
                let hits = scene.trace(&flat).ok_or(GpuErr::Failed)?;
                drop(flat);
                // hits -> surface -> secondary ray inputs
                let sec: Vec<(Vec<Pre>, Vec<[u32; 4]>, Vec<[u32; 4]>)> = par_vec(n, |k| {
                    let mut pv = Vec::with_capacity(samples[k].len());
                    let (mut ao, mut th) = (Vec::new(), Vec::new());
                    let y = (y0 + k) as u32;
                    for (j, &(x, sub, ti, b)) in samples[k].iter().enumerate() {
                        let lt = &self.lows[ti as usize];
                        let r = hits[offs[k] + j];
                        let t = f32::from_bits(r[0]);
                        let hit = if t >= 0.0 { Some(Hit { t, tri: r[1], u: f32::from_bits(r[2]), v: f32::from_bits(r[3]) }) } else { None };
                        let (o, d, _) = self.primary(lt, b);
                        let (hp, hn) = self.surface(lt, b, o, d, &hit);
                        if want_ao {
                            let org = add(hp, mul(hn, self.bias));
                            ao.push([org[0].to_bits(), org[1].to_bits(), org[2].to_bits(), x]);
                            ao.push([hn[0].to_bits(), hn[1].to_bits(), hn[2].to_bits(), y | ((sub as u32) << 16)]);
                        }
                        if want_th {
                            let inward = mul(hn, -1.0);
                            let org = add(hp, mul(inward, self.bias));
                            th.push([org[0].to_bits(), org[1].to_bits(), org[2].to_bits(), x]);
                            th.push([inward[0].to_bits(), inward[1].to_bits(), inward[2].to_bits(), y | ((sub as u32) << 16)]);
                        }
                        pv.push(Pre { hit, ..Pre::default() });
                    }
                    (pv, ao, th)
                });
                let (mut ao_in, mut th_in) = (Vec::new(), Vec::new());
                let mut at = 0;
                for (pv, ao, th) in sec {
                    for p in pv {
                        pre[at] = p;
                        at += 1;
                    }
                    ao_in.extend(ao);
                    th_in.extend(th);
                }
                if want_ao {
                    let r = scene.hemi(&ao_in, HemiParams { rays: ao_m, dist: self.ao_dist, falloff: s.ao_falloff, thickness: false, seed: 71 }).ok_or(GpuErr::Failed)?;
                    for (p, v) in pre.iter_mut().zip(r) {
                        p.occ = v[0];
                        p.dsum = [v[1], v[2], v[3]];
                    }
                }
                if want_th {
                    let r = scene.hemi(&th_in, HemiParams { rays: th_m, dist: self.thick_dist, falloff: 0.0, thickness: true, seed: 91 }).ok_or(GpuErr::Failed)?;
                    for (p, v) in pre.iter_mut().zip(r) {
                        p.thick = v[0];
                    }
                }
            }
            let rows: Vec<RowOut> = par_vec(n, |k| self.bake_row(y0 + k, &samples[k], Some(&pre[offs[k]..offs[k + 1]])));
            outs.extend(rows);
            if let Some(p) = prog {
                p.done.fetch_add(n as u64, Ordering::Relaxed);
            }
            y0 = y1;
        }
        Ok(outs)
    }
}

/// Name of the GPU the baker would use (None: the CPU traces the rays).
pub fn gpu_name() -> Option<String> {
    crate::gpu::BakeScene::adapter()
}

/// Bake with the default progress handling: no progress object and no custom cage.
pub fn bake(low: &Mesh, high: &Mesh, s: &BakeSettings) -> Result<BTreeMap<String, Tex>, String> {
    Ok(bake_full(low, Some(high), None, s, None)?.maps)
}

/// The full baker. `high` of None bakes the low-poly mesh against itself (ambient occlusion, curvature, ... of one
/// mesh); `cage` is an optional mesh with the same triangles as `low`, pushed out to where rays should start.
pub fn bake_full(low: &Mesh, high: Option<&Mesh>, cage: Option<&Mesh>, s: &BakeSettings, prog: Option<&Progress>) -> Result<BakeResult, String> {
    let t0 = Instant::now();
    let (w, h) = (s.width.clamp(16, 16384), s.height.clamp(16, 16384));
    if w * h > 150_000_000 {
        return Err("the output is larger than 150 megapixels".into());
    }
    if s.maps.is_empty() {
        return Err("choose at least one map to bake".into());
    }
    if low.tris.is_empty() {
        return Err("the low-poly mesh has no faces".into());
    }
    let has_uv = if s.uv_set == 1 { low.has_uv1 } else { low.has_uv };
    if !has_uv {
        return Err(format!("the low-poly mesh has no UV coordinates{}", if s.uv_set == 1 { " in the second UV set" } else { "" }));
    }
    let mut warnings: Vec<String> = Vec::new();
    if let Some(p) = prog {
        p.set_stage("Preparing meshes", 1);
    }
    let mut low_m = low.clone();
    low_m.apply_normals(s.low_normals);
    let high_owned: Mesh = match high {
        Some(hm) if !hm.tris.is_empty() => {
            let mut m = hm.clone();
            m.apply_normals(s.high_normals);
            m
        }
        Some(_) => return Err("the high-poly mesh has no faces".into()),
        None => low_m.clone(),
    };
    if let Some(c) = cage {
        if c.tris.len() != low.tris.len() {
            return Err(format!("the cage has {} triangles but the low-poly mesh has {}: a cage must be a copy of the low-poly mesh with the same faces", c.tris.len(), low.tris.len()));
        }
    }
    let diag = high_owned.diagonal();
    let q = (diag * 1e-6).max(1e-12);
    // ---- low-poly triangles
    let mat_idx = if s.material.is_empty() { None } else { low_m.materials.iter().position(|m| *m == s.material).map(|i| i as u32) };
    if !s.material.is_empty() && mat_idx.is_none() {
        return Err(format!("the low-poly mesh has no material called '{}'", s.material));
    }
    let basis = tangent_basis(&low_m.tris, s.uv_set, s.basis, q);
    let cage_n = if s.cage_normals == CageNormals::Averaged { averaged_normals(&low_m.tris, q) } else { low_m.tris.iter().map(|t| t.n).collect() };
    let mut lows: Vec<LowTri> = Vec::with_capacity(low_m.tris.len());
    let (mut degenerate, mut outside) = (0usize, 0usize);
    for (i, t) in low_m.tris.iter().enumerate() {
        if let Some(m) = mat_idx {
            if t.material != m {
                continue;
            }
        }
        let uv = if s.uv_set == 1 { t.uv1 } else { t.uv };
        if uv.iter().any(|u| u[0] < -1e-4 || u[0] > 1.0 + 1e-4 || u[1] < -1e-4 || u[1] > 1.0 + 1e-4) {
            outside += 1;
            continue;
        }
        let px = [[uv[0][0] * w as f32, (1.0 - uv[0][1]) * h as f32], [uv[1][0] * w as f32, (1.0 - uv[1][1]) * h as f32], [uv[2][0] * w as f32, (1.0 - uv[2][1]) * h as f32]];
        let den = (px[1][1] - px[2][1]) * (px[0][0] - px[2][0]) + (px[2][0] - px[1][0]) * (px[0][1] - px[2][1]);
        if den.abs() < 1e-9 || t.area() < 1e-14 {
            degenerate += 1;
            continue;
        }
        let area_px = den.abs() * 0.5;
        let el = |a: usize, b: usize| ((px[a][0] - px[b][0]).powi(2) + (px[a][1] - px[b][1]).powi(2)).sqrt().max(1e-9);
        let alt = [2.0 * area_px / el(1, 2), 2.0 * area_px / el(2, 0), 2.0 * area_px / el(0, 1)];
        lows.push(LowTri {
            p: t.p,
            n: t.n,
            t: basis[i].0,
            s: basis[i].1,
            cn: cage_n[i],
            cage: cage.map(|c| c.tris[i].p),
            col: t.col,
            px,
            alt,
            den,
            obj: t.object,
        });
    }
    if outside > 0 {
        warnings.push(format!("{outside} low-poly triangles have UVs outside 0..1 and were skipped (bake one UV tile at a time)"));
    }
    if degenerate > 0 {
        warnings.push(format!("{degenerate} low-poly triangles have no area in the UV layout or in 3D and were skipped"));
    }
    if lows.is_empty() {
        return Err("no low-poly triangle can be baked: check the UV layout, the UV set and the material filter".into());
    }
    // ---- matching by name
    let mut allowed: Vec<Option<Vec<bool>>> = vec![None; low_m.objects.len().max(1)];
    if s.match_by_name && high.is_some() {
        let hb: Vec<String> = high_owned.objects.iter().map(|n| base_name(n, &s.high_suffixes)).collect();
        let mut unmatched: Vec<String> = Vec::new();
        for (li, name) in low_m.objects.iter().enumerate() {
            let lb = base_name(name, &s.low_suffixes);
            let mask: Vec<bool> = hb.iter().map(|b| *b == lb).collect();
            if mask.iter().any(|m| *m) {
                allowed[li] = Some(mask);
            } else {
                unmatched.push(name.clone());
            }
        }
        if !unmatched.is_empty() && allowed.iter().any(|a| a.is_some()) {
            warnings.push(format!("no high-poly object matches {}; they bake against every high-poly object", unmatched.join(", ")));
        }
    }
    // ---- acceleration structure
    if let Some(p) = prog {
        p.set_stage("Building the ray-tracing structure", 1);
    }
    let bvh = Bvh::build(&high_owned.tris);
    let (hmin, hmax) = high_owned.bounds();
    let (pos_min, pos_max) = s.pos_range.unwrap_or((hmin, hmax));
    let pos_ext = [(pos_max[0] - pos_min[0]).max(1e-9), (pos_max[1] - pos_min[1]).max(1e-9), (pos_max[2] - pos_min[2]).max(1e-9)];
    let curv = if s.wants(MapKind::Curvature) { mesh_curvature(&high_owned, s.curv_smooth.min(32)) } else { Vec::new() };
    // ---- row bins
    let mut rows: Vec<Vec<u32>> = vec![Vec::new(); h];
    for (i, lt) in lows.iter().enumerate() {
        let (mut y0, mut y1) = (f32::MAX, f32::MIN);
        for c in &lt.px {
            y0 = y0.min(c[1]);
            y1 = y1.max(c[1]);
        }
        for y in (y0.floor().max(0.0) as usize)..((y1.ceil().min(h as f32)) as usize).min(h) {
            rows[y].push(i as u32);
        }
    }
    let mut maps: Vec<MapKind> = s.maps.clone();
    maps.sort();
    maps.dedup();
    let slot: HashMap<MapKind, usize> = maps.iter().enumerate().map(|(i, m)| (*m, i)).collect();
    let aa = s.aa.clamp(1, 8) as usize;
    let mut dir = s.direction;
    if len(dir) < 1e-6 {
        dir = [0.0, 1.0, 0.0];
    }
    let ctx = Ctx {
        s,
        w,
        h,
        spp: aa * aa,
        lows,
        high: &high_owned,
        bvh,
        rows,
        allowed,
        curv,
        diag,
        front: s.front.max(1e-6) * diag,
        rear: s.rear.max(0.0) * diag,
        bias: s.bias.max(1e-7) * diag,
        ao_dist: s.ao_dist.max(1e-6) * diag,
        thick_dist: s.thick_dist.max(1e-6) * diag,
        pos_min,
        pos_ext,
        dir: norm(dir),
        maps: maps.clone(),
        slot,
        prog,
    };
    let _ = (ctx.h, ctx.prog.is_some());
    // ---- rays, row by row on every core
    if let Some(p) = prog {
        p.set_stage("Baking", h as u64);
    }
    let mut device = "CPU".to_string();
    let mut gpu_rows: Option<Vec<RowOut>> = None;
    if s.device != Device::Cpu {
        // masks: one row of bits per low-poly object that has a match list
        let nh = high_owned.objects.len().max(1);
        let words = (nh + 31) / 32;
        let (mut masks, mut mask_idx) = (Vec::<u32>::new(), vec![u32::MAX; ctx.allowed.len()]);
        for (li, a) in ctx.allowed.iter().enumerate() {
            if let Some(m) = a {
                mask_idx[li] = (masks.len() / words) as u32;
                for wi in 0..words {
                    let mut v = 0u32;
                    for bit in 0..32 {
                        if m.get(wi * 32 + bit).copied().unwrap_or(true) {
                            v |= 1 << bit;
                        }
                    }
                    masks.push(v);
                }
            }
        }
        if let Some(p) = prog {
            p.set_stage("Preparing the GPU", 1);
        }
        let (nodes, tris) = ctx.bvh.export();
        let scene = crate::gpu::BakeScene::new(&crate::gpu::SceneData { nodes, tris, masks, words: words as u32, cull_back: s.ignore_backfaces });
        match scene {
            Some(sc) => {
                if let Some(p) = prog {
                    p.set_stage("Baking on the GPU", h as u64);
                }
                match ctx.bake_gpu(&sc, &mask_idx, prog) {
                    Ok(rows) => {
                        gpu_rows = Some(rows);
                        device = crate::gpu::BakeScene::adapter().unwrap_or_else(|| "GPU".into());
                    }
                    Err(GpuErr::Cancelled) => return Err("the bake was cancelled".into()),
                    Err(GpuErr::Failed) => {
                        if s.device == Device::Gpu {
                            return Err("the GPU could not finish the bake (out of memory or device lost); choose the CPU or a smaller size".into());
                        }
                        warnings.push("the GPU could not finish the bake, so the CPU did it".into());
                    }
                }
            }
            None => {
                if s.device == Device::Gpu {
                    return Err("no usable GPU was found (or the mesh is too large for its memory); choose the CPU".into());
                }
            }
        }
    }
    let rows_out: Vec<Option<RowOut>> = match gpu_rows {
        Some(rows) => rows.into_iter().map(Some).collect(),
        None => {
            if let Some(p) = prog {
                p.set_stage("Baking", h as u64);
            }
            let next = AtomicUsize::new(0);
            let results: Mutex<Vec<Option<RowOut>>> = Mutex::new((0..h).map(|_| None).collect());
            let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1).max(1);
            let cancelled = AtomicBool::new(false);
            std::thread::scope(|sc| {
                for _ in 0..threads {
                    sc.spawn(|| loop {
                        let y = next.fetch_add(1, Ordering::Relaxed);
                        if y >= h {
                            break;
                        }
                        if prog.map(|p| p.cancel.load(Ordering::Relaxed)).unwrap_or(false) {
                            cancelled.store(true, Ordering::Relaxed);
                            break;
                        }
                        let r = ctx.bake_row(y, &ctx.row_samples(y), None);
                        if let Ok(mut g) = results.lock() {
                            g[y] = Some(r);
                        }
                        if let Some(p) = prog {
                            p.done.fetch_add(1, Ordering::Relaxed);
                        }
                    });
                }
            });
            if cancelled.load(Ordering::Relaxed) {
                return Err("the bake was cancelled".into());
            }
            results.into_inner().map_err(|_| "a baking thread failed".to_string())?
        }
    };
    // ---- assemble
    if let Some(p) = prog {
        p.set_stage("Padding", 1);
    }
    let mut report = BakeReport { low_tris: ctx.lows.len(), high_tris: high_owned.tris.len(), texels: w * h, diagonal: diag, ..Default::default() };
    let mut cover = vec![false; w * h];
    let mut data: Vec<Vec<[f32; 4]>> = maps.iter().map(|m| vec![m.background(); w * h]).collect();
    for (y, r) in rows_out.into_iter().enumerate() {
        let Some(r) = r else { return Err("a row was not baked".into()) };
        report.samples += r.samples;
        report.hits += r.hits;
        report.misses += r.misses;
        report.rays += r.rays;
        report.overlapped += r.overlapped;
        for x in 0..w {
            if r.cover[x] {
                cover[y * w + x] = true;
                for (k, d) in data.iter_mut().enumerate() {
                    d[y * w + x] = r.data[k][x];
                }
            }
        }
    }
    report.covered = cover.iter().filter(|c| **c).count();
    if report.covered == 0 {
        return Err("no texel was covered: the UV layout is empty".into());
    }
    if report.overlapped > 0 {
        warnings.push(format!("{} texels are shared by overlapping UV triangles; their values are averaged", report.overlapped));
    }
    if report.hits == 0 {
        warnings.push("no ray hit the high-poly mesh: check the front and rear distances, the matching by name and that both meshes use the same position and scale".into());
    } else if report.misses as f32 > 0.25 * report.samples as f32 {
        warnings.push(format!("{:.0}% of the rays missed the high-poly mesh; the low-poly surface was used there (raise the front or rear distance)", 100.0 * report.misses as f32 / report.samples as f32));
    }
    // ---- padding: every empty texel within `padding` takes the nearest island texel
    let src = dilate(&cover, w, h, s.padding.min(256));
    let mut out = BTreeMap::new();
    for (k, m) in maps.iter().enumerate() {
        let mut px = std::mem::take(&mut data[k]);
        for i in 0..w * h {
            if !cover[i] && src[i] != u32::MAX {
                px[i] = px[src[i] as usize];
            }
        }
        out.insert(m.id().to_string(), Tex { w, h, px });
    }
    report.seconds = t0.elapsed().as_secs_f32();
    report.warnings = warnings;
    report.device = device;
    if let Some(p) = prog {
        p.set_stage("Done", 1);
        p.done.store(1, Ordering::Relaxed);
    }
    Ok(BakeResult { maps: out, report })
}

/// For every texel within `pad` steps of covered texels, the index of the nearest covered texel (8-connected rings);
/// u32::MAX elsewhere.
fn dilate(cover: &[bool], w: usize, h: usize, pad: usize) -> Vec<u32> {
    let mut src: Vec<u32> = (0..w * h).map(|i| if cover[i] { i as u32 } else { u32::MAX }).collect();
    if pad == 0 {
        return src;
    }
    let mut frontier: Vec<usize> = Vec::new();
    for y in 0..h {
        for x in 0..w {
            if cover[y * w + x] {
                frontier.push(y * w + x);
            }
        }
    }
    for _ in 0..pad {
        let mut next = Vec::new();
        for &i in &frontier {
            let (x, y) = ((i % w) as i64, (i / w) as i64);
            for dy in -1i64..=1 {
                for dx in -1i64..=1 {
                    let (nx, ny) = (x + dx, y + dy);
                    if nx < 0 || ny < 0 || nx >= w as i64 || ny >= h as i64 {
                        continue;
                    }
                    let j = ny as usize * w + nx as usize;
                    if src[j] == u32::MAX {
                        src[j] = src[i];
                        next.push(j);
                    }
                }
            }
        }
        if next.is_empty() {
            break;
        }
        frontier = next;
    }
    src
}

/// Load several mesh files (OBJ, PLY, STL, glTF, GLB) and merge them into one mesh.
pub fn load_meshes<P: AsRef<std::path::Path>>(paths: &[P]) -> Result<Mesh, String> {
    if paths.is_empty() {
        return Err("no mesh file given".into());
    }
    let mut parts = Vec::new();
    for p in paths {
        parts.push(crate::meshio::load_file(p.as_ref())?);
    }
    Ok(Mesh::merge(&parts))
}

/// Bake a stored set: its mesh files are resolved against `base` (the folder of the package).
pub fn run_set(set: &BakeSet, base: &std::path::Path, prog: Option<&Progress>) -> Result<BakeResult, String> {
    let abs = |f: &String| -> std::path::PathBuf {
        let p = std::path::Path::new(f);
        if p.is_absolute() { p.to_path_buf() } else { base.join(p) }
    };
    let low = load_meshes(&set.low.iter().map(abs).collect::<Vec<_>>()).map_err(|e| format!("low-poly: {e}"))?;
    let high = if set.high.is_empty() { None } else { Some(load_meshes(&set.high.iter().map(abs).collect::<Vec<_>>()).map_err(|e| format!("high-poly: {e}"))?) };
    let cage = match &set.cage {
        Some(c) => Some(crate::meshio::load_file(&abs(c)).map_err(|e| format!("cage: {e}"))?),
        None => None,
    };
    bake_full(&low, high.as_ref(), cage.as_ref(), &set.settings, prog)
}

/// A unit sphere with UVs and normals, handy for testing the baker and the 3D viewer. `bump` ripples the surface.
pub fn uv_sphere_obj(lon: usize, lat: usize, radius: f32, bump: f32) -> String {
    let (lon, lat) = (lon.max(3), lat.max(2));
    let mut s = String::from("o sphere\n");
    for j in 0..=lat {
        for i in 0..=lon {
            let (u, v) = (i as f32 / lon as f32, j as f32 / lat as f32);
            let (th, ph) = (u * std::f32::consts::TAU, v * std::f32::consts::PI);
            let r = radius * (1.0 + bump * (th * 6.0).sin() * (ph * 5.0).sin());
            if j == 0 || j == lat {
                s.push_str(&format!("v 0 {} 0\n", if j == 0 { r } else { -r }));
            } else {
                s.push_str(&format!("v {} {} {}\n", r * ph.sin() * th.sin(), r * ph.cos(), r * ph.sin() * th.cos()));
            }
            s.push_str(&format!("vt {} {}\n", u, 1.0 - v));
            if bump == 0.0 {
                s.push_str(&format!("vn {} {} {}\n", ph.sin() * th.sin(), ph.cos(), ph.sin() * th.cos()));
            }
        }
    }
    let w = lon + 1;
    for j in 0..lat {
        for i in 0..lon {
            let (a, b, c, d) = (j * w + i + 1, j * w + i + 2, (j + 1) * w + i + 2, (j + 1) * w + i + 1);
            if bump == 0.0 {
                s.push_str(&format!("f {a}/{a}/{a} {d}/{d}/{d} {c}/{c}/{c}\nf {a}/{a}/{a} {c}/{c}/{c} {b}/{b}/{b}\n"));
            } else {
                s.push_str(&format!("f {a}/{a} {d}/{d} {c}/{c}\nf {a}/{a} {c}/{c} {b}/{b}\n"));
            }
        }
    }
    s
}
