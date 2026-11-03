// SPDX-License-Identifier: MIT OR Apache-2.0
//! Meshes loaded from files and the scenes the viewer builds around the material: floor, look-dev balls, pedestal.
use crate::meshio;
use crate::render3d::{self, Mesh, Raw, View, RELIEF_DISPLACE, SHAPE_CUSTOM};
use crate::scene::{Instance, Material, SceneKind};
use crate::tex::Tex;
use crate::v3::*;
use std::collections::HashMap;

/// A mesh from a file, prepared for the viewer: indexed, normalised to fit the view, with tangents and UV scales.
#[derive(Clone, Debug)]
pub struct CustomMesh {
    pub name: String,
    pub pos: Vec<V>,
    pub nrm: Vec<V>,
    pub tan: Vec<V>,
    /// texture coordinates, v running downwards like the image rows
    pub uv: Vec<[f32; 2]>,
    pub su: Vec<f32>,
    pub sv: Vec<f32>,
    /// vertices at the same place share an id
    pub weld: Vec<u32>,
    /// smooth direction the vertex moves in when displaced
    pub disp: Vec<V>,
    pub tris: Vec<[u32; 3]>,
    /// the mesh had no UVs and a box projection was generated
    pub generated_uv: bool,
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

impl CustomMesh {
    pub fn from_mesh(src: &meshio::Mesh) -> Result<CustomMesh, String> {
        if src.tris.is_empty() {
            return Err("the mesh has no triangles".into());
        }
        let mut m = src.clone();
        if !m.has_normals {
            m.apply_normals(meshio::NormalMode::Angle(60.0));
        }
        let (mn, mx) = m.bounds();
        let centre = mul(add(mn, mx), 0.5);
        let half_diag = len(sub(mx, mn)) * 0.5;
        if !(half_diag.is_finite() && half_diag > 1e-12) {
            return Err("the mesh has no size".into());
        }
        let k = 1.1 / half_diag;
        let generated = !m.has_uv;
        let mut map: HashMap<Vec<u32>, u32> = HashMap::new();
        let (mut pos, mut nrm, mut uv): (Vec<V>, Vec<V>, Vec<[f32; 2]>) = (vec![], vec![], vec![]);
        let mut tris: Vec<[u32; 3]> = Vec::with_capacity(m.tris.len());
        for t in &m.tris {
            let fnrm = cross(sub(t.p[1], t.p[0]), sub(t.p[2], t.p[0]));
            let ax = {
                let a = [fnrm[0].abs(), fnrm[1].abs(), fnrm[2].abs()];
                if a[0] >= a[1] && a[0] >= a[2] { 0 } else if a[1] >= a[2] { 1 } else { 2 }
            };
            let mut idx = [0u32; 3];
            for c in 0..3 {
                let p = mul(sub(t.p[c], centre), k);
                let n = norm(t.n[c]);
                let q = if generated {
                    // box projection along the dominant axis of the face
                    let (u, v) = match ax {
                        0 => (p[2], p[1]),
                        1 => (p[0], p[2]),
                        _ => (p[0], p[1]),
                    };
                    [u * 0.5 + 0.5, 0.5 - v * 0.5]
                } else {
                    [t.uv[c][0], 1.0 - t.uv[c][1]]
                };
                let mut key = bits(&p);
                key.extend(bits(&n));
                key.extend(bits(&q));
                let id = *map.entry(key).or_insert_with(|| {
                    pos.push(p);
                    nrm.push(n);
                    uv.push(q);
                    (pos.len() - 1) as u32
                });
                idx[c] = id;
            }
            if idx[0] != idx[1] && idx[1] != idx[2] && idx[0] != idx[2] {
                tris.push(idx);
            }
        }
        if tris.is_empty() {
            return Err("every triangle of the mesh is degenerate".into());
        }
        let nv = pos.len();
        // tangent frame and uv scale from the uv gradients, area weighted
        let mut tan = vec![[0.0f32; 3]; nv];
        let mut su = vec![0.0f32; nv];
        let mut sv = vec![0.0f32; nv];
        let mut wsum = vec![0.0f32; nv];
        for t in &tris {
            let (i0, i1, i2) = (t[0] as usize, t[1] as usize, t[2] as usize);
            let (e1, e2) = (sub(pos[i1], pos[i0]), sub(pos[i2], pos[i0]));
            let (d1, d2) = ([uv[i1][0] - uv[i0][0], uv[i1][1] - uv[i0][1]], [uv[i2][0] - uv[i0][0], uv[i2][1] - uv[i0][1]]);
            let det = d1[0] * d2[1] - d2[0] * d1[1];
            let area = len(cross(e1, e2));
            if det.abs() < 1e-12 || area < 1e-14 {
                continue;
            }
            let r = 1.0 / det;
            let dpdu = mul(sub(mul(e1, d2[1]), mul(e2, d1[1])), r);
            let dpdv = mul(sub(mul(e2, d1[0]), mul(e1, d2[0])), r);
            for &i in t {
                let i = i as usize;
                tan[i] = add(tan[i], mul(norm(dpdu), area));
                su[i] += len(dpdu) * area;
                sv[i] += len(dpdv) * area;
                wsum[i] += area;
            }
        }
        for i in 0..nv {
            let n = nrm[i];
            let t = sub(tan[i], mul(n, dot(tan[i], n)));
            tan[i] = if dot(t, t) > 1e-12 {
                norm(t)
            } else {
                let a = if n[0].abs() < 0.9 { [1.0, 0.0, 0.0] } else { [0.0, 1.0, 0.0] };
                norm(cross(a, n))
            };
            if wsum[i] > 0.0 {
                su[i] = (su[i] / wsum[i]).max(1e-4);
                sv[i] = (sv[i] / wsum[i]).max(1e-4);
            } else {
                su[i] = 1.0;
                sv[i] = 1.0;
            }
        }
        // weld by position so hard edges do not open up when displaced
        let mut wm: HashMap<[i64; 3], u32> = HashMap::new();
        let weld: Vec<u32> = pos
            .iter()
            .map(|p| {
                let key = [(p[0] * 20000.0).round() as i64, (p[1] * 20000.0).round() as i64, (p[2] * 20000.0).round() as i64];
                let n = wm.len() as u32;
                *wm.entry(key).or_insert(n)
            })
            .collect();
        let mut acc = vec![[0.0f32; 3]; wm.len()];
        for i in 0..nv {
            acc[weld[i] as usize] = add(acc[weld[i] as usize], nrm[i]);
        }
        let disp: Vec<V> = (0..nv)
            .map(|i| {
                let a = acc[weld[i] as usize];
                if dot(a, a) > 1e-10 { norm(a) } else { nrm[i] }
            })
            .collect();
        Ok(CustomMesh { name: src.name.clone(), pos, nrm, tan, uv, su, sv, weld, disp, tris, generated_uv: generated })
    }

    pub fn load(path: &std::path::Path) -> Result<CustomMesh, String> {
        let m = meshio::load_file(path)?;
        CustomMesh::from_mesh(&m)
    }

    pub(crate) fn to_raw(&self, tiling: f32) -> Raw {
        let mut r = Raw::new();
        r.pos = self.pos.clone();
        r.nrm = self.nrm.clone();
        r.tan = self.tan.clone();
        r.uv = self.uv.iter().map(|q| [q[0] * tiling, q[1] * tiling]).collect();
        r.su = self.su.iter().map(|s| s / tiling).collect();
        r.sv = self.sv.iter().map(|s| s / tiling).collect();
        r.tris = self.tris.clone();
        r.weld = self.weld.clone();
        r.fade = vec![1.0; self.pos.len()];
        r.disp = self.disp.clone();
        r
    }
}

/// The objects of a scene, with the shapes in front of the camera.
pub struct SceneMesh {
    pub mesh: Mesh,
    /// one entry per object (`Mesh::obj` indexes this)
    pub objects: Vec<Instance>,
}

fn rot(p: V, pitch: f32, yaw: f32) -> V {
    let (sp, cp) = pitch.sin_cos();
    let q = [p[0], p[1] * cp - p[2] * sp, p[1] * sp + p[2] * cp];
    let (sy, cy) = yaw.sin_cos();
    [q[0] * cy + q[2] * sy, q[1], -q[0] * sy + q[2] * cy]
}

const GREY: Material = Material::Fixed([0.18, 0.18, 0.18], 0.5, 0.0);
const CHROME: Material = Material::Fixed([0.95, 0.95, 0.95], 0.03, 1.0);
const FLOOR: Material = Material::Fixed([0.16, 0.16, 0.17], 0.6, 0.0);
const BACKDROP: Material = Material::Fixed([0.55, 0.55, 0.56], 0.9, 0.0);
const PLINTH: Material = Material::Fixed([0.3, 0.3, 0.31], 0.35, 0.0);

fn inst(shape: u32, pos: V, pitch: f32, scale: f32, material: Material, tiling: f32) -> Instance {
    Instance { shape, pos, yaw: 0.0, pitch, scale, material, tiling }
}

/// The objects around the main one, given the lowest point (y) of the main object.
pub fn companions(kind: SceneKind, ymin: f32) -> Vec<Instance> {
    let flat = -std::f32::consts::FRAC_PI_2;
    match kind {
        SceneKind::Single => vec![],
        SceneKind::Floor => vec![inst(2, [0.0, ymin - 0.002, 0.0], flat, 4.0, FLOOR, 8.0)],
        SceneKind::LookDev => vec![
            inst(2, [0.0, ymin - 0.002, 0.0], flat, 4.0, FLOOR, 8.0),
            inst(0, [-2.0, ymin + 0.55, 0.3], 0.0, 0.55, GREY, 1.0),
            inst(0, [2.0, ymin + 0.55, 0.3], 0.0, 0.55, CHROME, 1.0),
        ],
        SceneKind::Pedestal => {
            let plinth_top = ymin - 0.002;
            let h = 1.2;
            let floor_y = plinth_top - 2.0 * h;
            vec![
                inst(3, [0.0, plinth_top - h, 0.0], 0.0, 1.5, PLINTH, 1.0),
                inst(2, [0.0, floor_y, 0.0], flat, 4.0, FLOOR, 8.0),
                inst(2, [0.0, floor_y + 6.4, -5.0], 0.0, 5.0, BACKDROP, 2.0),
            ]
        }
    }
}

/// Build every triangle of the scene in world space. `height` drives the displacement of the objects that use the
/// graph's material.
pub fn build_scene(v: &View, height: Option<&Tex>) -> SceneMesh {
    build_scene_ex(v, height, None)
}

/// Like [`build_scene`], with the lowest point of the main object given by the caller (the GPU viewport displaces in
/// the vertex shader and works it out itself).
pub fn build_scene_ex(v: &View, height: Option<&Tex>, ymin_override: Option<f32>) -> SceneMesh {
    let displace = v.relief == RELIEF_DISPLACE && v.height_scale > 0.0 && height.is_some();
    let amount = if displace { v.height_scale } else { 0.0 };
    let custom = v.custom.as_deref();
    let shape = if v.shape == SHAPE_CUSTOM && custom.is_none() { 0 } else { v.shape };
    let main_mesh = render3d::build_mesh_ex(shape, v.tess, v.tiling, height, amount, v.height_level, custom);
    let ymin = ymin_override.unwrap_or_else(|| main_mesh.pos.iter().map(|p| p[1]).fold(f32::MAX, f32::min));
    let mut list = vec![(Instance { shape, pos: [0.0; 3], yaw: 0.0, pitch: 0.0, scale: 1.0, material: Material::Graph, tiling: 0.0 }, Some(main_mesh))];
    for i in companions(v.scene, if ymin.is_finite() { ymin } else { -1.0 }) {
        list.push((i, None));
    }
    let mut out = Mesh { pos: vec![], nrm: vec![], tan: vec![], uv: vec![], su: vec![], sv: vec![], fade: vec![], base_nrm: vec![], disp: vec![], tris: vec![], obj: vec![], cull: vec![] };
    let mut objects = Vec::new();
    for (oi, (i, pre)) in list.into_iter().enumerate() {
        let m = pre.unwrap_or_else(|| {
            let tess = v.tess.min(64);
            let tiling = if i.tiling > 0.0 { i.tiling } else { v.tiling };
            render3d::build_mesh_ex(i.shape, tess, tiling, None, 0.0, 0.0, None)
        });
        let base = out.pos.len() as u32;
        let r = |p: V| rot(p, i.pitch, i.yaw);
        for k in 0..m.pos.len() {
            out.pos.push(add(mul(r(m.pos[k]), i.scale), i.pos));
            out.nrm.push(r(m.nrm[k]));
            out.tan.push(r(m.tan[k]));
            out.base_nrm.push(r(m.base_nrm[k]));
            out.disp.push(r(m.disp[k]));
            out.uv.push(m.uv[k]);
            out.su.push(m.su[k] * i.scale);
            out.sv.push(m.sv[k] * i.scale);
            out.fade.push(m.fade[k]);
        }
        for t in &m.tris {
            out.tris.push([t[0] + base, t[1] + base, t[2] + base]);
            out.obj.push(oi as u32);
        }
        out.cull.push(m.cull.first().copied().unwrap_or(false));
        objects.push(i);
    }
    SceneMesh { mesh: out, objects }
}
