// SPDX-License-Identifier: MIT OR Apache-2.0
//! A bounding volume hierarchy for ray casts against triangle soups (binned surface-area-heuristic build, nearest
//! child first traversal, triangles packed in leaf order). Used by the baker.
use crate::meshio::Tri;
use crate::v3::*;

#[derive(Clone, Copy)]
struct Node {
    min: V,
    max: V,
    /// leaf: first triangle slot; inner: index of the left child (the right child is the next node)
    first: u32,
    /// leaf: triangle count (> 0); inner: 0
    count: u32,
}

#[derive(Clone, Copy)]
struct PTri {
    p0: V,
    e1: V,
    e2: V,
    /// index in the mesh the BVH was built from
    id: u32,
    object: u32,
}

#[derive(Clone, Copy, Debug)]
pub struct Hit {
    pub t: f32,
    /// triangle index in the mesh the BVH was built from
    pub tri: u32,
    /// barycentric coordinates of the hit for corners 1 and 2
    pub u: f32,
    pub v: f32,
}

/// Which triangles a ray may hit.
#[derive(Clone, Copy)]
pub struct Filter<'a> {
    /// `objects[o]` tells whether triangles of object `o` count (None = all)
    pub objects: Option<&'a [bool]>,
    /// ignore triangles that face away from the ray
    pub cull_back: bool,
}

impl Filter<'_> {
    pub const ALL: Filter<'static> = Filter { objects: None, cull_back: false };
}

pub struct Bvh {
    nodes: Vec<Node>,
    tris: Vec<PTri>,
}

const LEAF: usize = 4;
const BINS: usize = 16;

#[derive(Clone, Copy)]
struct Aabb {
    min: V,
    max: V,
}

impl Aabb {
    const EMPTY: Aabb = Aabb { min: [f32::MAX; 3], max: [f32::MIN; 3] };
    fn grow(&mut self, p: V) {
        for k in 0..3 {
            self.min[k] = self.min[k].min(p[k]);
            self.max[k] = self.max[k].max(p[k]);
        }
    }
    fn join(&mut self, o: &Aabb) {
        for k in 0..3 {
            self.min[k] = self.min[k].min(o.min[k]);
            self.max[k] = self.max[k].max(o.max[k]);
        }
    }
    fn area(&self) -> f32 {
        if self.min[0] > self.max[0] {
            return 0.0;
        }
        let d = sub(self.max, self.min);
        2.0 * (d[0] * d[1] + d[1] * d[2] + d[2] * d[0])
    }
}

impl Bvh {
    pub fn build(tris: &[Tri]) -> Bvh {
        let n = tris.len();
        let boxes: Vec<Aabb> = tris
            .iter()
            .map(|t| {
                let mut b = Aabb::EMPTY;
                for p in &t.p {
                    b.grow(*p);
                }
                b
            })
            .collect();
        let cent: Vec<V> = boxes.iter().map(|b| mul(add(b.min, b.max), 0.5)).collect();
        let mut order: Vec<u32> = (0..n as u32).collect();
        let mut nodes: Vec<Node> = Vec::with_capacity(n / 2 + 2);
        nodes.push(Node { min: [0.0; 3], max: [0.0; 3], first: 0, count: 0 });
        if n > 0 {
            // explicit work list instead of recursion so very large meshes cannot overflow the stack
            let mut work: Vec<(usize, usize, usize)> = vec![(0, 0, n)];
            while let Some((ni, lo, hi)) = work.pop() {
                let mut b = Aabb::EMPTY;
                let mut cb = Aabb::EMPTY;
                for &i in &order[lo..hi] {
                    b.join(&boxes[i as usize]);
                    cb.grow(cent[i as usize]);
                }
                nodes[ni].min = b.min;
                nodes[ni].max = b.max;
                let count = hi - lo;
                if count <= LEAF {
                    nodes[ni].first = lo as u32;
                    nodes[ni].count = count as u32;
                    continue;
                }
                let ext = sub(cb.max, cb.min);
                let axis = if ext[0] >= ext[1] && ext[0] >= ext[2] { 0 } else if ext[1] >= ext[2] { 1 } else { 2 };
                let mut mid = lo + count / 2;
                if ext[axis] > 1e-12 {
                    // binned SAH along the widest centroid axis
                    let scale = BINS as f32 / ext[axis];
                    let bin_of = |c: f32| (((c - cb.min[axis]) * scale) as usize).min(BINS - 1);
                    let mut bins = [(Aabb::EMPTY, 0usize); BINS];
                    for &i in &order[lo..hi] {
                        let k = bin_of(cent[i as usize][axis]);
                        bins[k].0.join(&boxes[i as usize]);
                        bins[k].1 += 1;
                    }
                    let mut left_area = [0.0f32; BINS];
                    let mut left_cnt = [0usize; BINS];
                    let (mut acc, mut c) = (Aabb::EMPTY, 0usize);
                    for k in 0..BINS - 1 {
                        acc.join(&bins[k].0);
                        c += bins[k].1;
                        left_area[k] = acc.area();
                        left_cnt[k] = c;
                    }
                    let (mut best, mut best_k) = (f32::MAX, 0usize);
                    let (mut acc, mut c) = (Aabb::EMPTY, 0usize);
                    for k in (1..BINS).rev() {
                        acc.join(&bins[k].0);
                        c += bins[k].1;
                        let cost = left_area[k - 1] * left_cnt[k - 1] as f32 + acc.area() * c as f32;
                        if left_cnt[k - 1] > 0 && c > 0 && cost < best {
                            best = cost;
                            best_k = k - 1;
                        }
                    }
                    if best < f32::MAX {
                        // partition in place
                        let (mut i, mut j) = (lo, hi);
                        while i < j {
                            if bin_of(cent[order[i] as usize][axis]) <= best_k {
                                i += 1;
                            } else {
                                j -= 1;
                                order.swap(i, j);
                            }
                        }
                        if i > lo && i < hi {
                            mid = i;
                        } else {
                            order[lo..hi].sort_unstable_by(|&a, &b| cent[a as usize][axis].partial_cmp(&cent[b as usize][axis]).unwrap_or(std::cmp::Ordering::Equal));
                        }
                    } else {
                        order[lo..hi].sort_unstable_by(|&a, &b| cent[a as usize][axis].partial_cmp(&cent[b as usize][axis]).unwrap_or(std::cmp::Ordering::Equal));
                    }
                } else {
                    // all centroids coincide: split by count
                }
                let left = nodes.len();
                nodes.push(Node { min: [0.0; 3], max: [0.0; 3], first: 0, count: 0 });
                nodes.push(Node { min: [0.0; 3], max: [0.0; 3], first: 0, count: 0 });
                nodes[ni].first = left as u32;
                nodes[ni].count = 0;
                work.push((left, lo, mid));
                work.push((left + 1, mid, hi));
            }
        }
        let packed: Vec<PTri> = order
            .iter()
            .map(|&i| {
                let t = &tris[i as usize];
                PTri { p0: t.p[0], e1: sub(t.p[1], t.p[0]), e2: sub(t.p[2], t.p[0]), id: i, object: t.object }
            })
            .collect();
        Bvh { nodes, tris: packed }
    }

    /// The tree as plain words for a GPU buffer: two `[u32; 4]` per node (`min.xyz, first` and `max.xyz, count`, floats as
    /// bits) and three per triangle (`p0.xyz, id`, `e1.xyz, object`, `e2.xyz, 0`), in the order the tree uses.
    pub fn export(&self) -> (Vec<[u32; 4]>, Vec<[u32; 4]>) {
        let b = |x: f32| x.to_bits();
        let mut nodes = Vec::with_capacity(self.nodes.len() * 2);
        for n in &self.nodes {
            nodes.push([b(n.min[0]), b(n.min[1]), b(n.min[2]), n.first]);
            nodes.push([b(n.max[0]), b(n.max[1]), b(n.max[2]), n.count]);
        }
        let mut tris = Vec::with_capacity(self.tris.len() * 3);
        for t in &self.tris {
            tris.push([b(t.p0[0]), b(t.p0[1]), b(t.p0[2]), t.id]);
            tris.push([b(t.e1[0]), b(t.e1[1]), b(t.e1[2]), t.object]);
            tris.push([b(t.e2[0]), b(t.e2[1]), b(t.e2[2]), 0]);
        }
        (nodes, tris)
    }

    pub fn is_empty(&self) -> bool {
        self.tris.is_empty()
    }

    #[inline]
    fn slab(n: &Node, o: V, inv: V, tmax: f32) -> Option<f32> {
        let (mut t0, mut t1) = (0.0f32, tmax);
        for k in 0..3 {
            let a = (n.min[k] - o[k]) * inv[k];
            let b = (n.max[k] - o[k]) * inv[k];
            t0 = t0.max(a.min(b));
            t1 = t1.min(a.max(b));
        }
        if t0 <= t1 {
            Some(t0)
        } else {
            None
        }
    }

    /// Nearest hit with `tmin < t < tmax` along the unit direction `d`.
    pub fn closest(&self, o: V, d: V, tmin: f32, tmax: f32, f: &Filter) -> Option<Hit> {
        self.trace(o, d, tmin, tmax, f, false)
    }

    /// True when anything is hit before `tmax`.
    pub fn any(&self, o: V, d: V, tmin: f32, tmax: f32, f: &Filter) -> bool {
        self.trace(o, d, tmin, tmax, f, true).is_some()
    }

    fn trace(&self, o: V, d: V, tmin: f32, tmax: f32, f: &Filter, first_hit: bool) -> Option<Hit> {
        if self.tris.is_empty() {
            return None;
        }
        let fix = |x: f32| if x.abs() < 1e-12 { 1e-12f32.copysign(x) } else { x };
        let inv = [1.0 / fix(d[0]), 1.0 / fix(d[1]), 1.0 / fix(d[2])];
        let mut best: Option<Hit> = None;
        let mut tm = tmax;
        let mut stack = [0u32; 96];
        let mut sp = 0usize;
        if Self::slab(&self.nodes[0], o, inv, tm).is_none() {
            return None;
        }
        stack[0] = 0;
        sp += 1;
        while sp > 0 {
            sp -= 1;
            let n = &self.nodes[stack[sp] as usize];
            if n.count > 0 {
                for ps in &self.tris[n.first as usize..(n.first + n.count) as usize] {
                    if let Some(objs) = f.objects {
                        if !objs.get(ps.object as usize).copied().unwrap_or(true) {
                            continue;
                        }
                    }
                    let pv = cross(d, ps.e2);
                    let det = dot(ps.e1, pv);
                    if det.abs() < 1e-20 || (f.cull_back && det < 0.0) {
                        continue;
                    }
                    let id = 1.0 / det;
                    let tv = sub(o, ps.p0);
                    let u = dot(tv, pv) * id;
                    if !(-1e-6..=1.0 + 1e-6).contains(&u) {
                        continue;
                    }
                    let q = cross(tv, ps.e1);
                    let v = dot(d, q) * id;
                    if v < -1e-6 || u + v > 1.0 + 1e-6 {
                        continue;
                    }
                    let t = dot(ps.e2, q) * id;
                    if t > tmin && t < tm {
                        tm = t;
                        best = Some(Hit { t, tri: ps.id, u: u.clamp(0.0, 1.0), v: v.clamp(0.0, 1.0) });
                        if first_hit {
                            return best;
                        }
                    }
                }
            } else {
                let (l, r) = (n.first as usize, n.first as usize + 1);
                let (tl, tr) = (Self::slab(&self.nodes[l], o, inv, tm), Self::slab(&self.nodes[r], o, inv, tm));
                match (tl, tr) {
                    (Some(a), Some(b)) => {
                        // visit the nearer child first (it is pushed last)
                        let (far, near) = if a <= b { (r, l) } else { (l, r) };
                        if sp + 2 <= stack.len() {
                            stack[sp] = far as u32;
                            stack[sp + 1] = near as u32;
                            sp += 2;
                        }
                    }
                    (Some(_), None) => {
                        stack[sp] = l as u32;
                        sp += 1;
                    }
                    (None, Some(_)) => {
                        stack[sp] = r as u32;
                        sp += 1;
                    }
                    _ => {}
                }
            }
        }
        best
    }
}
