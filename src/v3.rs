// SPDX-License-Identifier: MIT OR Apache-2.0
//! Tiny vec3 helpers on [f32; 3].
pub type V = [f32; 3];
pub fn add(a: V, b: V) -> V { [a[0] + b[0], a[1] + b[1], a[2] + b[2]] }
pub fn sub(a: V, b: V) -> V { [a[0] - b[0], a[1] - b[1], a[2] - b[2]] }
pub fn mul(a: V, s: f32) -> V { [a[0] * s, a[1] * s, a[2] * s] }
pub fn dot(a: V, b: V) -> f32 { a[0] * b[0] + a[1] * b[1] + a[2] * b[2] }
pub fn cross(a: V, b: V) -> V { [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]] }
pub fn len(a: V) -> f32 { dot(a, a).sqrt() }
pub fn norm(a: V) -> V {
    let l = len(a);
    if l < 1e-12 { [0.0, 0.0, 1.0] } else { mul(a, 1.0 / l) }
}
