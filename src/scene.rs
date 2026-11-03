// SPDX-License-Identifier: MIT OR Apache-2.0
//! Settings of the 3D viewer that are not about the mesh itself: light rig, environment, post-processing, channel
//! isolation, scene layouts and camera presets. Everything here is plain data with a text form (`key value` lines) so
//! the editor can store it and the command line can override it.
use crate::v3::*;

// ------------------------------------------------------------------------------------------------ lights
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LightKind {
    /// Parallel light from a direction (sun, softbox far away). `intensity` is the irradiance on a surface facing it.
    Directional,
    /// Light from a point at `distance` from the middle of the scene; `intensity` is radiant intensity (falls off with
    /// the square of the distance).
    Point,
    /// Point light aimed at the middle of the scene with a cone.
    Spot,
}

impl LightKind {
    pub const ALL: [LightKind; 3] = [LightKind::Directional, LightKind::Point, LightKind::Spot];
    pub fn name(self) -> &'static str {
        match self {
            LightKind::Directional => "Directional",
            LightKind::Point => "Point",
            LightKind::Spot => "Spot",
        }
    }
    pub fn id(self) -> &'static str {
        match self {
            LightKind::Directional => "directional",
            LightKind::Point => "point",
            LightKind::Spot => "spot",
        }
    }
    pub fn parse(s: &str) -> Option<LightKind> {
        Self::ALL.into_iter().find(|k| k.id() == s.to_lowercase())
    }
}

pub const MAX_LIGHTS: usize = 6;

#[derive(Clone, Debug, PartialEq)]
pub struct Light {
    pub name: String,
    pub enabled: bool,
    pub kind: LightKind,
    /// direction towards the light: azimuth and elevation in radians (see `follows_camera`)
    pub yaw: f32,
    pub pitch: f32,
    /// distance from the middle of the scene (point and spot lights)
    pub distance: f32,
    pub color: [f32; 3],
    pub intensity: f32,
    /// outer half angle of a spot light in radians
    pub cone: f32,
    /// 0 = hard, 1 = very soft: size of the light source, which blurs shadows
    pub softness: f32,
    /// yaw / pitch are measured from the viewing direction, so the light stays put on screen while the camera orbits
    pub follows_camera: bool,
    pub shadows: bool,
}

impl Light {
    pub fn key() -> Light {
        Light { name: "Key".into(), enabled: true, kind: LightKind::Directional, yaw: 0.9, pitch: 0.7, distance: 4.0, color: [1.0, 0.98, 0.95], intensity: 3.5, cone: 0.7, softness: 0.25, follows_camera: true, shadows: true }
    }
    pub fn fill() -> Light {
        Light { name: "Fill".into(), enabled: true, kind: LightKind::Directional, yaw: -2.2, pitch: -0.2, distance: 4.0, color: [0.88, 0.93, 1.0], intensity: 1.2, cone: 0.7, softness: 0.6, follows_camera: true, shadows: false }
    }
    pub fn rim() -> Light {
        Light { name: "Rim".into(), enabled: true, kind: LightKind::Directional, yaw: 2.6, pitch: 0.5, distance: 4.0, color: [1.0, 0.95, 0.9], intensity: 2.0, cone: 0.7, softness: 0.3, follows_camera: true, shadows: false }
    }
    pub fn point() -> Light {
        Light { name: "Point".into(), enabled: true, kind: LightKind::Point, yaw: -0.6, pitch: 0.5, distance: 3.0, color: [1.0, 0.9, 0.75], intensity: 25.0, cone: 0.7, softness: 0.15, follows_camera: false, shadows: true }
    }
    pub fn spot() -> Light {
        Light { name: "Spot".into(), enabled: true, kind: LightKind::Spot, yaw: 0.5, pitch: 0.8, distance: 4.0, color: [1.0, 1.0, 1.0], intensity: 60.0, cone: 0.45, softness: 0.2, follows_camera: false, shadows: true }
    }
    pub fn new_of(kind: LightKind) -> Light {
        match kind {
            LightKind::Directional => Light { name: "Light".into(), follows_camera: false, ..Light::rim() },
            LightKind::Point => Light::point(),
            LightKind::Spot => Light::spot(),
        }
    }
}

/// The default three-point rig: key, fill and a dim rim light.
pub fn default_rig() -> Vec<Light> {
    vec![Light::key(), Light::fill(), Light { enabled: false, ..Light::rim() }]
}

/// Named rigs for the "Lights" menu.
pub const RIGS: &[(&str, &str)] = &[("three_point", "Three point"), ("key_only", "Single key light"), ("top", "Overhead softbox"), ("sun", "Sun and sky"), ("none", "No lights (environment only)")];

pub fn rig(name: &str) -> Vec<Light> {
    match name {
        "key_only" => vec![Light::key()],
        "top" => vec![Light { name: "Softbox".into(), yaw: 0.0, pitch: 1.3, intensity: 3.0, softness: 0.9, follows_camera: false, ..Light::key() }],
        "sun" => vec![Light { name: "Sun".into(), yaw: 0.6, pitch: 0.75, intensity: 4.5, color: [1.0, 0.93, 0.82], softness: 0.08, follows_camera: false, ..Light::key() }],
        "none" => vec![],
        _ => vec![Light::key(), Light::fill(), Light { enabled: true, intensity: 1.6, ..Light::rim() }],
    }
}

/// The camera frame lights are placed in.
#[derive(Clone, Copy, Debug)]
pub struct CamFrame {
    pub right: V,
    pub up: V,
    /// unit vector from the camera towards the scene
    pub fwd: V,
}

/// Where a light is and how it shines: direction towards it from the middle of the scene (and, for point and spot
/// lights, its position).
pub fn light_axis(l: &Light, cam: &CamFrame) -> V {
    let (sy, cy) = l.yaw.sin_cos();
    let (sp, cp) = l.pitch.sin_cos();
    let local = [sy * cp, sp, cy * cp];
    if l.follows_camera {
        norm(add(add(mul(cam.right, local[0]), mul(cam.up, local[1])), mul(cam.fwd, -local[2])))
    } else {
        norm(local)
    }
}

/// Light arriving at the world point `p`: (unit vector towards the light, irradiance colour, distance to the light or
/// infinity). The colour is `intensity * colour`, divided by the squared distance and shaped by the cone where needed.
pub fn light_at(l: &Light, cam: &CamFrame, p: V) -> (V, V, f32) {
    let axis = light_axis(l, cam);
    let c = mul(l.color, l.intensity);
    match l.kind {
        LightKind::Directional => (axis, c, f32::INFINITY),
        LightKind::Point | LightKind::Spot => {
            let pos = mul(axis, l.distance.max(0.2));
            let v = sub(pos, p);
            let d2 = dot(v, v).max(1e-4);
            let d = d2.sqrt();
            let dir = mul(v, 1.0 / d);
            let mut k = 1.0 / d2;
            if l.kind == LightKind::Spot {
                // aimed at the middle of the scene
                let cos_out = l.cone.clamp(0.05, 1.5).cos();
                let cos_in = (l.cone * (1.0 - 0.6 * l.softness.clamp(0.0, 1.0))).clamp(0.02, 1.5).cos();
                let cd = dot(mul(dir, -1.0), mul(axis, -1.0));
                let t = ((cd - cos_out) / (cos_in - cos_out).max(1e-4)).clamp(0.0, 1.0);
                k *= t * t * (3.0 - 2.0 * t);
            }
            (dir, mul(c, k), d)
        }
    }
}

// ------------------------------------------------------------------------------------------------ environment
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Background {
    /// the environment image (blurred by `blur`)
    Environment,
    /// a flat colour
    Color,
    /// the dark neutral gradient of the classic view
    Gradient,
}

impl Background {
    pub const ALL: [Background; 3] = [Background::Gradient, Background::Environment, Background::Color];
    pub fn name(self) -> &'static str {
        match self {
            Background::Environment => "Environment",
            Background::Color => "Colour",
            Background::Gradient => "Dark gradient",
        }
    }
    pub fn id(self) -> &'static str {
        match self {
            Background::Environment => "environment",
            Background::Color => "color",
            Background::Gradient => "gradient",
        }
    }
    pub fn parse(s: &str) -> Option<Background> {
        Self::ALL.into_iter().find(|b| b.id() == s.to_lowercase())
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct EnvSettings {
    /// name of a built-in environment (see `hdri::PRESETS`); ignored when `file` is set
    pub preset: String,
    /// path of an .hdr / .exr / .png environment image
    pub file: Option<String>,
    /// rotation around the vertical axis in radians
    pub rotation: f32,
    /// brightness of the environment lighting in stops (powers of two)
    pub exposure: f32,
    pub background: Background,
    pub bg_color: [f32; 3],
    /// blur of the environment in the background, 0 sharp .. 1 fully blurred
    pub blur: f32,
}

impl Default for EnvSettings {
    fn default() -> Self {
        EnvSettings { preset: "studio".into(), file: None, rotation: 0.0, exposure: 0.0, background: Background::Gradient, bg_color: [0.14, 0.14, 0.15], blur: 0.0 }
    }
}

impl EnvSettings {
    /// Identity of the lighting image, for caches: changes when the image would.
    pub fn image_key(&self) -> String {
        match &self.file {
            Some(f) => format!("file:{f}"),
            None => format!("preset:{}", self.preset),
        }
    }
}

// ------------------------------------------------------------------------------------------------ post-processing
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ToneMap {
    /// hue-preserving filmic curve (the classic view)
    Filmic,
    /// ACES fit applied per channel
    Aces,
    /// John Hable's "Uncharted 2" curve
    Hable,
    /// x / (1 + x)
    Reinhard,
    /// clip
    Linear,
}

impl ToneMap {
    pub const ALL: [ToneMap; 5] = [ToneMap::Filmic, ToneMap::Aces, ToneMap::Hable, ToneMap::Reinhard, ToneMap::Linear];
    pub fn name(self) -> &'static str {
        match self {
            ToneMap::Filmic => "Filmic (hue preserving)",
            ToneMap::Aces => "ACES",
            ToneMap::Hable => "Hable (Uncharted 2)",
            ToneMap::Reinhard => "Reinhard",
            ToneMap::Linear => "Linear (clip)",
        }
    }
    pub fn id(self) -> &'static str {
        match self {
            ToneMap::Filmic => "filmic",
            ToneMap::Aces => "aces",
            ToneMap::Hable => "hable",
            ToneMap::Reinhard => "reinhard",
            ToneMap::Linear => "linear",
        }
    }
    pub fn index(self) -> i32 {
        Self::ALL.iter().position(|t| *t == self).unwrap_or(0) as i32
    }
    pub fn parse(s: &str) -> Option<ToneMap> {
        Self::ALL.into_iter().find(|t| t.id() == s.to_lowercase())
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Post {
    pub tonemap: ToneMap,
    /// glow of bright areas: strength (0 = off), luminance threshold, radius (0..1 of the image)
    pub bloom: f32,
    pub bloom_threshold: f32,
    pub bloom_radius: f32,
    /// darkening of the corners (0 = off)
    pub vignette: f32,
    /// contrast around middle grey (1 = unchanged)
    pub contrast: f32,
    pub saturation: f32,
    /// white balance: -1 cool .. 1 warm, tint -1 green .. 1 magenta
    pub temperature: f32,
    pub tint: f32,
    /// ambient occlusion computed from the depth buffer: strength (0 = off) and radius in scene units
    pub ssao: f32,
    pub ssao_radius: f32,
    /// depth of field: aperture (0 = off) and focus distance from the camera (0 = the middle of the scene)
    pub dof: f32,
    pub focus: f32,
}

impl Default for Post {
    fn default() -> Self {
        Post { tonemap: ToneMap::Filmic, bloom: 0.0, bloom_threshold: 1.0, bloom_radius: 0.35, vignette: 0.0, contrast: 1.0, saturation: 1.0, temperature: 0.0, tint: 0.0, ssao: 0.0, ssao_radius: 0.25, dof: 0.0, focus: 0.0 }
    }
}

pub const POST_PRESETS: &[(&str, &str)] = &[("neutral", "Neutral"), ("cinematic", "Cinematic"), ("punchy", "Punchy product shot"), ("soft", "Soft glow")];

pub fn post_preset(name: &str) -> Post {
    match name {
        "cinematic" => Post { tonemap: ToneMap::Aces, bloom: 0.25, bloom_threshold: 1.2, vignette: 0.35, contrast: 1.12, saturation: 0.92, temperature: 0.12, ssao: 0.5, ..Post::default() },
        "punchy" => Post { tonemap: ToneMap::Hable, contrast: 1.18, saturation: 1.18, ssao: 0.6, vignette: 0.12, ..Post::default() },
        "soft" => Post { tonemap: ToneMap::Filmic, bloom: 0.45, bloom_threshold: 0.8, bloom_radius: 0.5, contrast: 0.95, saturation: 1.05, ..Post::default() },
        _ => Post::default(),
    }
}

// ------------------------------------------------------------------------------------------------ channels, scenes, camera
/// What the view shows: the lit material or one channel of it, unlit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Channel {
    Lit,
    BaseColor,
    Roughness,
    Metallic,
    Normal,
    Ao,
    Height,
    Emissive,
    /// the shading normal in world space
    WorldNormal,
    /// the texture coordinates as colours with a checker
    Uv,
}

impl Channel {
    pub const ALL: [Channel; 10] = [Channel::Lit, Channel::BaseColor, Channel::Roughness, Channel::Metallic, Channel::Normal, Channel::Ao, Channel::Height, Channel::Emissive, Channel::WorldNormal, Channel::Uv];
    pub fn name(self) -> &'static str {
        match self {
            Channel::Lit => "Lit material",
            Channel::BaseColor => "Base colour",
            Channel::Roughness => "Roughness",
            Channel::Metallic => "Metallic",
            Channel::Normal => "Normal map",
            Channel::Ao => "Ambient occlusion",
            Channel::Height => "Height",
            Channel::Emissive => "Emissive",
            Channel::WorldNormal => "World normal",
            Channel::Uv => "UV layout",
        }
    }
    pub fn id(self) -> &'static str {
        match self {
            Channel::Lit => "lit",
            Channel::BaseColor => "base",
            Channel::Roughness => "roughness",
            Channel::Metallic => "metallic",
            Channel::Normal => "normal",
            Channel::Ao => "ao",
            Channel::Height => "height",
            Channel::Emissive => "emissive",
            Channel::WorldNormal => "world_normal",
            Channel::Uv => "uv",
        }
    }
    pub fn index(self) -> i32 {
        Self::ALL.iter().position(|c| *c == self).unwrap_or(0) as i32
    }
    pub fn parse(s: &str) -> Option<Channel> {
        Self::ALL.into_iter().find(|c| c.id() == s.to_lowercase())
    }
}

/// How the shape is placed in the scene.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SceneKind {
    /// the shape alone
    Single,
    /// the shape standing on a floor, which catches shadows and reflections
    Floor,
    /// look-dev line-up: the shape in the middle between a grey and a chrome reference ball
    LookDev,
    /// the shape on a round plinth in front of a backdrop
    Pedestal,
}

impl SceneKind {
    pub const ALL: [SceneKind; 4] = [SceneKind::Single, SceneKind::Floor, SceneKind::LookDev, SceneKind::Pedestal];
    pub fn name(self) -> &'static str {
        match self {
            SceneKind::Single => "Single object",
            SceneKind::Floor => "Object on a floor",
            SceneKind::LookDev => "Look-dev balls",
            SceneKind::Pedestal => "Pedestal and backdrop",
        }
    }
    pub fn id(self) -> &'static str {
        match self {
            SceneKind::Single => "single",
            SceneKind::Floor => "floor",
            SceneKind::LookDev => "lookdev",
            SceneKind::Pedestal => "pedestal",
        }
    }
    pub fn parse(s: &str) -> Option<SceneKind> {
        Self::ALL.into_iter().find(|k| k.id() == s.to_lowercase())
    }
}

/// Camera presets: (id, name, yaw, pitch, distance)
pub const CAMERAS: &[(&str, &str, f32, f32, f32)] = &[
    ("three_quarter", "Three quarter", 0.5, 0.3, 3.4),
    ("front", "Front", 0.0, 0.0, 3.4),
    ("back", "Back", std::f32::consts::PI, 0.0, 3.4),
    ("left", "Left", -std::f32::consts::FRAC_PI_2, 0.0, 3.4),
    ("right", "Right", std::f32::consts::FRAC_PI_2, 0.0, 3.4),
    ("top", "Top", 0.0, 1.5, 3.4),
    ("bottom", "Bottom", 0.0, -1.4, 3.4),
    ("close_up", "Close-up", 0.4, 0.2, 1.8),
    ("wide", "Wide", 0.7, 0.35, 6.0),
];

/// A scene object other than the main shape: where it stands and what it is made of.
#[derive(Clone, Debug)]
pub struct Instance {
    /// shape id (see `render3d::SHAPES`), `render3d::SHAPE_CUSTOM` for the loaded mesh
    pub shape: u32,
    pub pos: V,
    /// rotation around the vertical axis (radians), applied after `pitch`
    pub yaw: f32,
    /// rotation around the x axis (radians)
    pub pitch: f32,
    pub scale: f32,
    pub material: Material,
    /// UV tiling override (0 = the view's tiling)
    pub tiling: f32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Material {
    /// the material of the graph
    Graph,
    /// a constant material: base colour (linear), roughness, metallic
    Fixed([f32; 3], f32, f32),
}

// ------------------------------------------------------------------------------------------------ text form
pub fn parse_bool(s: &str) -> Result<bool, String> {
    match s.to_lowercase().as_str() {
        "1" | "true" | "yes" | "on" => Ok(true),
        "0" | "false" | "no" | "off" => Ok(false),
        _ => Err(format!("'{s}' is not a yes / no value")),
    }
}

pub fn parse_f32(s: &str) -> Result<f32, String> {
    s.trim().parse::<f32>().map_err(|_| format!("'{s}' is not a number"))
}

pub fn parse_color(s: &str) -> Result<[f32; 3], String> {
    let t: Vec<&str> = s.split(',').collect();
    if t.len() == 1 && s.starts_with('#') && s.len() == 7 {
        let h = |i: usize| u8::from_str_radix(&s[i..i + 2], 16).map(|v| (v as f32 / 255.0).powf(2.2)).map_err(|_| format!("bad colour '{s}'"));
        return Ok([h(1)?, h(3)?, h(5)?]);
    }
    if t.len() != 3 {
        return Err(format!("'{s}' is not a colour (use r,g,b or #rrggbb)"));
    }
    Ok([parse_f32(t[0])?, parse_f32(t[1])?, parse_f32(t[2])?])
}

pub fn fmt_color(c: [f32; 3]) -> String {
    format!("{},{},{}", c[0], c[1], c[2])
}

impl Light {
    /// `key value` lines for one light, prefixed with `light.<i>.`
    pub fn to_text(&self, i: usize, out: &mut String) {
        let p = format!("light.{i}.");
        let mut put = |k: &str, v: String| {
            out.push_str(&p);
            out.push_str(k);
            out.push(' ');
            out.push_str(&v);
            out.push('\n');
        };
        put("name", self.name.clone());
        put("enabled", (self.enabled as u8).to_string());
        put("kind", self.kind.id().to_string());
        put("yaw", self.yaw.to_string());
        put("pitch", self.pitch.to_string());
        put("distance", self.distance.to_string());
        put("color", fmt_color(self.color));
        put("intensity", self.intensity.to_string());
        put("cone", self.cone.to_string());
        put("softness", self.softness.to_string());
        put("follows_camera", (self.follows_camera as u8).to_string());
        put("shadows", (self.shadows as u8).to_string());
    }

    pub fn set(&mut self, key: &str, val: &str) -> Result<(), String> {
        match key {
            "name" => self.name = val.to_string(),
            "enabled" => self.enabled = parse_bool(val)?,
            "kind" => self.kind = LightKind::parse(val).ok_or_else(|| format!("unknown light kind '{val}' (directional, point, spot)"))?,
            "yaw" => self.yaw = parse_f32(val)?,
            "pitch" => self.pitch = parse_f32(val)?.clamp(-1.57, 1.57),
            "distance" => self.distance = parse_f32(val)?.clamp(0.3, 40.0),
            "color" => self.color = parse_color(val)?,
            "intensity" => self.intensity = parse_f32(val)?.clamp(0.0, 10000.0),
            "cone" => self.cone = parse_f32(val)?.clamp(0.05, 1.5),
            "softness" => self.softness = parse_f32(val)?.clamp(0.0, 1.0),
            "follows_camera" => self.follows_camera = parse_bool(val)?,
            "shadows" => self.shadows = parse_bool(val)?,
            _ => return Err(format!("unknown light setting '{key}'")),
        }
        Ok(())
    }
}

impl EnvSettings {
    pub fn to_text(&self, out: &mut String) {
        out.push_str(&format!("env.preset {}\n", self.preset));
        if let Some(f) = &self.file {
            out.push_str(&format!("env.file {f}\n"));
        }
        out.push_str(&format!("env.rotation {}\nenv.exposure {}\nenv.background {}\nenv.bg_color {}\nenv.blur {}\n", self.rotation, self.exposure, self.background.id(), fmt_color(self.bg_color), self.blur));
    }
    pub fn set(&mut self, key: &str, val: &str) -> Result<(), String> {
        match key {
            "preset" => {
                if !crate::hdri::PRESETS.iter().any(|p| p.0 == val) {
                    return Err(format!("unknown environment '{val}' ({})", crate::hdri::PRESETS.iter().map(|p| p.0).collect::<Vec<_>>().join(", ")));
                }
                self.preset = val.to_string();
                self.file = None;
            }
            "file" => self.file = if val.is_empty() { None } else { Some(val.to_string()) },
            "rotation" => self.rotation = parse_f32(val)?,
            "exposure" => self.exposure = parse_f32(val)?.clamp(-8.0, 8.0),
            "background" => self.background = Background::parse(val).ok_or_else(|| format!("unknown background '{val}' (environment, color, gradient)"))?,
            "bg_color" => self.bg_color = parse_color(val)?,
            "blur" => self.blur = parse_f32(val)?.clamp(0.0, 1.0),
            _ => return Err(format!("unknown environment setting '{key}'")),
        }
        Ok(())
    }
}

impl Post {
    pub fn to_text(&self, out: &mut String) {
        out.push_str(&format!(
            "post.tonemap {}\npost.bloom {}\npost.bloom_threshold {}\npost.bloom_radius {}\npost.vignette {}\npost.contrast {}\npost.saturation {}\npost.temperature {}\npost.tint {}\npost.ssao {}\npost.ssao_radius {}\npost.dof {}\npost.focus {}\n",
            self.tonemap.id(), self.bloom, self.bloom_threshold, self.bloom_radius, self.vignette, self.contrast, self.saturation, self.temperature, self.tint, self.ssao, self.ssao_radius, self.dof, self.focus
        ));
    }
    pub fn set(&mut self, key: &str, val: &str) -> Result<(), String> {
        match key {
            "preset" => {
                if !POST_PRESETS.iter().any(|p| p.0 == val) {
                    return Err(format!("unknown post-processing preset '{val}'"));
                }
                *self = post_preset(val);
            }
            "tonemap" => self.tonemap = ToneMap::parse(val).ok_or_else(|| format!("unknown tone mapping '{val}' (filmic, aces, hable, reinhard, linear)"))?,
            "bloom" => self.bloom = parse_f32(val)?.clamp(0.0, 4.0),
            "bloom_threshold" => self.bloom_threshold = parse_f32(val)?.clamp(0.0, 20.0),
            "bloom_radius" => self.bloom_radius = parse_f32(val)?.clamp(0.0, 1.0),
            "vignette" => self.vignette = parse_f32(val)?.clamp(0.0, 1.0),
            "contrast" => self.contrast = parse_f32(val)?.clamp(0.2, 3.0),
            "saturation" => self.saturation = parse_f32(val)?.clamp(0.0, 3.0),
            "temperature" => self.temperature = parse_f32(val)?.clamp(-1.0, 1.0),
            "tint" => self.tint = parse_f32(val)?.clamp(-1.0, 1.0),
            "ssao" => self.ssao = parse_f32(val)?.clamp(0.0, 2.0),
            "ssao_radius" => self.ssao_radius = parse_f32(val)?.clamp(0.02, 2.0),
            "dof" => self.dof = parse_f32(val)?.clamp(0.0, 4.0),
            "focus" => self.focus = parse_f32(val)?.clamp(0.0, 60.0),
            _ => return Err(format!("unknown post-processing setting '{key}'")),
        }
        Ok(())
    }
}
