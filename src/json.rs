// SPDX-License-Identifier: MIT OR Apache-2.0
//! A small, dependency-free JSON reader and writer for the command interface (MCP server, control channel, CLI).
//!
//! Everything that arrives here comes from another program, so the parser never panics: it returns an error for
//! malformed text, nesting deeper than [`MAX_DEPTH`], and numbers it cannot represent. Object keys keep their order.

use std::fmt::Write as _;

/// Deepest nesting the parser accepts.
pub const MAX_DEPTH: usize = 64;

#[derive(Debug, Clone, PartialEq, Default)]
pub enum Json {
    #[default]
    Null,
    Bool(bool),
    Num(f64),
    Str(String),
    Arr(Vec<Json>),
    Obj(Vec<(String, Json)>),
}

impl From<&str> for Json {
    fn from(s: &str) -> Json {
        Json::Str(s.to_string())
    }
}
impl From<String> for Json {
    fn from(s: String) -> Json {
        Json::Str(s)
    }
}
impl From<bool> for Json {
    fn from(b: bool) -> Json {
        Json::Bool(b)
    }
}
impl From<f64> for Json {
    fn from(n: f64) -> Json {
        Json::Num(n)
    }
}
impl From<f32> for Json {
    fn from(n: f32) -> Json {
        // go through the shortest decimal text so 0.1f32 prints as 0.1, not 0.10000000149011612
        Json::Num(n.to_string().parse::<f64>().unwrap_or(n as f64))
    }
}
impl From<u32> for Json {
    fn from(n: u32) -> Json {
        Json::Num(n as f64)
    }
}
impl From<i32> for Json {
    fn from(n: i32) -> Json {
        Json::Num(n as f64)
    }
}
impl From<usize> for Json {
    fn from(n: usize) -> Json {
        Json::Num(n as f64)
    }
}
impl<T: Into<Json>> From<Vec<T>> for Json {
    fn from(v: Vec<T>) -> Json {
        Json::Arr(v.into_iter().map(Into::into).collect())
    }
}
impl<T: Into<Json>> From<Option<T>> for Json {
    fn from(v: Option<T>) -> Json {
        v.map(Into::into).unwrap_or(Json::Null)
    }
}

impl Json {
    /// An empty object.
    pub fn obj() -> Json {
        Json::Obj(Vec::new())
    }

    /// Builder: add (or replace) a key of an object.
    pub fn with(mut self, key: &str, v: impl Into<Json>) -> Json {
        self.set(key, v);
        self
    }

    pub fn set(&mut self, key: &str, v: impl Into<Json>) {
        if let Json::Obj(o) = self {
            let v = v.into();
            if let Some(slot) = o.iter_mut().find(|(k, _)| k == key) {
                slot.1 = v;
            } else {
                o.push((key.to_string(), v));
            }
        }
    }

    pub fn get(&self, key: &str) -> Option<&Json> {
        match self {
            Json::Obj(o) => o.iter().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Json::Str(s) => Some(s),
            _ => None,
        }
    }

    /// A finite number.
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Json::Num(n) if n.is_finite() => Some(*n),
            _ => None,
        }
    }

    pub fn as_f32(&self) -> Option<f32> {
        self.as_f64().map(|n| n as f32).filter(|n| n.is_finite())
    }

    /// A whole, non-negative number that fits a `u32`.
    pub fn as_u32(&self) -> Option<u32> {
        let n = self.as_f64()?;
        (n >= 0.0 && n <= u32::MAX as f64 && n.fract() == 0.0).then_some(n as u32)
    }

    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Json::Bool(b) => Some(*b),
            _ => None,
        }
    }

    pub fn as_arr(&self) -> Option<&[Json]> {
        match self {
            Json::Arr(a) => Some(a),
            _ => None,
        }
    }

    pub fn as_obj(&self) -> Option<&[(String, Json)]> {
        match self {
            Json::Obj(o) => Some(o),
            _ => None,
        }
    }

    pub fn is_null(&self) -> bool {
        matches!(self, Json::Null)
    }

    /// Compact text on one line (what the control channel sends).
    pub fn to_text(&self) -> String {
        let mut s = String::new();
        write_value(&mut s, self, None, 0);
        s
    }

    /// Indented text for people.
    pub fn to_pretty(&self) -> String {
        let mut s = String::new();
        write_value(&mut s, self, Some(2), 0);
        s
    }

    pub fn parse(text: &str) -> Result<Json, String> {
        let mut p = Parser { b: text.as_bytes(), i: 0 };
        p.ws();
        let v = p.value(0)?;
        p.ws();
        if p.i != p.b.len() {
            return Err(p.err("unexpected text after the value"));
        }
        Ok(v)
    }
}

fn write_str(out: &mut String, s: &str) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

fn write_num(out: &mut String, n: f64) {
    if !n.is_finite() {
        out.push_str("null");
    } else if n.fract() == 0.0 && n.abs() < 1e15 {
        let _ = write!(out, "{}", n as i64);
    } else {
        let _ = write!(out, "{n}");
    }
}

fn write_value(out: &mut String, v: &Json, indent: Option<usize>, level: usize) {
    let nl = |out: &mut String, level: usize| {
        if let Some(step) = indent {
            out.push('\n');
            for _ in 0..level * step {
                out.push(' ');
            }
        }
    };
    match v {
        Json::Null => out.push_str("null"),
        Json::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        Json::Num(n) => write_num(out, *n),
        Json::Str(s) => write_str(out, s),
        Json::Arr(a) => {
            if a.is_empty() {
                out.push_str("[]");
                return;
            }
            // short arrays of numbers stay on one line, which keeps lists of ids readable
            let flat = indent.is_some() && a.iter().all(|x| matches!(x, Json::Num(_) | Json::Bool(_) | Json::Null)) && a.len() <= 16;
            out.push('[');
            for (i, x) in a.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                    if flat {
                        out.push(' ');
                    }
                }
                if !flat {
                    nl(out, level + 1);
                }
                write_value(out, x, indent, level + 1);
            }
            if !flat {
                nl(out, level);
            }
            out.push(']');
        }
        Json::Obj(o) => {
            if o.is_empty() {
                out.push_str("{}");
                return;
            }
            out.push('{');
            for (i, (k, x)) in o.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                nl(out, level + 1);
                write_str(out, k);
                out.push(':');
                if indent.is_some() {
                    out.push(' ');
                }
                write_value(out, x, indent, level + 1);
            }
            nl(out, level);
            out.push('}');
        }
    }
}

struct Parser<'a> {
    b: &'a [u8],
    i: usize,
}

impl Parser<'_> {
    fn err(&self, m: &str) -> String {
        format!("invalid JSON at byte {}: {m}", self.i)
    }

    fn ws(&mut self) {
        while matches!(self.b.get(self.i), Some(b' ' | b'\t' | b'\n' | b'\r')) {
            self.i += 1;
        }
    }

    fn eat(&mut self, lit: &str) -> bool {
        if self.b.get(self.i..self.i + lit.len()) == Some(lit.as_bytes()) {
            self.i += lit.len();
            true
        } else {
            false
        }
    }

    fn value(&mut self, depth: usize) -> Result<Json, String> {
        if depth > MAX_DEPTH {
            return Err(self.err("nested too deeply"));
        }
        match self.b.get(self.i) {
            None => Err(self.err("unexpected end of text")),
            Some(b'{') => {
                self.i += 1;
                let mut o: Vec<(String, Json)> = Vec::new();
                self.ws();
                if self.b.get(self.i) == Some(&b'}') {
                    self.i += 1;
                    return Ok(Json::Obj(o));
                }
                loop {
                    self.ws();
                    if self.b.get(self.i) != Some(&b'"') {
                        return Err(self.err("expected a string key"));
                    }
                    let k = self.string()?;
                    self.ws();
                    if self.b.get(self.i) != Some(&b':') {
                        return Err(self.err("expected ':'"));
                    }
                    self.i += 1;
                    self.ws();
                    let v = self.value(depth + 1)?;
                    // a repeated key: the last one wins, as in most parsers
                    if let Some(slot) = o.iter_mut().find(|(kk, _)| *kk == k) {
                        slot.1 = v;
                    } else {
                        o.push((k, v));
                    }
                    self.ws();
                    match self.b.get(self.i) {
                        Some(b',') => self.i += 1,
                        Some(b'}') => {
                            self.i += 1;
                            return Ok(Json::Obj(o));
                        }
                        _ => return Err(self.err("expected ',' or '}'")),
                    }
                }
            }
            Some(b'[') => {
                self.i += 1;
                let mut a = Vec::new();
                self.ws();
                if self.b.get(self.i) == Some(&b']') {
                    self.i += 1;
                    return Ok(Json::Arr(a));
                }
                loop {
                    self.ws();
                    a.push(self.value(depth + 1)?);
                    self.ws();
                    match self.b.get(self.i) {
                        Some(b',') => self.i += 1,
                        Some(b']') => {
                            self.i += 1;
                            return Ok(Json::Arr(a));
                        }
                        _ => return Err(self.err("expected ',' or ']'")),
                    }
                }
            }
            Some(b'"') => Ok(Json::Str(self.string()?)),
            Some(b't') if self.eat("true") => Ok(Json::Bool(true)),
            Some(b'f') if self.eat("false") => Ok(Json::Bool(false)),
            Some(b'n') if self.eat("null") => Ok(Json::Null),
            Some(c) if *c == b'-' || c.is_ascii_digit() => self.number(),
            Some(_) => Err(self.err("unexpected character")),
        }
    }

    fn number(&mut self) -> Result<Json, String> {
        let start = self.i;
        if self.b.get(self.i) == Some(&b'-') {
            self.i += 1;
        }
        let digits = |p: &mut Self| {
            let s = p.i;
            while matches!(p.b.get(p.i), Some(b'0'..=b'9')) {
                p.i += 1;
            }
            p.i > s
        };
        if !digits(self) {
            return Err(self.err("expected digits"));
        }
        if self.b.get(self.i) == Some(&b'.') {
            self.i += 1;
            if !digits(self) {
                return Err(self.err("expected digits after '.'"));
            }
        }
        if matches!(self.b.get(self.i), Some(b'e' | b'E')) {
            self.i += 1;
            if matches!(self.b.get(self.i), Some(b'+' | b'-')) {
                self.i += 1;
            }
            if !digits(self) {
                return Err(self.err("expected digits in the exponent"));
            }
        }
        let text = std::str::from_utf8(&self.b[start..self.i]).map_err(|_| self.err("bad number"))?;
        let n: f64 = text.parse().map_err(|_| self.err("bad number"))?;
        if !n.is_finite() {
            return Err(self.err("number out of range"));
        }
        Ok(Json::Num(n))
    }

    fn hex4(&mut self) -> Result<u32, String> {
        let s = self.b.get(self.i..self.i + 4).ok_or_else(|| self.err("short \\u escape"))?;
        let mut v = 0u32;
        for &c in s {
            let d = (c as char).to_digit(16).ok_or_else(|| self.err("bad \\u escape"))?;
            v = v * 16 + d;
        }
        self.i += 4;
        Ok(v)
    }

    fn string(&mut self) -> Result<String, String> {
        self.i += 1; // opening quote
        let mut out: Vec<u8> = Vec::new();
        loop {
            let c = *self.b.get(self.i).ok_or_else(|| self.err("unterminated string"))?;
            self.i += 1;
            match c {
                b'"' => break,
                b'\\' => {
                    let e = *self.b.get(self.i).ok_or_else(|| self.err("unterminated escape"))?;
                    self.i += 1;
                    let ch = match e {
                        b'"' => '"',
                        b'\\' => '\\',
                        b'/' => '/',
                        b'b' => '\u{8}',
                        b'f' => '\u{c}',
                        b'n' => '\n',
                        b'r' => '\r',
                        b't' => '\t',
                        b'u' => {
                            let hi = self.hex4()?;
                            let cp = if (0xD800..0xDC00).contains(&hi) {
                                if !self.eat("\\u") {
                                    return Err(self.err("lone surrogate"));
                                }
                                let lo = self.hex4()?;
                                if !(0xDC00..0xE000).contains(&lo) {
                                    return Err(self.err("bad surrogate pair"));
                                }
                                0x10000 + ((hi - 0xD800) << 10) + (lo - 0xDC00)
                            } else if (0xDC00..0xE000).contains(&hi) {
                                return Err(self.err("lone surrogate"));
                            } else {
                                hi
                            };
                            char::from_u32(cp).ok_or_else(|| self.err("bad code point"))?
                        }
                        _ => return Err(self.err("bad escape")),
                    };
                    let mut buf = [0u8; 4];
                    out.extend_from_slice(ch.encode_utf8(&mut buf).as_bytes());
                }
                c if c < 0x20 => return Err(self.err("control character in a string")),
                c => out.push(c),
            }
        }
        String::from_utf8(out).map_err(|_| self.err("invalid UTF-8 in a string"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip() {
        let t = r#"{"a":1,"b":[true,null,"x\n\u00e9\ud83d\ude00"],"c":{"d":-2.5e1}}"#;
        let v = Json::parse(t).unwrap();
        assert_eq!(v.get("a").and_then(Json::as_u32), Some(1));
        assert_eq!(v.get("c").and_then(|c| c.get("d")).and_then(Json::as_f64), Some(-25.0));
        assert_eq!(Json::parse(&v.to_text()).unwrap(), v);
        assert_eq!(Json::parse(&v.to_pretty()).unwrap(), v);
    }

    #[test]
    fn rejects_bad_text_without_panicking() {
        for t in ["", "{", "[1,", "{\"a\"}", "\"\\ud800\"", "nul", "1e999", "[1 2]", "{\"a\":1}x", "\"\u{1}\"", "--1", "01.", "{1:2}"] {
            assert!(Json::parse(t).is_err(), "{t:?} should fail");
        }
        let deep = "[".repeat(10_000);
        assert!(Json::parse(&deep).is_err());
    }

    #[test]
    fn floats_print_short() {
        assert_eq!(Json::from(0.1f32).to_text(), "0.1");
        assert_eq!(Json::from(3.0f64).to_text(), "3");
        assert_eq!(Json::Num(f64::NAN).to_text(), "null");
    }
}
