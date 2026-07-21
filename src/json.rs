//! Minimal strict JSON subset — the ADR-0017 decision, hand-rolled.
//!
//! Parses exactly what kenny consumes (safetensors headers, kenny manifests):
//! objects, arrays, strings (full escape handling incl. surrogate pairs),
//! unsigned integers, `true`/`false`/`null`. Floats, negative numbers,
//! duplicate keys, and trailing data are rejected with positioned errors — a
//! loud failure beats silent widening of the subset.
//!
//! The canonical writer emits sorted keys, no whitespace, minimal escapes
//! (`\u00xx` lowercase for bare controls), decimal integers: same `Value` in,
//! same bytes out, always. The manifest identity is blake3 over these bytes
//! (ADR-0005), which makes this writer consensus surface — locked by golden
//! tests in `tests/roundtrip.rs`.

use crate::error::{Error, Result};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Value {
    Null,
    Bool(bool),
    Int(u64),
    Str(String),
    Arr(Vec<Value>),
    /// Parse order preserved; the canonical writer sorts keys bytewise.
    Obj(Vec<(String, Value)>),
}

impl Value {
    pub fn get(&self, key: &str) -> Option<&Value> {
        match self {
            Value::Obj(o) => o.iter().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }

    pub fn as_u64(&self) -> Option<u64> {
        match self {
            Value::Int(n) => Some(*n),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Value::Str(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_arr(&self) -> Option<&[Value]> {
        match self {
            Value::Arr(a) => Some(a),
            _ => None,
        }
    }

    pub fn as_obj(&self) -> Option<&[(String, Value)]> {
        match self {
            Value::Obj(o) => Some(o),
            _ => None,
        }
    }
}

const MAX_DEPTH: usize = 64;

pub fn parse(bytes: &[u8]) -> Result<Value> {
    let text = std::str::from_utf8(bytes)
        .map_err(|e| Error::parse(format!("json: invalid utf-8: {e}")))?;
    let mut p = Parser {
        b: text.as_bytes(),
        pos: 0,
    };
    p.skip_ws();
    let v = p.value(0)?;
    p.skip_ws();
    if p.pos != p.b.len() {
        return Err(p.err("trailing data after top-level value"));
    }
    Ok(v)
}

struct Parser<'a> {
    b: &'a [u8],
    pos: usize,
}

impl Parser<'_> {
    fn err(&self, msg: &str) -> Error {
        Error::parse(format!("json at byte {}: {}", self.pos, msg))
    }

    fn skip_ws(&mut self) {
        while self.pos < self.b.len() && matches!(self.b[self.pos], b' ' | b'\t' | b'\n' | b'\r') {
            self.pos += 1;
        }
    }

    fn peek(&self) -> Option<u8> {
        self.b.get(self.pos).copied()
    }

    fn expect(&mut self, c: u8) -> Result<()> {
        if self.peek() == Some(c) {
            self.pos += 1;
            Ok(())
        } else {
            Err(self.err(&format!("expected '{}'", c as char)))
        }
    }

    fn value(&mut self, depth: usize) -> Result<Value> {
        if depth >= MAX_DEPTH {
            return Err(self.err("nesting too deep"));
        }
        match self.peek() {
            Some(b'{') => self.object(depth),
            Some(b'[') => self.array(depth),
            Some(b'"') => Ok(Value::Str(self.string()?)),
            Some(b't') => self.literal("true", Value::Bool(true)),
            Some(b'f') => self.literal("false", Value::Bool(false)),
            Some(b'n') => self.literal("null", Value::Null),
            Some(b'0'..=b'9') => self.number(),
            Some(b'-') => Err(self.err("negative numbers are not in the kenny json subset")),
            Some(c) => Err(self.err(&format!("unexpected byte {:#04x}", c))),
            None => Err(self.err("unexpected end of input")),
        }
    }

    fn literal(&mut self, lit: &str, v: Value) -> Result<Value> {
        if self.b[self.pos..].starts_with(lit.as_bytes()) {
            self.pos += lit.len();
            Ok(v)
        } else {
            Err(self.err(&format!("expected '{lit}'")))
        }
    }

    fn number(&mut self) -> Result<Value> {
        let start = self.pos;
        while matches!(self.peek(), Some(b'0'..=b'9')) {
            self.pos += 1;
        }
        let digits = &self.b[start..self.pos];
        if digits.len() > 1 && digits[0] == b'0' {
            return Err(self.err("leading zero in number"));
        }
        if matches!(self.peek(), Some(b'.') | Some(b'e') | Some(b'E')) {
            return Err(self.err("floats are not in the kenny json subset"));
        }
        let mut n: u64 = 0;
        for &d in digits {
            n = n
                .checked_mul(10)
                .and_then(|n| n.checked_add((d - b'0') as u64))
                .ok_or_else(|| self.err("integer overflows u64"))?;
        }
        Ok(Value::Int(n))
    }

    fn string(&mut self) -> Result<String> {
        self.expect(b'"')?;
        let mut out: Vec<u8> = Vec::new();
        loop {
            let c = self.peek().ok_or_else(|| self.err("unterminated string"))?;
            self.pos += 1;
            match c {
                b'"' => break,
                b'\\' => {
                    let e = self.peek().ok_or_else(|| self.err("unterminated escape"))?;
                    self.pos += 1;
                    match e {
                        b'"' => out.push(b'"'),
                        b'\\' => out.push(b'\\'),
                        b'/' => out.push(b'/'),
                        b'b' => out.push(0x08),
                        b'f' => out.push(0x0C),
                        b'n' => out.push(b'\n'),
                        b'r' => out.push(b'\r'),
                        b't' => out.push(b'\t'),
                        b'u' => {
                            let hi = self.hex4()?;
                            let ch = if (0xD800..=0xDBFF).contains(&hi) {
                                // High surrogate: a low surrogate must follow.
                                if self.peek() != Some(b'\\') {
                                    return Err(self.err("lone high surrogate"));
                                }
                                self.pos += 1;
                                if self.peek() != Some(b'u') {
                                    return Err(self.err("lone high surrogate"));
                                }
                                self.pos += 1;
                                let lo = self.hex4()?;
                                if !(0xDC00..=0xDFFF).contains(&lo) {
                                    return Err(self.err("invalid low surrogate"));
                                }
                                let code = 0x10000 + ((hi - 0xD800) << 10) + (lo - 0xDC00);
                                char::from_u32(code)
                                    .ok_or_else(|| self.err("invalid surrogate pair"))?
                            } else if (0xDC00..=0xDFFF).contains(&hi) {
                                return Err(self.err("lone low surrogate"));
                            } else {
                                char::from_u32(hi).ok_or_else(|| self.err("invalid \\u escape"))?
                            };
                            let mut buf = [0u8; 4];
                            out.extend_from_slice(ch.encode_utf8(&mut buf).as_bytes());
                        }
                        _ => return Err(self.err("invalid escape character")),
                    }
                }
                c if c < 0x20 => return Err(self.err("unescaped control character in string")),
                c => out.push(c),
            }
        }
        // Input was validated UTF-8 and escapes emit valid UTF-8, so this
        // cannot fail; the error path is kept for defense in depth.
        String::from_utf8(out).map_err(|e| Error::parse(format!("json string: {e}")))
    }

    fn hex4(&mut self) -> Result<u32> {
        if self.pos + 4 > self.b.len() {
            return Err(self.err("truncated \\u escape"));
        }
        let mut v: u32 = 0;
        for _ in 0..4 {
            let c = self.b[self.pos];
            self.pos += 1;
            let d = match c {
                b'0'..=b'9' => c - b'0',
                b'a'..=b'f' => c - b'a' + 10,
                b'A'..=b'F' => c - b'A' + 10,
                _ => return Err(self.err("invalid hex digit in \\u escape")),
            };
            v = (v << 4) | d as u32;
        }
        Ok(v)
    }

    fn object(&mut self, depth: usize) -> Result<Value> {
        self.expect(b'{')?;
        let mut entries: Vec<(String, Value)> = Vec::new();
        self.skip_ws();
        if self.peek() == Some(b'}') {
            self.pos += 1;
            return Ok(Value::Obj(entries));
        }
        loop {
            self.skip_ws();
            let key = self.string()?;
            self.skip_ws();
            self.expect(b':')?;
            self.skip_ws();
            let v = self.value(depth + 1)?;
            entries.push((key, v));
            self.skip_ws();
            match self.peek() {
                Some(b',') => self.pos += 1,
                Some(b'}') => {
                    self.pos += 1;
                    break;
                }
                _ => return Err(self.err("expected ',' or '}' in object")),
            }
        }
        // Duplicate keys would make canonical encoding ambiguous — reject.
        let mut idx: Vec<usize> = (0..entries.len()).collect();
        idx.sort_by(|&a, &b| entries[a].0.cmp(&entries[b].0));
        for w in idx.windows(2) {
            if entries[w[0]].0 == entries[w[1]].0 {
                return Err(Error::parse(format!(
                    "json: duplicate object key {:?}",
                    entries[w[0]].0
                )));
            }
        }
        Ok(Value::Obj(entries))
    }

    fn array(&mut self, depth: usize) -> Result<Value> {
        self.expect(b'[')?;
        let mut items = Vec::new();
        self.skip_ws();
        if self.peek() == Some(b']') {
            self.pos += 1;
            return Ok(Value::Arr(items));
        }
        loop {
            self.skip_ws();
            items.push(self.value(depth + 1)?);
            self.skip_ws();
            match self.peek() {
                Some(b',') => self.pos += 1,
                Some(b']') => {
                    self.pos += 1;
                    break;
                }
                _ => return Err(self.err("expected ',' or ']' in array")),
            }
        }
        Ok(Value::Arr(items))
    }
}

/// Canonical bytes: sorted keys, no whitespace, minimal escapes. Consensus
/// surface — see module docs.
pub fn to_canonical(v: &Value) -> Vec<u8> {
    let mut out = Vec::new();
    write_value(v, &mut out);
    out
}

fn write_value(v: &Value, out: &mut Vec<u8>) {
    match v {
        Value::Null => out.extend_from_slice(b"null"),
        Value::Bool(true) => out.extend_from_slice(b"true"),
        Value::Bool(false) => out.extend_from_slice(b"false"),
        Value::Int(n) => out.extend_from_slice(n.to_string().as_bytes()),
        Value::Str(s) => write_str(s, out),
        Value::Arr(items) => {
            out.push(b'[');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push(b',');
                }
                write_value(item, out);
            }
            out.push(b']');
        }
        Value::Obj(entries) => {
            let mut idx: Vec<usize> = (0..entries.len()).collect();
            idx.sort_by(|&a, &b| entries[a].0.cmp(&entries[b].0));
            debug_assert!(
                idx.windows(2).all(|w| entries[w[0]].0 != entries[w[1]].0),
                "duplicate keys reach the canonical writer"
            );
            out.push(b'{');
            for (i, &k) in idx.iter().enumerate() {
                if i > 0 {
                    out.push(b',');
                }
                write_str(&entries[k].0, out);
                out.push(b':');
                write_value(&entries[k].1, out);
            }
            out.push(b'}');
        }
    }
}

fn write_str(s: &str, out: &mut Vec<u8>) {
    out.push(b'"');
    for ch in s.chars() {
        match ch {
            '"' => out.extend_from_slice(b"\\\""),
            '\\' => out.extend_from_slice(b"\\\\"),
            '\n' => out.extend_from_slice(b"\\n"),
            '\r' => out.extend_from_slice(b"\\r"),
            '\t' => out.extend_from_slice(b"\\t"),
            '\u{8}' => out.extend_from_slice(b"\\b"),
            '\u{c}' => out.extend_from_slice(b"\\f"),
            c if (c as u32) < 0x20 => {
                out.extend_from_slice(format!("\\u{:04x}", c as u32).as_bytes());
            }
            c => {
                let mut buf = [0u8; 4];
                out.extend_from_slice(c.encode_utf8(&mut buf).as_bytes());
            }
        }
    }
    out.push(b'"');
}

#[cfg(test)]
mod tests {
    use super::*;

    fn p(s: &str) -> Result<Value> {
        parse(s.as_bytes())
    }

    #[test]
    fn parses_safetensors_style_header() {
        let v = p(r#"{"__metadata__":{"format":"pt"},"t":{"dtype":"BF16","shape":[2,4],"data_offsets":[0,16]}}"#).unwrap();
        let t = v.get("t").unwrap();
        assert_eq!(t.get("dtype").unwrap().as_str(), Some("BF16"));
        let shape: Vec<u64> = t
            .get("shape")
            .unwrap()
            .as_arr()
            .unwrap()
            .iter()
            .map(|x| x.as_u64().unwrap())
            .collect();
        assert_eq!(shape, [2, 4]);
    }

    #[test]
    fn string_escapes() {
        let v = p(r#""a\"b\\c\/\n\tAé😀""#).unwrap();
        assert_eq!(v.as_str(), Some("a\"b\\c/\n\tAé😀"));
    }

    #[test]
    fn rejects_out_of_subset() {
        assert!(p("1.5").is_err(), "float");
        assert!(p("1e3").is_err(), "exponent");
        assert!(p("-1").is_err(), "negative");
        assert!(p("01").is_err(), "leading zero");
        assert!(p("18446744073709551616").is_err(), "u64 overflow");
        assert!(p(r#"{"a":1,"a":2}"#).is_err(), "duplicate key");
        assert!(p("{} x").is_err(), "trailing data");
        assert!(p(r#""\ud800x""#).is_err(), "lone surrogate");
        assert!(p("\"a\nb\"").is_err(), "raw control char");
        let deep = "[".repeat(100) + &"]".repeat(100);
        assert!(p(&deep).is_err(), "depth cap");
    }

    #[test]
    fn boundary_values() {
        assert_eq!(p("0").unwrap(), Value::Int(0));
        assert_eq!(
            p("18446744073709551615").unwrap(),
            Value::Int(u64::MAX),
            "u64::MAX parses"
        );
        assert_eq!(p("[]").unwrap(), Value::Arr(vec![]));
        assert_eq!(p("{}").unwrap(), Value::Obj(vec![]));
        assert_eq!(p(" true ").unwrap(), Value::Bool(true));
    }

    #[test]
    fn canonical_sorts_and_is_stable() {
        let v = p(r#"{ "b" : 2 , "a" : { "z" : [1, 2] , "y" : "s" } }"#).unwrap();
        let bytes = to_canonical(&v);
        assert_eq!(bytes, br#"{"a":{"y":"s","z":[1,2]},"b":2}"#);
        // Re-parsing the canonical form and re-writing it is a fixed point.
        assert_eq!(to_canonical(&parse(&bytes).unwrap()), bytes);
    }

    #[test]
    fn canonical_escapes() {
        let v = Value::Str("q\"\\\n\u{1}é".to_string());
        assert_eq!(to_canonical(&v), "\"q\\\"\\\\\\n\\u0001é\"".as_bytes());
    }
}
