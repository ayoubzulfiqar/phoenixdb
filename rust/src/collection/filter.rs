//! Metadata filters: a MongoDB-style query language over document metadata.
//!
//! ```json
//! {
//!   "category": "news",
//!   "year": { "$gte": 2020, "$lt": 2025 },
//!   "tags": { "$in": ["ai", "rust"] },
//!   "$or": [ { "lang": "en" }, { "lang": "fr" } ],
//!   "draft": { "$exists": false }
//! }
//! ```
//!
//! * A bare value means equality; several keys in one object are ANDed.
//! * Operators: `$eq $ne $gt $gte $lt $lte $in $nin $exists`, and the logical
//!   `$and $or $not`.
//! * Nested objects are addressed with dot paths (`"author.name"`).
//! * An array field matches when **any** element matches, as in MongoDB:
//!   `{"tags": "ai"}` matches `{"tags": ["ai", "ml"]}`.
//! * `$ne` / `$nin` also match documents that lack the field.
//! * Ranges compare numbers with numbers and strings with strings; a range
//!   never matches a value of another type.
//! * Strings longer than [`MAX_INDEXED_STRING`] bytes are stored and returned
//!   but not indexed, so comparisons never match them (`$exists` still sees
//!   the field), and filter strings are limited to the same length.
//! * Metadata keys may not contain `.` (it separates path segments) or NUL.

use crate::error::{Error, Result};
use serde_json::Value;
use std::cmp::Ordering;
use std::collections::BTreeSet;

/// Longest string value that is indexed (and that a filter may compare
/// against), in bytes.
pub const MAX_INDEXED_STRING: usize = 512;

/// A scalar a filter compares against.
#[derive(Debug, Clone, PartialEq)]
pub enum Scalar {
    /// JSON `null`.
    Null,
    /// A boolean.
    Bool(bool),
    /// Any JSON number, as `f64`.
    Number(f64),
    /// A string.
    Str(String),
}

impl Scalar {
    fn from_json(v: &Value, context: &str) -> Result<Self> {
        Ok(match v {
            Value::Null => Scalar::Null,
            Value::Bool(b) => Scalar::Bool(*b),
            Value::Number(n) => Scalar::Number(
                n.as_f64()
                    .filter(|f| f.is_finite())
                    .ok_or_else(|| Error::invalid(format!("{context}: number out of range")))?,
            ),
            Value::String(s) => Scalar::Str(s.clone()),
            _ => {
                return Err(Error::invalid(format!(
                    "{context}: expected a scalar (null, boolean, number or string)"
                )));
            }
        })
    }

    /// A scalar used as a filter operand.
    fn operand(v: &Value, field: &str) -> Result<Self> {
        let s = Self::from_json(v, field)?;
        if !s.indexable() {
            return Err(Error::invalid(format!(
                "`{field}`: filter strings are limited to {MAX_INDEXED_STRING} bytes"
            )));
        }
        Ok(s)
    }

    /// Compares two scalars of the same type; `None` across types.
    fn compare(&self, other: &Scalar) -> Option<Ordering> {
        match (self, other) {
            (Scalar::Number(a), Scalar::Number(b)) => a.partial_cmp(b),
            (Scalar::Str(a), Scalar::Str(b)) => Some(a.as_str().cmp(b.as_str())),
            (Scalar::Bool(a), Scalar::Bool(b)) => Some(a.cmp(b)),
            (Scalar::Null, Scalar::Null) => Some(Ordering::Equal),
            _ => None,
        }
    }

    /// Order-preserving, self-delimiting byte encoding for index keys.
    ///
    /// Each encoding carries a type tag and ends unambiguously (numbers are
    /// fixed width; strings escape `0x00` and end with `0x00 0x01`), so the
    /// id appended after it can never be mistaken for part of the value, and
    /// byte order equals value order within a type.
    #[must_use]
    pub fn encode(&self) -> Vec<u8> {
        match self {
            Scalar::Null => vec![0x01],
            Scalar::Bool(b) => vec![0x02, u8::from(*b)],
            Scalar::Number(f) => {
                let f = if *f == 0.0 { 0.0 } else { *f }; // -0.0 == 0.0
                let bits = f.to_bits();
                let ordered = if bits >> 63 == 1 {
                    !bits
                } else {
                    bits ^ (1 << 63)
                };
                let mut out = vec![0x03];
                out.extend_from_slice(&ordered.to_be_bytes());
                out
            }
            Scalar::Str(s) => {
                let mut out = Vec::with_capacity(s.len() + 3);
                out.push(0x04);
                for &b in s.as_bytes() {
                    out.push(b);
                    if b == 0 {
                        out.push(0xFF); // escape
                    }
                }
                out.extend_from_slice(&[0x00, 0x01]);
                out
            }
        }
    }

    /// Tag byte shared by every encoding of this scalar's type.
    #[must_use]
    pub fn type_tag(&self) -> u8 {
        match self {
            Scalar::Null => 0x01,
            Scalar::Bool(_) => 0x02,
            Scalar::Number(_) => 0x03,
            Scalar::Str(_) => 0x04,
        }
    }

    /// Whether this value can be indexed (long strings are not).
    #[must_use]
    pub fn indexable(&self) -> bool {
        !matches!(self, Scalar::Str(s) if s.len() > MAX_INDEXED_STRING)
    }
}

/// A comparison operator on one field.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RangeOp {
    /// `$gt`
    Gt,
    /// `$gte`
    Gte,
    /// `$lt`
    Lt,
    /// `$lte`
    Lte,
}

/// A parsed filter.
#[derive(Debug, Clone, PartialEq)]
pub enum Filter {
    /// Matches everything.
    All,
    /// `field == value` (any element, for arrays).
    Eq(String, Scalar),
    /// `field != value`, including documents without the field.
    Ne(String, Scalar),
    /// `field op value` for numbers or strings.
    Range(String, RangeOp, Scalar),
    /// `field` equals one of the values.
    In(String, Vec<Scalar>),
    /// `field` equals none of the values (or is missing).
    Nin(String, Vec<Scalar>),
    /// `field` is present (`true`) or absent (`false`).
    Exists(String, bool),
    /// Every sub-filter matches.
    And(Vec<Filter>),
    /// Some sub-filter matches.
    Or(Vec<Filter>),
    /// The sub-filter does not match.
    Not(Box<Filter>),
}

impl Filter {
    /// Parses a filter document (see the module docs).
    pub fn parse(value: &Value) -> Result<Self> {
        Self::parse_at(value, 0)
    }

    /// Parses a filter from JSON text; an empty or `{}` document matches all.
    pub fn parse_str(text: &str) -> Result<Self> {
        if text.trim().is_empty() {
            return Ok(Filter::All);
        }
        let value: Value = serde_json::from_str(text)
            .map_err(|e| Error::invalid(format!("filter is not valid JSON: {e}")))?;
        Self::parse(&value)
    }

    fn parse_at(value: &Value, depth: usize) -> Result<Self> {
        if depth > 64 {
            return Err(Error::invalid("filter is nested too deeply"));
        }
        let Value::Object(map) = value else {
            return Err(Error::invalid("a filter must be a JSON object"));
        };
        let mut clauses = Vec::with_capacity(map.len());
        for (key, v) in map {
            clauses.push(match key.as_str() {
                "$and" | "$or" => {
                    let Value::Array(items) = v else {
                        return Err(Error::invalid(format!("`{key}` takes an array of filters")));
                    };
                    if items.is_empty() {
                        return Err(Error::invalid(format!("`{key}` needs at least one filter")));
                    }
                    let parts = items
                        .iter()
                        .map(|f| Self::parse_at(f, depth + 1))
                        .collect::<Result<Vec<_>>>()?;
                    if key == "$and" {
                        Filter::And(parts)
                    } else {
                        Filter::Or(parts)
                    }
                }
                "$not" => Filter::Not(Box::new(Self::parse_at(v, depth + 1)?)),
                op if op.starts_with('$') => {
                    return Err(Error::invalid(format!("unknown top-level operator `{op}`")));
                }
                field => Self::parse_field(field, v)?,
            });
        }
        Ok(match clauses.len() {
            0 => Filter::All,
            1 => clauses.pop().expect("one clause"),
            _ => Filter::And(clauses),
        })
    }

    fn parse_field(field: &str, v: &Value) -> Result<Self> {
        validate_field(field)?;
        let ops = match v {
            Value::Object(ops) if ops.keys().all(|k| k.starts_with('$')) && !ops.is_empty() => ops,
            Value::Object(_) => {
                return Err(Error::invalid(format!(
                    "field `{field}`: match nested values with a dot path (\"{field}.child\"), \
                     not an object literal"
                )));
            }
            Value::Array(_) => {
                return Err(Error::invalid(format!(
                    "field `{field}`: use {{\"$in\": [...]}} to match one of several values"
                )));
            }
            scalar => {
                return Ok(Filter::Eq(
                    field.to_string(),
                    Scalar::operand(scalar, field)?,
                ));
            }
        };
        let list = |v: &Value, op: &str| -> Result<Vec<Scalar>> {
            let Value::Array(items) = v else {
                return Err(Error::invalid(format!(
                    "`{op}` on `{field}` takes an array"
                )));
            };
            items.iter().map(|i| Scalar::operand(i, field)).collect()
        };
        let mut parts = Vec::with_capacity(ops.len());
        for (op, arg) in ops {
            parts.push(match op.as_str() {
                "$eq" => Filter::Eq(field.to_string(), Scalar::operand(arg, field)?),
                "$ne" => Filter::Ne(field.to_string(), Scalar::operand(arg, field)?),
                "$gt" | "$gte" | "$lt" | "$lte" => {
                    let scalar = Scalar::operand(arg, field)?;
                    if !matches!(scalar, Scalar::Number(_) | Scalar::Str(_)) {
                        return Err(Error::invalid(format!(
                            "`{op}` on `{field}` compares numbers or strings"
                        )));
                    }
                    let range = match op.as_str() {
                        "$gt" => RangeOp::Gt,
                        "$gte" => RangeOp::Gte,
                        "$lt" => RangeOp::Lt,
                        _ => RangeOp::Lte,
                    };
                    Filter::Range(field.to_string(), range, scalar)
                }
                "$in" => Filter::In(field.to_string(), list(arg, "$in")?),
                "$nin" => Filter::Nin(field.to_string(), list(arg, "$nin")?),
                "$exists" => match arg {
                    Value::Bool(b) => Filter::Exists(field.to_string(), *b),
                    _ => {
                        return Err(Error::invalid(format!(
                            "`$exists` on `{field}` takes true or false"
                        )));
                    }
                },
                other => {
                    return Err(Error::invalid(format!(
                        "unknown operator `{other}` on `{field}`"
                    )));
                }
            });
        }
        Ok(if parts.len() == 1 {
            parts.pop().expect("one part")
        } else {
            Filter::And(parts)
        })
    }

    /// Evaluates the filter against a document's metadata, in memory.
    ///
    /// This is the reference semantics; the index-backed evaluation in the
    /// collection must agree with it.
    #[must_use]
    pub fn matches(&self, metadata: &Value) -> bool {
        match self {
            Filter::All => true,
            Filter::Eq(field, v) => values_at(metadata, field)
                .iter()
                .any(|x| x.compare(v) == Some(Ordering::Equal)),
            Filter::Ne(field, v) => !Filter::Eq(field.clone(), v.clone()).matches(metadata),
            Filter::Range(field, op, v) => values_at(metadata, field).iter().any(|x| {
                matches!(
                    (op, x.compare(v)),
                    (RangeOp::Gt, Some(Ordering::Greater))
                        | (RangeOp::Gte, Some(Ordering::Greater | Ordering::Equal))
                        | (RangeOp::Lt, Some(Ordering::Less))
                        | (RangeOp::Lte, Some(Ordering::Less | Ordering::Equal))
                )
            }),
            Filter::In(field, vs) => {
                let have = values_at(metadata, field);
                vs.iter()
                    .any(|v| have.iter().any(|x| x.compare(v) == Some(Ordering::Equal)))
            }
            Filter::Nin(field, vs) => !Filter::In(field.clone(), vs.clone()).matches(metadata),
            Filter::Exists(field, want) => field_present(metadata, field) == *want,
            Filter::And(fs) => fs.iter().all(|f| f.matches(metadata)),
            Filter::Or(fs) => fs.iter().any(|f| f.matches(metadata)),
            Filter::Not(f) => !f.matches(metadata),
        }
    }
}

/// Field names may not be empty or contain NUL (the index separator).
pub fn validate_field(field: &str) -> Result<()> {
    if field.is_empty() || field.len() > 256 || field.contains('\0') {
        return Err(Error::invalid(format!(
            "metadata field name {field:?} must be 1..=256 bytes without NUL"
        )));
    }
    Ok(())
}

/// Every scalar reachable at `path` (arrays contribute each element).
fn values_at(metadata: &Value, path: &str) -> Vec<Scalar> {
    let mut out = Vec::new();
    collect_at(metadata, &path.split('.').collect::<Vec<_>>(), &mut out);
    out
}

fn collect_at(value: &Value, path: &[&str], out: &mut Vec<Scalar>) {
    match value {
        Value::Array(items) => {
            for item in items {
                collect_at(item, path, out);
            }
        }
        Value::Object(map) if !path.is_empty() => {
            if let Some(child) = map.get(path[0]) {
                collect_at(child, &path[1..], out);
            }
        }
        _ if path.is_empty() => {
            if let Ok(s) = Scalar::from_json(value, "")
                && s.indexable()
            {
                out.push(s);
            }
        }
        _ => {}
    }
}

fn field_present(metadata: &Value, path: &str) -> bool {
    fn walk(value: &Value, path: &[&str]) -> bool {
        match value {
            Value::Array(items) => items.iter().any(|i| walk(i, path)),
            Value::Object(map) if !path.is_empty() => {
                map.get(path[0]).is_some_and(|c| walk(c, &path[1..]))
            }
            _ => path.is_empty(),
        }
    }
    walk(metadata, &path.split('.').collect::<Vec<_>>())
}

/// What a document's metadata contributes to the index.
#[derive(Debug, Default, PartialEq)]
pub struct Flattened {
    /// `(dot.path, scalar)` for every indexable scalar, deduplicated, in path
    /// order. Arrays contribute each element under the array's own path.
    pub values: Vec<(String, Scalar)>,
    /// Every path at which the field counts as present for `$exists` — the
    /// same rule as the in-memory check: the path reaches a non-array value.
    pub present: BTreeSet<String>,
}

/// Flattens metadata for indexing; at most `limit` entries (values plus
/// presence paths).
pub fn flatten(metadata: &Value, limit: usize) -> Result<Flattened> {
    fn walk(value: &Value, path: &mut String, out: &mut Flattened, limit: usize) -> Result<()> {
        if !path.is_empty() && !value.is_array() {
            out.present.insert(path.clone());
        }
        match value {
            Value::Object(map) => {
                for (k, v) in map {
                    validate_field(k)?;
                    if k.contains('.') {
                        return Err(Error::invalid(format!(
                            "metadata key {k:?} must not contain `.` (dot paths address \
                             nested objects)"
                        )));
                    }
                    let len = path.len();
                    if !path.is_empty() {
                        path.push('.');
                    }
                    path.push_str(k);
                    walk(v, path, out, limit)?;
                    path.truncate(len);
                }
            }
            Value::Array(items) => {
                for item in items {
                    walk(item, path, out, limit)?;
                }
            }
            scalar => {
                if !path.is_empty() {
                    let s = Scalar::from_json(scalar, path)?;
                    if s.indexable() {
                        out.values.push((path.clone(), s));
                    }
                }
            }
        }
        if out.values.len() + out.present.len() > limit {
            return Err(Error::invalid(format!(
                "metadata has more than {limit} indexed entries"
            )));
        }
        Ok(())
    }
    let mut out = Flattened::default();
    walk(metadata, &mut String::new(), &mut out, limit)?;
    out.values
        .sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.encode().cmp(&b.1.encode())));
    out.values
        .dedup_by(|a, b| a.0 == b.0 && a.1.encode() == b.1.encode());
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn f(v: Value) -> Filter {
        Filter::parse(&v).unwrap()
    }

    #[test]
    fn parses_the_documented_forms() {
        assert_eq!(f(json!({})), Filter::All);
        assert_eq!(
            f(json!({"a": 1})),
            Filter::Eq("a".into(), Scalar::Number(1.0))
        );
        assert!(matches!(f(json!({"a": 1, "b": 2})), Filter::And(v) if v.len() == 2));
        assert!(matches!(
            f(json!({"y": {"$gte": 2020, "$lt": 2025}})),
            Filter::And(v) if v.len() == 2
        ));
        assert!(matches!(
            f(json!({"$or": [{"a": 1}, {"b": 2}]})),
            Filter::Or(_)
        ));
        assert!(matches!(f(json!({"$not": {"a": 1}})), Filter::Not(_)));
        for bad in [
            json!([]),
            json!({"$xor": []}),
            json!({"a": {"$regex": "x"}}),
            json!({"a": [1, 2]}),
            json!({"a": {"b": 1}}),
            json!({"a": {"$gt": true}}),
            json!({"$or": []}),
            json!({"a": {"$exists": 1}}),
        ] {
            assert!(Filter::parse(&bad).is_err(), "{bad} should be rejected");
        }
    }

    #[test]
    fn in_memory_semantics_follow_mongodb() {
        let doc = json!({
            "cat": "news", "year": 2021, "tags": ["ai", "rust"],
            "author": {"name": "ada", "age": 36}, "none": null
        });
        assert!(f(json!({"cat": "news"})).matches(&doc));
        assert!(f(json!({"tags": "ai"})).matches(&doc), "array contains");
        assert!(f(json!({"author.name": "ada"})).matches(&doc), "dot path");
        assert!(f(json!({"year": {"$gt": 2020, "$lte": 2021}})).matches(&doc));
        assert!(
            !f(json!({"year": {"$gt": "2020"}})).matches(&doc),
            "no cross-type range"
        );
        assert!(f(json!({"tags": {"$in": ["go", "rust"]}})).matches(&doc));
        assert!(f(json!({"tags": {"$nin": ["go"]}})).matches(&doc));
        assert!(
            f(json!({"missing": {"$ne": 1}})).matches(&doc),
            "$ne matches absent"
        );
        assert!(f(json!({"missing": {"$exists": false}})).matches(&doc));
        assert!(f(json!({"none": null})).matches(&doc));
        assert!(f(json!({"none": {"$exists": true}})).matches(&doc));
        assert!(!f(json!({"$not": {"cat": "news"}})).matches(&doc));
    }

    #[test]
    fn long_strings_are_stored_but_never_compared() {
        let long = "x".repeat(MAX_INDEXED_STRING + 1);
        let doc = json!({"body": long, "tag": "t"});
        assert!(
            Filter::parse(&json!({"body": long})).is_err(),
            "operand too long"
        );
        assert!(!f(json!({"body": {"$gt": "a"}})).matches(&doc));
        assert!(f(json!({"body": {"$exists": true}})).matches(&doc));
        let flat = flatten(&doc, 100).unwrap();
        assert_eq!(flat.values.len(), 1);
        assert!(flat.present.contains("body"));
    }

    #[test]
    fn encodings_are_ordered_and_prefix_free() {
        let nums = [-1e9, -2.5, -0.0, 0.0, 1.0, 2.5, 1e9];
        for w in nums.windows(2) {
            let (a, b) = (Scalar::Number(w[0]).encode(), Scalar::Number(w[1]).encode());
            assert!(a <= b, "{} vs {}", w[0], w[1]);
        }
        assert_eq!(Scalar::Number(-0.0).encode(), Scalar::Number(0.0).encode());
        let strs = ["", "a", "a\0", "a\0b", "ab", "b"];
        for w in strs.windows(2) {
            let (a, b) = (
                Scalar::Str(w[0].into()).encode(),
                Scalar::Str(w[1].into()).encode(),
            );
            assert!(a < b, "{:?} vs {:?}", w[0], w[1]);
            assert!(!b.starts_with(&a), "{:?} must not prefix {:?}", w[0], w[1]);
        }
    }

    #[test]
    fn flatten_walks_objects_and_arrays() {
        let doc = json!({
            "a": 1, "b": {"c": "x"}, "t": ["p", "q", "p"], "o": [{"k": 1}], "e": [], "z": {}
        });
        let flat = flatten(&doc, 100).unwrap();
        let paths: Vec<&str> = flat.values.iter().map(|(p, _)| p.as_str()).collect();
        assert_eq!(paths, ["a", "b.c", "o.k", "t", "t"], "duplicates collapse");
        let present: Vec<&str> = flat.present.iter().map(String::as_str).collect();
        assert_eq!(present, ["a", "b", "b.c", "o", "o.k", "t", "z"]);
        // The presence index agrees with the in-memory rule, path by path.
        for path in ["a", "b", "b.c", "o", "o.k", "t", "e", "z", "missing", "a.x"] {
            assert_eq!(
                flat.present.contains(path),
                f(json!({path: {"$exists": true}})).matches(&doc),
                "{path}"
            );
        }
        assert!(flatten(&json!({"bad\u{0}": 1}), 100).is_err());
        assert!(
            flatten(&json!({"a.b": 1}), 100).is_err(),
            "dotted keys are ambiguous"
        );
        let many: serde_json::Map<String, Value> =
            (0..20).map(|i| (format!("k{i}"), json!(i))).collect();
        assert!(flatten(&Value::Object(many), 10).is_err());
    }
}
