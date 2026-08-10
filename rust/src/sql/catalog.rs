//! Row encoding and the table catalog.
//!
//! # Key layout
//!
//! Rows and schemas share the underlying key/value store, so keys are
//! namespaced by prefix:
//!
//! ```text
//!   \x00schema\x00<table>              -> bincode(TableSchema)
//!   \x01row\x00<table>\x00<rowid u64>  -> bincode(Row)
//! ```
//!
//! The prefixes start with control bytes that ordinary user keys are unlikely
//! to contain, and the big-endian row id means a prefix scan returns rows in
//! insertion order. Big-endian matters: little-endian would order row 256
//! before row 2.
//!
//! # Why bincode
//!
//! The engine already depends on it for the WAL, so no new dependency, and it
//! round-trips `Value` losslessly — a stringly-typed row format would turn
//! `42` and `'42'` into the same thing.

use crate::error::{Error, Result};
use crate::sql::ast::{ColumnDef, Value};
use serde::{Deserialize, Serialize};

/// Key prefix for schema entries.
pub const SCHEMA_PREFIX: &[u8] = b"\x00schema\x00";

/// Key prefix for row entries.
pub const ROW_PREFIX: &[u8] = b"\x01row\x00";

/// Separator between a table name and the row id.
const SEP: u8 = 0x00;

/// A stored value. Mirrors [`Value`] but owns its data and is serialisable.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Cell {
    /// Text.
    Text(String),
    /// 64-bit signed integer.
    Integer(i64),
    /// Double-precision float.
    Float(f64),
    /// SQL `NULL`.
    Null,
}

impl From<&Value> for Cell {
    fn from(v: &Value) -> Self {
        match v {
            Value::Text(s) => Cell::Text(s.clone()),
            Value::Integer(i) => Cell::Integer(*i),
            Value::Float(f) => Cell::Float(*f),
            Value::Null => Cell::Null,
        }
    }
}

impl Cell {
    /// Renders the cell for display in a result set.
    #[must_use]
    pub fn display(&self) -> String {
        match self {
            Cell::Text(s) => s.clone(),
            Cell::Integer(i) => i.to_string(),
            Cell::Float(f) => f.to_string(),
            Cell::Null => "NULL".to_string(),
        }
    }

    /// Compares against `other`, or `None` when the types are not comparable.
    ///
    /// Integers and floats compare numerically with each other. Text compares
    /// with text. `NULL` compares with nothing — matching SQL's three-valued
    /// logic, where any comparison against `NULL` is unknown, not false.
    #[must_use]
    pub fn partial_compare(&self, other: &Cell) -> Option<std::cmp::Ordering> {
        match (self, other) {
            (Cell::Null, _) | (_, Cell::Null) => None,
            (Cell::Integer(a), Cell::Integer(b)) => Some(a.cmp(b)),
            (Cell::Text(a), Cell::Text(b)) => Some(a.cmp(b)),
            (Cell::Float(a), Cell::Float(b)) => a.partial_cmp(b),
            // Mixed numeric: promote to f64 so `age > 30.5` works on an
            // integer column.
            (Cell::Integer(a), Cell::Float(b)) => (*a as f64).partial_cmp(b),
            (Cell::Float(a), Cell::Integer(b)) => a.partial_cmp(&(*b as f64)),
            // Text vs number is a type error, reported as "not comparable"
            // rather than silently ordering by some arbitrary rule.
            _ => None,
        }
    }
}

/// Decodes `bytes`, reporting corruption with a type label.
///
/// Shared by [`Row`] and [`TableSchema`]: both are plain bincode payloads, and
/// a malformed one always means the store is damaged rather than the caller
/// being wrong.
fn decode_stored<T: serde::de::DeserializeOwned>(bytes: &[u8], what: &str) -> Result<T> {
    bincode::deserialize(bytes).map_err(|e| Error::corrupt(format!("malformed {what}: {e}")))
}

/// One stored row.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Row {
    /// Cells, positionally matching the table's column order.
    pub cells: Vec<Cell>,
}

impl Row {
    /// Creates a row from `cells`.
    #[must_use]
    pub fn new(cells: Vec<Cell>) -> Self {
        Row { cells }
    }

    /// Encodes the row for storage.
    pub fn encode(&self) -> Result<Vec<u8>> {
        Ok(bincode::serialize(self)?)
    }

    /// Decodes a row produced by [`Row::encode`].
    pub fn decode(bytes: &[u8]) -> Result<Self> {
        decode_stored(bytes, "row")
    }
}

/// A table's column layout.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TableSchema {
    /// Table name.
    pub name: String,
    /// Column names, in declaration order.
    pub columns: Vec<String>,
    /// Declared types, parallel to `columns`. Advisory only.
    pub types: Vec<String>,
    /// Index of the primary-key column, if one was declared.
    pub primary_key: Option<usize>,
    /// Columns declared `NOT NULL`, as indices.
    pub not_null: Vec<usize>,
    /// Next row id to assign.
    pub next_row_id: u64,
}

impl TableSchema {
    /// Builds a schema from parsed column definitions.
    ///
    /// Rejects duplicate column names: silently keeping the last one would make
    /// `SELECT` ambiguous.
    pub fn from_defs(name: &str, defs: &[ColumnDef]) -> Result<Self> {
        let mut columns = Vec::with_capacity(defs.len());
        let mut types = Vec::with_capacity(defs.len());
        let mut primary_key = None;
        let mut not_null = Vec::new();

        for (i, d) in defs.iter().enumerate() {
            let lower = d.name.to_lowercase();
            if columns.iter().any(|c: &String| c.to_lowercase() == lower) {
                return Err(Error::invalid(format!(
                    "duplicate column `{}` in table `{name}`",
                    d.name
                )));
            }
            columns.push(d.name.clone());
            types.push(d.data_type.clone());
            if d.primary_key {
                if primary_key.is_some() {
                    return Err(Error::invalid(format!(
                        "table `{name}` declares more than one PRIMARY KEY"
                    )));
                }
                primary_key = Some(i);
            }
            if d.not_null || d.primary_key {
                not_null.push(i);
            }
        }

        Ok(TableSchema {
            name: name.to_string(),
            columns,
            types,
            primary_key,
            not_null,
            next_row_id: 1,
        })
    }

    /// Index of `column`, matched case-insensitively as SQL requires.
    pub fn column_index(&self, column: &str) -> Result<usize> {
        self.columns
            .iter()
            .position(|c| c.eq_ignore_ascii_case(column))
            .ok_or_else(|| {
                Error::invalid(format!(
                    "no column `{column}` in table `{}` (have: {})",
                    self.name,
                    self.columns.join(", ")
                ))
            })
    }

    /// Number of columns.
    #[must_use]
    pub fn width(&self) -> usize {
        self.columns.len()
    }

    /// Encodes the schema for storage.
    pub fn encode(&self) -> Result<Vec<u8>> {
        Ok(bincode::serialize(self)?)
    }

    /// Decodes a schema produced by [`TableSchema::encode`].
    pub fn decode(bytes: &[u8]) -> Result<Self> {
        decode_stored(bytes, "schema")
    }
}

/// Storage key for a table's schema.
#[must_use]
pub fn schema_key(table: &str) -> Vec<u8> {
    let mut k = SCHEMA_PREFIX.to_vec();
    // Table names are case-insensitive, so normalise before building the key.
    k.extend_from_slice(table.to_lowercase().as_bytes());
    k
}

/// Storage key for one row.
///
/// The row id is big-endian so a prefix scan yields insertion order.
#[must_use]
pub fn row_key(table: &str, row_id: u64) -> Vec<u8> {
    let mut k = row_prefix(table);
    k.extend_from_slice(&row_id.to_be_bytes());
    k
}

/// Key prefix covering every row of `table`.
#[must_use]
pub fn row_prefix(table: &str) -> Vec<u8> {
    let mut k = ROW_PREFIX.to_vec();
    k.extend_from_slice(table.to_lowercase().as_bytes());
    k.push(SEP);
    k
}

/// Extracts the row id from a row key, or `None` when it is malformed.
#[must_use]
pub fn row_id_of(table: &str, key: &[u8]) -> Option<u64> {
    let prefix = row_prefix(table);
    let rest = key.strip_prefix(prefix.as_slice())?;
    if rest.len() != 8 {
        return None;
    }
    Some(u64::from_be_bytes(rest.try_into().ok()?))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn defs() -> Vec<ColumnDef> {
        vec![
            ColumnDef {
                name: "id".into(),
                data_type: "INTEGER".into(),
                primary_key: true,
                not_null: false,
            },
            ColumnDef {
                name: "name".into(),
                data_type: "TEXT".into(),
                primary_key: false,
                not_null: true,
            },
        ]
    }

    #[test]
    fn schema_roundtrips() {
        let s = TableSchema::from_defs("users", &defs()).unwrap();
        let back = TableSchema::decode(&s.encode().unwrap()).unwrap();
        assert_eq!(s, back);
        assert_eq!(back.primary_key, Some(0));
        assert_eq!(back.not_null, vec![0, 1], "PRIMARY KEY implies NOT NULL");
    }

    #[test]
    fn column_lookup_is_case_insensitive() {
        let s = TableSchema::from_defs("users", &defs()).unwrap();
        assert_eq!(s.column_index("id").unwrap(), 0);
        assert_eq!(s.column_index("ID").unwrap(), 0);
        assert_eq!(s.column_index("NaMe").unwrap(), 1);
        assert!(s.column_index("missing").is_err());
    }

    #[test]
    fn unknown_column_error_lists_the_real_ones() {
        let s = TableSchema::from_defs("users", &defs()).unwrap();
        let msg = format!("{}", s.column_index("nmae").unwrap_err());
        assert!(msg.contains("id, name"), "should help with the typo: {msg}");
    }

    #[test]
    fn duplicate_columns_are_rejected() {
        let dupes = vec![
            ColumnDef {
                name: "a".into(),
                data_type: String::new(),
                primary_key: false,
                not_null: false,
            },
            ColumnDef {
                name: "A".into(),
                data_type: String::new(),
                primary_key: false,
                not_null: false,
            },
        ];
        assert!(TableSchema::from_defs("t", &dupes).is_err());
    }

    #[test]
    fn two_primary_keys_are_rejected() {
        let mut d = defs();
        d[1].primary_key = true;
        assert!(TableSchema::from_defs("t", &d).is_err());
    }

    #[test]
    fn row_roundtrips_every_cell_type() {
        let r = Row::new(vec![
            Cell::Integer(42),
            Cell::Text("hello".into()),
            Cell::Float(3.5),
            Cell::Null,
        ]);
        assert_eq!(Row::decode(&r.encode().unwrap()).unwrap(), r);
    }

    #[test]
    fn row_keys_sort_in_insertion_order() {
        // Big-endian is what makes this hold; little-endian would fail at 256.
        let mut keys: Vec<Vec<u8>> = vec![
            row_key("t", 300),
            row_key("t", 2),
            row_key("t", 1),
            row_key("t", 256),
        ];
        keys.sort();
        let ids: Vec<u64> = keys.iter().filter_map(|k| row_id_of("t", k)).collect();
        assert_eq!(ids, vec![1, 2, 256, 300]);
    }

    #[test]
    fn table_names_are_case_insensitive_in_keys() {
        assert_eq!(schema_key("Users"), schema_key("users"));
        assert_eq!(row_key("Users", 1), row_key("USERS", 1));
    }

    #[test]
    fn row_prefix_does_not_match_another_table() {
        let k = row_key("users", 1);
        assert!(k.starts_with(&row_prefix("users")));
        assert!(!k.starts_with(&row_prefix("user")), "prefix must not bleed");
        assert_eq!(row_id_of("user", &k), None);
    }

    #[test]
    fn cell_comparison_follows_sql_semantics() {
        use std::cmp::Ordering;
        assert_eq!(
            Cell::Integer(1).partial_compare(&Cell::Integer(2)),
            Some(Ordering::Less)
        );
        assert_eq!(
            Cell::Text("a".into()).partial_compare(&Cell::Text("b".into())),
            Some(Ordering::Less)
        );
        // Mixed numeric promotes.
        assert_eq!(
            Cell::Integer(2).partial_compare(&Cell::Float(2.5)),
            Some(Ordering::Less)
        );
        // NULL is not comparable with anything, including itself.
        assert_eq!(Cell::Null.partial_compare(&Cell::Integer(1)), None);
        assert_eq!(Cell::Null.partial_compare(&Cell::Null), None);
        // Text vs number is a type error, not an arbitrary ordering.
        assert_eq!(
            Cell::Text("1".into()).partial_compare(&Cell::Integer(1)),
            None
        );
    }

    #[test]
    fn malformed_bytes_are_rejected() {
        assert!(Row::decode(&[0xFF; 4]).is_err());
        assert!(TableSchema::decode(&[]).is_err());
    }
}
