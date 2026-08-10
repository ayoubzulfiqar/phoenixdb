//! SQL abstract syntax tree.
//!
//! Deliberately small: the statements the design directive names
//! (`CREATE TABLE`, `INSERT INTO`, `SELECT ... WHERE`, `UPDATE`), plus
//! `DELETE` and `DROP TABLE` because they fall out of the same grammar for
//! almost no extra code.
//!
//! Values are typed at parse time rather than kept as strings, so the executor
//! never re-parses and a type error surfaces before any storage work happens.

use std::fmt;

/// A literal value appearing in a statement.
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    /// Text.
    Text(String),
    /// 64-bit signed integer.
    Integer(i64),
    /// Double-precision float.
    Float(f64),
    /// SQL `NULL`.
    Null,
}

impl Value {
    /// Name of this value's type, for error messages.
    #[must_use]
    pub fn type_name(&self) -> &'static str {
        match self {
            Value::Text(_) => "text",
            Value::Integer(_) => "integer",
            Value::Float(_) => "float",
            Value::Null => "null",
        }
    }

    /// Renders the value the way it is stored in a row.
    #[must_use]
    pub fn to_storage_string(&self) -> String {
        match self {
            Value::Text(s) => s.clone(),
            Value::Integer(i) => i.to_string(),
            Value::Float(f) => f.to_string(),
            Value::Null => String::new(),
        }
    }
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Value::Text(s) => write!(f, "'{s}'"),
            Value::Integer(i) => write!(f, "{i}"),
            Value::Float(x) => write!(f, "{x}"),
            Value::Null => write!(f, "NULL"),
        }
    }
}

/// A comparison operator in a `WHERE` clause.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComparisonOp {
    /// `=`
    Eq,
    /// `<>` / `!=`
    NotEq,
    /// `<`
    Lt,
    /// `<=`
    LtEq,
    /// `>`
    Gt,
    /// `>=`
    GtEq,
}

impl ComparisonOp {
    /// SQL spelling of the operator.
    #[must_use]
    pub fn symbol(&self) -> &'static str {
        match self {
            ComparisonOp::Eq => "=",
            ComparisonOp::NotEq => "<>",
            ComparisonOp::Lt => "<",
            ComparisonOp::LtEq => "<=",
            ComparisonOp::Gt => ">",
            ComparisonOp::GtEq => ">=",
        }
    }
}

/// A single `column <op> value` predicate.
#[derive(Debug, Clone, PartialEq)]
pub struct Predicate {
    /// Column being tested.
    pub column: String,
    /// Comparison to apply.
    pub op: ComparisonOp,
    /// Value to compare against.
    pub value: Value,
}

/// A `WHERE` clause: predicates joined by `AND` or `OR`.
///
/// Mixed `AND`/`OR` without parentheses is rejected by the parser rather than
/// silently guessing a precedence, which is the kind of ambiguity that produces
/// wrong answers instead of errors.
#[derive(Debug, Clone, PartialEq)]
pub enum WhereClause {
    /// A single predicate.
    Single(Predicate),
    /// Every predicate must hold.
    And(Vec<Predicate>),
    /// At least one predicate must hold.
    Or(Vec<Predicate>),
}

impl WhereClause {
    /// Every predicate in the clause, regardless of connective.
    #[must_use]
    pub fn predicates(&self) -> &[Predicate] {
        match self {
            WhereClause::Single(p) => std::slice::from_ref(p),
            WhereClause::And(v) | WhereClause::Or(v) => v.as_slice(),
        }
    }
}

/// One column in a `CREATE TABLE`.
#[derive(Debug, Clone, PartialEq)]
pub struct ColumnDef {
    /// Column name.
    pub name: String,
    /// Declared type, uppercased. Advisory: storage is schemaless.
    pub data_type: String,
    /// Whether the column carries `PRIMARY KEY`.
    pub primary_key: bool,
    /// Whether the column carries `NOT NULL`.
    pub not_null: bool,
}

/// A parsed SQL statement.
#[derive(Debug, Clone, PartialEq)]
pub enum Statement {
    /// `CREATE TABLE name (cols…)`
    CreateTable {
        /// Table name.
        table: String,
        /// Column definitions, in declaration order.
        columns: Vec<ColumnDef>,
        /// Whether `IF NOT EXISTS` was given.
        if_not_exists: bool,
    },
    /// `DROP TABLE name`
    DropTable {
        /// Table name.
        table: String,
        /// Whether `IF EXISTS` was given.
        if_exists: bool,
    },
    /// `INSERT INTO name (cols…) VALUES (…), (…)`
    Insert {
        /// Target table.
        table: String,
        /// Target columns; empty means "every column, in declaration order".
        columns: Vec<String>,
        /// One `Vec<Value>` per row.
        rows: Vec<Vec<Value>>,
    },
    /// `SELECT cols FROM name WHERE … ORDER BY … LIMIT …`
    Select {
        /// Table to read.
        table: String,
        /// Projection; empty means `*`.
        columns: Vec<String>,
        /// Optional filter.
        filter: Option<WhereClause>,
        /// Optional `(column, descending)` ordering.
        order_by: Option<(String, bool)>,
        /// Optional row cap.
        limit: Option<usize>,
    },
    /// `UPDATE name SET col = val, … WHERE …`
    Update {
        /// Table to modify.
        table: String,
        /// Assignments, in statement order.
        assignments: Vec<(String, Value)>,
        /// Optional filter; `None` updates every row.
        filter: Option<WhereClause>,
    },
    /// `DELETE FROM name WHERE …`
    Delete {
        /// Table to delete from.
        table: String,
        /// Optional filter; `None` deletes every row.
        filter: Option<WhereClause>,
    },
}

impl Statement {
    /// The table this statement operates on.
    #[must_use]
    pub fn table(&self) -> &str {
        match self {
            Statement::CreateTable { table, .. }
            | Statement::DropTable { table, .. }
            | Statement::Insert { table, .. }
            | Statement::Select { table, .. }
            | Statement::Update { table, .. }
            | Statement::Delete { table, .. } => table,
        }
    }

    /// True when the statement modifies data or schema.
    ///
    /// Used by the executor to pick a read-only or read-write transaction, and
    /// by RBAC to choose between the read and write permissions.
    #[must_use]
    pub fn is_mutation(&self) -> bool {
        !matches!(self, Statement::Select { .. })
    }

    /// A short name for tracing spans and audit records.
    #[must_use]
    pub fn kind_name(&self) -> &'static str {
        match self {
            Statement::CreateTable { .. } => "create_table",
            Statement::DropTable { .. } => "drop_table",
            Statement::Insert { .. } => "insert",
            Statement::Select { .. } => "select",
            Statement::Update { .. } => "update",
            Statement::Delete { .. } => "delete",
        }
    }
}
