//! SQL abstract syntax tree.
//!
//! Covers `CREATE TABLE`, `DROP TABLE`, `INSERT`, `SELECT` (with `WHERE`,
//! `GROUP BY`, aggregates, multi-column `ORDER BY`, `LIMIT`/`OFFSET`),
//! `UPDATE` and `DELETE`.
//!
//! Values are typed at parse time rather than kept as strings, so the executor
//! never re-parses and a type error surfaces before any storage work happens.
//! Bound parameters (`?`, `?N`) stay symbolic in the tree and are resolved by
//! the executor against the caller's parameter list, so user data never has
//! to be spliced into SQL text.

use std::fmt;

/// A literal value.
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    /// Text.
    Text(String),
    /// 64-bit signed integer.
    Integer(i64),
    /// Double-precision float (always finite).
    Float(f64),
    /// SQL `NULL`.
    Null,
}

impl Value {
    /// Short type label for messages.
    #[must_use]
    pub fn type_name(&self) -> &'static str {
        match self {
            Value::Text(_) => "text",
            Value::Integer(_) => "integer",
            Value::Float(_) => "float",
            Value::Null => "null",
        }
    }

    /// Text form used for display.
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
            Value::Text(s) => write!(f, "'{}'", s.replace('\'', "''")),
            Value::Integer(i) => write!(f, "{i}"),
            Value::Float(x) => write!(f, "{x:?}"),
            Value::Null => write!(f, "NULL"),
        }
    }
}

/// A binary comparison.
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
    /// The operator's SQL spelling.
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

/// A scalar or boolean expression.
#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    /// A literal.
    Literal(Value),
    /// A column reference.
    Column(String),
    /// A bound parameter, as a 0-based index into the parameter list.
    Param(usize),
    /// `left op right`.
    Compare {
        /// Left operand.
        left: Box<Expr>,
        /// Operator.
        op: ComparisonOp,
        /// Right operand.
        right: Box<Expr>,
    },
    /// `left AND right`.
    And(Box<Expr>, Box<Expr>),
    /// `left OR right`.
    Or(Box<Expr>, Box<Expr>),
    /// `NOT expr`.
    Not(Box<Expr>),
    /// `expr IS [NOT] NULL`.
    IsNull {
        /// Tested expression.
        expr: Box<Expr>,
        /// `IS NOT NULL`.
        negated: bool,
    },
    /// `expr [NOT] IN (list…)`.
    InList {
        /// Tested expression.
        expr: Box<Expr>,
        /// Candidate values.
        list: Vec<Expr>,
        /// `NOT IN`.
        negated: bool,
    },
    /// `expr [NOT] BETWEEN low AND high` (inclusive).
    Between {
        /// Tested expression.
        expr: Box<Expr>,
        /// Lower bound.
        low: Box<Expr>,
        /// Upper bound.
        high: Box<Expr>,
        /// `NOT BETWEEN`.
        negated: bool,
    },
    /// `expr [NOT] LIKE|ILIKE pattern`, with `%` and `_` wildcards.
    Like {
        /// Tested expression.
        expr: Box<Expr>,
        /// Pattern.
        pattern: Box<Expr>,
        /// `NOT LIKE`.
        negated: bool,
        /// `ILIKE`: case-insensitive.
        case_insensitive: bool,
    },
}

impl Expr {
    /// Every column this expression references.
    pub fn columns(&self, out: &mut Vec<String>) {
        match self {
            Expr::Column(c) => out.push(c.clone()),
            Expr::Literal(_) | Expr::Param(_) => {}
            Expr::Compare { left, right, .. } | Expr::And(left, right) | Expr::Or(left, right) => {
                left.columns(out);
                right.columns(out);
            }
            Expr::Not(e) | Expr::IsNull { expr: e, .. } => e.columns(out),
            Expr::InList { expr, list, .. } => {
                expr.columns(out);
                for e in list {
                    e.columns(out);
                }
            }
            Expr::Between {
                expr, low, high, ..
            } => {
                expr.columns(out);
                low.columns(out);
                high.columns(out);
            }
            Expr::Like { expr, pattern, .. } => {
                expr.columns(out);
                pattern.columns(out);
            }
        }
    }
}

/// An aggregate function.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AggFunc {
    /// `COUNT`
    Count,
    /// `SUM`
    Sum,
    /// `AVG`
    Avg,
    /// `MIN`
    Min,
    /// `MAX`
    Max,
}

impl AggFunc {
    /// Lower-case SQL name.
    #[must_use]
    pub fn name(&self) -> &'static str {
        match self {
            AggFunc::Count => "count",
            AggFunc::Sum => "sum",
            AggFunc::Avg => "avg",
            AggFunc::Min => "min",
            AggFunc::Max => "max",
        }
    }
}

/// One entry of a `SELECT` list.
#[derive(Debug, Clone, PartialEq)]
pub enum SelectItem {
    /// `*`
    Wildcard,
    /// A column, with an optional `AS alias`.
    Column {
        /// Column name.
        name: String,
        /// Output name.
        alias: Option<String>,
    },
    /// An aggregate: `COUNT(*)` (`column == None`), `SUM(col)`, ...
    Aggregate {
        /// Function.
        func: AggFunc,
        /// Argument column; `None` for `COUNT(*)`.
        column: Option<String>,
        /// `COUNT(DISTINCT col)` etc.
        distinct: bool,
        /// Output name.
        alias: Option<String>,
    },
}

/// One `ORDER BY` key.
#[derive(Debug, Clone, PartialEq)]
pub struct OrderItem {
    /// Output column name, alias or table column.
    pub column: String,
    /// Descending.
    pub desc: bool,
}

/// A column definition in `CREATE TABLE`.
#[derive(Debug, Clone, PartialEq)]
pub struct ColumnDef {
    /// Column name.
    pub name: String,
    /// Declared type (advisory), e.g. `INTEGER`, `VARCHAR(255)`.
    pub data_type: String,
    /// `PRIMARY KEY`.
    pub primary_key: bool,
    /// `NOT NULL`.
    pub not_null: bool,
}

/// A parsed statement.
#[derive(Debug, Clone, PartialEq)]
pub enum Statement {
    /// `CREATE TABLE [IF NOT EXISTS] t (...)`
    CreateTable {
        /// Table name.
        table: String,
        /// Columns.
        columns: Vec<ColumnDef>,
        /// `IF NOT EXISTS`.
        if_not_exists: bool,
    },
    /// `DROP TABLE [IF EXISTS] t`
    DropTable {
        /// Table name.
        table: String,
        /// `IF EXISTS`.
        if_exists: bool,
    },
    /// `INSERT INTO t [(cols)] VALUES (...), ...`
    Insert {
        /// Table name.
        table: String,
        /// Named columns; empty means every column in order.
        columns: Vec<String>,
        /// Rows of literal or parameter expressions.
        rows: Vec<Vec<Expr>>,
    },
    /// `SELECT ... FROM t [WHERE] [GROUP BY] [ORDER BY] [LIMIT] [OFFSET]`
    Select {
        /// Table name.
        table: String,
        /// Select list.
        items: Vec<SelectItem>,
        /// `WHERE` condition.
        filter: Option<Expr>,
        /// `GROUP BY` columns.
        group_by: Vec<String>,
        /// `ORDER BY` keys, most significant first.
        order_by: Vec<OrderItem>,
        /// `LIMIT`.
        limit: Option<usize>,
        /// `OFFSET`.
        offset: Option<usize>,
    },
    /// `UPDATE t SET col = expr, ... [WHERE]`
    Update {
        /// Table name.
        table: String,
        /// Assignments of literal or parameter expressions.
        assignments: Vec<(String, Expr)>,
        /// `WHERE` condition.
        filter: Option<Expr>,
    },
    /// `DELETE FROM t [WHERE]`
    Delete {
        /// Table name.
        table: String,
        /// `WHERE` condition.
        filter: Option<Expr>,
    },
}

impl Statement {
    /// The table the statement targets.
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

    /// True for statements that write.
    #[must_use]
    pub fn is_mutation(&self) -> bool {
        !matches!(self, Statement::Select { .. })
    }

    /// Short statement kind, for logs and metrics.
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
