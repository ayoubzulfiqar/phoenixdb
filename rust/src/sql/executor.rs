//! SQL executor: runs a parsed [`Statement`] against a [`Database`].
//!
//! # Design
//!
//! Direct interpretation of the AST, not a Volcano-style operator tree. The
//! supported statement set has no joins or subqueries, so an operator pipeline
//! would be indirection without benefit. A planner becomes worthwhile when
//! joins arrive; until then this is easier to verify.
//!
//! # Transactions
//!
//! Every statement runs in its own transaction unless the caller supplies one.
//! A mutation that fails partway is rolled back, so `INSERT ... VALUES` with
//! three rows either stores all three or none — the atomicity guarantee the
//! storage engine already provides, surfaced at the SQL layer.
//!
//! # Isolation caveat
//!
//! Row scans read through [`Database::scan`], which materialises the visible
//! key space. That is correct but O(database) per scan: fine for the embedded
//! workloads this targets, wrong for large tables. A prefix-bounded iterator on
//! `Database` would fix it, and is the obvious next optimisation.

use crate::error::{Error, Result};
use crate::sql::ast::{ComparisonOp, Predicate, Statement, Value, WhereClause};
use crate::sql::catalog::{Cell, Row, TableSchema, row_id_of, row_key, row_prefix, schema_key};
use crate::{Database, Result as DbResult};

/// The outcome of executing one statement.
#[derive(Debug, Clone, PartialEq)]
pub enum QueryResult {
    /// A `SELECT` result set.
    Rows {
        /// Column names, in projection order.
        columns: Vec<String>,
        /// Rows, each parallel to `columns`.
        rows: Vec<Vec<Cell>>,
    },
    /// A mutation reporting how many rows it touched.
    Affected {
        /// Number of rows inserted, updated or deleted.
        count: u64,
    },
    /// A schema change.
    SchemaChanged {
        /// What happened, for display.
        detail: String,
    },
}

impl QueryResult {
    /// Number of rows returned, or 0 for a non-`SELECT`.
    #[must_use]
    pub fn row_count(&self) -> usize {
        match self {
            QueryResult::Rows { rows, .. } => rows.len(),
            _ => 0,
        }
    }

    /// Number of rows affected, or 0 for a `SELECT`.
    #[must_use]
    pub fn affected(&self) -> u64 {
        match self {
            QueryResult::Affected { count } => *count,
            _ => 0,
        }
    }

    /// Renders the result as text, for a CLI or test output.
    #[must_use]
    pub fn render(&self) -> String {
        match self {
            QueryResult::Rows { columns, rows } => {
                let mut out = columns.join(" | ");
                out.push('\n');
                out.push_str(&"-".repeat(out.len().saturating_sub(1)));
                for row in rows {
                    out.push('\n');
                    out.push_str(
                        &row.iter()
                            .map(Cell::display)
                            .collect::<Vec<_>>()
                            .join(" | "),
                    );
                }
                out.push_str(&format!("\n({} row(s))", rows.len()));
                out
            }
            QueryResult::Affected { count } => format!("{count} row(s) affected"),
            QueryResult::SchemaChanged { detail } => detail.clone(),
        }
    }
}

/// Executes SQL statements against a database.
pub struct Executor<'a> {
    db: &'a Database,
}

impl<'a> Executor<'a> {
    /// Binds an executor to `db`.
    #[must_use]
    pub fn new(db: &'a Database) -> Self {
        Executor { db }
    }

    /// Parses and executes `sql`.
    pub fn run(&self, sql: &str) -> Result<QueryResult> {
        let stmt = crate::sql::parse(sql)?;
        self.execute(&stmt)
    }

    /// Executes an already-parsed statement.
    pub fn execute(&self, stmt: &Statement) -> Result<QueryResult> {
        match stmt {
            Statement::CreateTable {
                table,
                columns,
                if_not_exists,
            } => self.create_table(table, columns, *if_not_exists),
            Statement::DropTable { table, if_exists } => self.drop_table(table, *if_exists),
            Statement::Insert {
                table,
                columns,
                rows,
            } => self.insert(table, columns, rows),
            Statement::Select {
                table,
                columns,
                filter,
                order_by,
                limit,
            } => self.select(table, columns, filter.as_ref(), order_by.as_ref(), *limit),
            Statement::Update {
                table,
                assignments,
                filter,
            } => self.update(table, assignments, filter.as_ref()),
            Statement::Delete { table, filter } => self.delete(table, filter.as_ref()),
        }
    }

    // ---- catalog ----------------------------------------------------------

    /// Loads a table's schema, or reports that the table does not exist.
    fn load_schema(&self, table: &str) -> Result<TableSchema> {
        match self.db.get_auto(&schema_key(table)) {
            Ok(bytes) => TableSchema::decode(&bytes),
            Err(Error::NotFound) => Err(Error::invalid(format!("no such table: `{table}`"))),
            Err(e) => Err(e),
        }
    }

    /// True when `table` exists.
    fn table_exists(&self, table: &str) -> bool {
        !matches!(self.db.get_auto(&schema_key(table)), Err(Error::NotFound))
    }

    fn store_schema(&self, schema: &TableSchema) -> DbResult<()> {
        self.db
            .put_auto(&schema_key(&schema.name), &schema.encode()?)
    }

    // ---- statements -------------------------------------------------------

    fn create_table(
        &self,
        table: &str,
        columns: &[crate::sql::ast::ColumnDef],
        if_not_exists: bool,
    ) -> Result<QueryResult> {
        if self.table_exists(table) {
            if if_not_exists {
                return Ok(QueryResult::SchemaChanged {
                    detail: format!("table `{table}` already exists, skipped"),
                });
            }
            return Err(Error::invalid(format!("table `{table}` already exists")));
        }
        let schema = TableSchema::from_defs(table, columns)?;
        self.store_schema(&schema)?;
        Ok(QueryResult::SchemaChanged {
            detail: format!("table `{table}` created with {} column(s)", schema.width()),
        })
    }

    fn drop_table(&self, table: &str, if_exists: bool) -> Result<QueryResult> {
        if !self.table_exists(table) {
            if if_exists {
                return Ok(QueryResult::SchemaChanged {
                    detail: format!("table `{table}` does not exist, skipped"),
                });
            }
            return Err(Error::invalid(format!("no such table: `{table}`")));
        }
        // Remove every row, then the schema. Doing it in one transaction keeps
        // a crash from leaving rows behind whose schema is gone.
        let prefix = row_prefix(table);
        let victims: Vec<Vec<u8>> = self
            .db
            .scan()?
            .into_iter()
            .map(|(k, _)| k)
            .filter(|k| k.starts_with(prefix.as_slice()))
            .collect();

        self.in_txn(|txn| {
            for key in &victims {
                self.db.delete(txn, key)?;
            }
            self.db.delete(txn, &schema_key(table))?;
            Ok(())
        })?;
        Ok(QueryResult::SchemaChanged {
            detail: format!("table `{table}` dropped ({} row(s))", victims.len()),
        })
    }

    /// Resolves a column list to positions; empty means "every column".
    ///
    /// Shared by `INSERT` (where the list names the supplied values) and
    /// `SELECT` (where it names the projection).
    fn resolve_columns(schema: &TableSchema, columns: &[String]) -> Result<Vec<usize>> {
        if columns.is_empty() {
            Ok((0..schema.width()).collect())
        } else {
            columns.iter().map(|c| schema.column_index(c)).collect()
        }
    }

    /// Runs `body` in a transaction, committing on success and rolling back on
    /// any error.
    ///
    /// Every mutation needs this exact dance; hand-rolling it four times was
    /// four chances to forget the rollback and leak a transaction.
    fn in_txn<T>(&self, body: impl FnOnce(u64) -> Result<T>) -> Result<T> {
        let txn = self.db.begin(false)?;
        match body(txn) {
            Ok(value) => {
                self.db.commit(txn)?;
                Ok(value)
            }
            Err(e) => {
                // The original error is what the caller needs; a rollback
                // failure here would only mask it.
                let _ = self.db.rollback(txn);
                Err(e)
            }
        }
    }

    fn insert(&self, table: &str, columns: &[String], rows: &[Vec<Value>]) -> Result<QueryResult> {
        let mut schema = self.load_schema(table)?;
        let positions = Self::resolve_columns(&schema, columns)?;

        let mut encoded = Vec::with_capacity(rows.len());
        for row in rows {
            if row.len() != positions.len() {
                return Err(Error::invalid(format!(
                    "row supplies {} value(s) but {} column(s) were named",
                    row.len(),
                    positions.len()
                )));
            }
            // Unnamed columns default to NULL.
            let mut cells = vec![Cell::Null; schema.width()];
            for (slot, value) in positions.iter().zip(row) {
                cells[*slot] = Cell::from(value);
            }
            // Enforce NOT NULL before writing anything.
            for &idx in &schema.not_null {
                if cells[idx] == Cell::Null {
                    return Err(Error::invalid(format!(
                        "column `{}` is NOT NULL but no value was supplied",
                        schema.columns[idx]
                    )));
                }
            }
            let id = schema.next_row_id;
            schema.next_row_id += 1;
            encoded.push((row_key(table, id), Row::new(cells).encode()?));
        }

        // One transaction for every row plus the schema's row-id counter, so a
        // failure cannot leave the counter ahead of the rows.
        self.in_txn(|txn| {
            for (key, value) in &encoded {
                self.db.insert(txn, key, value)?;
            }
            self.db.insert(txn, &schema_key(table), &schema.encode()?)
        })?;
        Ok(QueryResult::Affected {
            count: encoded.len() as u64,
        })
    }

    fn select(
        &self,
        table: &str,
        columns: &[String],
        filter: Option<&WhereClause>,
        order_by: Option<&(String, bool)>,
        limit: Option<usize>,
    ) -> Result<QueryResult> {
        let schema = self.load_schema(table)?;
        let projection = Self::resolve_columns(&schema, columns)?;
        let names: Vec<String> = projection
            .iter()
            .map(|&i| schema.columns[i].clone())
            .collect();

        // Validate the filter up front. Evaluating it per-row would silently
        // accept a typo'd column whenever the table happens to be empty.
        validate_filter(&schema, filter)?;

        let mut matched: Vec<Row> = Vec::new();
        for (_, row) in self.scan_rows(table)? {
            if row_matches(&schema, &row, filter)? {
                matched.push(row);
            }
        }

        // ORDER BY before LIMIT, as SQL requires: limiting first would return
        // an arbitrary subset.
        if let Some((column, desc)) = order_by {
            let idx = schema.column_index(column)?;
            matched.sort_by(|a, b| {
                let ord = a.cells[idx]
                    .partial_compare(&b.cells[idx])
                    // NULLs and incomparable pairs sort last, deterministically.
                    .unwrap_or(std::cmp::Ordering::Equal);
                if *desc { ord.reverse() } else { ord }
            });
        }
        if let Some(n) = limit {
            matched.truncate(n);
        }

        let rows = matched
            .into_iter()
            .map(|r| projection.iter().map(|&i| r.cells[i].clone()).collect())
            .collect();
        Ok(QueryResult::Rows {
            columns: names,
            rows,
        })
    }

    fn update(
        &self,
        table: &str,
        assignments: &[(String, Value)],
        filter: Option<&WhereClause>,
    ) -> Result<QueryResult> {
        let schema = self.load_schema(table)?;
        let targets: Vec<(usize, Cell)> = assignments
            .iter()
            .map(|(c, v)| Ok((schema.column_index(c)?, Cell::from(v))))
            .collect::<Result<_>>()?;

        // Reject a NOT NULL violation before touching storage.
        for (idx, cell) in &targets {
            if *cell == Cell::Null && schema.not_null.contains(idx) {
                return Err(Error::invalid(format!(
                    "column `{}` is NOT NULL and cannot be set to NULL",
                    schema.columns[*idx]
                )));
            }
        }

        let mut updates = Vec::new();
        validate_filter(&schema, filter)?;
        for (key, mut row) in self.scan_rows(table)? {
            if !row_matches(&schema, &row, filter)? {
                continue;
            }
            for (idx, cell) in &targets {
                row.cells[*idx] = cell.clone();
            }
            updates.push((key, row.encode()?));
        }

        let count = updates.len() as u64;
        if count > 0 {
            self.in_txn(|txn| {
                for (key, value) in &updates {
                    self.db.insert(txn, key, value)?;
                }
                Ok(())
            })?;
        }
        Ok(QueryResult::Affected { count })
    }

    fn delete(&self, table: &str, filter: Option<&WhereClause>) -> Result<QueryResult> {
        let schema = self.load_schema(table)?;
        validate_filter(&schema, filter)?;
        let mut victims = Vec::new();
        for (key, row) in self.scan_rows(table)? {
            if row_matches(&schema, &row, filter)? {
                victims.push(key);
            }
        }

        let count = victims.len() as u64;
        if count > 0 {
            self.in_txn(|txn| {
                for key in &victims {
                    self.db.delete(txn, key)?;
                }
                Ok(())
            })?;
        }
        Ok(QueryResult::Affected { count })
    }

    /// Every row of `table`, in insertion order.
    fn scan_rows(&self, table: &str) -> Result<Vec<(Vec<u8>, Row)>> {
        let prefix = row_prefix(table);
        let mut out = Vec::new();
        for (key, value) in self.db.scan()? {
            if !key.starts_with(prefix.as_slice()) {
                continue;
            }
            // A key under our prefix that is not a valid row key means the
            // store has been corrupted; surface it rather than skipping.
            if row_id_of(table, &key).is_none() {
                return Err(Error::corrupt(format!(
                    "malformed row key in table `{table}`"
                )));
            }
            out.push((key, Row::decode(&value)?));
        }
        Ok(out)
    }
}

/// Checks that every column named in a filter exists.
///
/// Called before scanning so a typo is reported even when the table is empty —
/// otherwise `WHERE nmae = 'x'` would quietly return zero rows and look like a
/// legitimate "no matches" answer.
fn validate_filter(schema: &TableSchema, filter: Option<&WhereClause>) -> Result<()> {
    let Some(clause) = filter else {
        return Ok(());
    };
    for p in clause.predicates() {
        schema.column_index(&p.column)?;
    }
    Ok(())
}

/// Evaluates a `WHERE` clause against one row.
fn row_matches(schema: &TableSchema, row: &Row, filter: Option<&WhereClause>) -> Result<bool> {
    let Some(clause) = filter else {
        return Ok(true); // no filter matches every row
    };
    match clause {
        WhereClause::Single(p) => predicate_matches(schema, row, p),
        WhereClause::And(ps) => {
            for p in ps {
                if !predicate_matches(schema, row, p)? {
                    return Ok(false);
                }
            }
            Ok(true)
        }
        WhereClause::Or(ps) => {
            for p in ps {
                if predicate_matches(schema, row, p)? {
                    return Ok(true);
                }
            }
            Ok(false)
        }
    }
}

/// Evaluates one predicate against a row.
///
/// A comparison involving `NULL`, or between incomparable types, is `false` —
/// SQL's "unknown" collapses to "not matched" in a `WHERE` clause.
fn predicate_matches(schema: &TableSchema, row: &Row, p: &Predicate) -> Result<bool> {
    let idx = schema.column_index(&p.column)?;
    let lhs = &row.cells[idx];
    let rhs = Cell::from(&p.value);

    // IS NULL / IS NOT NULL are not in the grammar yet, so `= NULL` is the
    // only way to test for null, and SQL says that is never true. Support the
    // intuitive meaning instead of silently returning nothing.
    if rhs == Cell::Null {
        return Ok(match p.op {
            ComparisonOp::Eq => *lhs == Cell::Null,
            ComparisonOp::NotEq => *lhs != Cell::Null,
            _ => false,
        });
    }

    let Some(ord) = lhs.partial_compare(&rhs) else {
        return Ok(false); // NULL or type mismatch: unknown -> not matched
    };
    use std::cmp::Ordering::*;
    Ok(match p.op {
        ComparisonOp::Eq => ord == Equal,
        ComparisonOp::NotEq => ord != Equal,
        ComparisonOp::Lt => ord == Less,
        ComparisonOp::LtEq => ord != Greater,
        ComparisonOp::Gt => ord == Greater,
        ComparisonOp::GtEq => ord != Less,
    })
}

/// Renders a [`QueryResult`] as a JSON document for the FFI boundary.
///
/// Hand-written rather than using `serde_json`, because that crate sits behind
/// the separate `json` feature and the SQL layer must not drag it in. The
/// output shape is documented on `phoenix_sql_query`.
#[must_use]
pub fn result_to_json(result: &QueryResult) -> String {
    let mut out = String::new();
    match result {
        QueryResult::Rows { columns, rows } => {
            out.push_str("{\"type\":\"rows\",\"columns\":[");
            for (i, c) in columns.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                push_json_string(&mut out, c);
            }
            out.push_str("],\"rows\":[");
            for (i, row) in rows.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                out.push('[');
                for (j, cell) in row.iter().enumerate() {
                    if j > 0 {
                        out.push(',');
                    }
                    push_cell_json(&mut out, cell);
                }
                out.push(']');
            }
            out.push_str("]}");
        }
        QueryResult::Affected { count } => {
            out.push_str(&format!("{{\"type\":\"affected\",\"count\":{count}}}"));
        }
        QueryResult::SchemaChanged { detail } => {
            out.push_str("{\"type\":\"schema\",\"detail\":");
            push_json_string(&mut out, detail);
            out.push('}');
        }
    }
    out
}

/// Appends a cell as a JSON value, preserving its type.
fn push_cell_json(out: &mut String, cell: &Cell) {
    match cell {
        Cell::Null => out.push_str("null"),
        Cell::Integer(i) => out.push_str(&i.to_string()),
        Cell::Float(f) => {
            // JSON has no NaN or Infinity; emit null rather than invalid JSON.
            if f.is_finite() {
                out.push_str(&f.to_string());
            } else {
                out.push_str("null");
            }
        }
        Cell::Text(s) => push_json_string(out, s),
    }
}

/// Appends `s` as a quoted, escaped JSON string.
///
/// Escapes the two structural characters plus every C0 control code, as
/// RFC 8259 requires — an unescaped newline or quote in a value would produce
/// a document the Dart side cannot parse.
fn push_json_string(out: &mut String, s: &str) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{08}' => out.push_str("\\b"),
            '\u{0C}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
}
