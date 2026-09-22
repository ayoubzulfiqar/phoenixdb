//! SQL executor: runs a parsed [`Statement`] against a [`Database`].
//!
//! # Design
//!
//! Direct interpretation of the AST, not a Volcano-style operator tree: the
//! supported statement set has no joins or subqueries, so an operator
//! pipeline would be indirection without benefit. Conditions are compiled
//! once per statement (column names resolved to positions, parameters bound)
//! and then evaluated per row with SQL's three-valued logic.
//!
//! # Transactions and concurrency
//!
//! Every statement reads *and* writes through one transaction, so its reads
//! come from the same snapshot its writes are validated against. That is what
//! makes concurrent statements safe: two `INSERT`s racing for the next row id,
//! or two `UPDATE`s of one row, write the same key, and the loser's commit
//! fails with a write-write conflict instead of silently overwriting the
//! winner. Autocommit statements (the [`Executor::run`] family) retry such a
//! conflict transparently; statements run inside a caller's transaction
//! ([`Executor::run_in`]) surface it, and the caller retries the transaction.
//!
//! A mutation validates everything — types, `NOT NULL`, primary-key
//! uniqueness — before it stages any write, so a failed statement leaves
//! nothing behind, even inside a caller's transaction.
//!
//! # Primary keys
//!
//! A `PRIMARY KEY` is enforced through a unique index (`catalog::pk_key`).
//! Because each key value maps to exactly one index entry, two transactions
//! inserting the same key also collide on that entry and one of them fails,
//! which a check-then-insert without the index could not guarantee under
//! snapshot isolation. Tables created before the index existed are indexed
//! on their first write. `WHERE pk = value` uses the index directly.

use crate::error::{Error, Result};
use crate::sql::ast::{AggFunc, ComparisonOp, Expr, OrderItem, SelectItem, Statement, Value};
use crate::sql::catalog::{
    Cell, Row, TableSchema, fold, pk_key, pk_prefix, pk_ready_key, row_id_of, row_key, row_prefix,
    schema_key, validate_name,
};
use crate::{Database, prefix_successor};
use std::collections::{HashMap, HashSet};
use std::ops::Bound;

/// How many times an autocommit statement is retried after a conflict.
const MAX_CONFLICT_RETRIES: usize = 32;

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

    /// Parses and executes `sql` in its own transaction.
    pub fn run(&self, sql: &str) -> Result<QueryResult> {
        self.run_with(sql, &[])
    }

    /// Parses and executes `sql` with bound parameters (`?`, `?N`), in its
    /// own transaction. Parameters are values, never SQL text, so they cannot
    /// change the statement's meaning.
    pub fn run_with(&self, sql: &str, params: &[Value]) -> Result<QueryResult> {
        let (stmt, needed) = crate::sql::parser::parse_with_params(sql)?;
        check_param_count(needed, params.len())?;
        self.execute_with(&stmt, params)
    }

    /// Parses and executes `sql` inside the caller's transaction `txn`.
    ///
    /// The statement is atomic within the transaction (a failure stages
    /// nothing); committing, rolling back and retrying on
    /// [`Error::Conflict`] are the caller's.
    pub fn run_in(&self, txn: u64, sql: &str, params: &[Value]) -> Result<QueryResult> {
        let (stmt, needed) = crate::sql::parser::parse_with_params(sql)?;
        check_param_count(needed, params.len())?;
        self.execute_in(txn, &stmt, params)
    }

    /// Executes an already-parsed statement in its own transaction.
    pub fn execute(&self, stmt: &Statement) -> Result<QueryResult> {
        self.execute_with(stmt, &[])
    }

    /// Executes an already-parsed statement with parameters, in its own
    /// transaction, retrying transparently on write-write conflicts.
    pub fn execute_with(&self, stmt: &Statement, params: &[Value]) -> Result<QueryResult> {
        if !stmt.is_mutation() {
            let txn = self.db.begin(true)?;
            let result = self.execute_in(txn, stmt, params);
            let _ = self.db.rollback(txn); // read-only: nothing to commit
            return result;
        }
        // Autocommit statements take turns, so they never conflict with each
        // other (a hot row-id counter would otherwise make concurrent INSERTs
        // retry each other into starvation). A residual conflict with a
        // non-SQL writer is retried after a short, jittered backoff.
        let _turn = self.db.serialize_writes();
        for attempt in 0..MAX_CONFLICT_RETRIES {
            if attempt > 0 {
                backoff(attempt);
            }
            let txn = self.db.begin(false)?;
            let result = match self.execute_in(txn, stmt, params) {
                Ok(result) => result,
                Err(e) => {
                    let _ = self.db.rollback(txn);
                    return Err(e);
                }
            };
            match self.db.commit(txn) {
                Ok(()) => return Ok(result),
                // The engine already aborted the loser; start over from a
                // fresh snapshot that includes the winner's write.
                Err(Error::Conflict) => continue,
                Err(e) => {
                    let _ = self.db.rollback(txn);
                    return Err(e);
                }
            }
        }
        Err(Error::Conflict)
    }

    /// Executes an already-parsed statement inside transaction `txn`.
    pub fn execute_in(&self, txn: u64, stmt: &Statement, params: &[Value]) -> Result<QueryResult> {
        let ctx = Ctx {
            db: self.db,
            txn,
            params,
        };
        match stmt {
            Statement::CreateTable {
                table,
                columns,
                if_not_exists,
            } => ctx.create_table(table, columns, *if_not_exists),
            Statement::DropTable { table, if_exists } => ctx.drop_table(table, *if_exists),
            Statement::Insert {
                table,
                columns,
                rows,
            } => ctx.insert(table, columns, rows),
            Statement::Select {
                table,
                items,
                filter,
                group_by,
                order_by,
                limit,
                offset,
            } => ctx.select(&SelectSpec {
                table,
                items,
                filter: filter.as_ref(),
                group_by,
                order_by,
                limit: *limit,
                offset: offset.unwrap_or(0),
            }),
            Statement::Update {
                table,
                assignments,
                filter,
            } => ctx.update(table, assignments, filter.as_ref()),
            Statement::Delete { table, filter } => ctx.delete(table, filter.as_ref()),
        }
    }
}

/// Sleeps a little longer on each retry, with jitter so racing writers
/// spread out instead of colliding again in lockstep.
fn backoff(attempt: usize) {
    let jitter = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.subsec_nanos() as u64 % 50);
    let micros = (10u64 << attempt.min(10)).min(5_000) + jitter;
    std::thread::sleep(std::time::Duration::from_micros(micros));
}

fn check_param_count(needed: usize, supplied: usize) -> Result<()> {
    if needed != supplied {
        return Err(Error::invalid(format!(
            "statement uses {needed} parameter(s) but {supplied} were supplied"
        )));
    }
    Ok(())
}

/// Borrowed pieces of a `SELECT`.
struct SelectSpec<'s> {
    table: &'s str,
    items: &'s [SelectItem],
    filter: Option<&'s Expr>,
    group_by: &'s [String],
    order_by: &'s [OrderItem],
    limit: Option<usize>,
    offset: usize,
}

/// One statement's execution context.
struct Ctx<'a> {
    db: &'a Database,
    txn: u64,
    params: &'a [Value],
}

impl Ctx<'_> {
    // ---- storage through the transaction -----------------------------------

    fn get(&self, key: &[u8]) -> Result<Option<Vec<u8>>> {
        match self.db.get(self.txn, key) {
            Ok(v) => Ok(Some(v)),
            Err(Error::NotFound) => Ok(None),
            Err(e) => Err(e),
        }
    }

    fn put(&self, key: &[u8], value: &[u8]) -> Result<()> {
        self.db.insert(self.txn, key, value)
    }

    fn remove(&self, key: &[u8]) -> Result<()> {
        match self.db.delete(self.txn, key) {
            Ok(()) | Err(Error::NotFound) => Ok(()),
            Err(e) => Err(e),
        }
    }

    /// Every `(key, value)` under `prefix`, via this transaction's snapshot.
    fn scan_prefix(&self, prefix: &[u8]) -> Result<Vec<(Vec<u8>, Vec<u8>)>> {
        let end = prefix_successor(prefix);
        let hi = match &end {
            Some(e) => Bound::Excluded(e.as_slice()),
            None => Bound::Unbounded,
        };
        let mut out = Vec::new();
        self.db
            .scan_txn(self.txn, Bound::Included(prefix), hi, |k, v| {
                out.push((k, v));
                Ok(true)
            })?;
        Ok(out)
    }

    fn load_schema(&self, table: &str) -> Result<TableSchema> {
        match self.get(&schema_key(table))? {
            Some(bytes) => TableSchema::decode(&bytes),
            None => Err(Error::invalid(format!("no such table: `{table}`"))),
        }
    }

    fn table_exists(&self, table: &str) -> Result<bool> {
        Ok(self.get(&schema_key(table))?.is_some())
    }

    /// Every row of `table`, in insertion order.
    fn scan_rows(&self, table: &str) -> Result<Vec<(Vec<u8>, Row)>> {
        let mut out = Vec::new();
        for (key, value) in self.scan_prefix(&row_prefix(table))? {
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

    /// Whether `table`'s primary-key index is complete (built at creation,
    /// or backfilled for a table from before the index existed).
    fn pk_index_ready(&self, table: &str) -> Result<bool> {
        Ok(self.get(&pk_ready_key(table))?.is_some())
    }

    /// Builds the primary-key index for a legacy table, inside this
    /// transaction. Fails if stored rows already violate uniqueness.
    fn ensure_pk_index(&self, schema: &TableSchema) -> Result<()> {
        let Some(pk) = schema.primary_key else {
            return Ok(());
        };
        if self.pk_index_ready(&schema.name)? {
            return Ok(());
        }
        let mut seen = HashSet::new();
        let mut entries = Vec::new();
        for (key, row) in self.scan_rows(&schema.name)? {
            let index_key = pk_key(&schema.name, &row.cells[pk])?;
            if !seen.insert(index_key.clone()) {
                return Err(Error::invalid(format!(
                    "table `{}` already holds duplicate {} values, so its PRIMARY KEY \
                     cannot be enforced; remove the duplicates first",
                    schema.name, schema.columns[pk]
                )));
            }
            let id = row_id_of(&schema.name, &key).expect("validated by scan_rows");
            entries.push((index_key, id));
        }
        for (index_key, id) in entries {
            self.put(&index_key, &id.to_be_bytes())?;
        }
        self.put(&pk_ready_key(&schema.name), &[1])
    }

    // ---- statements -------------------------------------------------------

    fn create_table(
        &self,
        table: &str,
        columns: &[crate::sql::ast::ColumnDef],
        if_not_exists: bool,
    ) -> Result<QueryResult> {
        if self.table_exists(table)? {
            if if_not_exists {
                return Ok(QueryResult::SchemaChanged {
                    detail: format!("table `{table}` already exists, skipped"),
                });
            }
            return Err(Error::invalid(format!("table `{table}` already exists")));
        }
        let schema = TableSchema::from_defs(table, columns)?;
        self.put(&schema_key(&schema.name), &schema.encode()?)?;
        if schema.primary_key.is_some() {
            // An empty table's index is trivially complete.
            self.put(&pk_ready_key(table), &[1])?;
        }
        Ok(QueryResult::SchemaChanged {
            detail: format!("table `{table}` created with {} column(s)", schema.width()),
        })
    }

    fn drop_table(&self, table: &str, if_exists: bool) -> Result<QueryResult> {
        validate_name(table, "table")?;
        if !self.table_exists(table)? {
            if if_exists {
                return Ok(QueryResult::SchemaChanged {
                    detail: format!("table `{table}` does not exist, skipped"),
                });
            }
            return Err(Error::invalid(format!("no such table: `{table}`")));
        }
        let rows = self.scan_rows(table)?;
        let index = self.scan_prefix(&pk_prefix(table))?;
        for (key, _) in &rows {
            self.remove(key)?;
        }
        for (key, _) in &index {
            self.remove(key)?;
        }
        self.remove(&pk_ready_key(table))?;
        self.remove(&schema_key(table))?;
        Ok(QueryResult::SchemaChanged {
            detail: format!("table `{table}` dropped ({} row(s))", rows.len()),
        })
    }

    fn insert(&self, table: &str, columns: &[String], rows: &[Vec<Expr>]) -> Result<QueryResult> {
        let mut schema = self.load_schema(table)?;
        let positions = resolve_columns(&schema, columns)?;
        self.ensure_pk_index(&schema)?;

        // Validate and encode everything first: nothing is staged unless the
        // whole statement is valid.
        let mut new_keys: HashSet<Vec<u8>> = HashSet::new();
        let mut writes: Vec<(Vec<u8>, Vec<u8>)> = Vec::with_capacity(rows.len() * 2 + 1);
        for row in rows {
            if row.len() != positions.len() {
                return Err(Error::invalid(format!(
                    "row supplies {} value(s) but {} column(s) were named",
                    row.len(),
                    positions.len()
                )));
            }
            let mut cells = vec![Cell::Null; schema.width()];
            for (slot, expr) in positions.iter().zip(row) {
                cells[*slot] = self.value_of(expr)?;
            }
            check_not_null(&schema, &cells)?;
            let id = schema.next_row_id;
            schema.next_row_id += 1;
            if let Some(pk) = schema.primary_key {
                let index_key = pk_key(table, &cells[pk])?;
                if !new_keys.insert(index_key.clone()) || self.get(&index_key)?.is_some() {
                    return Err(duplicate_key(&schema, &cells[pk]));
                }
                writes.push((index_key, id.to_be_bytes().to_vec()));
            }
            writes.push((row_key(table, id), Row::new(cells).encode()?));
        }
        // The row-id counter moves with the rows, so a failure cannot leave it
        // ahead of them — and two concurrent inserts collide on it.
        writes.push((schema_key(table), schema.encode()?));
        for (key, value) in &writes {
            self.put(key, value)?;
        }
        Ok(QueryResult::Affected {
            count: rows.len() as u64,
        })
    }

    fn update(
        &self,
        table: &str,
        assignments: &[(String, Expr)],
        filter: Option<&Expr>,
    ) -> Result<QueryResult> {
        let schema = self.load_schema(table)?;
        let mut targets: Vec<(usize, Cell)> = Vec::with_capacity(assignments.len());
        for (column, expr) in assignments {
            let idx = schema.column_index(column)?;
            let cell = self.value_of(expr)?;
            if cell == Cell::Null && schema.not_null.contains(&idx) {
                return Err(Error::invalid(format!(
                    "column `{}` is NOT NULL and cannot be set to NULL",
                    schema.columns[idx]
                )));
            }
            targets.push((idx, cell));
        }
        let condition = self.compile_filter(&schema, filter)?;
        self.ensure_pk_index(&schema)?;
        let matched: Vec<(Vec<u8>, Row)> = self
            .candidate_rows(&schema, filter)?
            .into_iter()
            .filter(|(_, row)| {
                condition
                    .as_ref()
                    .is_none_or(|c| c.eval(&row.cells) == Some(true))
            })
            .collect();

        let pk_target = schema.primary_key.and_then(|pk| {
            targets
                .iter()
                .find(|(idx, _)| *idx == pk)
                .map(|(_, c)| c.clone())
        });
        let mut writes: Vec<(Vec<u8>, Option<Vec<u8>>)> = Vec::new();
        if let (Some(pk), Some(new_value)) = (schema.primary_key, &pk_target) {
            let new_key = pk_key(table, new_value)?;
            let changing: Vec<&(Vec<u8>, Row)> = matched
                .iter()
                .filter(|(_, row)| pk_key(table, &row.cells[pk]).ok().as_ref() != Some(&new_key))
                .collect();
            if matched.len() > 1 && !changing.is_empty() {
                return Err(duplicate_key(&schema, new_value));
            }
            if let Some((row_k, row)) = changing.first() {
                if self.get(&new_key)?.is_some() {
                    return Err(duplicate_key(&schema, new_value));
                }
                let id = row_id_of(table, row_k).expect("validated by scan_rows");
                writes.push((pk_key(table, &row.cells[pk])?, None));
                writes.push((new_key, Some(id.to_be_bytes().to_vec())));
            }
        }
        for (key, mut row) in matched.iter().cloned() {
            for (idx, cell) in &targets {
                row.cells[*idx] = cell.clone();
            }
            writes.push((key, Some(row.encode()?)));
        }
        for (key, value) in &writes {
            match value {
                Some(v) => self.put(key, v)?,
                None => self.remove(key)?,
            }
        }
        Ok(QueryResult::Affected {
            count: matched.len() as u64,
        })
    }

    fn delete(&self, table: &str, filter: Option<&Expr>) -> Result<QueryResult> {
        let schema = self.load_schema(table)?;
        let condition = self.compile_filter(&schema, filter)?;
        self.ensure_pk_index(&schema)?;
        let mut victims = Vec::new();
        for (key, row) in self.candidate_rows(&schema, filter)? {
            if condition
                .as_ref()
                .is_none_or(|c| c.eval(&row.cells) == Some(true))
            {
                if let Some(pk) = schema.primary_key {
                    victims.push(pk_key(table, &row.cells[pk])?);
                }
                victims.push(key);
            }
        }
        let rows = if schema.primary_key.is_some() {
            victims.len() / 2
        } else {
            victims.len()
        };
        for key in &victims {
            self.remove(key)?;
        }
        Ok(QueryResult::Affected { count: rows as u64 })
    }

    fn select(&self, spec: &SelectSpec<'_>) -> Result<QueryResult> {
        let schema = self.load_schema(spec.table)?;
        let condition = self.compile_filter(&schema, spec.filter)?;
        for column in spec.group_by {
            schema.column_index(column)?;
        }
        let aggregated = !spec.group_by.is_empty()
            || spec
                .items
                .iter()
                .any(|i| matches!(i, SelectItem::Aggregate { .. }));

        let mut rows: Vec<Row> = self
            .candidate_rows(&schema, spec.filter)?
            .into_iter()
            .map(|(_, row)| row)
            .filter(|row| {
                condition
                    .as_ref()
                    .is_none_or(|c| c.eval(&row.cells) == Some(true))
            })
            .collect();

        let (columns, mut output) = if aggregated {
            aggregate(&schema, spec.items, spec.group_by, &rows)?
        } else {
            // Plain projection. ORDER BY may name any table column (or an
            // output alias), so rows are sorted before projecting.
            let projection = project(&schema, spec.items)?;
            let keys = order_keys_for_rows(&schema, &projection, spec.order_by)?;
            sort_by_keys(&mut rows, &keys, |row, idx| &row.cells[idx]);
            let names = projection.iter().map(|(_, name)| name.clone()).collect();
            let out = rows
                .into_iter()
                .map(|r| {
                    projection
                        .iter()
                        .map(|(i, _)| r.cells[*i].clone())
                        .collect()
                })
                .collect();
            (names, out)
        };
        if aggregated && !spec.order_by.is_empty() {
            let keys = order_keys_for_output(&columns, spec.order_by)?;
            sort_by_keys(&mut output, &keys, |row: &Vec<Cell>, idx| &row[idx]);
        }
        let output: Vec<Vec<Cell>> = output
            .into_iter()
            .skip(spec.offset)
            .take(spec.limit.unwrap_or(usize::MAX))
            .collect();
        Ok(QueryResult::Rows {
            columns,
            rows: output,
        })
    }

    /// Rows that can match `filter`: one index lookup for `pk = value`, a
    /// scan of the table otherwise.
    fn candidate_rows(
        &self,
        schema: &TableSchema,
        filter: Option<&Expr>,
    ) -> Result<Vec<(Vec<u8>, Row)>> {
        if let (Some(pk), Some(Expr::Compare { left, op, right })) = (schema.primary_key, filter)
            && *op == ComparisonOp::Eq
            && self.pk_index_ready(&schema.name)?
        {
            let value = match (left.as_ref(), right.as_ref()) {
                (Expr::Column(c), v) | (v, Expr::Column(c))
                    if schema.column_index(c)? == pk && !matches!(v, Expr::Column(_)) =>
                {
                    Some(self.value_of(v)?)
                }
                _ => None,
            };
            if let Some(value) = value {
                if value == Cell::Null {
                    return Ok(Vec::new());
                }
                let Some(id) = self.get(&pk_key(&schema.name, &value)?)? else {
                    return Ok(Vec::new());
                };
                let id = u64::from_be_bytes(
                    id.as_slice()
                        .try_into()
                        .map_err(|_| Error::corrupt("malformed primary-key index entry"))?,
                );
                let key = row_key(&schema.name, id);
                return Ok(match self.get(&key)? {
                    Some(bytes) => vec![(key, Row::decode(&bytes)?)],
                    None => Vec::new(),
                });
            }
        }
        self.scan_rows(&schema.name)
    }

    // ---- expressions --------------------------------------------------------

    /// Evaluates a literal or parameter.
    fn value_of(&self, expr: &Expr) -> Result<Cell> {
        match expr {
            Expr::Literal(v) => Ok(Cell::from(v)),
            Expr::Param(i) => self
                .params
                .get(*i)
                .map(Cell::from)
                .ok_or_else(|| Error::invalid(format!("parameter ?{} was not supplied", i + 1))),
            other => Err(Error::invalid(format!(
                "expected a literal or parameter, found {other:?}"
            ))),
        }
    }

    fn compile_filter(&self, schema: &TableSchema, filter: Option<&Expr>) -> Result<Option<Cond>> {
        filter.map(|f| self.compile(schema, f)).transpose()
    }

    /// Resolves column names and binds parameters. Validation happens here,
    /// before any row is read, so `WHERE nmae = 'x'` is an error even on an
    /// empty table instead of a silent "no matches".
    fn compile(&self, schema: &TableSchema, expr: &Expr) -> Result<Cond> {
        let operand = |e: &Expr| -> Result<Operand> {
            match e {
                Expr::Column(c) => Ok(Operand::Column(schema.column_index(c)?)),
                other => Ok(Operand::Value(self.value_of(other)?)),
            }
        };
        Ok(match expr {
            Expr::Compare { left, op, right } => {
                Cond::Compare(operand(left)?, *op, operand(right)?)
            }
            Expr::And(a, b) => Cond::And(
                Box::new(self.compile(schema, a)?),
                Box::new(self.compile(schema, b)?),
            ),
            Expr::Or(a, b) => Cond::Or(
                Box::new(self.compile(schema, a)?),
                Box::new(self.compile(schema, b)?),
            ),
            Expr::Not(e) => Cond::Not(Box::new(self.compile(schema, e)?)),
            Expr::IsNull { expr, negated } => Cond::IsNull(operand(expr)?, *negated),
            Expr::InList {
                expr,
                list,
                negated,
            } => Cond::In(
                operand(expr)?,
                list.iter().map(operand).collect::<Result<_>>()?,
                *negated,
            ),
            Expr::Between {
                expr,
                low,
                high,
                negated,
            } => Cond::Between(operand(expr)?, operand(low)?, operand(high)?, *negated),
            Expr::Like {
                expr,
                pattern,
                negated,
                case_insensitive,
            } => Cond::Like(
                operand(expr)?,
                operand(pattern)?,
                *negated,
                *case_insensitive,
            ),
            Expr::Literal(_) | Expr::Column(_) | Expr::Param(_) => {
                return Err(Error::invalid(
                    "a condition needs a comparison (e.g. `active = 1`), not a bare value",
                ));
            }
        })
    }
}

// ---------------------------------------------------------------------------
// Compiled conditions and three-valued logic
// ---------------------------------------------------------------------------

/// An operand with its column resolved or its value bound.
#[derive(Debug, Clone)]
enum Operand {
    Column(usize),
    Value(Cell),
}

impl Operand {
    fn get<'c>(&'c self, row: &'c [Cell]) -> &'c Cell {
        match self {
            Operand::Column(i) => &row[*i],
            Operand::Value(v) => v,
        }
    }
}

/// A compiled condition. `eval` returns `Some(true)`, `Some(false)` or `None`
/// (SQL's UNKNOWN); a `WHERE` keeps only `Some(true)`.
#[derive(Debug, Clone)]
enum Cond {
    Compare(Operand, ComparisonOp, Operand),
    And(Box<Cond>, Box<Cond>),
    Or(Box<Cond>, Box<Cond>),
    Not(Box<Cond>),
    IsNull(Operand, bool),
    In(Operand, Vec<Operand>, bool),
    Between(Operand, Operand, Operand, bool),
    Like(Operand, Operand, bool, bool),
}

fn compare(a: &Cell, op: ComparisonOp, b: &Cell) -> Option<bool> {
    use std::cmp::Ordering::*;
    // NULL, or text against a number: unknown.
    let ord = a.partial_compare(b)?;
    Some(match op {
        ComparisonOp::Eq => ord == Equal,
        ComparisonOp::NotEq => ord != Equal,
        ComparisonOp::Lt => ord == Less,
        ComparisonOp::LtEq => ord != Greater,
        ComparisonOp::Gt => ord == Greater,
        ComparisonOp::GtEq => ord != Less,
    })
}

fn negate(v: Option<bool>, negated: bool) -> Option<bool> {
    if negated { v.map(|b| !b) } else { v }
}

impl Cond {
    fn eval(&self, row: &[Cell]) -> Option<bool> {
        match self {
            Cond::Compare(a, op, b) => compare(a.get(row), *op, b.get(row)),
            Cond::And(a, b) => match (a.eval(row), b.eval(row)) {
                (Some(false), _) | (_, Some(false)) => Some(false),
                (Some(true), Some(true)) => Some(true),
                _ => None,
            },
            Cond::Or(a, b) => match (a.eval(row), b.eval(row)) {
                (Some(true), _) | (_, Some(true)) => Some(true),
                (Some(false), Some(false)) => Some(false),
                _ => None,
            },
            Cond::Not(c) => c.eval(row).map(|b| !b),
            Cond::IsNull(a, negated) => Some((*a.get(row) == Cell::Null) != *negated),
            Cond::In(a, list, negated) => {
                let value = a.get(row);
                let mut saw_unknown = false;
                for candidate in list {
                    match compare(value, ComparisonOp::Eq, candidate.get(row)) {
                        Some(true) => return negate(Some(true), *negated),
                        Some(false) => {}
                        None => saw_unknown = true,
                    }
                }
                negate(if saw_unknown { None } else { Some(false) }, *negated)
            }
            Cond::Between(a, low, high, negated) => {
                let v = a.get(row);
                let ge = compare(v, ComparisonOp::GtEq, low.get(row));
                let le = compare(v, ComparisonOp::LtEq, high.get(row));
                let both = match (ge, le) {
                    (Some(false), _) | (_, Some(false)) => Some(false),
                    (Some(true), Some(true)) => Some(true),
                    _ => None,
                };
                negate(both, *negated)
            }
            Cond::Like(a, pattern, negated, ci) => {
                let (value, pattern) = (a.get(row), pattern.get(row));
                let (Some(value), Some(pattern)) = (like_text(value), like_text(pattern)) else {
                    return None;
                };
                let matched = if *ci {
                    like_match(&value.to_lowercase(), &pattern.to_lowercase())
                } else {
                    like_match(&value, &pattern)
                };
                negate(Some(matched), *negated)
            }
        }
    }
}

/// Text form of a cell for `LIKE` (numbers match by their text), `None` for
/// `NULL`.
fn like_text(cell: &Cell) -> Option<String> {
    match cell {
        Cell::Null => None,
        Cell::Text(s) => Some(s.clone()),
        other => Some(other.display()),
    }
}

/// SQL `LIKE`: `%` matches any run of characters, `_` exactly one. Iterative
/// with single backtracking, so it is `O(len(value) * len(pattern))` at worst
/// and cannot blow the stack.
fn like_match(value: &str, pattern: &str) -> bool {
    let v: Vec<char> = value.chars().collect();
    let p: Vec<char> = pattern.chars().collect();
    let (mut i, mut j) = (0usize, 0usize);
    let mut star: Option<usize> = None;
    let mut resume = 0usize;
    while i < v.len() {
        if j < p.len() && (p[j] == '_' || (p[j] != '%' && p[j] == v[i])) {
            i += 1;
            j += 1;
        } else if j < p.len() && p[j] == '%' {
            star = Some(j);
            resume = i;
            j += 1;
        } else if let Some(s) = star {
            j = s + 1;
            resume += 1;
            i = resume;
        } else {
            return false;
        }
    }
    while j < p.len() && p[j] == '%' {
        j += 1;
    }
    j == p.len()
}

// ---------------------------------------------------------------------------
// Projection, ordering and aggregation
// ---------------------------------------------------------------------------

fn resolve_columns(schema: &TableSchema, columns: &[String]) -> Result<Vec<usize>> {
    if columns.is_empty() {
        Ok((0..schema.width()).collect())
    } else {
        columns.iter().map(|c| schema.column_index(c)).collect()
    }
}

fn check_not_null(schema: &TableSchema, cells: &[Cell]) -> Result<()> {
    for &idx in &schema.not_null {
        if cells[idx] == Cell::Null {
            return Err(Error::invalid(format!(
                "column `{}` is NOT NULL but no value was supplied",
                schema.columns[idx]
            )));
        }
    }
    Ok(())
}

fn duplicate_key(schema: &TableSchema, value: &Cell) -> Error {
    let column = schema
        .primary_key
        .map_or("primary key", |pk| schema.columns[pk].as_str());
    Error::invalid(format!(
        "duplicate PRIMARY KEY {column} = {} in table `{}`",
        value.display(),
        schema.name
    ))
}

/// `(table column index, output name)` for a non-aggregate select list.
fn project(schema: &TableSchema, items: &[SelectItem]) -> Result<Vec<(usize, String)>> {
    let mut out = Vec::new();
    for item in items {
        match item {
            SelectItem::Wildcard => {
                out.extend(schema.columns.iter().cloned().enumerate());
            }
            SelectItem::Column { name, alias } => {
                let idx = schema.column_index(name)?;
                out.push((
                    idx,
                    alias.clone().unwrap_or_else(|| schema.columns[idx].clone()),
                ));
            }
            SelectItem::Aggregate { .. } => unreachable!("handled by aggregate()"),
        }
    }
    Ok(out)
}

/// ORDER BY for a plain select: each key names an output alias or a table
/// column, resolved to a table column index.
fn order_keys_for_rows(
    schema: &TableSchema,
    projection: &[(usize, String)],
    order_by: &[OrderItem],
) -> Result<Vec<(usize, bool)>> {
    order_by
        .iter()
        .map(|item| {
            let wanted = fold(&item.column);
            let idx = match projection.iter().find(|(_, name)| fold(name) == wanted) {
                Some((idx, _)) => *idx,
                None => schema.column_index(&item.column)?,
            };
            Ok((idx, item.desc))
        })
        .collect()
}

/// ORDER BY for an aggregate select: keys must name output columns.
fn order_keys_for_output(columns: &[String], order_by: &[OrderItem]) -> Result<Vec<(usize, bool)>> {
    order_by
        .iter()
        .map(|item| {
            let wanted = fold(&item.column);
            columns
                .iter()
                .position(|c| fold(c) == wanted)
                .map(|idx| (idx, item.desc))
                .ok_or_else(|| {
                    Error::invalid(format!(
                        "ORDER BY `{}` must name a selected column (have: {})",
                        item.column,
                        columns.join(", ")
                    ))
                })
        })
        .collect()
}

/// Stable multi-key sort under [`Cell::total_cmp`]. Ascending puts `NULL`
/// last and descending puts it first (NULL counts as the largest value, as
/// in PostgreSQL).
fn sort_by_keys<T>(rows: &mut [T], keys: &[(usize, bool)], cell: impl Fn(&T, usize) -> &Cell) {
    if keys.is_empty() {
        return;
    }
    rows.sort_by(|a, b| {
        for &(idx, desc) in keys {
            let ord = cell(a, idx).total_cmp(cell(b, idx));
            let ord = if desc { ord.reverse() } else { ord };
            if ord != std::cmp::Ordering::Equal {
                return ord;
            }
        }
        std::cmp::Ordering::Equal
    });
}

/// Running state of one aggregate.
#[derive(Debug, Clone)]
struct AggState {
    func: AggFunc,
    column: Option<usize>,
    distinct: Option<HashSet<Vec<u8>>>,
    count: u64,
    int_sum: i128,
    float_sum: f64,
    saw_float: bool,
    extreme: Option<Cell>,
}

impl AggState {
    fn new(func: AggFunc, column: Option<usize>, distinct: bool) -> Self {
        AggState {
            func,
            column,
            distinct: distinct.then(HashSet::new),
            count: 0,
            int_sum: 0,
            float_sum: 0.0,
            saw_float: false,
            extreme: None,
        }
    }

    fn feed(&mut self, row: &[Cell], name: &str) -> Result<()> {
        let Some(idx) = self.column else {
            self.count += 1; // COUNT(*)
            return Ok(());
        };
        let cell = &row[idx];
        if *cell == Cell::Null {
            return Ok(()); // aggregates skip NULLs
        }
        if let Some(seen) = &mut self.distinct
            && !seen.insert(cell_key(cell))
        {
            return Ok(());
        }
        self.count += 1;
        match self.func {
            AggFunc::Count => {}
            AggFunc::Sum | AggFunc::Avg => match cell {
                Cell::Integer(i) => self.int_sum += i128::from(*i),
                Cell::Float(f) => {
                    self.float_sum += f;
                    self.saw_float = true;
                }
                _ => {
                    return Err(Error::invalid(format!(
                        "{}({name}) needs numbers, found text",
                        self.func.name().to_uppercase()
                    )));
                }
            },
            AggFunc::Min | AggFunc::Max => {
                let better = match &self.extreme {
                    None => true,
                    Some(current) => {
                        let ord = cell.total_cmp(current);
                        if self.func == AggFunc::Min {
                            ord == std::cmp::Ordering::Less
                        } else {
                            ord == std::cmp::Ordering::Greater
                        }
                    }
                };
                if better {
                    self.extreme = Some(cell.clone());
                }
            }
        }
        Ok(())
    }

    fn finish(&self) -> Cell {
        match self.func {
            AggFunc::Count => Cell::Integer(self.count as i64),
            _ if self.count == 0 => Cell::Null,
            AggFunc::Sum => {
                if self.saw_float {
                    Cell::Float(self.int_sum as f64 + self.float_sum)
                } else {
                    i64::try_from(self.int_sum)
                        .map_or_else(|_| Cell::Float(self.int_sum as f64), Cell::Integer)
                }
            }
            AggFunc::Avg => Cell::Float((self.int_sum as f64 + self.float_sum) / self.count as f64),
            AggFunc::Min | AggFunc::Max => self.extreme.clone().unwrap_or(Cell::Null),
        }
    }
}

/// Grouping/equality key for a cell (numerically equal values collide).
fn cell_key(cell: &Cell) -> Vec<u8> {
    match cell {
        Cell::Null => vec![0],
        other => pk_key("", other).unwrap_or_default(),
    }
}

/// Computes an aggregate select: `GROUP BY` groups (or one group for the
/// whole table) and the select list's aggregates per group.
fn aggregate(
    schema: &TableSchema,
    items: &[SelectItem],
    group_by: &[String],
    rows: &[Row],
) -> Result<(Vec<String>, Vec<Vec<Cell>>)> {
    let group_idx: Vec<usize> = group_by
        .iter()
        .map(|c| schema.column_index(c))
        .collect::<Result<_>>()?;

    // Output plan: each item is a group column or an aggregate.
    enum Out {
        Group(usize), // position within group_idx
        Agg(usize),   // position within the aggregate list
    }
    let mut plan = Vec::new();
    let mut names = Vec::new();
    let mut aggs: Vec<(AggFunc, Option<usize>, bool, String)> = Vec::new();
    for item in items {
        match item {
            SelectItem::Wildcard => {
                return Err(Error::invalid(
                    "`*` cannot be combined with aggregates or GROUP BY",
                ));
            }
            SelectItem::Column { name, alias } => {
                let idx = schema.column_index(name)?;
                let Some(pos) = group_idx.iter().position(|g| *g == idx) else {
                    return Err(Error::invalid(format!(
                        "column `{name}` must appear in GROUP BY or be used in an aggregate"
                    )));
                };
                plan.push(Out::Group(pos));
                names.push(alias.clone().unwrap_or_else(|| schema.columns[idx].clone()));
            }
            SelectItem::Aggregate {
                func,
                column,
                distinct,
                alias,
            } => {
                let col_idx = column
                    .as_ref()
                    .map(|c| schema.column_index(c))
                    .transpose()?;
                let label = match column {
                    Some(c) if *distinct => format!("{}(distinct {c})", func.name()),
                    Some(c) => format!("{}({c})", func.name()),
                    None => format!("{}(*)", func.name()),
                };
                plan.push(Out::Agg(aggs.len()));
                names.push(alias.clone().unwrap_or_else(|| label.clone()));
                aggs.push((*func, col_idx, *distinct, label));
            }
        }
    }

    let fresh = || -> Vec<AggState> {
        aggs.iter()
            .map(|(f, c, d, _)| AggState::new(*f, *c, *d))
            .collect()
    };
    let mut groups: Vec<(Vec<Cell>, Vec<AggState>)> = Vec::new();
    let mut by_key: HashMap<Vec<u8>, usize> = HashMap::new();
    for row in rows {
        let values: Vec<Cell> = group_idx.iter().map(|i| row.cells[*i].clone()).collect();
        let key: Vec<u8> = values
            .iter()
            .flat_map(|c| {
                let mut k = cell_key(c);
                k.push(0xFF); // separator
                k
            })
            .collect();
        let slot = *by_key.entry(key).or_insert_with(|| {
            groups.push((values, fresh()));
            groups.len() - 1
        });
        for (state, (_, _, _, label)) in groups[slot].1.iter_mut().zip(&aggs) {
            state.feed(&row.cells, label)?;
        }
    }
    // An aggregate over no rows (and no GROUP BY) still yields one row.
    if groups.is_empty() && group_idx.is_empty() {
        groups.push((Vec::new(), fresh()));
    }
    // Deterministic group order: by the group values.
    groups.sort_by(|a, b| {
        for (x, y) in a.0.iter().zip(&b.0) {
            let ord = x.total_cmp(y);
            if ord != std::cmp::Ordering::Equal {
                return ord;
            }
        }
        std::cmp::Ordering::Equal
    });

    let out = groups
        .into_iter()
        .map(|(values, states)| {
            plan.iter()
                .map(|o| match o {
                    Out::Group(pos) => values[*pos].clone(),
                    Out::Agg(pos) => states[*pos].finish(),
                })
                .collect()
        })
        .collect();
    Ok((names, out))
}

// ---------------------------------------------------------------------------
// Parameters and JSON
// ---------------------------------------------------------------------------

/// Decodes bound parameters from a JSON array of scalars: `null`, booleans
/// (as 0/1), numbers (integers stay integers) and strings.
pub fn params_from_json(json: &str) -> Result<Vec<Value>> {
    let parsed: serde_json::Value = serde_json::from_str(json)
        .map_err(|e| Error::invalid(format!("parameters are not valid JSON: {e}")))?;
    let serde_json::Value::Array(items) = parsed else {
        return Err(Error::invalid("parameters must be a JSON array"));
    };
    items
        .into_iter()
        .enumerate()
        .map(|(i, item)| match item {
            serde_json::Value::Null => Ok(Value::Null),
            serde_json::Value::Bool(b) => Ok(Value::Integer(i64::from(b))),
            serde_json::Value::Number(n) => {
                if let Some(i) = n.as_i64() {
                    Ok(Value::Integer(i))
                } else {
                    n.as_f64()
                        .filter(|f| f.is_finite())
                        .map(Value::Float)
                        .ok_or_else(|| {
                            Error::invalid(format!("parameter {} is out of range", i + 1))
                        })
                }
            }
            serde_json::Value::String(s) => Ok(Value::Text(s)),
            _ => Err(Error::invalid(format!(
                "parameter {} must be null, a boolean, a number or a string",
                i + 1
            ))),
        })
        .collect()
}

/// Renders a [`QueryResult`] as a JSON document for the FFI boundary.
///
/// The output shape is documented on `phoenix_sql_query`.
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
            // `{:?}` always keeps a fraction or exponent (`3.0`, `1e20`), so a
            // float never reaches Dart as an int. JSON has no NaN/Infinity.
            if f.is_finite() {
                out.push_str(&format!("{f:?}"));
            } else {
                out.push_str("null");
            }
        }
        Cell::Text(s) => push_json_string(out, s),
    }
}

/// Appends `s` as a quoted, escaped JSON string (RFC 8259).
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn like_matches_wildcards() {
        assert!(like_match("hello", "h%o"));
        assert!(like_match("hello", "_ello"));
        assert!(like_match("hello", "%"));
        assert!(like_match("", "%"));
        assert!(!like_match("hello", "h_o"));
        assert!(like_match("abcabc", "%bc%bc"));
        assert!(!like_match("abc", "abcd"));
        assert!(like_match("a%b", "a%b"));
        assert!(like_match("日本語", "日_語"));
    }

    #[test]
    fn params_decode_every_scalar_type() {
        let p = params_from_json(r#"[null, true, 7, -2.5, "x", 9007199254740993]"#).unwrap();
        assert_eq!(
            p,
            vec![
                Value::Null,
                Value::Integer(1),
                Value::Integer(7),
                Value::Float(-2.5),
                Value::Text("x".into()),
                Value::Integer(9_007_199_254_740_993),
            ]
        );
        assert!(params_from_json("{}").is_err());
        assert!(params_from_json("[[1]]").is_err());
        assert!(params_from_json("[1,").is_err());
    }

    #[test]
    fn floats_keep_their_type_in_json() {
        let r = QueryResult::Rows {
            columns: vec!["f".into()],
            rows: vec![
                vec![Cell::Float(3.0)],
                vec![Cell::Float(-0.0)],
                vec![Cell::Float(1e20)],
            ],
        };
        assert_eq!(
            result_to_json(&r),
            r#"{"type":"rows","columns":["f"],"rows":[[3.0],[-0.0],[1e20]]}"#
        );
    }
}
