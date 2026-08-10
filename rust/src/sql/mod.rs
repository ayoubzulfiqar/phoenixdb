//! Minimal SQL layer: lexer, AST, parser.
//!
//! # Crate substitution
//!
//! The directive specified `sqlparser-rs`. It pulls in `stacker`, which needs a
//! C compiler; that breaks Flutter/Android cross-compilation and does not build
//! on this host at all (verified: `cargo add sqlparser` then build fails with
//! `failed to find tool "gcc.exe"`). This module is a hand-written replacement
//! with no non-Rust dependencies.
//!
//! # Prior art
//!
//! MagnumDB <https://github.com/sohamdev77/MagnumDB> (MIT, © 2026 MagnumDB
//! Contributors) demonstrated that a hand-rolled SQL front end is practical for
//! an embedded engine, and its `Statement` enum informed the shape of
//! [`ast::Statement`].
//!
//! Its parser was evaluated and **not** adopted. It dispatches with
//! `sql.to_uppercase().starts_with("CREATE TABLE")` and splits on single space
//! characters, so measured against it:
//!
//! ```text
//!   "CREATE  TABLE t (id)"   -> Syntax Error   (two spaces)
//!   "CREATE\nTABLE t (id)"   -> Syntax Error   (newline)
//!   "CREATE TABLE t (id"     -> accepted       (unclosed paren)
//! ```
//!
//! Multi-line SQL is the norm, so that behaviour was disqualifying. Tokenizing
//! first ([`lexer`]) makes whitespace and case insignificant by construction and
//! lets errors carry a byte offset.
//!
//! # Scope
//!
//! `CREATE TABLE`, `DROP TABLE`, `INSERT`, `SELECT … WHERE … ORDER BY … LIMIT`,
//! `UPDATE` and `DELETE`. No joins, subqueries, or aggregates: those need a
//! planner, and the directive asked for the statement set above.
//!
//! ```
//! use phoenixdb::sql::{parse, Statement};
//!
//! # fn main() -> phoenixdb::Result<()> {
//! let stmt = parse("SELECT name FROM users WHERE id = 42")?;
//! assert_eq!(stmt.table(), "users");
//! assert!(!stmt.is_mutation());
//! # Ok(())
//! # }
//! ```

pub mod ast;
pub mod catalog;
pub mod executor;
pub mod lexer;
pub mod parser;

pub use ast::{ColumnDef, ComparisonOp, Predicate, Statement, Value, WhereClause};
pub use catalog::{Cell, Row, TableSchema};
pub use executor::{Executor, QueryResult};
pub use lexer::{Token, TokenKind, tokenize};
pub use parser::parse;
