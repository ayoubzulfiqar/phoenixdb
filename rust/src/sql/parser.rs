//! Recursive-descent SQL parser.
//!
//! Consumes the token stream from [`crate::sql::lexer`] and produces a
//! [`Statement`]. Because it works on tokens rather than raw text, whitespace
//! and letter case are handled by construction: `CREATE TABLE`,
//! `create  table` and a statement split across five lines all parse
//! identically.
//!
//! # Error reporting
//!
//! Every failure names what was expected and what was found, with the byte
//! offset from the token. "Expected `(` after table name, found identifier
//! `id` at offset 18" is actionable; "syntax error" is not.
//!
//! # Grammar
//!
//! ```text
//!   statement   := create | drop | insert | select | update | delete
//!   create      := CREATE TABLE [IF NOT EXISTS] ident '(' coldef {',' coldef} ')'
//!   coldef      := ident [type] {PRIMARY KEY | NOT NULL}
//!   insert      := INSERT INTO ident ['(' ident {',' ident} ')'] VALUES row {',' row}
//!   row         := '(' value {',' value} ')'
//!   select      := SELECT (‘*’ | ident {',' ident}) FROM ident [where]
//!                  [ORDER BY ident [ASC|DESC]] [LIMIT integer]
//!   update      := UPDATE ident SET assign {',' assign} [where]
//!   delete      := DELETE FROM ident [where]
//!   where       := WHERE predicate {(AND|OR) predicate}
//!   predicate   := ident op value
//! ```

use crate::error::{Error, Result};
use crate::sql::ast::{ColumnDef, ComparisonOp, Predicate, Statement, Value, WhereClause};
use crate::sql::lexer::{Token, TokenKind, tokenize};

/// Parses one SQL statement.
///
/// A single trailing semicolon is permitted. Multiple statements in one string
/// are rejected: batching belongs to a higher layer that can decide on
/// transaction boundaries.
pub fn parse(sql: &str) -> Result<Statement> {
    let tokens = tokenize(sql)?;
    if tokens.is_empty() {
        return Err(Error::invalid("empty SQL statement"));
    }
    let mut p = Parser { tokens, pos: 0 };
    let stmt = p.parse_statement()?;
    p.accept_kind(&TokenKind::Semicolon);
    if let Some(tok) = p.peek() {
        return Err(Error::invalid(format!(
            "unexpected {} at offset {} after the end of the statement",
            tok.kind.describe(),
            tok.offset
        )));
    }
    Ok(stmt)
}

/// Cursor over the token stream.
struct Parser {
    tokens: Vec<Token>,
    pos: usize,
}

impl Parser {
    // ---- cursor helpers ---------------------------------------------------

    fn peek(&self) -> Option<&Token> {
        self.tokens.get(self.pos)
    }

    fn next(&mut self) -> Option<Token> {
        let t = self.tokens.get(self.pos).cloned();
        if t.is_some() {
            self.pos += 1;
        }
        t
    }

    /// Byte offset just past the last token, for "unexpected end" errors.
    fn end_offset(&self) -> usize {
        self.tokens.last().map_or(0, |t| t.offset)
    }

    /// Builds the "expected X, found Y" error every `expect_*` reports.
    ///
    /// Centralised so the wording — and the byte offset that makes it
    /// actionable — cannot drift between call sites.
    fn unexpected(&self, want: &str, found: Option<&Token>) -> Error {
        match found {
            Some(t) => Error::invalid(format!(
                "expected {want}, found {} at offset {}",
                t.kind.describe(),
                t.offset
            )),
            None => Error::invalid(format!(
                "expected {want}, found end of input at offset {}",
                self.end_offset()
            )),
        }
    }

    /// Consumes the next token, or reports `expected {want}` at end of input.
    fn take(&mut self, want: &str) -> Result<Token> {
        match self.next() {
            Some(t) => Ok(t),
            None => Err(self.unexpected(want, None)),
        }
    }

    /// Parses `item` one or more times, separated by commas.
    ///
    /// Every comma-separated list in the grammar — column definitions, target
    /// columns, VALUES rows, SET assignments, the select list — is this.
    fn comma_separated<T>(
        &mut self,
        mut item: impl FnMut(&mut Self) -> Result<T>,
    ) -> Result<Vec<T>> {
        let mut out = vec![item(self)?];
        while self.accept_kind(&TokenKind::Comma) {
            out.push(item(self)?);
        }
        Ok(out)
    }

    /// Consumes the next token if it is the keyword `word`.
    fn accept_keyword(&mut self, word: &str) -> bool {
        match self.peek() {
            Some(t) if t.kind.is_keyword(word) => {
                self.pos += 1;
                true
            }
            _ => false,
        }
    }

    /// Consumes the next token if it equals `kind`.
    fn accept_kind(&mut self, kind: &TokenKind) -> bool {
        match self.peek() {
            Some(t) if &t.kind == kind => {
                self.pos += 1;
                true
            }
            _ => false,
        }
    }

    /// Requires the keyword `word`.
    fn expect_keyword(&mut self, word: &str, context: &str) -> Result<()> {
        let want = format!("`{word}` {context}");
        let t = self.take(&want)?;
        if t.kind.is_keyword(word) {
            Ok(())
        } else {
            Err(self.unexpected(&want, Some(&t)))
        }
    }

    /// Requires the exact token `kind`.
    fn expect_kind(&mut self, kind: &TokenKind, context: &str) -> Result<()> {
        let want = format!("{} {context}", kind.describe());
        let t = self.take(&want)?;
        if t.kind == *kind {
            Ok(())
        } else {
            Err(self.unexpected(&want, Some(&t)))
        }
    }

    /// Requires an identifier and returns it.
    fn expect_ident(&mut self, want: &str) -> Result<String> {
        let t = self.take(want)?;
        match t.kind {
            TokenKind::Ident(name) => Ok(name),
            _ => Err(self.unexpected(want, Some(&t))),
        }
    }

    /// Requires a literal value.
    fn expect_value(&mut self, want: &str) -> Result<Value> {
        let t = self.take(want)?;
        match &t.kind {
            TokenKind::String(s) => Ok(Value::Text(s.clone())),
            TokenKind::Integer(i) => Ok(Value::Integer(*i)),
            TokenKind::Float(f) => Ok(Value::Float(*f)),
            TokenKind::Ident(w) if w.eq_ignore_ascii_case("null") => Ok(Value::Null),
            // A bare word where a value belongs is almost always a missing
            // quote, so say so rather than accepting it silently.
            TokenKind::Ident(w) => Err(Error::invalid(format!(
                "expected {want}, found bare identifier `{w}` at offset {} \
                 (string literals need single quotes)",
                t.offset
            ))),
            _ => Err(self.unexpected(want, Some(&t))),
        }
    }

    // ---- statements -------------------------------------------------------

    fn parse_statement(&mut self) -> Result<Statement> {
        let Some(first) = self.peek() else {
            return Err(Error::invalid("empty SQL statement"));
        };
        let offset = first.offset;
        let TokenKind::Ident(word) = &first.kind else {
            return Err(Error::invalid(format!(
                "a statement must begin with a keyword, found {} at offset {offset}",
                first.kind.describe()
            )));
        };

        // Dispatch on the leading keyword; each parser re-consumes it so the
        // grammar rules read top-down.
        match word.to_ascii_uppercase().as_str() {
            "CREATE" => self.parse_create(),
            "DROP" => self.parse_drop(),
            "INSERT" => self.parse_insert(),
            "SELECT" => self.parse_select(),
            "UPDATE" => self.parse_update(),
            "DELETE" => self.parse_delete(),
            other => Err(Error::invalid(format!(
                "unsupported statement `{other}` at offset {offset}; expected CREATE, DROP, \
                 INSERT, SELECT, UPDATE or DELETE"
            ))),
        }
    }

    fn parse_create(&mut self) -> Result<Statement> {
        self.expect_keyword("create", "at the start of the statement")?;
        self.expect_keyword("table", "after CREATE")?;
        let if_not_exists = if self.accept_keyword("if") {
            self.expect_keyword("not", "after IF")?;
            self.expect_keyword("exists", "after IF NOT")?;
            true
        } else {
            false
        };
        let table = self.expect_ident("a table name after CREATE TABLE")?;
        self.expect_kind(&TokenKind::LParen, "after the table name")?;
        let columns = self.comma_separated(Self::parse_column_def)?;
        self.expect_kind(&TokenKind::RParen, "to close the column list")?;

        Ok(Statement::CreateTable {
            table,
            columns,
            if_not_exists,
        })
    }

    /// `ident [type] {PRIMARY KEY | NOT NULL}`
    fn parse_column_def(&mut self) -> Result<ColumnDef> {
        let name = self.expect_ident("a column name")?;
        // The type word is advisory: storage is schemaless, but recording it
        // lets a future planner type-check without a schema migration.
        let mut data_type = String::new();
        let mut primary_key = false;
        let mut not_null = false;
        while let Some(tok) = self.peek() {
            match &tok.kind {
                TokenKind::Ident(w) if w.eq_ignore_ascii_case("primary") => {
                    self.pos += 1;
                    self.expect_keyword("key", "after PRIMARY")?;
                    primary_key = true;
                }
                TokenKind::Ident(w) if w.eq_ignore_ascii_case("not") => {
                    self.pos += 1;
                    self.expect_keyword("null", "after NOT")?;
                    not_null = true;
                }
                TokenKind::Ident(w) if data_type.is_empty() => {
                    data_type = w.to_uppercase();
                    self.pos += 1;
                    // Skip a parenthesised width like VARCHAR(255).
                    if self.accept_kind(&TokenKind::LParen) {
                        while let Some(t) = self.next() {
                            if t.kind == TokenKind::RParen {
                                break;
                            }
                        }
                    }
                }
                _ => break,
            }
        }
        Ok(ColumnDef {
            name,
            data_type,
            primary_key,
            not_null,
        })
    }

    fn parse_drop(&mut self) -> Result<Statement> {
        self.expect_keyword("drop", "at the start of the statement")?;
        self.expect_keyword("table", "after DROP")?;
        let if_exists = if self.accept_keyword("if") {
            self.expect_keyword("exists", "after IF")?;
            true
        } else {
            false
        };
        let table = self.expect_ident("a table name after DROP TABLE")?;
        Ok(Statement::DropTable { table, if_exists })
    }

    fn parse_insert(&mut self) -> Result<Statement> {
        self.expect_keyword("insert", "at the start of the statement")?;
        self.expect_keyword("into", "after INSERT")?;
        let table = self.expect_ident("a table name after INSERT INTO")?;

        // Optional column list.
        let mut columns = Vec::new();
        if self.accept_kind(&TokenKind::LParen) {
            columns = self.comma_separated(|p| p.expect_ident("a column name"))?;
            self.expect_kind(&TokenKind::RParen, "to close the column list")?;
        }

        self.expect_keyword("values", "after the table name")?;
        let rows = self.comma_separated(Self::parse_values_row)?;

        // Arity is checked after parsing so the error names the offending row.
        if !columns.is_empty()
            && let Some(bad) = rows.iter().find(|r| r.len() != columns.len())
        {
            return Err(Error::invalid(format!(
                "INSERT lists {} column(s) but this row supplies {} value(s)",
                columns.len(),
                bad.len()
            )));
        }
        Ok(Statement::Insert {
            table,
            columns,
            rows,
        })
    }

    /// `'(' value {',' value} ')'`
    fn parse_values_row(&mut self) -> Result<Vec<Value>> {
        self.expect_kind(&TokenKind::LParen, "to open a VALUES row")?;
        let row = self.comma_separated(|p| p.expect_value("a literal value"))?;
        self.expect_kind(&TokenKind::RParen, "to close a VALUES row")?;
        Ok(row)
    }

    fn parse_select(&mut self) -> Result<Statement> {
        self.expect_keyword("select", "at the start of the statement")?;
        // An empty projection means `*`.
        let columns = if self.accept_kind(&TokenKind::Star) {
            Vec::new()
        } else {
            self.comma_separated(|p| p.expect_ident("a column name or `*`"))?
        };
        self.expect_keyword("from", "after the select list")?;
        let table = self.expect_ident("a table name after FROM")?;
        let filter = self.parse_optional_where()?;

        let mut order_by = None;
        if self.accept_keyword("order") {
            self.expect_keyword("by", "after ORDER")?;
            let col = self.expect_ident("a column name after ORDER BY")?;
            let desc = if self.accept_keyword("desc") {
                true
            } else {
                self.accept_keyword("asc");
                false
            };
            order_by = Some((col, desc));
        }

        let mut limit = None;
        if self.accept_keyword("limit") {
            match self.next() {
                Some(Token {
                    kind: TokenKind::Integer(n),
                    offset,
                }) => {
                    if n < 0 {
                        return Err(Error::invalid(format!(
                            "LIMIT must not be negative, found {n} at offset {offset}"
                        )));
                    }
                    limit = Some(n as usize);
                }
                Some(t) => {
                    return Err(Error::invalid(format!(
                        "expected an integer after LIMIT, found {} at offset {}",
                        t.kind.describe(),
                        t.offset
                    )));
                }
                None => return Err(Error::invalid("expected an integer after LIMIT")),
            }
        }

        Ok(Statement::Select {
            table,
            columns,
            filter,
            order_by,
            limit,
        })
    }

    fn parse_update(&mut self) -> Result<Statement> {
        self.expect_keyword("update", "at the start of the statement")?;
        let table = self.expect_ident("a table name after UPDATE")?;
        self.expect_keyword("set", "after the table name")?;

        let assignments = self.comma_separated(|p| {
            let column = p.expect_ident("a column name in SET")?;
            p.expect_kind(&TokenKind::Eq, "after the column name in SET")?;
            let value = p.expect_value("a literal value in SET")?;
            Ok((column, value))
        })?;
        let filter = self.parse_optional_where()?;
        Ok(Statement::Update {
            table,
            assignments,
            filter,
        })
    }

    fn parse_delete(&mut self) -> Result<Statement> {
        self.expect_keyword("delete", "at the start of the statement")?;
        self.expect_keyword("from", "after DELETE")?;
        let table = self.expect_ident("a table name after DELETE FROM")?;
        let filter = self.parse_optional_where()?;
        Ok(Statement::Delete { table, filter })
    }

    // ---- WHERE ------------------------------------------------------------

    fn parse_optional_where(&mut self) -> Result<Option<WhereClause>> {
        if !self.accept_keyword("where") {
            return Ok(None);
        }
        let first = self.parse_predicate()?;
        let mut predicates = vec![first];
        let mut connective: Option<bool> = None; // Some(true) = AND

        loop {
            let is_and = if self.accept_keyword("and") {
                true
            } else if self.accept_keyword("or") {
                false
            } else {
                break;
            };
            // Refuse to guess precedence for `a AND b OR c`.
            match connective {
                Some(prev) if prev != is_and => {
                    return Err(Error::invalid(
                        "mixing AND and OR in one WHERE clause is ambiguous; \
                         parentheses are not yet supported",
                    ));
                }
                _ => connective = Some(is_and),
            }
            predicates.push(self.parse_predicate()?);
        }

        Ok(Some(match connective {
            None => WhereClause::Single(predicates.remove(0)),
            Some(true) => WhereClause::And(predicates),
            Some(false) => WhereClause::Or(predicates),
        }))
    }

    fn parse_predicate(&mut self) -> Result<Predicate> {
        let column = self.expect_ident("a column name in WHERE")?;
        let op = match self.next() {
            Some(t) => match t.kind {
                TokenKind::Eq => ComparisonOp::Eq,
                TokenKind::NotEq => ComparisonOp::NotEq,
                TokenKind::Lt => ComparisonOp::Lt,
                TokenKind::LtEq => ComparisonOp::LtEq,
                TokenKind::Gt => ComparisonOp::Gt,
                TokenKind::GtEq => ComparisonOp::GtEq,
                other => {
                    return Err(Error::invalid(format!(
                        "expected a comparison operator after `{column}`, found {} at offset {}",
                        other.describe(),
                        t.offset
                    )));
                }
            },
            None => {
                return Err(Error::invalid(format!(
                    "expected a comparison operator after `{column}`"
                )));
            }
        };
        let value = self.expect_value("a literal value in WHERE")?;
        Ok(Predicate { column, op, value })
    }
}
