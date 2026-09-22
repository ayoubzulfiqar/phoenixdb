//! Recursive-descent SQL parser.
//!
//! # Grammar (informal)
//!
//! ```text
//! create  := CREATE TABLE [IF NOT EXISTS] name '(' coldef {',' coldef} ')'
//! coldef  := name [type ['(' int [',' int] ')']] {PRIMARY KEY | NOT NULL}
//! drop    := DROP TABLE [IF EXISTS] name
//! insert  := INSERT INTO name ['(' name {',' name} ')'] VALUES row {',' row}
//! row     := '(' value {',' value} ')'
//! select  := SELECT items FROM name [WHERE expr] [GROUP BY name {',' name}]
//!            [ORDER BY name [ASC|DESC] {',' ...}] [LIMIT int] [OFFSET int]
//! items   := '*' | item {',' item}
//! item    := (agg '(' [DISTINCT] ('*' | name) ')' | name) [AS name]
//! update  := UPDATE name SET name '=' value {',' ...} [WHERE expr]
//! delete  := DELETE FROM name [WHERE expr]
//! expr    := and {OR and}
//! and     := not {AND not}
//! not     := NOT not | pred
//! pred    := '(' expr ')'
//!          | operand ( cmp operand
//!                    | IS [NOT] NULL
//!                    | [NOT] IN '(' operand {',' operand} ')'
//!                    | [NOT] BETWEEN operand AND operand
//!                    | [NOT] (LIKE | ILIKE) operand )
//! operand := literal | NULL | '?' | '?N' | name
//! ```
//!
//! `AND` binds tighter than `OR`, as in standard SQL. Bare reserved words
//! cannot be used as names — quote them (`"order"`) instead.
//!
//! `col = NULL` / `col <> NULL` are read as `IS NULL` / `IS NOT NULL`: standard
//! SQL says they are never true, which silently returns nothing, while this
//! reading is what the author of such a query means.

use crate::error::{Error, Result};
use crate::sql::ast::{
    AggFunc, ColumnDef, ComparisonOp, Expr, OrderItem, SelectItem, Statement, Value,
};
use crate::sql::lexer::{Token, TokenKind, tokenize};

/// Longest identifier accepted, in bytes.
pub const MAX_IDENT_LEN: usize = 128;

/// Words that cannot be used as bare names.
const RESERVED: &[&str] = &[
    "select", "from", "where", "and", "or", "not", "insert", "into", "values", "update", "set",
    "delete", "create", "table", "drop", "order", "group", "by", "limit", "offset", "null", "is",
    "in", "between", "like", "ilike", "as",
];

/// Parses exactly one statement (an optional trailing `;` is allowed).
pub fn parse(sql: &str) -> Result<Statement> {
    parse_with_params(sql).map(|(stmt, _)| stmt)
}

/// Parses one statement and reports how many bound parameters it uses.
pub fn parse_with_params(sql: &str) -> Result<(Statement, usize)> {
    let tokens = tokenize(sql)?;
    if tokens.is_empty() {
        return Err(Error::invalid("empty SQL statement"));
    }
    let mut p = Parser {
        tokens,
        pos: 0,
        next_param: 0,
        param_count: 0,
    };
    let stmt = p.parse_statement()?;
    p.accept_kind(&TokenKind::Semicolon);
    if let Some(tok) = p.peek() {
        return Err(Error::invalid(format!(
            "unexpected {} at offset {} after the end of the statement",
            tok.kind.describe(),
            tok.offset
        )));
    }
    Ok((stmt, p.param_count))
}

struct Parser {
    tokens: Vec<Token>,
    pos: usize,
    /// Index the next `?` receives.
    next_param: usize,
    /// One past the highest parameter index used.
    param_count: usize,
}

impl Parser {
    // ---- token helpers ----------------------------------------------------

    fn peek(&self) -> Option<&Token> {
        self.tokens.get(self.pos)
    }

    fn peek_at(&self, ahead: usize) -> Option<&Token> {
        self.tokens.get(self.pos + ahead)
    }

    fn next(&mut self) -> Option<Token> {
        let t = self.tokens.get(self.pos).cloned();
        if t.is_some() {
            self.pos += 1;
        }
        t
    }

    fn end_offset(&self) -> usize {
        self.tokens.last().map_or(0, |t| t.offset)
    }

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

    fn take(&mut self, want: &str) -> Result<Token> {
        match self.next() {
            Some(t) => Ok(t),
            None => Err(self.unexpected(want, None)),
        }
    }

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

    fn peek_keyword(&self, word: &str) -> bool {
        self.peek().is_some_and(|t| t.kind.is_keyword(word))
    }

    fn accept_keyword(&mut self, word: &str) -> bool {
        if self.peek_keyword(word) {
            self.pos += 1;
            true
        } else {
            false
        }
    }

    fn accept_kind(&mut self, kind: &TokenKind) -> bool {
        match self.peek() {
            Some(t) if &t.kind == kind => {
                self.pos += 1;
                true
            }
            _ => false,
        }
    }

    fn expect_keyword(&mut self, word: &str, context: &str) -> Result<()> {
        let want = format!("`{word}` {context}");
        let t = self.take(&want)?;
        if t.kind.is_keyword(word) {
            Ok(())
        } else {
            Err(self.unexpected(&want, Some(&t)))
        }
    }

    fn expect_kind(&mut self, kind: &TokenKind, context: &str) -> Result<()> {
        let want = format!("{} {context}", kind.describe());
        let t = self.take(&want)?;
        if t.kind == *kind {
            Ok(())
        } else {
            Err(self.unexpected(&want, Some(&t)))
        }
    }

    /// A table or column name: a bare non-reserved word or a quoted name.
    fn expect_ident(&mut self, want: &str) -> Result<String> {
        let t = self.take(want)?;
        let name = match &t.kind {
            TokenKind::Ident(name) => {
                if RESERVED.contains(&name.to_ascii_lowercase().as_str()) {
                    return Err(Error::invalid(format!(
                        "expected {want}, found the reserved word `{name}` at offset {} \
                         (quote it as \"{name}\" to use it as a name)",
                        t.offset
                    )));
                }
                name.clone()
            }
            TokenKind::QuotedIdent(name) => name.clone(),
            _ => return Err(self.unexpected(want, Some(&t))),
        };
        if name.is_empty() || name.len() > MAX_IDENT_LEN {
            return Err(Error::invalid(format!(
                "name at offset {} must be 1..={MAX_IDENT_LEN} bytes",
                t.offset
            )));
        }
        Ok(name)
    }

    fn param(&mut self, explicit: Option<usize>) -> usize {
        let index = match explicit {
            Some(n) => n - 1,
            None => {
                let i = self.next_param;
                self.next_param += 1;
                i
            }
        };
        self.param_count = self.param_count.max(index + 1);
        index
    }

    /// A literal, `NULL` or a parameter: what `VALUES` and `SET` accept.
    fn expect_value(&mut self, want: &str) -> Result<Expr> {
        let t = self.take(want)?;
        match &t.kind {
            TokenKind::String(s) => Ok(Expr::Literal(Value::Text(s.clone()))),
            TokenKind::Integer(i) => Ok(Expr::Literal(Value::Integer(*i))),
            TokenKind::Float(f) => Ok(Expr::Literal(Value::Float(*f))),
            TokenKind::Param(n) => Ok(Expr::Param(self.param(*n))),
            TokenKind::Ident(w) if w.eq_ignore_ascii_case("null") => Ok(Expr::Literal(Value::Null)),
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

    fn parse_column_def(&mut self) -> Result<ColumnDef> {
        let name = self.expect_ident("a column name")?;
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
                    // Size parameters: `VARCHAR(255)`, `DECIMAL(10, 2)`.
                    // Parsed strictly — skipping to the next `)` used to
                    // swallow the rest of the column list.
                    if self.accept_kind(&TokenKind::LParen) {
                        let mut sizes = Vec::new();
                        loop {
                            match self.next() {
                                Some(Token {
                                    kind: TokenKind::Integer(n),
                                    ..
                                }) if n >= 0 => sizes.push(n.to_string()),
                                other => {
                                    return Err(self.unexpected(
                                        "a non-negative size in the type's parentheses",
                                        other.as_ref(),
                                    ));
                                }
                            }
                            if sizes.len() == 2 || !self.accept_kind(&TokenKind::Comma) {
                                break;
                            }
                        }
                        self.expect_kind(&TokenKind::RParen, "to close the type's size")?;
                        data_type = format!("{data_type}({})", sizes.join(","));
                    }
                }
                TokenKind::Comma | TokenKind::RParen => break,
                _ => {
                    let t = tok.clone();
                    return Err(self.unexpected(
                        "a type, PRIMARY KEY, NOT NULL, `,` or `)` in a column definition",
                        Some(&t),
                    ));
                }
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

        let mut columns = Vec::new();
        if self.accept_kind(&TokenKind::LParen) {
            columns = self.comma_separated(|p| p.expect_ident("a column name"))?;
            self.expect_kind(&TokenKind::RParen, "to close the column list")?;
            reject_duplicates(&columns, "INSERT column list")?;
        }

        self.expect_keyword("values", "after the table name")?;
        let rows = self.comma_separated(Self::parse_values_row)?;

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

    fn parse_values_row(&mut self) -> Result<Vec<Expr>> {
        self.expect_kind(&TokenKind::LParen, "to open a VALUES row")?;
        let row = self.comma_separated(|p| p.expect_value("a literal value"))?;
        self.expect_kind(&TokenKind::RParen, "to close a VALUES row")?;
        Ok(row)
    }

    fn parse_select(&mut self) -> Result<Statement> {
        self.expect_keyword("select", "at the start of the statement")?;
        let items = if self.accept_kind(&TokenKind::Star) {
            vec![SelectItem::Wildcard]
        } else {
            self.comma_separated(Self::parse_select_item)?
        };
        self.expect_keyword("from", "after the select list")?;
        let table = self.expect_ident("a table name after FROM")?;
        let filter = self.parse_optional_where()?;

        let mut group_by = Vec::new();
        if self.accept_keyword("group") {
            self.expect_keyword("by", "after GROUP")?;
            group_by = self.comma_separated(|p| p.expect_ident("a column name after GROUP BY"))?;
        }

        let mut order_by = Vec::new();
        if self.accept_keyword("order") {
            self.expect_keyword("by", "after ORDER")?;
            order_by = self.comma_separated(|p| {
                let column = p.expect_ident("a column name after ORDER BY")?;
                let desc = if p.accept_keyword("desc") {
                    true
                } else {
                    p.accept_keyword("asc");
                    false
                };
                Ok(OrderItem { column, desc })
            })?;
        }

        let limit = if self.accept_keyword("limit") {
            Some(self.expect_count("LIMIT")?)
        } else {
            None
        };
        let offset = if self.accept_keyword("offset") {
            Some(self.expect_count("OFFSET")?)
        } else {
            None
        };

        Ok(Statement::Select {
            table,
            items,
            filter,
            group_by,
            order_by,
            limit,
            offset,
        })
    }

    fn expect_count(&mut self, what: &str) -> Result<usize> {
        match self.next() {
            Some(Token {
                kind: TokenKind::Integer(n),
                offset,
            }) => {
                if n < 0 {
                    return Err(Error::invalid(format!(
                        "{what} must not be negative, found {n} at offset {offset}"
                    )));
                }
                Ok(n as usize)
            }
            Some(t) => Err(Error::invalid(format!(
                "expected an integer after {what}, found {} at offset {}",
                t.kind.describe(),
                t.offset
            ))),
            None => Err(Error::invalid(format!("expected an integer after {what}"))),
        }
    }

    fn parse_select_item(&mut self) -> Result<SelectItem> {
        // An aggregate is a function name immediately followed by `(`.
        let func = match (self.peek(), self.peek_at(1)) {
            (
                Some(Token {
                    kind: TokenKind::Ident(w),
                    ..
                }),
                Some(Token {
                    kind: TokenKind::LParen,
                    ..
                }),
            ) => match w.to_ascii_lowercase().as_str() {
                "count" => Some(AggFunc::Count),
                "sum" => Some(AggFunc::Sum),
                "avg" => Some(AggFunc::Avg),
                "min" => Some(AggFunc::Min),
                "max" => Some(AggFunc::Max),
                other => {
                    return Err(Error::invalid(format!(
                        "unknown function `{other}`; expected COUNT, SUM, AVG, MIN or MAX"
                    )));
                }
            },
            _ => None,
        };
        let item = if let Some(func) = func {
            self.pos += 2; // name and `(`
            let distinct = self.accept_keyword("distinct");
            let column = if self.accept_kind(&TokenKind::Star) {
                if func != AggFunc::Count || distinct {
                    return Err(Error::invalid(format!(
                        "`*` is only valid in COUNT(*), not {}(*)",
                        func.name().to_uppercase()
                    )));
                }
                None
            } else {
                Some(self.expect_ident("a column name in the aggregate")?)
            };
            self.expect_kind(&TokenKind::RParen, "to close the aggregate")?;
            let alias = self.parse_alias()?;
            SelectItem::Aggregate {
                func,
                column,
                distinct,
                alias,
            }
        } else {
            let name = self.expect_ident("a column name, aggregate or `*`")?;
            let alias = self.parse_alias()?;
            SelectItem::Column { name, alias }
        };
        Ok(item)
    }

    fn parse_alias(&mut self) -> Result<Option<String>> {
        if self.accept_keyword("as") {
            Ok(Some(self.expect_ident("an alias after AS")?))
        } else {
            Ok(None)
        }
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
        let names: Vec<String> = assignments.iter().map(|(c, _)| c.clone()).collect();
        reject_duplicates(&names, "SET list")?;
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

    // ---- expressions --------------------------------------------------------

    fn parse_optional_where(&mut self) -> Result<Option<Expr>> {
        if !self.accept_keyword("where") {
            return Ok(None);
        }
        Ok(Some(self.parse_or(0)?))
    }

    fn parse_or(&mut self, depth: usize) -> Result<Expr> {
        guard_depth(depth)?;
        let mut left = self.parse_and(depth + 1)?;
        while self.accept_keyword("or") {
            let right = self.parse_and(depth + 1)?;
            left = Expr::Or(Box::new(left), Box::new(right));
        }
        Ok(left)
    }

    fn parse_and(&mut self, depth: usize) -> Result<Expr> {
        guard_depth(depth)?;
        let mut left = self.parse_not(depth + 1)?;
        while self.accept_keyword("and") {
            let right = self.parse_not(depth + 1)?;
            left = Expr::And(Box::new(left), Box::new(right));
        }
        Ok(left)
    }

    fn parse_not(&mut self, depth: usize) -> Result<Expr> {
        guard_depth(depth)?;
        if self.accept_keyword("not") {
            return Ok(Expr::Not(Box::new(self.parse_not(depth + 1)?)));
        }
        self.parse_predicate(depth + 1)
    }

    fn parse_predicate(&mut self, depth: usize) -> Result<Expr> {
        guard_depth(depth)?;
        if self.accept_kind(&TokenKind::LParen) {
            let inner = self.parse_or(depth + 1)?;
            self.expect_kind(&TokenKind::RParen, "to close the parenthesised condition")?;
            return Ok(inner);
        }
        let left = self.parse_operand()?;

        // `IS [NOT] NULL`
        if self.accept_keyword("is") {
            let negated = self.accept_keyword("not");
            self.expect_keyword("null", "after IS")?;
            return Ok(Expr::IsNull {
                expr: Box::new(left),
                negated,
            });
        }

        // `[NOT] IN | BETWEEN | LIKE | ILIKE`
        let negated = self.peek_keyword("not")
            && self.peek_at(1).is_some_and(|t| {
                ["in", "between", "like", "ilike"]
                    .iter()
                    .any(|w| t.kind.is_keyword(w))
            });
        if negated {
            self.pos += 1;
        }
        if self.accept_keyword("in") {
            self.expect_kind(&TokenKind::LParen, "after IN")?;
            let list = self.comma_separated(Self::parse_operand)?;
            self.expect_kind(&TokenKind::RParen, "to close the IN list")?;
            return Ok(Expr::InList {
                expr: Box::new(left),
                list,
                negated,
            });
        }
        if self.accept_keyword("between") {
            let low = self.parse_operand()?;
            self.expect_keyword("and", "between the BETWEEN bounds")?;
            let high = self.parse_operand()?;
            return Ok(Expr::Between {
                expr: Box::new(left),
                low: Box::new(low),
                high: Box::new(high),
                negated,
            });
        }
        let like = if self.accept_keyword("like") {
            Some(false)
        } else if self.accept_keyword("ilike") {
            Some(true)
        } else {
            None
        };
        if let Some(case_insensitive) = like {
            let pattern = self.parse_operand()?;
            return Ok(Expr::Like {
                expr: Box::new(left),
                pattern: Box::new(pattern),
                negated,
                case_insensitive,
            });
        }

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
                        "expected a comparison operator, IS, IN, BETWEEN or LIKE, found {} at offset {}",
                        other.describe(),
                        t.offset
                    )));
                }
            },
            None => {
                return Err(Error::invalid(
                    "expected a comparison operator at the end of the condition",
                ));
            }
        };
        let right = self.parse_operand()?;

        // `x = NULL` / `x <> NULL` mean IS [NOT] NULL (see the module docs).
        if matches!(right, Expr::Literal(Value::Null)) {
            match op {
                ComparisonOp::Eq => {
                    return Ok(Expr::IsNull {
                        expr: Box::new(left),
                        negated: false,
                    });
                }
                ComparisonOp::NotEq => {
                    return Ok(Expr::IsNull {
                        expr: Box::new(left),
                        negated: true,
                    });
                }
                _ => {}
            }
        }
        Ok(Expr::Compare {
            left: Box::new(left),
            op,
            right: Box::new(right),
        })
    }

    /// A literal, `NULL`, a parameter or a column name.
    fn parse_operand(&mut self) -> Result<Expr> {
        let Some(t) = self.peek().cloned() else {
            return Err(self.unexpected("a value or column name", None));
        };
        match &t.kind {
            TokenKind::Ident(w) if w.eq_ignore_ascii_case("null") => {
                self.pos += 1;
                Ok(Expr::Literal(Value::Null))
            }
            TokenKind::Ident(_) | TokenKind::QuotedIdent(_) => {
                Ok(Expr::Column(self.expect_ident("a column name")?))
            }
            _ => self.expect_value("a value or column name"),
        }
    }
}

/// Bounds nesting so a hostile `((((…))))` cannot overflow the stack.
fn guard_depth(depth: usize) -> Result<()> {
    if depth > 256 {
        return Err(Error::invalid("condition is nested too deeply"));
    }
    Ok(())
}

fn reject_duplicates(names: &[String], what: &str) -> Result<()> {
    for (i, name) in names.iter().enumerate() {
        let folded = name.to_lowercase();
        if names[..i].iter().any(|n| n.to_lowercase() == folded) {
            return Err(Error::invalid(format!(
                "column `{name}` appears twice in the {what}"
            )));
        }
    }
    Ok(())
}
