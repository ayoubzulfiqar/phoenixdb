//! SQL tokenizer.
//!
//! Byte-oriented and allocation-light: identifiers and literals are the only
//! tokens that own data. Every token records its byte offset so parse errors
//! can point at the exact position.
//!
//! Recognised:
//!
//! * bare identifiers and keywords (`[A-Za-z_][A-Za-z0-9_]*`), matched
//!   case-insensitively as keywords;
//! * quoted identifiers (`"like this"`, `""` escapes a quote) — always names,
//!   never keywords;
//! * string literals (`'text'`, `''` escapes a quote);
//! * numbers: `42`, `-7`, `+3`, `3.5`, `.5`, `1e3`, `2.5E-4`;
//! * bound parameters: `?` (next in order) and `?3` (explicit, 1-based);
//! * `( ) , ; * .` and the comparison operators `= <> != < <= > >=`;
//! * `--` line comments and `/* */` block comments.

use crate::error::{Error, Result};

/// One lexical token with its source offset.
#[derive(Debug, Clone, PartialEq)]
pub struct Token {
    /// What the token is.
    pub kind: TokenKind,
    /// Byte offset of the token's first character.
    pub offset: usize,
}

impl Token {
    /// Creates a token.
    #[must_use]
    pub fn new(kind: TokenKind, offset: usize) -> Self {
        Token { kind, offset }
    }
}

/// Token categories.
#[derive(Debug, Clone, PartialEq)]
pub enum TokenKind {
    /// A bare word: a keyword or an identifier.
    Ident(String),
    /// A `"quoted"` identifier: always a name, never a keyword.
    QuotedIdent(String),
    /// A `'string'` literal, unescaped.
    String(String),
    /// An integer literal (sign included).
    Integer(i64),
    /// A finite floating-point literal (sign included).
    Float(f64),
    /// A bound parameter: `?` (`None`, next in order) or `?N` (`Some(N)`,
    /// 1-based).
    Param(Option<usize>),
    /// `(`
    LParen,
    /// `)`
    RParen,
    /// `,`
    Comma,
    /// `;`
    Semicolon,
    /// `*`
    Star,
    /// `.`
    Dot,
    /// `=`
    Eq,
    /// `<>` or `!=`
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

impl TokenKind {
    /// Human-readable description for error messages.
    #[must_use]
    pub fn describe(&self) -> String {
        match self {
            TokenKind::Ident(s) => format!("identifier `{s}`"),
            TokenKind::QuotedIdent(s) => format!("quoted identifier \"{s}\""),
            TokenKind::String(s) => format!("string '{s}'"),
            TokenKind::Integer(n) => format!("integer {n}"),
            TokenKind::Float(f) => format!("float {f}"),
            TokenKind::Param(None) => "parameter `?`".to_string(),
            TokenKind::Param(Some(n)) => format!("parameter `?{n}`"),
            TokenKind::LParen => "`(`".to_string(),
            TokenKind::RParen => "`)`".to_string(),
            TokenKind::Comma => "`,`".to_string(),
            TokenKind::Semicolon => "`;`".to_string(),
            TokenKind::Star => "`*`".to_string(),
            TokenKind::Dot => "`.`".to_string(),
            TokenKind::Eq => "`=`".to_string(),
            TokenKind::NotEq => "`<>`".to_string(),
            TokenKind::Lt => "`<`".to_string(),
            TokenKind::LtEq => "`<=`".to_string(),
            TokenKind::Gt => "`>`".to_string(),
            TokenKind::GtEq => "`>=`".to_string(),
        }
    }

    /// True when this is the bare keyword `word` (case-insensitive). A quoted
    /// identifier is never a keyword.
    #[must_use]
    pub fn is_keyword(&self, word: &str) -> bool {
        match self {
            TokenKind::Ident(s) => s.eq_ignore_ascii_case(word),
            _ => false,
        }
    }
}

/// Splits `sql` into tokens.
pub fn tokenize(sql: &str) -> Result<Vec<Token>> {
    let bytes = sql.as_bytes();
    let mut tokens: Vec<Token> = Vec::new();
    let mut i = 0usize;

    while i < bytes.len() {
        let c = bytes[i];

        // --- whitespace: insignificant, in any amount or kind --------------
        if c.is_ascii_whitespace() {
            i += 1;
            continue;
        }

        // --- comments -------------------------------------------------------
        if c == b'-' && i + 1 < bytes.len() && bytes[i + 1] == b'-' {
            while i < bytes.len() && bytes[i] != b'\n' {
                i += 1;
            }
            continue;
        }
        if c == b'/' && i + 1 < bytes.len() && bytes[i + 1] == b'*' {
            let start = i;
            i += 2;
            loop {
                if i + 1 >= bytes.len() {
                    return Err(Error::invalid(format!(
                        "unterminated block comment starting at offset {start}"
                    )));
                }
                if bytes[i] == b'*' && bytes[i + 1] == b'/' {
                    i += 2;
                    break;
                }
                i += 1;
            }
            continue;
        }

        let start = i;

        // --- string literal -------------------------------------------------
        if c == b'\'' {
            let (value, end) = quoted(sql, i, b'\'', "string literal")?;
            tokens.push(Token::new(TokenKind::String(value), start));
            i = end;
            continue;
        }

        // --- quoted identifier ------------------------------------------------
        if c == b'"' {
            let (name, end) = quoted(sql, i, b'"', "quoted identifier")?;
            if let Some(bad) = name.chars().find(|ch| ch.is_control()) {
                return Err(Error::invalid(format!(
                    "quoted identifier at offset {start} contains the control character {:?}",
                    bad
                )));
            }
            tokens.push(Token::new(TokenKind::QuotedIdent(name), start));
            i = end;
            continue;
        }

        // --- number -----------------------------------------------------------
        // The grammar has no arithmetic, so `-`/`+` can only be a sign. Lexing
        // the sign with the digits lets `-9223372036854775808` parse, which
        // would overflow if read as a positive literal and then negated.
        let signed = (c == b'-' || c == b'+')
            && i + 1 < bytes.len()
            && (bytes[i + 1].is_ascii_digit()
                || (bytes[i + 1] == b'.' && i + 2 < bytes.len() && bytes[i + 2].is_ascii_digit()));
        let leading_dot = c == b'.' && i + 1 < bytes.len() && bytes[i + 1].is_ascii_digit();
        if c.is_ascii_digit() || signed || leading_dot {
            let (kind, end) = number(sql, i)?;
            tokens.push(Token::new(kind, start));
            i = end;
            continue;
        }

        // --- identifier / keyword --------------------------------------------
        if c.is_ascii_alphabetic() || c == b'_' {
            while i < bytes.len() && (bytes[i].is_ascii_alphanumeric() || bytes[i] == b'_') {
                i += 1;
            }
            tokens.push(Token::new(
                TokenKind::Ident(sql[start..i].to_string()),
                start,
            ));
            continue;
        }

        // --- bound parameter --------------------------------------------------
        if c == b'?' {
            i += 1;
            let digits_start = i;
            while i < bytes.len() && bytes[i].is_ascii_digit() {
                i += 1;
            }
            let kind = if i == digits_start {
                TokenKind::Param(None)
            } else {
                let n: usize = sql[digits_start..i].parse().map_err(|_| {
                    Error::invalid(format!("parameter number at offset {start} is too large"))
                })?;
                if n == 0 || n > 65_535 {
                    return Err(Error::invalid(format!(
                        "parameter `?{n}` at offset {start}: numbers run from ?1 to ?65535"
                    )));
                }
                TokenKind::Param(Some(n))
            };
            tokens.push(Token::new(kind, start));
            continue;
        }

        // --- operators and punctuation ----------------------------------------
        let (kind, width) = match c {
            b'(' => (TokenKind::LParen, 1),
            b')' => (TokenKind::RParen, 1),
            b',' => (TokenKind::Comma, 1),
            b';' => (TokenKind::Semicolon, 1),
            b'*' => (TokenKind::Star, 1),
            b'.' => (TokenKind::Dot, 1),
            b'=' => (TokenKind::Eq, 1),
            b'<' if i + 1 < bytes.len() && bytes[i + 1] == b'>' => (TokenKind::NotEq, 2),
            b'<' if i + 1 < bytes.len() && bytes[i + 1] == b'=' => (TokenKind::LtEq, 2),
            b'<' => (TokenKind::Lt, 1),
            b'>' if i + 1 < bytes.len() && bytes[i + 1] == b'=' => (TokenKind::GtEq, 2),
            b'>' => (TokenKind::Gt, 1),
            b'!' if i + 1 < bytes.len() && bytes[i + 1] == b'=' => (TokenKind::NotEq, 2),
            _ => {
                // Decode the whole character so the message shows what the
                // user actually typed, not one byte of it.
                let ch = sql[start..].chars().next().unwrap_or('\u{FFFD}');
                return Err(Error::invalid(format!(
                    "unexpected character `{ch}` at offset {start}"
                )));
            }
        };
        tokens.push(Token::new(kind, start));
        i += width;
    }

    Ok(tokens)
}

/// Reads a `quote`-delimited run starting at `start`, where a doubled quote
/// is an escaped quote. Returns the unescaped text and the offset just past
/// the closing quote.
fn quoted(sql: &str, start: usize, quote: u8, what: &str) -> Result<(String, usize)> {
    let bytes = sql.as_bytes();
    let mut i = start + 1;
    let mut value = String::new();
    let mut run = i;
    loop {
        if i >= bytes.len() {
            return Err(Error::invalid(format!(
                "unterminated {what} starting at offset {start}"
            )));
        }
        if bytes[i] == quote {
            value.push_str(&sql[run..i]);
            if i + 1 < bytes.len() && bytes[i + 1] == quote {
                value.push(quote as char);
                i += 2;
                run = i;
                continue;
            }
            return Ok((value, i + 1));
        }
        i += 1;
    }
}

/// Reads a number starting at `start`: `[+-]digits[.digits][e[+-]digits]` or
/// `[+-].digits…`. The literal must not run into an identifier character, so
/// `1OR` or `3abc` is rejected instead of silently split.
fn number(sql: &str, start: usize) -> Result<(TokenKind, usize)> {
    let bytes = sql.as_bytes();
    let mut i = start;
    if bytes[i] == b'-' || bytes[i] == b'+' {
        i += 1;
    }
    let mut is_float = false;
    while i < bytes.len() && bytes[i].is_ascii_digit() {
        i += 1;
    }
    if i < bytes.len() && bytes[i] == b'.' && i + 1 < bytes.len() && bytes[i + 1].is_ascii_digit() {
        is_float = true;
        i += 1;
        while i < bytes.len() && bytes[i].is_ascii_digit() {
            i += 1;
        }
    } else if i < bytes.len() && bytes[i] == b'.' {
        // `5.` — a trailing dot with no fraction.
        is_float = true;
        i += 1;
    }
    if i < bytes.len() && (bytes[i] == b'e' || bytes[i] == b'E') {
        let mut j = i + 1;
        if j < bytes.len() && (bytes[j] == b'+' || bytes[j] == b'-') {
            j += 1;
        }
        if j < bytes.len() && bytes[j].is_ascii_digit() {
            is_float = true;
            while j < bytes.len() && bytes[j].is_ascii_digit() {
                j += 1;
            }
            i = j;
        }
    }
    if i < bytes.len() && (bytes[i].is_ascii_alphanumeric() || bytes[i] == b'_') {
        let mut end = i;
        while end < bytes.len() && (bytes[end].is_ascii_alphanumeric() || bytes[end] == b'_') {
            end += 1;
        }
        return Err(Error::invalid(format!(
            "malformed number `{}` at offset {start} (a separator is missing?)",
            &sql[start..end]
        )));
    }
    let text = &sql[start..i];
    let digits = text.strip_prefix('+').unwrap_or(text);
    let kind = if is_float {
        let value: f64 = digits
            .parse()
            .map_err(|_| Error::invalid(format!("malformed number `{text}` at offset {start}")))?;
        if !value.is_finite() {
            return Err(Error::invalid(format!(
                "number `{text}` at offset {start} is out of the finite range"
            )));
        }
        TokenKind::Float(value)
    } else {
        TokenKind::Integer(digits.parse::<i64>().map_err(|_| {
            Error::invalid(format!(
                "integer `{text}` at offset {start} is out of range"
            ))
        })?)
    };
    Ok((kind, i))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn kinds(sql: &str) -> Vec<TokenKind> {
        tokenize(sql).unwrap().into_iter().map(|t| t.kind).collect()
    }

    #[test]
    fn negative_numbers_are_single_tokens() {
        assert_eq!(kinds("-42"), vec![TokenKind::Integer(-42)]);
        assert_eq!(kinds("-3.5"), vec![TokenKind::Float(-3.5)]);
        assert_eq!(kinds("+7"), vec![TokenKind::Integer(7)]);
    }

    #[test]
    fn the_most_negative_i64_parses() {
        assert_eq!(
            kinds("-9223372036854775808"),
            vec![TokenKind::Integer(i64::MIN)]
        );
        assert!(tokenize("9223372036854775808").is_err());
    }

    #[test]
    fn a_bare_minus_is_still_an_error() {
        assert!(tokenize("-").is_err());
        assert!(tokenize("a - b").is_err());
    }

    #[test]
    fn number_forms() {
        assert_eq!(kinds(".5"), vec![TokenKind::Float(0.5)]);
        assert_eq!(kinds("-.25"), vec![TokenKind::Float(-0.25)]);
        assert_eq!(kinds("1e3"), vec![TokenKind::Float(1000.0)]);
        assert_eq!(kinds("2.5E-2"), vec![TokenKind::Float(0.025)]);
        assert_eq!(kinds("5."), vec![TokenKind::Float(5.0)]);
        assert_eq!(kinds("3.0"), vec![TokenKind::Float(3.0)]);
    }

    #[test]
    fn a_number_glued_to_a_word_is_rejected() {
        let err = tokenize("1OR 2").unwrap_err();
        assert!(format!("{err}").contains("malformed number `1OR`"), "{err}");
        assert!(tokenize("3abc").is_err());
    }

    #[test]
    fn infinite_literals_are_rejected() {
        let huge = format!("1{}.0", "0".repeat(400));
        assert!(tokenize(&huge).is_err());
        assert!(tokenize("1e999").is_err());
    }

    #[test]
    fn negative_numbers_work_in_context() {
        let toks = kinds("WHERE a > -5 AND b = -0.5");
        assert!(toks.contains(&TokenKind::Integer(-5)));
        assert!(toks.contains(&TokenKind::Float(-0.5)));
    }

    #[test]
    fn tokenizes_a_simple_statement() {
        let toks = kinds("SELECT a, b FROM t WHERE a = 1;");
        assert_eq!(toks.len(), 11);
        assert!(toks[0].is_keyword("select"));
        assert_eq!(toks[10], TokenKind::Semicolon);
    }

    #[test]
    fn whitespace_is_insignificant_in_any_amount_or_kind() {
        assert_eq!(kinds("SELECT\t\n  *\r\nFROM   t"), kinds("SELECT * FROM t"));
    }

    #[test]
    fn identifiers_keep_their_case_but_match_keywords_insensitively() {
        let toks = kinds("SeLeCt MyTable");
        assert!(toks[0].is_keyword("SELECT"));
        assert_eq!(toks[1], TokenKind::Ident("MyTable".into()));
    }

    #[test]
    fn string_literals_may_contain_anything() {
        assert_eq!(
            kinds("'SELECT * FROM x; -- not a comment'"),
            vec![TokenKind::String(
                "SELECT * FROM x; -- not a comment".into()
            )]
        );
    }

    #[test]
    fn doubled_quote_is_an_escape() {
        assert_eq!(kinds("'it''s'"), vec![TokenKind::String("it's".into())]);
        assert_eq!(
            kinds("\"say \"\"hi\"\"\""),
            vec![TokenKind::QuotedIdent("say \"hi\"".into())]
        );
    }

    #[test]
    fn unicode_survives_a_string_literal() {
        assert_eq!(
            kinds("'héllo 🌍 日本語'"),
            vec![TokenKind::String("héllo 🌍 日本語".into())]
        );
    }

    #[test]
    fn numbers_are_typed() {
        assert_eq!(kinds("42"), vec![TokenKind::Integer(42)]);
        assert_eq!(kinds("4.25"), vec![TokenKind::Float(4.25)]);
    }

    #[test]
    fn comparison_operators_are_recognised() {
        assert_eq!(
            kinds("= <> != < <= > >="),
            vec![
                TokenKind::Eq,
                TokenKind::NotEq,
                TokenKind::NotEq,
                TokenKind::Lt,
                TokenKind::LtEq,
                TokenKind::Gt,
                TokenKind::GtEq,
            ]
        );
    }

    #[test]
    fn parameters_are_recognised() {
        assert_eq!(
            kinds("? ?2 ?"),
            vec![
                TokenKind::Param(None),
                TokenKind::Param(Some(2)),
                TokenKind::Param(None)
            ]
        );
        assert!(tokenize("?0").is_err());
    }

    #[test]
    fn comments_are_skipped() {
        assert_eq!(
            kinds("SELECT -- trailing\n* /* inline */ FROM t"),
            kinds("SELECT * FROM t")
        );
        assert_eq!(
            kinds("/* leading */ SELECT"),
            vec![TokenKind::Ident("SELECT".into())]
        );
    }

    #[test]
    fn quoted_identifiers_allow_reserved_words() {
        let toks = kinds("\"select\"");
        assert_eq!(toks, vec![TokenKind::QuotedIdent("select".into())]);
        assert!(
            !toks[0].is_keyword("select"),
            "a quoted name is not a keyword"
        );
    }

    #[test]
    fn control_characters_in_identifiers_are_rejected() {
        assert!(tokenize("\"a\u{0}b\"").is_err());
        assert!(tokenize("\"tab\there\"").is_err());
    }

    #[test]
    fn offsets_point_at_the_token() {
        let toks = tokenize("SELECT  *").unwrap();
        assert_eq!(toks[0].offset, 0);
        assert_eq!(toks[1].offset, 8, "offset must survive the double space");
    }

    #[test]
    fn malformed_input_is_rejected_with_a_position() {
        let err = tokenize("'unterminated").unwrap_err();
        assert!(format!("{err}").contains("unterminated string"));

        let err = tokenize("SELECT @").unwrap_err();
        assert!(format!("{err}").contains("offset 7"), "got: {err}");

        let err = tokenize("SELECT é").unwrap_err();
        assert!(
            format!("{err}").contains("`é`"),
            "the whole character: {err}"
        );

        assert!(tokenize("/* never closed").is_err());
        assert!(tokenize("\"unterminated").is_err());
    }

    #[test]
    fn empty_input_yields_no_tokens() {
        assert!(tokenize("").unwrap().is_empty());
        assert!(tokenize("   \n\t  ").unwrap().is_empty());
        assert!(tokenize("-- only a comment").unwrap().is_empty());
    }

    #[test]
    fn a_realistic_multiline_statement_tokenizes() {
        let sql = "CREATE TABLE users (\n  id,\n  name,\n  email\n);";
        let toks = tokenize(sql).unwrap();
        assert!(toks[0].kind.is_keyword("create"));
        assert!(toks[1].kind.is_keyword("table"));
        assert_eq!(toks[2].kind, TokenKind::Ident("users".into()));
        assert_eq!(toks[3].kind, TokenKind::LParen);
        assert_eq!(
            *toks.last().unwrap(),
            Token::new(TokenKind::Semicolon, sql.len() - 1)
        );
    }
}
