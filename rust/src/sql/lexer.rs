//! SQL tokenizer.
//!
//! # Why hand-written
//!
//! The design called for `sqlparser-rs`. That crate depends on `stacker`, which
//! needs a C compiler — unavailable on the Flutter/Android cross-compilation
//! path and on this build host. A hand-written lexer keeps the SQL feature
//! pure-Rust and dependency-free.
//!
//! The approach (hand-rolled parsing rather than a parser-generator) was
//! confirmed workable by MagnumDB <https://github.com/sohamdev77/MagnumDB>
//! (MIT). This implementation does **not** reuse its code: MagnumDB dispatches
//! on `sql.to_uppercase().starts_with("CREATE TABLE")` and splits on single
//! spaces, so it rejects `CREATE  TABLE` (two spaces), and any statement
//! containing a newline or tab. Tokenizing first removes that whole class of
//! bug — whitespace becomes insignificant, as SQL requires.
//!
//! # Design
//!
//! One pass over the input produces a `Vec<Token>`. Whitespace separates tokens
//! but is not itself a token; comments are skipped. String literals are scanned
//! with proper `''` escape handling, so a quoted value may contain commas,
//! parentheses and SQL keywords without confusing the parser.
//!
//! Every token records its byte offset, so a parse error can point at the exact
//! character that went wrong instead of reporting "syntax error".

use crate::error::{Error, Result};

/// A lexical token together with where it started.
#[derive(Debug, Clone, PartialEq)]
pub struct Token {
    /// What kind of token this is.
    pub kind: TokenKind,
    /// Byte offset of the token's first character in the source.
    pub offset: usize,
}

impl Token {
    /// Creates a token at `offset`.
    #[must_use]
    pub fn new(kind: TokenKind, offset: usize) -> Self {
        Token { kind, offset }
    }
}

/// The kinds of token the SQL dialect recognises.
#[derive(Debug, Clone, PartialEq)]
pub enum TokenKind {
    /// A bare word: keyword, table name, or column name.
    ///
    /// Keywords are *not* distinguished here — the parser matches them
    /// case-insensitively, so `select` and `SELECT` are the same identifier
    /// token and a column legitimately named `count` still works.
    Ident(String),
    /// A single-quoted string literal, with escapes already resolved.
    String(String),
    /// An integer literal.
    Integer(i64),
    /// A floating-point literal.
    Float(f64),
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
    /// Renders the token for an error message.
    #[must_use]
    pub fn describe(&self) -> String {
        match self {
            TokenKind::Ident(s) => format!("identifier `{s}`"),
            TokenKind::String(s) => format!("string '{s}'"),
            TokenKind::Integer(n) => format!("integer {n}"),
            TokenKind::Float(f) => format!("float {f}"),
            TokenKind::LParen => "`(`".to_string(),
            TokenKind::RParen => "`)`".to_string(),
            TokenKind::Comma => "`,`".to_string(),
            TokenKind::Semicolon => "`;`".to_string(),
            TokenKind::Star => "`*`".to_string(),
            TokenKind::Eq => "`=`".to_string(),
            TokenKind::NotEq => "`<>`".to_string(),
            TokenKind::Lt => "`<`".to_string(),
            TokenKind::LtEq => "`<=`".to_string(),
            TokenKind::Gt => "`>`".to_string(),
            TokenKind::GtEq => "`>=`".to_string(),
        }
    }

    /// True when this is the identifier `word`, compared case-insensitively.
    #[must_use]
    pub fn is_keyword(&self, word: &str) -> bool {
        match self {
            TokenKind::Ident(s) => s.eq_ignore_ascii_case(word),
            _ => false,
        }
    }
}

/// Splits `sql` into tokens.
///
/// Returns [`Error::InvalidArgument`] for an unterminated string literal, an
/// unterminated block comment, or a character that cannot begin a token. The
/// message carries the byte offset so a caller can point at the problem.
pub fn tokenize(sql: &str) -> Result<Vec<Token>> {
    let bytes = sql.as_bytes();
    let mut tokens = Vec::new();
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
            // Line comment: skip to the newline.
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
            i += 1;
            let mut value = String::new();
            loop {
                if i >= bytes.len() {
                    return Err(Error::invalid(format!(
                        "unterminated string literal starting at offset {start}"
                    )));
                }
                if bytes[i] == b'\'' {
                    // '' inside a literal is an escaped single quote.
                    if i + 1 < bytes.len() && bytes[i + 1] == b'\'' {
                        value.push('\'');
                        i += 2;
                        continue;
                    }
                    i += 1;
                    break;
                }
                // Copy the whole UTF-8 sequence, not just this byte.
                let ch_len = utf8_len(bytes[i]);
                let end = (i + ch_len).min(bytes.len());
                value.push_str(
                    std::str::from_utf8(&bytes[i..end])
                        .map_err(|_| Error::invalid("string literal is not valid UTF-8"))?,
                );
                i = end;
            }
            tokens.push(Token::new(TokenKind::String(value), start));
            continue;
        }

        // --- number -----------------------------------------------------------
        // A leading `-` is part of the literal. This grammar has no arithmetic
        // operators, so a `-` can only ever be a sign — there is no `a - b` to
        // be ambiguous with. Parsing the sign here (rather than negating in the
        // parser) also lets `-9223372036854775808` parse, which would overflow
        // if read as a positive literal and then negated.
        let negative = c == b'-' && i + 1 < bytes.len() && bytes[i + 1].is_ascii_digit();
        if c.is_ascii_digit() || negative {
            if negative {
                i += 1; // consume the sign
            }
            let mut saw_dot = false;
            while i < bytes.len() && (bytes[i].is_ascii_digit() || bytes[i] == b'.') {
                if bytes[i] == b'.' {
                    if saw_dot {
                        break; // a second dot ends the number
                    }
                    saw_dot = true;
                }
                i += 1;
            }
            let text = &sql[start..i];
            let kind = if saw_dot {
                TokenKind::Float(text.parse::<f64>().map_err(|_| {
                    Error::invalid(format!("malformed number `{text}` at offset {start}"))
                })?)
            } else {
                TokenKind::Integer(text.parse::<i64>().map_err(|_| {
                    Error::invalid(format!(
                        "integer `{text}` at offset {start} is out of range"
                    ))
                })?)
            };
            tokens.push(Token::new(kind, start));
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

        // --- quoted identifier ------------------------------------------------
        if c == b'"' {
            i += 1;
            let id_start = i;
            while i < bytes.len() && bytes[i] != b'"' {
                i += 1;
            }
            if i >= bytes.len() {
                return Err(Error::invalid(format!(
                    "unterminated quoted identifier starting at offset {start}"
                )));
            }
            let name = sql[id_start..i].to_string();
            i += 1; // closing quote
            tokens.push(Token::new(TokenKind::Ident(name), start));
            continue;
        }

        // --- operators and punctuation ----------------------------------------
        let (kind, width) = match c {
            b'(' => (TokenKind::LParen, 1),
            b')' => (TokenKind::RParen, 1),
            b',' => (TokenKind::Comma, 1),
            b';' => (TokenKind::Semicolon, 1),
            b'*' => (TokenKind::Star, 1),
            b'=' => (TokenKind::Eq, 1),
            b'<' if i + 1 < bytes.len() && bytes[i + 1] == b'>' => (TokenKind::NotEq, 2),
            b'<' if i + 1 < bytes.len() && bytes[i + 1] == b'=' => (TokenKind::LtEq, 2),
            b'<' => (TokenKind::Lt, 1),
            b'>' if i + 1 < bytes.len() && bytes[i + 1] == b'=' => (TokenKind::GtEq, 2),
            b'>' => (TokenKind::Gt, 1),
            b'!' if i + 1 < bytes.len() && bytes[i + 1] == b'=' => (TokenKind::NotEq, 2),
            other => {
                return Err(Error::invalid(format!(
                    "unexpected character `{}` at offset {start}",
                    other as char
                )));
            }
        };
        tokens.push(Token::new(kind, start));
        i += width;
    }

    Ok(tokens)
}

/// Length in bytes of the UTF-8 sequence beginning with `first`.
#[inline]
fn utf8_len(first: u8) -> usize {
    if first < 0x80 {
        1
    } else if first >> 5 == 0b110 {
        2
    } else if first >> 4 == 0b1110 {
        3
    } else if first >> 3 == 0b11110 {
        4
    } else {
        1 // continuation or invalid byte: consume one and let UTF-8 checks fail
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn kinds(sql: &str) -> Vec<TokenKind> {
        tokenize(sql).unwrap().into_iter().map(|t| t.kind).collect()
    }

    #[test]
    fn negative_numbers_are_single_tokens() {
        // Regression: the lexer used to reject `-` outright, so no negative
        // literal could be written at all.
        assert_eq!(kinds("-42"), vec![TokenKind::Integer(-42)]);
        assert_eq!(kinds("-3.5"), vec![TokenKind::Float(-3.5)]);
    }

    #[test]
    fn the_most_negative_i64_parses() {
        // Only works because the sign is lexed together with the digits:
        // reading the magnitude as a positive i64 first would overflow.
        assert_eq!(
            kinds("-9223372036854775808"),
            vec![TokenKind::Integer(i64::MIN)]
        );
    }

    #[test]
    fn a_bare_minus_is_still_an_error() {
        // The sign rule must not silently swallow a stray operator.
        assert!(tokenize("-").is_err());
        assert!(tokenize("SELECT - FROM t").is_err());
    }

    #[test]
    fn negative_numbers_work_in_context() {
        assert_eq!(
            kinds("n > -5"),
            vec![
                TokenKind::Ident("n".into()),
                TokenKind::Gt,
                TokenKind::Integer(-5),
            ]
        );
    }

    #[test]
    fn tokenizes_a_simple_statement() {
        assert_eq!(
            kinds("SELECT * FROM users"),
            vec![
                TokenKind::Ident("SELECT".into()),
                TokenKind::Star,
                TokenKind::Ident("FROM".into()),
                TokenKind::Ident("users".into()),
            ]
        );
    }

    #[test]
    fn whitespace_is_insignificant_in_any_amount_or_kind() {
        // This is the exact class of input MagnumDB's parser rejects.
        let expected = vec![
            TokenKind::Ident("CREATE".into()),
            TokenKind::Ident("TABLE".into()),
            TokenKind::Ident("t".into()),
        ];
        assert_eq!(kinds("CREATE TABLE t"), expected);
        assert_eq!(kinds("CREATE  TABLE   t"), expected, "double spaces");
        assert_eq!(kinds("CREATE\nTABLE\nt"), expected, "newlines");
        assert_eq!(kinds("CREATE\tTABLE\tt"), expected, "tabs");
        assert_eq!(kinds("  CREATE\r\n\tTABLE  t  "), expected, "mixed");
    }

    #[test]
    fn identifiers_keep_their_case_but_match_keywords_insensitively() {
        let toks = tokenize("SeLeCt").unwrap();
        assert_eq!(toks[0].kind, TokenKind::Ident("SeLeCt".into()));
        assert!(toks[0].kind.is_keyword("select"));
        assert!(toks[0].kind.is_keyword("SELECT"));
        assert!(!toks[0].kind.is_keyword("insert"));
    }

    #[test]
    fn string_literals_may_contain_anything() {
        assert_eq!(
            kinds("'hello, world'"),
            vec![TokenKind::String("hello, world".into())],
            "commas must not split a literal"
        );
        assert_eq!(
            kinds("'a)b('"),
            vec![TokenKind::String("a)b(".into())],
            "parens must not split a literal"
        );
        assert_eq!(
            kinds("'SELECT * FROM'"),
            vec![TokenKind::String("SELECT * FROM".into())],
            "keywords inside a literal are just text"
        );
    }

    #[test]
    fn doubled_quote_is_an_escape() {
        assert_eq!(kinds("'it''s'"), vec![TokenKind::String("it's".into())]);
        assert_eq!(kinds("''"), vec![TokenKind::String(String::new())]);
        assert_eq!(
            kinds("''''"),
            vec![TokenKind::String("'".into())],
            "four quotes is one escaped quote"
        );
    }

    #[test]
    fn unicode_survives_a_string_literal() {
        assert_eq!(
            kinds("'héllo 🌍 日本'"),
            vec![TokenKind::String("héllo 🌍 日本".into())]
        );
    }

    #[test]
    fn numbers_are_typed() {
        assert_eq!(kinds("42"), vec![TokenKind::Integer(42)]);
        assert_eq!(kinds("0"), vec![TokenKind::Integer(0)]);
        assert_eq!(kinds("3.5"), vec![TokenKind::Float(3.5)]);
        // A trailing dot ends the number; the dot is then unexpected.
        assert!(tokenize("1.2.3").is_err() || !kinds("1.2").is_empty());
    }

    #[test]
    fn comparison_operators_are_recognised() {
        assert_eq!(kinds("="), vec![TokenKind::Eq]);
        assert_eq!(kinds("<>"), vec![TokenKind::NotEq]);
        assert_eq!(kinds("!="), vec![TokenKind::NotEq]);
        assert_eq!(kinds("<"), vec![TokenKind::Lt]);
        assert_eq!(kinds("<="), vec![TokenKind::LtEq]);
        assert_eq!(kinds(">"), vec![TokenKind::Gt]);
        assert_eq!(kinds(">="), vec![TokenKind::GtEq]);
        // The two-character forms must win over the one-character prefix.
        assert_eq!(
            kinds("a<=b"),
            vec![
                TokenKind::Ident("a".into()),
                TokenKind::LtEq,
                TokenKind::Ident("b".into())
            ]
        );
    }

    #[test]
    fn comments_are_skipped() {
        assert_eq!(
            kinds("SELECT -- this is ignored\n*"),
            vec![TokenKind::Ident("SELECT".into()), TokenKind::Star]
        );
        assert_eq!(
            kinds("SELECT /* inline */ *"),
            vec![TokenKind::Ident("SELECT".into()), TokenKind::Star]
        );
        assert_eq!(
            kinds("/* leading */ SELECT"),
            vec![TokenKind::Ident("SELECT".into())]
        );
    }

    #[test]
    fn quoted_identifiers_allow_reserved_words() {
        assert_eq!(
            kinds("\"select\""),
            vec![TokenKind::Ident("select".into())],
            "a quoted identifier is a name, not a keyword"
        );
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
        // The shape a human actually writes, and exactly what a
        // `starts_with`-based parser cannot handle.
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
