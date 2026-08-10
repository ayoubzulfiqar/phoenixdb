//! Parser tests.
//!
//! The `whitespace_and_case` group is the regression suite for the failures
//! measured in MagnumDB's parser (see the module docs on `crate::sql`): every
//! case there is one that a `starts_with` + `split(' ')` design rejects.
//!
//! The whole file is gated on the `sql` feature so the lean default build —
//! the one embedded in a Flutter app — still compiles its test suite.
#![cfg(feature = "sql")]

use phoenixdb::sql::{ColumnDef, ComparisonOp, Statement, Value, WhereClause, parse};

// ---- CREATE TABLE ---------------------------------------------------------

#[test]
fn create_table_basic() {
    let stmt = parse("CREATE TABLE users (id, name)").unwrap();
    match stmt {
        Statement::CreateTable {
            table,
            columns,
            if_not_exists,
        } => {
            assert_eq!(table, "users");
            assert_eq!(columns.len(), 2);
            assert_eq!(columns[0].name, "id");
            assert_eq!(columns[1].name, "name");
            assert!(!if_not_exists);
        }
        other => panic!("expected CreateTable, got {other:?}"),
    }
}

#[test]
fn create_table_with_types_and_constraints() {
    let stmt =
        parse("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT NOT NULL, bio VARCHAR(255))")
            .unwrap();
    let Statement::CreateTable { columns, .. } = stmt else {
        panic!("expected CreateTable");
    };
    assert_eq!(
        columns[0],
        ColumnDef {
            name: "id".into(),
            data_type: "INTEGER".into(),
            primary_key: true,
            not_null: false,
        }
    );
    assert_eq!(columns[1].data_type, "TEXT");
    assert!(columns[1].not_null);
    // VARCHAR(255): the width is skipped, the type is kept.
    assert_eq!(columns[2].data_type, "VARCHAR");
}

#[test]
fn create_table_if_not_exists() {
    let stmt = parse("CREATE TABLE IF NOT EXISTS t (id)").unwrap();
    let Statement::CreateTable { if_not_exists, .. } = stmt else {
        panic!("expected CreateTable");
    };
    assert!(if_not_exists);
}

// ---- whitespace and case: the MagnumDB regression suite -------------------

mod whitespace_and_case {
    use super::*;

    /// Every spelling below is valid SQL and must produce the same AST.
    #[test]
    fn equivalent_spellings_parse_identically() {
        let canonical = parse("CREATE TABLE users (id, name)").unwrap();
        for variant in [
            "CREATE  TABLE users (id, name)",      // double space
            "CREATE   TABLE   users   (id, name)", // many spaces
            "CREATE\nTABLE users (id, name)",      // newline
            "CREATE\tTABLE users (id, name)",      // tab
            "CREATE\r\nTABLE users (id, name)",    // CRLF
            "  CREATE TABLE users (id, name)  ",   // surrounding space
            "CREATE TABLE users (id,name)",        // no space after comma
            "CREATE TABLE users ( id , name )",    // spaces inside parens
            "CREATE TABLE users (id, name);",      // trailing semicolon
        ] {
            assert_eq!(
                parse(variant).unwrap(),
                canonical,
                "variant failed to match: {variant:?}"
            );
        }
    }

    #[test]
    fn a_realistic_multiline_statement_parses() {
        // The way a human actually writes it.
        let sql = "CREATE TABLE users (\n    id   INTEGER PRIMARY KEY,\n    \
                   name TEXT NOT NULL,\n    email TEXT\n);";
        let Statement::CreateTable { table, columns, .. } = parse(sql).unwrap() else {
            panic!("expected CreateTable");
        };
        assert_eq!(table, "users");
        assert_eq!(columns.len(), 3);
        assert!(columns[0].primary_key);
    }

    #[test]
    fn keywords_are_case_insensitive() {
        let canonical = parse("SELECT * FROM t WHERE id = 1").unwrap();
        for variant in [
            "select * from t where id = 1",
            "SeLeCt * FrOm t WhErE id = 1",
            "SELECT * FROM t WHERE id=1",
        ] {
            assert_eq!(parse(variant).unwrap(), canonical, "variant: {variant:?}");
        }
    }

    #[test]
    fn identifier_case_is_preserved() {
        // Keywords are case-insensitive, but names are not folded.
        let Statement::Select { table, columns, .. } =
            parse("select UserName from MyTable").unwrap()
        else {
            panic!("expected Select");
        };
        assert_eq!(table, "MyTable");
        assert_eq!(columns, vec!["UserName"]);
    }

    #[test]
    fn comments_are_ignored() {
        let stmt = parse("SELECT * FROM t -- trailing comment").unwrap();
        assert_eq!(stmt.table(), "t");
        let stmt = parse("SELECT /* inline */ * FROM t").unwrap();
        assert_eq!(stmt.table(), "t");
        let sql = "-- leading\nSELECT * FROM t";
        assert_eq!(parse(sql).unwrap().table(), "t");
    }
}

// ---- INSERT ---------------------------------------------------------------

#[test]
fn insert_with_and_without_columns() {
    let Statement::Insert {
        table,
        columns,
        rows,
    } = parse("INSERT INTO users VALUES (1, 'bob')").unwrap()
    else {
        panic!("expected Insert");
    };
    assert_eq!(table, "users");
    assert!(columns.is_empty(), "no column list means all columns");
    assert_eq!(
        rows,
        vec![vec![Value::Integer(1), Value::Text("bob".into())]]
    );

    let Statement::Insert { columns, .. } =
        parse("INSERT INTO users (id, name) VALUES (1, 'bob')").unwrap()
    else {
        panic!("expected Insert");
    };
    assert_eq!(columns, vec!["id", "name"]);
}

#[test]
fn insert_multiple_rows() {
    let Statement::Insert { rows, .. } =
        parse("INSERT INTO t VALUES (1, 'a'), (2, 'b'), (3, 'c')").unwrap()
    else {
        panic!("expected Insert");
    };
    assert_eq!(rows.len(), 3);
    assert_eq!(rows[2], vec![Value::Integer(3), Value::Text("c".into())]);
}

#[test]
fn insert_value_types_are_preserved() {
    let Statement::Insert { rows, .. } =
        parse("INSERT INTO t VALUES (42, 3.5, 'text', NULL)").unwrap()
    else {
        panic!("expected Insert");
    };
    assert_eq!(
        rows[0],
        vec![
            Value::Integer(42),
            Value::Float(3.5),
            Value::Text("text".into()),
            Value::Null,
        ]
    );
}

#[test]
fn string_literals_may_contain_sql_syntax() {
    // Commas, parens and keywords inside quotes must not confuse the parser.
    let Statement::Insert { rows, .. } =
        parse("INSERT INTO t VALUES ('a, b', 'c) FROM d', 'SELECT *')").unwrap()
    else {
        panic!("expected Insert");
    };
    assert_eq!(
        rows[0],
        vec![
            Value::Text("a, b".into()),
            Value::Text("c) FROM d".into()),
            Value::Text("SELECT *".into()),
        ]
    );
}

#[test]
fn escaped_quotes_in_literals() {
    let Statement::Insert { rows, .. } = parse("INSERT INTO t VALUES ('it''s here')").unwrap()
    else {
        panic!("expected Insert");
    };
    assert_eq!(rows[0], vec![Value::Text("it's here".into())]);
}

#[test]
fn insert_arity_mismatch_is_rejected() {
    let err = parse("INSERT INTO t (a, b) VALUES (1)").unwrap_err();
    let msg = format!("{err}");
    assert!(msg.contains("2 column"), "unhelpful message: {msg}");
    assert!(msg.contains("1 value"), "unhelpful message: {msg}");
}

// ---- SELECT ---------------------------------------------------------------

#[test]
fn select_star_and_projection() {
    let Statement::Select { columns, .. } = parse("SELECT * FROM t").unwrap() else {
        panic!("expected Select");
    };
    assert!(columns.is_empty(), "`*` is the empty projection");

    let Statement::Select { columns, .. } = parse("SELECT a, b, c FROM t").unwrap() else {
        panic!("expected Select");
    };
    assert_eq!(columns, vec!["a", "b", "c"]);
}

#[test]
fn select_with_every_comparison_operator() {
    for (sql_op, expected) in [
        ("=", ComparisonOp::Eq),
        ("<>", ComparisonOp::NotEq),
        ("!=", ComparisonOp::NotEq),
        ("<", ComparisonOp::Lt),
        ("<=", ComparisonOp::LtEq),
        (">", ComparisonOp::Gt),
        (">=", ComparisonOp::GtEq),
    ] {
        let sql = format!("SELECT * FROM t WHERE age {sql_op} 30");
        let Statement::Select { filter, .. } = parse(&sql).unwrap() else {
            panic!("expected Select");
        };
        let WhereClause::Single(p) = filter.unwrap() else {
            panic!("expected a single predicate");
        };
        assert_eq!(p.op, expected, "operator {sql_op} mis-parsed");
        assert_eq!(p.column, "age");
        assert_eq!(p.value, Value::Integer(30));
    }
}

#[test]
fn select_with_and_or() {
    let Statement::Select { filter, .. } =
        parse("SELECT * FROM t WHERE a = 1 AND b = 2 AND c = 3").unwrap()
    else {
        panic!("expected Select");
    };
    match filter.unwrap() {
        WhereClause::And(ps) => assert_eq!(ps.len(), 3),
        other => panic!("expected And, got {other:?}"),
    }

    let Statement::Select { filter, .. } = parse("SELECT * FROM t WHERE a = 1 OR b = 2").unwrap()
    else {
        panic!("expected Select");
    };
    match filter.unwrap() {
        WhereClause::Or(ps) => assert_eq!(ps.len(), 2),
        other => panic!("expected Or, got {other:?}"),
    }
}

#[test]
fn mixing_and_or_is_rejected_rather_than_guessed() {
    // Silently picking a precedence here would return wrong rows.
    let err = parse("SELECT * FROM t WHERE a = 1 AND b = 2 OR c = 3").unwrap_err();
    assert!(format!("{err}").contains("ambiguous"), "got: {err}");
}

#[test]
fn select_order_by_and_limit() {
    let Statement::Select {
        order_by, limit, ..
    } = parse("SELECT * FROM t ORDER BY name DESC LIMIT 10").unwrap()
    else {
        panic!("expected Select");
    };
    assert_eq!(order_by, Some(("name".to_string(), true)));
    assert_eq!(limit, Some(10));

    let Statement::Select { order_by, .. } = parse("SELECT * FROM t ORDER BY name").unwrap() else {
        panic!("expected Select");
    };
    assert_eq!(
        order_by,
        Some(("name".to_string(), false)),
        "ASC by default"
    );
}

#[test]
fn negative_limit_is_rejected() {
    assert!(parse("SELECT * FROM t LIMIT -1").is_err());
}

// ---- UPDATE / DELETE ------------------------------------------------------

#[test]
fn update_single_and_multiple_assignments() {
    let Statement::Update {
        table,
        assignments,
        filter,
    } = parse("UPDATE users SET name = 'x' WHERE id = 1").unwrap()
    else {
        panic!("expected Update");
    };
    assert_eq!(table, "users");
    assert_eq!(
        assignments,
        vec![("name".to_string(), Value::Text("x".into()))]
    );
    assert!(filter.is_some());

    let Statement::Update { assignments, .. } =
        parse("UPDATE t SET a = 1, b = 'two', c = 3.5").unwrap()
    else {
        panic!("expected Update");
    };
    assert_eq!(assignments.len(), 3);
    assert_eq!(assignments[2], ("c".to_string(), Value::Float(3.5)));
}

#[test]
fn update_and_delete_without_where_affect_everything() {
    let Statement::Update { filter, .. } = parse("UPDATE t SET a = 1").unwrap() else {
        panic!("expected Update");
    };
    assert!(filter.is_none());

    let Statement::Delete { table, filter } = parse("DELETE FROM t").unwrap() else {
        panic!("expected Delete");
    };
    assert_eq!(table, "t");
    assert!(filter.is_none());
}

#[test]
fn drop_table() {
    let Statement::DropTable { table, if_exists } = parse("DROP TABLE t").unwrap() else {
        panic!("expected DropTable");
    };
    assert_eq!(table, "t");
    assert!(!if_exists);

    let Statement::DropTable { if_exists, .. } = parse("DROP TABLE IF EXISTS t").unwrap() else {
        panic!("expected DropTable");
    };
    assert!(if_exists);
}

// ---- error handling -------------------------------------------------------

#[test]
fn malformed_statements_are_rejected_with_a_position() {
    for (sql, expected_fragment) in [
        ("", "empty"),
        ("CREATE", "expected `table`"),
        ("CREATE TABLE", "expected a table name"),
        ("CREATE TABLE t", "expected `(`"),
        ("CREATE TABLE t (id", "expected `)`"),
        ("SELECT", "expected a column name"),
        ("SELECT * FROM", "expected a table name"),
        ("INSERT INTO t", "expected `values`"),
        ("UPDATE t", "expected `set`"),
        ("UPDATE t SET a", "expected `=`"),
        ("DELETE", "expected `from`"),
        ("FROBNICATE t", "unsupported statement"),
        ("SELECT * FROM t WHERE a", "comparison operator"),
    ] {
        let err = parse(sql).unwrap_err();
        let msg = format!("{err}").to_lowercase();
        assert!(
            msg.contains(&expected_fragment.to_lowercase()),
            "for {sql:?}\n  expected message containing {expected_fragment:?}\n  got: {msg}"
        );
    }
}

#[test]
fn unclosed_paren_is_an_error_not_a_silent_accept() {
    // MagnumDB's parser accepts this as a valid CREATE TABLE.
    assert!(
        parse("CREATE TABLE t (id").is_err(),
        "an unclosed column list must be rejected"
    );
}

#[test]
fn unterminated_string_is_rejected() {
    let err = parse("INSERT INTO t VALUES ('abc").unwrap_err();
    assert!(
        format!("{err}").contains("unterminated string"),
        "got: {err}"
    );
}

#[test]
fn a_bare_word_where_a_value_belongs_suggests_quoting() {
    let err = parse("INSERT INTO t VALUES (abc)").unwrap_err();
    assert!(
        format!("{err}").contains("single quotes"),
        "the message should hint at the fix, got: {err}"
    );
}

#[test]
fn trailing_garbage_is_rejected() {
    assert!(parse("SELECT * FROM t EXTRA").is_err());
    assert!(
        parse("SELECT * FROM t; SELECT * FROM u").is_err(),
        "multiple statements need an explicit batching API"
    );
}

#[test]
fn parser_never_panics_on_arbitrary_input() {
    // Fixed adversarial corpus: the parser must always return, never unwind.
    for sql in [
        "((((",
        "))))",
        "''''",
        "\"\"\"",
        ";;;;",
        "SELECT SELECT SELECT",
        "WHERE = = =",
        "\0",
        "\u{1}\u{2}",
        "-- ",
        "/*",
        "SELECT * FROM t WHERE",
        "1 2 3",
        "....",
        "INSERT INTO VALUES",
        "\n\n\n",
        "SELECT * FROM t LIMIT abc",
    ] {
        let _ = parse(sql); // must not panic
    }
}

// ---- statement metadata ---------------------------------------------------

#[test]
fn mutation_flag_and_kind_name_are_correct() {
    assert!(!parse("SELECT * FROM t").unwrap().is_mutation());
    assert!(parse("INSERT INTO t VALUES (1)").unwrap().is_mutation());
    assert!(parse("UPDATE t SET a = 1").unwrap().is_mutation());
    assert!(parse("DELETE FROM t").unwrap().is_mutation());
    assert!(parse("CREATE TABLE t (a)").unwrap().is_mutation());
    assert!(parse("DROP TABLE t").unwrap().is_mutation());

    assert_eq!(parse("SELECT * FROM t").unwrap().kind_name(), "select");
    assert_eq!(parse("DROP TABLE t").unwrap().kind_name(), "drop_table");
}

#[test]
fn quoted_identifiers_allow_reserved_words_as_names() {
    let Statement::Select { table, columns, .. } =
        parse("SELECT \"select\" FROM \"from\"").unwrap()
    else {
        panic!("expected Select");
    };
    assert_eq!(table, "from");
    assert_eq!(columns, vec!["select"]);
}
