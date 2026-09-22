//! Parser tests.
//!
//! The `whitespace_and_case` group is the regression suite for the failures
//! measured in MagnumDB's parser (see the module docs on `crate::sql`): every
//! case there is one that a `starts_with` + `split(' ')` design rejects.
//!
//! The whole file is gated on the `sql` feature so the lean default build —
//! the one embedded in a Flutter app — still compiles its test suite.
#![cfg(feature = "sql")]

use phoenixdb::sql::{
    AggFunc, ColumnDef, ComparisonOp, Expr, OrderItem, SelectItem, Statement, Value, parse,
    parse_with_params,
};

fn lit(v: Value) -> Expr {
    Expr::Literal(v)
}

fn col(name: &str) -> Expr {
    Expr::Column(name.into())
}

fn columns_of(items: &[SelectItem]) -> Vec<String> {
    items
        .iter()
        .map(|i| match i {
            SelectItem::Column { name, .. } => name.clone(),
            other => panic!("expected a plain column, got {other:?}"),
        })
        .collect()
}

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
    // VARCHAR(255): the size is parsed strictly and kept with the type.
    assert_eq!(columns[2].data_type, "VARCHAR(255)");
}

#[test]
fn a_type_size_must_be_well_formed() {
    // Regression: the parser used to skip to the next `)`, silently
    // swallowing the rest of the column list.
    assert!(parse("CREATE TABLE s (a VARCHAR(, b INTEGER NOT NULL, c TEXT))").is_err());
    assert!(parse("CREATE TABLE s (a VARCHAR(x))").is_err());
    let Statement::CreateTable { columns, .. } =
        parse("CREATE TABLE s (a DECIMAL(10, 2), b INT)").unwrap()
    else {
        panic!("expected CreateTable");
    };
    assert_eq!(columns[0].data_type, "DECIMAL(10,2)");
    assert_eq!(columns.len(), 2);
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
        let Statement::Select { table, items, .. } = parse("select UserName from MyTable").unwrap()
        else {
            panic!("expected Select");
        };
        assert_eq!(table, "MyTable");
        assert_eq!(columns_of(&items), vec!["UserName"]);
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
        vec![vec![lit(Value::Integer(1)), lit(Value::Text("bob".into()))]]
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
    assert_eq!(
        rows[2],
        vec![lit(Value::Integer(3)), lit(Value::Text("c".into()))]
    );
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
            lit(Value::Integer(42)),
            lit(Value::Float(3.5)),
            lit(Value::Text("text".into())),
            lit(Value::Null),
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
            lit(Value::Text("a, b".into())),
            lit(Value::Text("c) FROM d".into())),
            lit(Value::Text("SELECT *".into())),
        ]
    );
}

#[test]
fn escaped_quotes_in_literals() {
    let Statement::Insert { rows, .. } = parse("INSERT INTO t VALUES ('it''s here')").unwrap()
    else {
        panic!("expected Insert");
    };
    assert_eq!(rows[0], vec![lit(Value::Text("it's here".into()))]);
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
    let Statement::Select { items, .. } = parse("SELECT * FROM t").unwrap() else {
        panic!("expected Select");
    };
    assert_eq!(items, vec![SelectItem::Wildcard]);

    let Statement::Select { items, .. } = parse("SELECT a, b, c FROM t").unwrap() else {
        panic!("expected Select");
    };
    assert_eq!(columns_of(&items), vec!["a", "b", "c"]);
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
        assert_eq!(
            filter.unwrap(),
            Expr::Compare {
                left: Box::new(col("age")),
                op: expected,
                right: Box::new(lit(Value::Integer(30))),
            },
            "operator {sql_op} mis-parsed"
        );
    }
}

#[test]
fn select_with_and_or() {
    let Statement::Select { filter, .. } =
        parse("SELECT * FROM t WHERE a = 1 AND b = 2 AND c = 3").unwrap()
    else {
        panic!("expected Select");
    };
    assert!(matches!(filter.unwrap(), Expr::And(_, _)));

    let Statement::Select { filter, .. } = parse("SELECT * FROM t WHERE a = 1 OR b = 2").unwrap()
    else {
        panic!("expected Select");
    };
    assert!(matches!(filter.unwrap(), Expr::Or(_, _)));
}

#[test]
fn and_binds_tighter_than_or_and_parentheses_override() {
    let eq = |c: &str, v: i64| Expr::Compare {
        left: Box::new(col(c)),
        op: ComparisonOp::Eq,
        right: Box::new(lit(Value::Integer(v))),
    };
    let Statement::Select { filter, .. } =
        parse("SELECT * FROM t WHERE a = 1 AND b = 2 OR c = 3").unwrap()
    else {
        panic!("expected Select");
    };
    assert_eq!(
        filter.unwrap(),
        Expr::Or(
            Box::new(Expr::And(Box::new(eq("a", 1)), Box::new(eq("b", 2)))),
            Box::new(eq("c", 3))
        )
    );
    let Statement::Select { filter, .. } =
        parse("SELECT * FROM t WHERE a = 1 AND (b = 2 OR c = 3)").unwrap()
    else {
        panic!("expected Select");
    };
    assert_eq!(
        filter.unwrap(),
        Expr::And(
            Box::new(eq("a", 1)),
            Box::new(Expr::Or(Box::new(eq("b", 2)), Box::new(eq("c", 3))))
        )
    );
}

#[test]
fn rich_predicates_parse() {
    let where_of = |sql: &str| -> Expr {
        let Statement::Select { filter, .. } = parse(sql).unwrap() else {
            panic!("expected Select");
        };
        filter.unwrap()
    };
    assert!(matches!(
        where_of("SELECT * FROM t WHERE a IS NOT NULL"),
        Expr::IsNull { negated: true, .. }
    ));
    assert!(matches!(
        where_of("SELECT * FROM t WHERE a = NULL"),
        Expr::IsNull { negated: false, .. }
    ));
    assert!(matches!(
        where_of("SELECT * FROM t WHERE a NOT IN (1, 2, 'x')"),
        Expr::InList { negated: true, ref list, .. } if list.len() == 3
    ));
    assert!(matches!(
        where_of("SELECT * FROM t WHERE a BETWEEN 1 AND 5"),
        Expr::Between { negated: false, .. }
    ));
    assert!(matches!(
        where_of("SELECT * FROM t WHERE name ILIKE 'a%'"),
        Expr::Like {
            case_insensitive: true,
            ..
        }
    ));
    assert!(matches!(
        where_of("SELECT * FROM t WHERE NOT (a = 1)"),
        Expr::Not(_)
    ));
    assert!(matches!(
        where_of("SELECT * FROM t WHERE a > b"),
        Expr::Compare { ref right, .. } if **right == col("b")
    ));
}

#[test]
fn parameters_are_numbered() {
    let (stmt, n) = parse_with_params("SELECT * FROM t WHERE a = ? AND b IN (?, ?3)").unwrap();
    assert_eq!(n, 3);
    let Statement::Select { filter, .. } = stmt else {
        panic!("expected Select");
    };
    let Expr::And(left, _) = filter.unwrap() else {
        panic!("expected And");
    };
    assert!(matches!(*left, Expr::Compare { ref right, .. } if **right == Expr::Param(0)));
    let (_, n) = parse_with_params("INSERT INTO t VALUES (?, ?)").unwrap();
    assert_eq!(n, 2);
}

#[test]
fn aggregates_group_by_and_offset_parse() {
    let Statement::Select {
        items,
        group_by,
        order_by,
        limit,
        offset,
        ..
    } = parse(
        "SELECT dept, COUNT(*) AS n, AVG(salary), MAX(DISTINCT age) FROM e \
         GROUP BY dept ORDER BY n DESC, dept LIMIT 5 OFFSET 10",
    )
    .unwrap()
    else {
        panic!("expected Select");
    };
    assert_eq!(items.len(), 4);
    assert_eq!(
        items[1],
        SelectItem::Aggregate {
            func: AggFunc::Count,
            column: None,
            distinct: false,
            alias: Some("n".into()),
        }
    );
    assert!(matches!(
        items[3],
        SelectItem::Aggregate { distinct: true, .. }
    ));
    assert_eq!(group_by, vec!["dept"]);
    assert_eq!(
        order_by,
        vec![
            OrderItem {
                column: "n".into(),
                desc: true
            },
            OrderItem {
                column: "dept".into(),
                desc: false
            },
        ]
    );
    assert_eq!((limit, offset), (Some(5), Some(10)));
    assert!(parse("SELECT SUM(*) FROM t").is_err());
    assert!(parse("SELECT nosuch(a) FROM t").is_err());
    // `count` alone is an ordinary column name.
    let Statement::Select { items, .. } = parse("SELECT count FROM t").unwrap() else {
        panic!("expected Select");
    };
    assert_eq!(columns_of(&items), vec!["count"]);
}

#[test]
fn reserved_words_and_duplicates_are_rejected() {
    // Regression: `SELECT FROM FROM k` used to project a column named FROM.
    assert!(parse("SELECT FROM FROM k").is_err());
    assert!(
        parse("SELECT \"from\" FROM k").is_ok(),
        "quoting makes it a name"
    );
    assert!(parse("INSERT INTO u (id, id) VALUES (1, 2)").is_err());
    assert!(parse("UPDATE u SET a = 1, A = 2").is_err());
    // Regression: `1OR` used to lex as `1` followed by `OR`.
    assert!(parse("SELECT * FROM t WHERE id = 1OR id = 2").is_err());
}

#[test]
fn select_order_by_and_limit() {
    let Statement::Select {
        order_by, limit, ..
    } = parse("SELECT * FROM t ORDER BY name DESC LIMIT 10").unwrap()
    else {
        panic!("expected Select");
    };
    assert_eq!(
        order_by,
        vec![OrderItem {
            column: "name".into(),
            desc: true
        }]
    );
    assert_eq!(limit, Some(10));

    let Statement::Select { order_by, .. } = parse("SELECT * FROM t ORDER BY name").unwrap() else {
        panic!("expected Select");
    };
    assert_eq!(
        order_by,
        vec![OrderItem {
            column: "name".into(),
            desc: false
        }],
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
        vec![("name".to_string(), lit(Value::Text("x".into())))]
    );
    assert!(filter.is_some());

    let Statement::Update { assignments, .. } =
        parse("UPDATE t SET a = 1, b = 'two', c = 3.5").unwrap()
    else {
        panic!("expected Update");
    };
    assert_eq!(assignments.len(), 3);
    assert_eq!(assignments[2], ("c".to_string(), lit(Value::Float(3.5))));
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
    let Statement::Select { table, items, .. } = parse("SELECT \"select\" FROM \"from\"").unwrap()
    else {
        panic!("expected Select");
    };
    assert_eq!(table, "from");
    assert_eq!(columns_of(&items), vec!["select"]);
}
