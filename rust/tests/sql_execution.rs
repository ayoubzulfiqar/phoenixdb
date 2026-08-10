//! End-to-end SQL execution tests.
//!
//! These run real statements against a real `Database` on disk, so they cover
//! the whole path: parse -> execute -> storage -> reopen. A parser-only test
//! cannot catch a row that never reaches the disk.
#![cfg(feature = "sql")]

use phoenixdb::sql::{Cell, Executor, QueryResult};
use phoenixdb::{Database, Options};

/// Opens a temporary database with an executor bound to it.
fn setup() -> (tempfile::TempDir, Database) {
    let dir = tempfile::tempdir().unwrap();
    let db = Database::open(dir.path().join("sql.pdb"), Options::default()).unwrap();
    (dir, db)
}

/// Runs `sql`, asserting it succeeds.
fn run(db: &Database, sql: &str) -> QueryResult {
    Executor::new(db)
        .run(sql)
        .unwrap_or_else(|e| panic!("{sql}\n  failed: {e}"))
}

/// Extracts the rows from a `SELECT` result.
fn rows(r: &QueryResult) -> &Vec<Vec<Cell>> {
    match r {
        QueryResult::Rows { rows, .. } => rows,
        other => panic!("expected rows, got {other:?}"),
    }
}

// ---- the core loop --------------------------------------------------------

#[test]
fn create_insert_select_roundtrip() {
    let (_d, db) = setup();
    run(&db, "CREATE TABLE users (id INTEGER, name TEXT)");
    assert_eq!(
        run(&db, "INSERT INTO users VALUES (1, 'alice')").affected(),
        1
    );
    run(&db, "INSERT INTO users VALUES (2, 'bob')");

    let r = run(&db, "SELECT * FROM users");
    assert_eq!(rows(&r).len(), 2);
    assert_eq!(
        rows(&r)[0],
        vec![Cell::Integer(1), Cell::Text("alice".into())]
    );
    assert_eq!(
        rows(&r)[1],
        vec![Cell::Integer(2), Cell::Text("bob".into())]
    );
}

#[test]
fn data_survives_reopen() {
    // The test that proves SQL rows reach durable storage.
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("sql.pdb");
    {
        let db = Database::open(&path, Options::default()).unwrap();
        run(&db, "CREATE TABLE t (id INTEGER, label TEXT)");
        run(
            &db,
            "INSERT INTO t VALUES (1, 'one'), (2, 'two'), (3, 'three')",
        );
        db.checkpoint().unwrap();
    }
    let db = Database::open(&path, Options::default()).unwrap();
    let r = run(&db, "SELECT label FROM t ORDER BY id");
    assert_eq!(rows(&r).len(), 3, "rows must survive a restart");
    assert_eq!(rows(&r)[0][0], Cell::Text("one".into()));
    assert_eq!(rows(&r)[2][0], Cell::Text("three".into()));
}

#[test]
fn multi_row_insert_is_atomic() {
    let (_d, db) = setup();
    run(&db, "CREATE TABLE t (a INTEGER, b INTEGER)");
    assert_eq!(
        run(&db, "INSERT INTO t VALUES (1, 1), (2, 2), (3, 3)").affected(),
        3
    );
    assert_eq!(rows(&run(&db, "SELECT * FROM t")).len(), 3);
}

#[test]
fn insert_with_named_columns_defaults_the_rest_to_null() {
    let (_d, db) = setup();
    run(&db, "CREATE TABLE t (a INTEGER, b TEXT, c INTEGER)");
    run(&db, "INSERT INTO t (a, c) VALUES (1, 3)");
    let r = run(&db, "SELECT * FROM t");
    assert_eq!(
        rows(&r)[0],
        vec![Cell::Integer(1), Cell::Null, Cell::Integer(3)]
    );
}

#[test]
fn column_order_follows_the_insert_list_not_the_schema() {
    let (_d, db) = setup();
    run(&db, "CREATE TABLE t (a INTEGER, b TEXT)");
    run(&db, "INSERT INTO t (b, a) VALUES ('x', 7)");
    let r = run(&db, "SELECT * FROM t");
    assert_eq!(rows(&r)[0], vec![Cell::Integer(7), Cell::Text("x".into())]);
}

// ---- projection, filtering, ordering --------------------------------------

#[test]
fn projection_selects_named_columns_in_order() {
    let (_d, db) = setup();
    run(&db, "CREATE TABLE t (a INTEGER, b TEXT, c INTEGER)");
    run(&db, "INSERT INTO t VALUES (1, 'x', 3)");

    let r = run(&db, "SELECT c, a FROM t");
    match &r {
        QueryResult::Rows { columns, rows } => {
            assert_eq!(columns, &vec!["c".to_string(), "a".to_string()]);
            assert_eq!(rows[0], vec![Cell::Integer(3), Cell::Integer(1)]);
        }
        other => panic!("expected rows, got {other:?}"),
    }
}

#[test]
fn where_filters_with_every_operator() {
    let (_d, db) = setup();
    run(&db, "CREATE TABLE t (n INTEGER)");
    run(&db, "INSERT INTO t VALUES (10), (20), (30)");

    for (sql, expected) in [
        ("SELECT n FROM t WHERE n = 20", 1),
        ("SELECT n FROM t WHERE n <> 20", 2),
        ("SELECT n FROM t WHERE n < 20", 1),
        ("SELECT n FROM t WHERE n <= 20", 2),
        ("SELECT n FROM t WHERE n > 20", 1),
        ("SELECT n FROM t WHERE n >= 20", 2),
    ] {
        assert_eq!(rows(&run(&db, sql)).len(), expected, "{sql}");
    }
}

#[test]
fn where_and_or() {
    let (_d, db) = setup();
    run(&db, "CREATE TABLE t (a INTEGER, b INTEGER)");
    run(&db, "INSERT INTO t VALUES (1, 1), (1, 2), (2, 2)");

    assert_eq!(
        rows(&run(&db, "SELECT * FROM t WHERE a = 1 AND b = 2")).len(),
        1
    );
    assert_eq!(
        rows(&run(&db, "SELECT * FROM t WHERE a = 2 OR b = 1")).len(),
        2
    );
}

#[test]
fn order_by_ascending_and_descending() {
    let (_d, db) = setup();
    run(&db, "CREATE TABLE t (n INTEGER)");
    run(&db, "INSERT INTO t VALUES (3), (1), (2)");

    let asc = run(&db, "SELECT n FROM t ORDER BY n");
    assert_eq!(
        rows(&asc).iter().map(|r| r[0].clone()).collect::<Vec<_>>(),
        vec![Cell::Integer(1), Cell::Integer(2), Cell::Integer(3)]
    );
    let desc = run(&db, "SELECT n FROM t ORDER BY n DESC");
    assert_eq!(
        rows(&desc).iter().map(|r| r[0].clone()).collect::<Vec<_>>(),
        vec![Cell::Integer(3), Cell::Integer(2), Cell::Integer(1)]
    );
}

#[test]
fn order_by_applies_before_limit() {
    // Limiting first would return an arbitrary subset.
    let (_d, db) = setup();
    run(&db, "CREATE TABLE t (n INTEGER)");
    run(&db, "INSERT INTO t VALUES (5), (1), (4), (2), (3)");
    let r = run(&db, "SELECT n FROM t ORDER BY n DESC LIMIT 2");
    assert_eq!(
        rows(&r).iter().map(|x| x[0].clone()).collect::<Vec<_>>(),
        vec![Cell::Integer(5), Cell::Integer(4)]
    );
}

#[test]
fn order_by_text_and_mixed_numerics() {
    let (_d, db) = setup();
    run(&db, "CREATE TABLE t (s TEXT, n INTEGER)");
    run(
        &db,
        "INSERT INTO t VALUES ('pear', 2), ('apple', 10), ('fig', 1)",
    );

    let r = run(&db, "SELECT s FROM t ORDER BY s");
    assert_eq!(rows(&r)[0][0], Cell::Text("apple".into()));
    // Numeric ordering must be numeric, not lexicographic (10 > 2).
    let r = run(&db, "SELECT n FROM t ORDER BY n DESC");
    assert_eq!(rows(&r)[0][0], Cell::Integer(10));
}

#[test]
fn limit_zero_and_oversized_limit() {
    let (_d, db) = setup();
    run(&db, "CREATE TABLE t (n INTEGER)");
    run(&db, "INSERT INTO t VALUES (1), (2)");
    assert_eq!(rows(&run(&db, "SELECT n FROM t LIMIT 0")).len(), 0);
    assert_eq!(rows(&run(&db, "SELECT n FROM t LIMIT 999")).len(), 2);
}

// ---- UPDATE / DELETE ------------------------------------------------------

#[test]
fn update_matching_rows_only() {
    let (_d, db) = setup();
    run(&db, "CREATE TABLE t (id INTEGER, name TEXT)");
    run(&db, "INSERT INTO t VALUES (1, 'a'), (2, 'b')");

    assert_eq!(
        run(&db, "UPDATE t SET name = 'changed' WHERE id = 1").affected(),
        1
    );
    let r = run(&db, "SELECT name FROM t ORDER BY id");
    assert_eq!(rows(&r)[0][0], Cell::Text("changed".into()));
    assert_eq!(rows(&r)[1][0], Cell::Text("b".into()), "row 2 untouched");
}

#[test]
fn update_without_where_touches_every_row() {
    let (_d, db) = setup();
    run(&db, "CREATE TABLE t (n INTEGER, flag TEXT)");
    run(&db, "INSERT INTO t VALUES (1, 'x'), (2, 'x'), (3, 'x')");
    assert_eq!(run(&db, "UPDATE t SET flag = 'y'").affected(), 3);
    let r = run(&db, "SELECT flag FROM t");
    assert!(rows(&r).iter().all(|x| x[0] == Cell::Text("y".into())));
}

#[test]
fn update_multiple_columns_at_once() {
    let (_d, db) = setup();
    run(&db, "CREATE TABLE t (a INTEGER, b TEXT, c INTEGER)");
    run(&db, "INSERT INTO t VALUES (1, 'x', 1)");
    run(&db, "UPDATE t SET a = 9, b = 'z' WHERE c = 1");
    let r = run(&db, "SELECT * FROM t");
    assert_eq!(
        rows(&r)[0],
        vec![Cell::Integer(9), Cell::Text("z".into()), Cell::Integer(1)]
    );
}

#[test]
fn delete_matching_rows_only() {
    let (_d, db) = setup();
    run(&db, "CREATE TABLE t (n INTEGER)");
    run(&db, "INSERT INTO t VALUES (1), (2), (3)");

    assert_eq!(run(&db, "DELETE FROM t WHERE n = 2").affected(), 1);
    let r = run(&db, "SELECT n FROM t ORDER BY n");
    assert_eq!(
        rows(&r).iter().map(|x| x[0].clone()).collect::<Vec<_>>(),
        vec![Cell::Integer(1), Cell::Integer(3)]
    );
}

#[test]
fn delete_without_where_empties_the_table() {
    let (_d, db) = setup();
    run(&db, "CREATE TABLE t (n INTEGER)");
    run(&db, "INSERT INTO t VALUES (1), (2)");
    assert_eq!(run(&db, "DELETE FROM t").affected(), 2);
    assert_eq!(rows(&run(&db, "SELECT * FROM t")).len(), 0);
    // The table still exists, just empty.
    run(&db, "INSERT INTO t VALUES (3)");
    assert_eq!(rows(&run(&db, "SELECT * FROM t")).len(), 1);
}

#[test]
fn deleted_rows_stay_deleted_after_reopen() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("d.pdb");
    {
        let db = Database::open(&path, Options::default()).unwrap();
        run(&db, "CREATE TABLE t (n INTEGER)");
        run(&db, "INSERT INTO t VALUES (1), (2), (3)");
        run(&db, "DELETE FROM t WHERE n = 2");
        db.checkpoint().unwrap();
    }
    let db = Database::open(&path, Options::default()).unwrap();
    let r = run(&db, "SELECT n FROM t ORDER BY n");
    assert_eq!(rows(&r).len(), 2, "a deleted row must not come back");
}

// ---- schema ---------------------------------------------------------------

#[test]
fn drop_table_removes_rows_and_schema() {
    let (_d, db) = setup();
    run(&db, "CREATE TABLE t (n INTEGER)");
    run(&db, "INSERT INTO t VALUES (1), (2)");
    run(&db, "DROP TABLE t");
    assert!(Executor::new(&db).run("SELECT * FROM t").is_err());

    // Recreating gives a clean table, not the old rows.
    run(&db, "CREATE TABLE t (n INTEGER)");
    assert_eq!(rows(&run(&db, "SELECT * FROM t")).len(), 0);
}

#[test]
fn if_not_exists_and_if_exists_are_idempotent() {
    let (_d, db) = setup();
    run(&db, "CREATE TABLE t (n INTEGER)");
    run(&db, "CREATE TABLE IF NOT EXISTS t (n INTEGER)"); // no error
    assert!(
        Executor::new(&db)
            .run("CREATE TABLE t (n INTEGER)")
            .is_err(),
        "a plain CREATE on an existing table must fail"
    );
    run(&db, "DROP TABLE IF EXISTS t");
    run(&db, "DROP TABLE IF EXISTS t"); // still no error
    assert!(Executor::new(&db).run("DROP TABLE t").is_err());
}

#[test]
fn tables_are_independent() {
    let (_d, db) = setup();
    run(&db, "CREATE TABLE a (n INTEGER)");
    run(&db, "CREATE TABLE b (n INTEGER)");
    run(&db, "INSERT INTO a VALUES (1), (2)");
    run(&db, "INSERT INTO b VALUES (3)");

    assert_eq!(rows(&run(&db, "SELECT * FROM a")).len(), 2);
    assert_eq!(rows(&run(&db, "SELECT * FROM b")).len(), 1);
    run(&db, "DELETE FROM a");
    assert_eq!(
        rows(&run(&db, "SELECT * FROM b")).len(),
        1,
        "b is unaffected"
    );
}

#[test]
fn table_and_column_names_are_case_insensitive() {
    let (_d, db) = setup();
    run(&db, "CREATE TABLE Users (Id INTEGER)");
    run(&db, "INSERT INTO users VALUES (1)");
    assert_eq!(rows(&run(&db, "SELECT ID FROM USERS")).len(), 1);
    assert_eq!(rows(&run(&db, "SELECT * FROM users WHERE id = 1")).len(), 1);
}

// ---- constraints and errors ----------------------------------------------

#[test]
fn not_null_is_enforced_on_insert_and_update() {
    let (_d, db) = setup();
    run(&db, "CREATE TABLE t (a INTEGER NOT NULL, b TEXT)");
    run(&db, "INSERT INTO t VALUES (1, 'x')");

    let e = Executor::new(&db);
    assert!(
        e.run("INSERT INTO t VALUES (NULL, 'y')").is_err(),
        "NOT NULL must reject a null insert"
    );
    assert!(
        e.run("UPDATE t SET a = NULL").is_err(),
        "NOT NULL must reject a null update"
    );
    // The rejected insert must not have landed.
    assert_eq!(rows(&run(&db, "SELECT * FROM t")).len(), 1);
}

#[test]
fn primary_key_implies_not_null() {
    let (_d, db) = setup();
    run(&db, "CREATE TABLE t (id INTEGER PRIMARY KEY, v TEXT)");
    assert!(
        Executor::new(&db)
            .run("INSERT INTO t (v) VALUES ('no id')")
            .is_err()
    );
}

#[test]
fn operations_on_a_missing_table_are_rejected() {
    let (_d, db) = setup();
    let e = Executor::new(&db);
    for sql in [
        "SELECT * FROM ghost",
        "INSERT INTO ghost VALUES (1)",
        "UPDATE ghost SET a = 1",
        "DELETE FROM ghost",
    ] {
        let err = e.run(sql).unwrap_err();
        assert!(format!("{err}").contains("no such table"), "{sql} -> {err}");
    }
}

#[test]
fn unknown_columns_are_rejected() {
    let (_d, db) = setup();
    run(&db, "CREATE TABLE t (a INTEGER)");
    let e = Executor::new(&db);
    assert!(e.run("SELECT nope FROM t").is_err());
    assert!(e.run("SELECT * FROM t WHERE nope = 1").is_err());
    assert!(e.run("UPDATE t SET nope = 1").is_err());
    assert!(e.run("INSERT INTO t (nope) VALUES (1)").is_err());
}

#[test]
fn a_typo_in_where_is_caught_even_on_an_empty_table() {
    // Regression: the filter used to be validated per-row, so with no rows the
    // predicate never ran and a misspelled column silently returned "0 rows" —
    // indistinguishable from a legitimate no-match.
    let (_d, db) = setup();
    run(&db, "CREATE TABLE t (name TEXT)");
    let e = Executor::new(&db);
    assert_eq!(
        rows(&run(&db, "SELECT * FROM t")).len(),
        0,
        "table is empty"
    );

    for sql in [
        "SELECT * FROM t WHERE nmae = 'x'",
        "UPDATE t SET name = 'y' WHERE nmae = 'x'",
        "DELETE FROM t WHERE nmae = 'x'",
    ] {
        let err = e.run(sql).expect_err(&format!(
            "{sql} must be rejected, not silently match nothing"
        ));
        assert!(format!("{err}").contains("no column"), "{sql} -> {err}");
    }

    // The same must hold once the table has rows.
    run(&db, "INSERT INTO t VALUES ('present')");
    assert!(e.run("DELETE FROM t WHERE nmae = 'x'").is_err());
    assert_eq!(rows(&run(&db, "SELECT * FROM t")).len(), 1, "row untouched");
}

#[test]
fn insert_arity_mismatch_is_rejected() {
    let (_d, db) = setup();
    run(&db, "CREATE TABLE t (a INTEGER, b INTEGER)");
    assert!(
        Executor::new(&db)
            .run("INSERT INTO t (a) VALUES (1, 2)")
            .is_err()
    );
}

// ---- SQL semantics --------------------------------------------------------

#[test]
fn null_comparisons_follow_sql_semantics() {
    let (_d, db) = setup();
    run(&db, "CREATE TABLE t (a INTEGER, b TEXT)");
    run(&db, "INSERT INTO t (b) VALUES ('has null a')");
    run(&db, "INSERT INTO t VALUES (1, 'has value')");

    // A NULL cell never satisfies an ordinary comparison.
    assert_eq!(rows(&run(&db, "SELECT * FROM t WHERE a > 0")).len(), 1);
    assert_eq!(rows(&run(&db, "SELECT * FROM t WHERE a < 999")).len(), 1);
    // `= NULL` is supported as an explicit null test.
    assert_eq!(rows(&run(&db, "SELECT * FROM t WHERE a = NULL")).len(), 1);
    assert_eq!(rows(&run(&db, "SELECT * FROM t WHERE a <> NULL")).len(), 1);
}

#[test]
fn type_mismatched_comparisons_match_nothing() {
    let (_d, db) = setup();
    run(&db, "CREATE TABLE t (n INTEGER)");
    run(&db, "INSERT INTO t VALUES (1)");
    // Comparing a number against text is unknown, so no rows — not an error,
    // and not a spurious match.
    assert_eq!(rows(&run(&db, "SELECT * FROM t WHERE n = 'one'")).len(), 0);
}

#[test]
fn integers_and_floats_compare_numerically() {
    let (_d, db) = setup();
    run(&db, "CREATE TABLE t (n INTEGER)");
    run(&db, "INSERT INTO t VALUES (2), (3)");
    assert_eq!(rows(&run(&db, "SELECT * FROM t WHERE n > 2.5")).len(), 1);
}

#[test]
fn string_values_preserve_special_characters() {
    let (_d, db) = setup();
    run(&db, "CREATE TABLE t (s TEXT)");
    run(
        &db,
        "INSERT INTO t VALUES ('a, b'), ('it''s'), ('SELECT *')",
    );
    let r = run(&db, "SELECT s FROM t");
    let got: Vec<String> = rows(&r).iter().map(|x| x[0].display()).collect();
    assert!(got.contains(&"a, b".to_string()));
    assert!(got.contains(&"it's".to_string()));
    assert!(got.contains(&"SELECT *".to_string()));
}

#[test]
fn unicode_survives_the_round_trip() {
    let (_d, db) = setup();
    run(&db, "CREATE TABLE t (s TEXT)");
    run(&db, "INSERT INTO t VALUES ('héllo 🌍 日本語')");
    assert_eq!(
        rows(&run(&db, "SELECT s FROM t"))[0][0].display(),
        "héllo 🌍 日本語"
    );
}

#[test]
fn empty_table_selects_return_no_rows() {
    let (_d, db) = setup();
    run(&db, "CREATE TABLE t (n INTEGER)");
    assert_eq!(rows(&run(&db, "SELECT * FROM t")).len(), 0);
    assert_eq!(run(&db, "UPDATE t SET n = 1").affected(), 0);
    assert_eq!(run(&db, "DELETE FROM t").affected(), 0);
}

#[test]
fn multiline_sql_executes() {
    let (_d, db) = setup();
    run(
        &db,
        "CREATE TABLE users (\n  id   INTEGER PRIMARY KEY,\n  name TEXT NOT NULL\n);",
    );
    run(
        &db,
        "INSERT INTO users\n  (id, name)\nVALUES\n  (1, 'alice');",
    );
    let r = run(&db, "SELECT name\nFROM users\nWHERE id = 1;");
    assert_eq!(rows(&r)[0][0], Cell::Text("alice".into()));
}

#[test]
fn a_larger_dataset_behaves() {
    let (_d, db) = setup();
    run(&db, "CREATE TABLE t (id INTEGER, bucket INTEGER)");
    for i in 0..200i64 {
        run(&db, &format!("INSERT INTO t VALUES ({i}, {})", i % 5));
    }
    assert_eq!(rows(&run(&db, "SELECT * FROM t")).len(), 200);
    assert_eq!(
        rows(&run(&db, "SELECT id FROM t WHERE bucket = 0")).len(),
        40
    );
    let top = run(&db, "SELECT id FROM t ORDER BY id DESC LIMIT 3");
    assert_eq!(
        rows(&top).iter().map(|r| r[0].clone()).collect::<Vec<_>>(),
        vec![Cell::Integer(199), Cell::Integer(198), Cell::Integer(197)]
    );
}

#[test]
fn negative_numbers_round_trip_and_compare() {
    // Regression: the lexer rejected `-`, so negative values were unwritable.
    let (_d, db) = setup();
    run(&db, "CREATE TABLE t (n INTEGER, f INTEGER)");
    run(&db, "INSERT INTO t VALUES (-5, -2.5), (0, 0), (5, 2.5)");

    let r = run(&db, "SELECT n FROM t ORDER BY n");
    assert_eq!(
        rows(&r).iter().map(|x| x[0].clone()).collect::<Vec<_>>(),
        vec![Cell::Integer(-5), Cell::Integer(0), Cell::Integer(5)],
        "negative values must sort below zero"
    );
    assert_eq!(rows(&run(&db, "SELECT * FROM t WHERE n < 0")).len(), 1);
    assert_eq!(rows(&run(&db, "SELECT * FROM t WHERE f = -2.5")).len(), 1);
    run(&db, "UPDATE t SET n = -100 WHERE n = 5");
    assert_eq!(rows(&run(&db, "SELECT * FROM t WHERE n = -100")).len(), 1);
}

#[test]
fn i64_bounds_survive_storage() {
    let (_d, db) = setup();
    run(&db, "CREATE TABLE t (n INTEGER)");
    run(
        &db,
        "INSERT INTO t VALUES (-9223372036854775808), (9223372036854775807)",
    );
    let r = run(&db, "SELECT n FROM t ORDER BY n");
    assert_eq!(rows(&r)[0][0], Cell::Integer(i64::MIN));
    assert_eq!(rows(&r)[1][0], Cell::Integer(i64::MAX));
}

#[test]
fn render_produces_readable_output() {
    let (_d, db) = setup();
    run(&db, "CREATE TABLE t (a INTEGER, b TEXT)");
    run(&db, "INSERT INTO t VALUES (1, 'x')");
    let text = run(&db, "SELECT * FROM t").render();
    assert!(text.contains("a | b"));
    assert!(text.contains("1 | x"));
    assert!(text.contains("(1 row(s))"));

    assert_eq!(run(&db, "DELETE FROM t").render(), "1 row(s) affected");
}
