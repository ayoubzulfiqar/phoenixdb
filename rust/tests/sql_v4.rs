//! SQL correctness tests: concurrency, primary keys, ordering, typing,
//! parameters, caller transactions, and the richer grammar.
#![cfg(feature = "sql")]

use phoenixdb::sql::{Cell, Executor, QueryResult, Value};
use phoenixdb::{Database, Options};
use std::sync::Arc;

fn db() -> (tempfile::TempDir, Database) {
    let dir = tempfile::tempdir().unwrap();
    let db = Database::open(dir.path().join("sql.pdb"), Options::default()).unwrap();
    (dir, db)
}

fn run(db: &Database, sql: &str) -> QueryResult {
    Executor::new(db)
        .run(sql)
        .unwrap_or_else(|e| panic!("{sql}: {e}"))
}

fn rows(db: &Database, sql: &str) -> Vec<Vec<Cell>> {
    match run(db, sql) {
        QueryResult::Rows { rows, .. } => rows,
        other => panic!("{sql}: expected rows, got {other:?}"),
    }
}

fn ints(db: &Database, sql: &str) -> Vec<i64> {
    rows(db, sql)
        .into_iter()
        .map(|r| match &r[0] {
            Cell::Integer(i) => *i,
            other => panic!("expected an integer, got {other:?}"),
        })
        .collect()
}

fn err(db: &Database, sql: &str) -> String {
    Executor::new(db)
        .run(sql)
        .err()
        .unwrap_or_else(|| panic!("{sql}: expected an error"))
        .to_string()
}

// ---------------------------------------------------------------------------
// Concurrency (audit #1, #2)
// ---------------------------------------------------------------------------

#[test]
fn concurrent_inserts_never_lose_rows() {
    // Regression: 8 threads x 200 INSERTs reported 645 successes but stored
    // 365 rows — two statements read the same next_row_id outside their
    // transaction and the second overwrote the first.
    let (_d, db) = db();
    let db = Arc::new(db);
    run(&db, "CREATE TABLE c (t INTEGER, i INTEGER)");
    let handles: Vec<_> = (0..8)
        .map(|t| {
            let db = Arc::clone(&db);
            std::thread::spawn(move || {
                for i in 0..100 {
                    Executor::new(&db)
                        .run(&format!("INSERT INTO c VALUES ({t}, {i})"))
                        .unwrap();
                }
            })
        })
        .collect();
    for h in handles {
        h.join().unwrap();
    }
    assert_eq!(ints(&db, "SELECT COUNT(*) FROM c"), vec![800]);
    assert_eq!(db.stats().active_txns, 0, "no transaction may leak");
}

#[test]
fn concurrent_updates_of_one_row_both_land() {
    let (_d, db) = db();
    let db = Arc::new(db);
    run(
        &db,
        "CREATE TABLE lu (id INTEGER PRIMARY KEY, a INTEGER, b INTEGER)",
    );
    for round in 0..50 {
        run(&db, "DELETE FROM lu");
        run(&db, "INSERT INTO lu VALUES (1, 0, 0)");
        let a = {
            let db = Arc::clone(&db);
            std::thread::spawn(move || {
                Executor::new(&db)
                    .run(&format!("UPDATE lu SET a = {} WHERE id = 1", round + 1))
                    .unwrap();
            })
        };
        let b = {
            let db = Arc::clone(&db);
            std::thread::spawn(move || {
                Executor::new(&db)
                    .run(&format!("UPDATE lu SET b = {} WHERE id = 1", round + 1))
                    .unwrap();
            })
        };
        a.join().unwrap();
        b.join().unwrap();
        let r = rows(&db, "SELECT a, b FROM lu");
        assert_eq!(
            r,
            vec![vec![Cell::Integer(round + 1), Cell::Integer(round + 1)]],
            "round {round}: one UPDATE was lost"
        );
    }
}

// ---------------------------------------------------------------------------
// Primary keys (audit #3)
// ---------------------------------------------------------------------------

#[test]
fn primary_keys_are_unique() {
    let (_d, db) = db();
    run(&db, "CREATE TABLE u (id INTEGER PRIMARY KEY, name TEXT)");
    run(&db, "INSERT INTO u VALUES (1, 'a')");
    assert!(err(&db, "INSERT INTO u VALUES (1, 'b')").contains("duplicate PRIMARY KEY"));
    assert!(
        err(&db, "INSERT INTO u VALUES (2, 'c'), (2, 'd')").contains("duplicate"),
        "duplicates within one statement"
    );
    assert_eq!(
        ints(&db, "SELECT COUNT(*) FROM u"),
        vec![1],
        "nothing staged"
    );
    // 1.0 is the same numeric key as 1.
    assert!(err(&db, "INSERT INTO u VALUES (1.0, 'e')").contains("duplicate"));

    run(&db, "INSERT INTO u VALUES (2, 'c'), (3, 'd')");
    assert!(err(&db, "UPDATE u SET id = 1 WHERE id = 2").contains("duplicate"));
    assert!(
        err(&db, "UPDATE u SET id = 9 WHERE id > 1").contains("duplicate"),
        "two rows cannot both take one key"
    );
    run(&db, "UPDATE u SET id = 9 WHERE id = 3");
    assert_eq!(ints(&db, "SELECT id FROM u ORDER BY id"), vec![1, 2, 9]);
    run(
        &db,
        "UPDATE u SET id = 9, name = 'same key is fine' WHERE id = 9",
    );
    run(&db, "DELETE FROM u WHERE id = 9");
    run(&db, "INSERT INTO u VALUES (9, 'reused')");
    assert_eq!(
        rows(&db, "SELECT name FROM u WHERE id = 9"),
        vec![vec![Cell::Text("reused".into())]]
    );
    assert!(err(&db, "INSERT INTO u (name) VALUES ('no id')").contains("NOT NULL"));
}

#[test]
fn racing_inserts_of_one_key_admit_exactly_one() {
    let (_d, db) = db();
    let db = Arc::new(db);
    run(&db, "CREATE TABLE k (id INTEGER PRIMARY KEY, who INTEGER)");
    for key in 0..30 {
        let handles: Vec<_> = (0..4)
            .map(|who| {
                let db = Arc::clone(&db);
                std::thread::spawn(move || {
                    Executor::new(&db)
                        .run(&format!("INSERT INTO k VALUES ({key}, {who})"))
                        .is_ok()
                })
            })
            .collect();
        let winners = handles
            .into_iter()
            .map(|h| h.join().unwrap())
            .filter(|won| *won)
            .count();
        assert_eq!(winners, 1, "key {key}: exactly one INSERT may succeed");
    }
    // Every key exists exactly once, whichever thread won it.
    assert_eq!(ints(&db, "SELECT COUNT(*) FROM k"), vec![30]);
    assert_eq!(
        ints(&db, "SELECT COUNT(DISTINCT id) FROM k"),
        vec![30],
        "a key admitted twice"
    );
}

#[test]
fn legacy_tables_get_their_primary_key_index_on_first_write() {
    let (_d, db) = db();
    run(&db, "CREATE TABLE old (id INTEGER PRIMARY KEY, v TEXT)");
    run(&db, "INSERT INTO old VALUES (1, 'a'), (2, 'b')");
    // Simulate a table written before the index existed: drop the index
    // entries and the completeness marker straight from the store.
    let stale: Vec<Vec<u8>> = db
        .scan()
        .unwrap()
        .into_iter()
        .map(|(k, _)| k)
        .filter(|k| k.starts_with(b"\x02"))
        .collect();
    assert!(!stale.is_empty());
    for k in stale {
        db.delete_auto(&k).unwrap();
    }
    assert!(err(&db, "INSERT INTO old VALUES (2, 'dup')").contains("duplicate"));
    run(&db, "INSERT INTO old VALUES (3, 'c')");
    assert_eq!(ints(&db, "SELECT id FROM old WHERE id = 3"), vec![3]);
}

// ---------------------------------------------------------------------------
// Ordering and typing (audit #4, #5, #9)
// ---------------------------------------------------------------------------

#[test]
fn order_by_is_a_total_order_with_nulls_last() {
    // Regression: `3, NULL, 1, 2` came back unsorted.
    let (_d, db) = db();
    run(&db, "CREATE TABLE n (v INTEGER)");
    run(&db, "INSERT INTO n VALUES (3), (NULL), (1), (2), (1.5)");
    assert_eq!(
        rows(&db, "SELECT v FROM n ORDER BY v"),
        vec![
            vec![Cell::Integer(1)],
            vec![Cell::Float(1.5)],
            vec![Cell::Integer(2)],
            vec![Cell::Integer(3)],
            vec![Cell::Null],
        ]
    );
    assert_eq!(
        rows(&db, "SELECT v FROM n ORDER BY v DESC")[0],
        vec![Cell::Null],
        "DESC puts NULL first"
    );
    run(&db, "CREATE TABLE mixed (v TEXT)");
    run(&db, "INSERT INTO mixed VALUES (5), ('b'), (1), ('a'), (3)");
    assert_eq!(
        rows(&db, "SELECT v FROM mixed ORDER BY v"),
        vec![
            vec![Cell::Integer(1)],
            vec![Cell::Integer(3)],
            vec![Cell::Integer(5)],
            vec![Cell::Text("a".into())],
            vec![Cell::Text("b".into())],
        ]
    );
}

#[test]
fn multi_column_order_offset_and_limit() {
    let (_d, db) = db();
    run(&db, "CREATE TABLE p (dept TEXT, age INTEGER, name TEXT)");
    run(
        &db,
        "INSERT INTO p VALUES ('b', 30, 'x'), ('a', 40, 'y'), ('b', 20, 'z'), ('a', 10, 'w')",
    );
    let names: Vec<Cell> = rows(&db, "SELECT name FROM p ORDER BY dept, age DESC")
        .into_iter()
        .map(|mut r| r.remove(0))
        .collect();
    assert_eq!(
        names,
        ["y", "w", "x", "z"].map(|s| Cell::Text(s.into())).to_vec()
    );
    let page = rows(&db, "SELECT name FROM p ORDER BY age LIMIT 2 OFFSET 1");
    assert_eq!(
        page,
        vec![vec![Cell::Text("z".into())], vec![Cell::Text("x".into())]]
    );
    // ORDER BY an alias.
    let r = rows(
        &db,
        "SELECT age AS years FROM p ORDER BY years DESC LIMIT 1",
    );
    assert_eq!(r, vec![vec![Cell::Integer(40)]]);
}

#[test]
fn mixed_integer_float_comparisons_are_exact() {
    let (_d, db) = db();
    run(&db, "CREATE TABLE big (n INTEGER)");
    run(&db, "INSERT INTO big VALUES (9007199254740993)");
    assert_eq!(
        ints(&db, "SELECT n FROM big WHERE n > 9007199254740992.0"),
        vec![9_007_199_254_740_993]
    );
    assert!(ints(&db, "SELECT n FROM big WHERE n = 9007199254740992.0").is_empty());
}

#[test]
fn floats_keep_their_type_through_json() {
    let (_d, db) = db();
    run(&db, "CREATE TABLE f (x TEXT)");
    run(&db, "INSERT INTO f VALUES (3.0)");
    let json = phoenixdb::sql::executor::result_to_json(&run(&db, "SELECT x FROM f"));
    assert!(json.contains("[[3.0]]"), "{json}");
}

// ---------------------------------------------------------------------------
// Grammar
// ---------------------------------------------------------------------------

#[test]
fn rich_conditions_follow_three_valued_logic() {
    let (_d, db) = db();
    run(&db, "CREATE TABLE t (id INTEGER, name TEXT, score INTEGER)");
    run(
        &db,
        "INSERT INTO t VALUES (1, 'Alice', 90), (2, 'bob', NULL), (3, 'Carol', 70), (4, NULL, 50)",
    );
    let ids = |sql: &str| ints(&db, &format!("SELECT id FROM t WHERE {sql} ORDER BY id"));
    assert_eq!(ids("score IS NULL"), vec![2]);
    assert_eq!(ids("score IS NOT NULL AND score >= 70"), vec![1, 3]);
    assert_eq!(
        ids("id = 1 OR id = 2 AND score > 0"),
        vec![1],
        "AND binds tighter"
    );
    assert_eq!(ids("(id = 1 OR id = 2) AND score > 0"), vec![1]);
    assert_eq!(ids("NOT (id = 1)"), vec![2, 3, 4]);
    assert_eq!(
        ids("NOT (score > 60)"),
        vec![4],
        "NOT UNKNOWN stays UNKNOWN"
    );
    assert_eq!(ids("id IN (1, 3, 99)"), vec![1, 3]);
    assert_eq!(ids("id NOT IN (1, 3)"), vec![2, 4]);
    assert!(
        ids("id NOT IN (1, NULL)").is_empty(),
        "NOT IN with NULL is never true"
    );
    assert_eq!(ids("score BETWEEN 50 AND 70"), vec![3, 4]);
    assert_eq!(ids("score NOT BETWEEN 50 AND 70"), vec![1]);
    assert_eq!(ids("name LIKE '%o%'"), vec![2, 3]);
    assert_eq!(
        ids("name LIKE 'a%'"),
        Vec::<i64>::new(),
        "LIKE is case-sensitive"
    );
    assert_eq!(ids("name ILIKE 'a%'"), vec![1]);
    assert_eq!(ids("name NOT LIKE 'C%'"), vec![1, 2]);
    assert_eq!(
        ids("id < score"),
        vec![1, 3, 4],
        "column-to-column comparison"
    );
    assert!(err(&db, "SELECT id FROM t WHERE id").contains("comparison"));
}

#[test]
fn aggregates_and_group_by() {
    let (_d, db) = db();
    run(
        &db,
        "CREATE TABLE e (dept TEXT, salary INTEGER, bonus INTEGER)",
    );
    assert_eq!(
        rows(
            &db,
            "SELECT COUNT(*), SUM(salary), AVG(salary), MIN(salary), MAX(salary) FROM e"
        ),
        vec![vec![
            Cell::Integer(0),
            Cell::Null,
            Cell::Null,
            Cell::Null,
            Cell::Null
        ]],
        "an aggregate over no rows still returns one row"
    );
    run(
        &db,
        "INSERT INTO e VALUES ('eng', 100, 5), ('eng', 200, NULL), ('ops', 50, 1), ('ops', 50, 2)",
    );
    assert_eq!(
        rows(
            &db,
            "SELECT COUNT(*), COUNT(bonus), SUM(salary), COUNT(DISTINCT salary) FROM e"
        ),
        vec![vec![
            Cell::Integer(4),
            Cell::Integer(3),
            Cell::Integer(400),
            Cell::Integer(3)
        ]]
    );
    let QueryResult::Rows { columns, rows: out } = run(
        &db,
        "SELECT dept, COUNT(*) AS n, AVG(salary) AS avg_pay FROM e GROUP BY dept ORDER BY avg_pay DESC",
    ) else {
        panic!("expected rows");
    };
    assert_eq!(columns, ["dept", "n", "avg_pay"]);
    assert_eq!(
        out,
        vec![
            vec![
                Cell::Text("eng".into()),
                Cell::Integer(2),
                Cell::Float(150.0)
            ],
            vec![
                Cell::Text("ops".into()),
                Cell::Integer(2),
                Cell::Float(50.0)
            ],
        ]
    );
    assert!(err(&db, "SELECT dept, salary FROM e GROUP BY dept").contains("GROUP BY"));
    assert!(err(&db, "SELECT SUM(dept) FROM e").contains("numbers"));
    assert!(err(&db, "SELECT * , COUNT(*) FROM e").contains("expected"));
}

// ---------------------------------------------------------------------------
// Parameters and caller transactions (audit #8, injection)
// ---------------------------------------------------------------------------

#[test]
fn parameters_bind_values_not_sql() {
    let (_d, db) = db();
    let ex = Executor::new(&db);
    run(
        &db,
        "CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)",
    );
    // `?2, ?1` reuses parameters by position: rows (1, 2) and (2, 1).
    ex.run_with(
        "INSERT INTO users VALUES (?, ?), (?2, ?1)",
        &[Value::Integer(1), Value::Integer(2)],
    )
    .unwrap();
    assert_eq!(
        rows(&db, "SELECT id, name FROM users ORDER BY id"),
        vec![
            vec![Cell::Integer(1), Cell::Integer(2)],
            vec![Cell::Integer(2), Cell::Integer(1)],
        ]
    );
    let hostile = "x'); DROP TABLE users; --";
    ex.run_with(
        "INSERT INTO users VALUES (?, ?)",
        &[Value::Integer(7), Value::Text(hostile.into())],
    )
    .unwrap();
    let r = ex
        .run_with("SELECT name FROM users WHERE id = ?", &[Value::Integer(7)])
        .unwrap();
    let QueryResult::Rows { rows, .. } = r else {
        panic!("expected rows");
    };
    assert_eq!(rows, vec![vec![Cell::Text(hostile.into())]]);
    let e = ex
        .run_with("SELECT * FROM users WHERE id = ?", &[])
        .unwrap_err();
    assert!(e.to_string().contains("1 parameter(s) but 0"), "{e}");
}

#[test]
fn statements_can_join_a_caller_transaction() {
    let (_d, db) = db();
    let ex = Executor::new(&db);
    run(&db, "CREATE TABLE a (id INTEGER PRIMARY KEY, v TEXT)");
    let txn = db.begin(false).unwrap();
    ex.run_in(txn, "INSERT INTO a VALUES (1, 'one')", &[])
        .unwrap();
    ex.run_in(txn, "INSERT INTO a VALUES (2, 'two')", &[])
        .unwrap();
    // A failing statement inside the transaction stages nothing...
    assert!(
        ex.run_in(txn, "INSERT INTO a VALUES (3, 'x'), (1, 'dup')", &[])
            .is_err()
    );
    // ...its own reads see its writes, others do not.
    let inside = ex.run_in(txn, "SELECT COUNT(*) FROM a", &[]).unwrap();
    assert_eq!(
        inside,
        QueryResult::Rows {
            columns: vec!["count(*)".into()],
            rows: vec![vec![Cell::Integer(2)]]
        }
    );
    assert_eq!(ints(&db, "SELECT COUNT(*) FROM a"), vec![0]);
    db.commit(txn).unwrap();
    assert_eq!(ints(&db, "SELECT id FROM a ORDER BY id"), vec![1, 2]);

    let txn = db.begin(false).unwrap();
    ex.run_in(txn, "DELETE FROM a", &[]).unwrap();
    db.rollback(txn).unwrap();
    assert_eq!(
        ints(&db, "SELECT COUNT(*) FROM a"),
        vec![2],
        "rollback discards"
    );
}

// ---------------------------------------------------------------------------
// Catalog hygiene (audit #7, #10, #11, #13)
// ---------------------------------------------------------------------------

#[test]
fn names_cannot_forge_key_separators() {
    let (_d, db) = db();
    run(&db, "CREATE TABLE a (x INTEGER)");
    run(&db, "INSERT INTO a VALUES (1)");
    // A quoted name with a NUL used to let one table's rows sit inside
    // another's key prefix (and DROP deleted both).
    assert!(
        Executor::new(&db)
            .run("CREATE TABLE \"a\u{0}zzz\" (x INTEGER)")
            .is_err()
    );
    assert_eq!(ints(&db, "SELECT x FROM a"), vec![1]);
}

#[test]
fn unicode_names_fold_consistently() {
    let (_d, db) = db();
    assert!(err(&db, "CREATE TABLE uc (\"É\" INTEGER, \"é\" INTEGER)").contains("duplicate"));
    run(&db, "CREATE TABLE uc2 (\"É\" INTEGER)");
    run(&db, "INSERT INTO uc2 VALUES (1)");
    assert_eq!(ints(&db, "SELECT \"é\" FROM uc2"), vec![1]);
}

#[test]
fn drop_removes_rows_and_index_so_the_name_can_be_reused() {
    let (_d, db) = db();
    run(&db, "CREATE TABLE d (id INTEGER PRIMARY KEY)");
    run(&db, "INSERT INTO d VALUES (1), (2)");
    run(&db, "DROP TABLE d");
    run(&db, "CREATE TABLE d (id INTEGER PRIMARY KEY, extra TEXT)");
    run(&db, "INSERT INTO d VALUES (1, 'fresh')");
    assert_eq!(ints(&db, "SELECT COUNT(*) FROM d"), vec![1]);
    assert!(
        db.scan().unwrap().iter().all(|(k, _)| !k.is_empty()),
        "store is consistent"
    );
}

#[test]
fn errors_matter_not_existence_guesses() {
    let (_d, db) = db();
    assert!(err(&db, "INSERT INTO ghost VALUES (1)").contains("no such table"));
    run(&db, "CREATE TABLE IF NOT EXISTS g (a INTEGER)");
    run(&db, "CREATE TABLE IF NOT EXISTS g (a INTEGER)");
    assert!(err(&db, "CREATE TABLE g (a INTEGER)").contains("already exists"));
    assert!(err(&db, "SELECT nmae FROM g").contains("no column"));
    assert!(err(&db, "SELECT a FROM g WHERE nmae = 1").contains("no column"));
    assert!(err(&db, "SELECT a FROM g ORDER BY nmae").contains("no column"));
}
