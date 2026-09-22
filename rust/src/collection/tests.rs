use super::*;
use serde_json::json;

fn open(dir: &Path, dim: usize) -> Collection {
    Collection::open(
        dir,
        CollectionOptions {
            dim,
            sync_on_write: false,
            ..CollectionOptions::default()
        },
    )
    .unwrap()
}

fn doc(id: &str, text: &str, metadata: Value, vector: Option<Vec<f32>>) -> Document {
    Document {
        id: id.into(),
        text: Some(text.into()),
        metadata,
        vector,
    }
}

fn ids(hits: &[Hit]) -> Vec<&str> {
    hits.iter().map(|h| h.id.as_str()).collect()
}

fn filter(v: Value) -> Filter {
    Filter::parse(&v).unwrap()
}

#[test]
fn documents_round_trip_and_replace_cleanly() {
    let dir = tempfile::tempdir().unwrap();
    let c = open(dir.path(), 2);
    c.upsert(&[
        doc(
            "a",
            "red apples",
            json!({"kind": "fruit", "n": 1}),
            Some(vec![1.0, 0.0]),
        ),
        doc(
            "b",
            "green pears",
            json!({"kind": "fruit", "n": 2}),
            Some(vec![0.0, 1.0]),
        ),
        doc("c", "blue cars", json!({"kind": "car"}), None),
    ])
    .unwrap();

    let a = c.get("a", true).unwrap().unwrap();
    assert_eq!(a.text.as_deref(), Some("red apples"));
    assert_eq!(a.metadata, json!({"kind": "fruit", "n": 1}));
    assert_eq!(a.vector, Some(vec![1.0, 0.0]));
    assert_eq!(c.get("a", false).unwrap().unwrap().vector, None);
    assert!(c.get("zzz", false).unwrap().is_none());
    assert_eq!(c.count(None).unwrap(), 3);
    assert_eq!(c.count(Some(&filter(json!({"kind": "fruit"})))).unwrap(), 2);

    // Replace `a`: its old metadata, terms and vector must all be gone.
    c.upsert(&[doc("a", "yellow bananas", json!({"kind": "snack"}), None)])
        .unwrap();
    assert_eq!(c.count(Some(&filter(json!({"kind": "fruit"})))).unwrap(), 1);
    assert_eq!(
        c.count(Some(&filter(json!({"n": {"$exists": true}}))))
            .unwrap(),
        1
    );
    let apples = c
        .search(&SearchRequest {
            text: Some("apples".into()),
            ..SearchRequest::new(5)
        })
        .unwrap();
    assert!(apples.is_empty(), "stale postings: {apples:?}");
    assert!(c.get("a", true).unwrap().unwrap().vector.is_none());
    let stats = c.stats().unwrap();
    assert_eq!(
        (stats.documents, stats.text_documents, stats.vectors),
        (3, 3, 1)
    );

    assert_eq!(c.delete(&["a", "missing", "b"]).unwrap(), 2);
    assert_eq!(c.count(None).unwrap(), 1);
    assert_eq!(c.stats().unwrap().vectors, 0);
    assert_eq!(
        c.count(Some(&filter(json!({"kind": {"$exists": true}}))))
            .unwrap(),
        1
    );
}

#[test]
fn list_paginates_in_id_order() {
    let dir = tempfile::tempdir().unwrap();
    let c = open(dir.path(), 0);
    let docs: Vec<Document> = (0..25)
        .map(|i| doc(&format!("d{i:02}"), "x", json!({"even": i % 2 == 0}), None))
        .collect();
    c.upsert(&docs).unwrap();
    let mut seen = Vec::new();
    let mut after: Option<String> = None;
    loop {
        let page = c.list(None, 10, after.as_deref()).unwrap();
        if page.is_empty() {
            break;
        }
        after = page.last().map(|d| d.id.clone());
        seen.extend(page.into_iter().map(|d| d.id));
    }
    let expected: Vec<String> = (0..25).map(|i| format!("d{i:02}")).collect();
    assert_eq!(seen, expected);
    let evens = c
        .list(Some(&filter(json!({"even": true}))), 0, None)
        .unwrap();
    assert_eq!(evens.len(), 13);
}

/// A small deterministic generator, so the property test needs no crate.
struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.0 >> 33
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
    fn pick<'a, T>(&mut self, items: &'a [T]) -> &'a T {
        &items[self.below(items.len() as u64) as usize]
    }
}

fn random_metadata(rng: &mut Lcg) -> Value {
    let mut m = serde_json::Map::new();
    let cats = ["a", "b", "c"];
    if rng.below(5) != 0 {
        m.insert("cat".into(), json!(rng.pick(&cats)));
    }
    if rng.below(4) != 0 {
        m.insert("n".into(), json!(rng.below(10) as i64 - 3));
    }
    if rng.below(3) == 0 {
        let tags: Vec<&str> = (0..rng.below(3))
            .map(|_| *rng.pick(&["x", "y", "z"]))
            .collect();
        m.insert("tags".into(), json!(tags));
    }
    if rng.below(3) == 0 {
        m.insert("flag".into(), json!(rng.below(2) == 0));
    }
    if rng.below(4) == 0 {
        m.insert(
            "nested".into(),
            json!({"v": rng.below(3), "s": rng.pick(&cats)}),
        );
    }
    if rng.below(6) == 0 {
        m.insert("maybe".into(), Value::Null);
    }
    if rng.below(8) == 0 {
        m.insert("mixed".into(), json!([1, "1", null, {"deep": 2}]));
    }
    Value::Object(m)
}

fn random_filter(rng: &mut Lcg, depth: u32) -> Value {
    let fields = [
        "cat",
        "n",
        "tags",
        "flag",
        "nested.v",
        "nested.s",
        "maybe",
        "mixed",
        "mixed.deep",
        "nested",
        "nope",
    ];
    let scalars = [
        json!("a"),
        json!("b"),
        json!("x"),
        json!(1),
        json!(0),
        json!(-2),
        json!(2.5),
        json!(true),
        json!(null),
        json!("1"),
    ];
    let field = *rng.pick(&fields);
    if depth < 2 && rng.below(4) == 0 {
        let op = *rng.pick(&["$and", "$or"]);
        let parts: Vec<Value> = (0..1 + rng.below(3))
            .map(|_| random_filter(rng, depth + 1))
            .collect();
        return json!({ op: parts });
    }
    if depth < 2 && rng.below(8) == 0 {
        return json!({"$not": random_filter(rng, depth + 1)});
    }
    match rng.below(9) {
        0 => json!({ field: rng.pick(&scalars).clone() }),
        1 => json!({ field: {"$ne": rng.pick(&scalars).clone()} }),
        2 => json!({ field: {"$gt": rng.below(6) as i64 - 3} }),
        3 => json!({ field: {"$lte": rng.pick(&["a", "b", "x"])} }),
        4 => json!({ field: {"$gte": 0, "$lt": 3} }),
        5 => json!({ field: {"$in": [rng.pick(&scalars).clone(), rng.pick(&scalars).clone()]} }),
        6 => json!({ field: {"$nin": [rng.pick(&scalars).clone()]} }),
        7 => json!({ field: {"$exists": rng.below(2) == 0} }),
        _ => json!({ field: {"$lt": 2.5} }),
    }
}

#[test]
fn index_filters_agree_with_the_reference_semantics() {
    let dir = tempfile::tempdir().unwrap();
    let c = open(dir.path(), 0);
    let mut rng = Lcg(0x5eed);
    let mut model: HashMap<String, Value> = HashMap::new();
    for round in 0..4 {
        let docs: Vec<Document> = (0..60)
            .map(|_| {
                let id = format!("doc{}", rng.below(80));
                Document {
                    id,
                    text: None,
                    metadata: random_metadata(&mut rng),
                    vector: None,
                }
            })
            .collect();
        c.upsert(&docs).unwrap();
        for d in docs {
            model.insert(d.id, d.metadata);
        }
        let doomed: Vec<String> = (0..8).map(|_| format!("doc{}", rng.below(80))).collect();
        let refs: Vec<&str> = doomed.iter().map(String::as_str).collect();
        c.delete(&refs).unwrap();
        for id in &doomed {
            model.remove(id);
        }
        assert_eq!(c.count(None).unwrap(), model.len() as u64, "round {round}");

        for _ in 0..150 {
            let spec = random_filter(&mut rng, 0);
            let f = Filter::parse(&spec).unwrap();
            let mut expected: Vec<&str> = model
                .iter()
                .filter(|(_, m)| f.matches(m))
                .map(|(id, _)| id.as_str())
                .collect();
            expected.sort_unstable();
            let got: Vec<String> = c
                .list(Some(&f), 0, None)
                .unwrap()
                .into_iter()
                .map(|d| d.id)
                .collect();
            assert_eq!(got, expected, "filter {spec}");
        }
    }
}

#[test]
fn bm25_ranks_by_relevance_with_stemming_and_cjk() {
    let dir = tempfile::tempdir().unwrap();
    let c = open(dir.path(), 0);
    c.upsert(&[
        doc(
            "db",
            "Embedded databases store data locally. A database file.",
            json!({}),
            None,
        ),
        doc(
            "vec",
            "Vector search finds nearest neighbours",
            json!({}),
            None,
        ),
        doc("mix", "A vector database combines both", json!({}), None),
        doc("zh", "东京是日本的首都", json!({}), None),
    ])
    .unwrap();
    let hits = c
        .search(&SearchRequest {
            text: Some("database".into()),
            ..SearchRequest::new(10)
        })
        .unwrap();
    assert_eq!(ids(&hits), ["db", "mix"], "tf and stemming favour `db`");
    assert!(hits[0].text_score.unwrap() > hits[1].text_score.unwrap());
    assert_eq!(
        hits[0].text.as_deref(),
        Some("Embedded databases store data locally. A database file.")
    );

    let zh = c
        .search(&SearchRequest {
            text: Some("日本首都".into()),
            ..SearchRequest::new(3)
        })
        .unwrap();
    assert_eq!(ids(&zh), ["zh"]);

    let none = c
        .search(&SearchRequest {
            text: Some("the and of".into()),
            ..SearchRequest::new(3)
        })
        .unwrap();
    assert!(none.is_empty(), "stopword-only queries match nothing");
}

#[test]
fn vector_search_respects_filters() {
    let dir = tempfile::tempdir().unwrap();
    let c = open(dir.path(), 2);
    let docs: Vec<Document> = (0..200)
        .map(|i| {
            // Angles stay below pi, so distance to (1, 0) grows with `i`.
            let angle = i as f32 * 0.015;
            Document {
                id: format!("p{i:03}"),
                text: None,
                metadata: json!({"bucket": i % 4}),
                vector: Some(vec![angle.cos(), angle.sin()]),
            }
        })
        .collect();
    c.upsert(&docs).unwrap();
    let hits = c
        .search(&SearchRequest {
            vector: Some(vec![1.0, 0.0]),
            filter: Some(filter(json!({"bucket": 3}))),
            ..SearchRequest::new(3)
        })
        .unwrap();
    assert_eq!(ids(&hits), ["p003", "p007", "p011"]);
    assert!(
        hits.iter()
            .all(|h| h.metadata.as_ref().unwrap()["bucket"] == 3)
    );
    assert!(hits[0].vector_score.unwrap() >= hits[1].vector_score.unwrap());

    let nothing = c
        .search(&SearchRequest {
            vector: Some(vec![1.0, 0.0]),
            filter: Some(filter(json!({"bucket": 9}))),
            ..SearchRequest::new(3)
        })
        .unwrap();
    assert!(nothing.is_empty());
}

#[test]
fn hybrid_fusion_and_mmr() {
    let dir = tempfile::tempdir().unwrap();
    let c = open(dir.path(), 2);
    c.upsert(&[
        // Closest vector, no text match.
        doc(
            "near",
            "unrelated words here",
            json!({}),
            Some(vec![1.0, 0.0]),
        ),
        // A near-duplicate of `near`.
        doc(
            "near2",
            "other unrelated words",
            json!({}),
            Some(vec![0.999, 0.01]),
        ),
        // Moderate vector, strong text match.
        doc(
            "both",
            "rust database engine",
            json!({}),
            Some(vec![0.8, 0.6]),
        ),
        // Text match only, far vector.
        doc("text", "rust database", json!({}), Some(vec![-1.0, 0.0])),
        // Fillers between `both` and `text` in the vector ranking.
        doc("f1", "filler", json!({}), Some(vec![0.5, 0.866])),
        doc("f2", "filler", json!({}), Some(vec![0.3, 0.954])),
        doc("f3", "filler", json!({}), Some(vec![0.0, 1.0])),
    ])
    .unwrap();
    let query = |fusion, mmr| {
        c.search(&SearchRequest {
            vector: Some(vec![1.0, 0.0]),
            text: Some("rust database".into()),
            fusion,
            mmr,
            candidates: Some(7),
            ..SearchRequest::new(2)
        })
        .unwrap()
    };
    // RRF rewards appearing high in both rankings: `both` is 3rd by vector
    // and 2nd by text, `text` is 7th and 1st.
    let rrf = query(Fusion::default(), None);
    assert_eq!(rrf[0].id, "both", "{rrf:?}");
    assert!(rrf[0].text_score.is_some() && rrf[0].vector_score.is_some());

    // alpha = 1 is pure vector ranking; alpha = 0 is pure text ranking.
    let vec_only = query(Fusion::Weighted { alpha: 1.0 }, None);
    assert_eq!(ids(&vec_only), ["near", "near2"]);
    let text_only = query(Fusion::Weighted { alpha: 0.0 }, None);
    assert_eq!(
        ids(&text_only),
        ["text", "both"],
        "the shorter document wins BM25"
    );

    // MMR with a diversity preference skips the near-duplicate.
    let diverse = query(Fusion::Weighted { alpha: 1.0 }, Some(0.3));
    assert_eq!(
        ids(&diverse),
        ["near", "f3"],
        "the orthogonal filler adds the most"
    );
    // lambda = 1 is plain relevance.
    assert_eq!(
        ids(&query(Fusion::Weighted { alpha: 1.0 }, Some(1.0))),
        ["near", "near2"]
    );
}

#[test]
fn collection_persists_and_checks_its_layout() {
    let dir = tempfile::tempdir().unwrap();
    {
        let c = open(dir.path(), 3);
        c.upsert(&[doc(
            "a",
            "persisted text",
            json!({"k": 1}),
            Some(vec![0.1, 0.2, 0.3]),
        )])
        .unwrap();
    }
    {
        let c = open(dir.path(), 0);
        assert_eq!(c.dim(), 3, "dim 0 adopts the stored layout");
        assert_eq!(
            c.stats().unwrap().repaired,
            0,
            "a clean close needs no repair"
        );
        let hits = c
            .search(&SearchRequest {
                vector: Some(vec![0.1, 0.2, 0.3]),
                text: Some("persisted".into()),
                filter: Some(filter(json!({"k": 1}))),
                include_vector: true,
                ..SearchRequest::new(1)
            })
            .unwrap();
        assert_eq!(hits[0].id, "a");
        assert_eq!(hits[0].vector.as_deref(), Some(&[0.1, 0.2, 0.3][..]));
    }
    let err = Collection::open(
        dir.path(),
        CollectionOptions {
            dim: 4,
            ..Default::default()
        },
    )
    .unwrap_err();
    assert!(err.to_string().contains("3-dimensional"), "{err}");
    let err = Collection::open(
        dir.path(),
        CollectionOptions {
            dim: 3,
            metric: Metric::Euclidean,
            ..Default::default()
        },
    )
    .unwrap_err();
    assert!(err.to_string().contains("metric"), "{err}");
}

#[test]
fn a_crash_between_vector_write_and_commit_is_repaired() {
    let dir = tempfile::tempdir().unwrap();
    {
        let c = open(dir.path(), 2);
        c.upsert(&[doc("kept", "t", json!({}), Some(vec![1.0, 0.0]))])
            .unwrap();
        c.inject_orphan_vector("orphan", &[0.0, 1.0]).unwrap();
        c.flush().unwrap();
        c.simulate_crash();
    }
    let c = open(dir.path(), 2);
    assert_eq!(c.stats().unwrap().repaired, 1);
    assert_eq!(c.stats().unwrap().vectors, 1);
    let hits = c
        .search(&SearchRequest {
            vector: Some(vec![0.0, 1.0]),
            ..SearchRequest::new(5)
        })
        .unwrap();
    assert_eq!(ids(&hits), ["kept"], "the orphan must never surface");
}

#[test]
fn upsert_is_all_or_nothing_and_last_duplicate_wins() {
    let dir = tempfile::tempdir().unwrap();
    let c = open(dir.path(), 2);
    let bad = [
        doc("ok", "fine", json!({}), Some(vec![1.0, 0.0])),
        doc("bad", "wrong dim", json!({}), Some(vec![1.0, 0.0, 0.0])),
    ];
    assert!(c.upsert(&bad).is_err());
    for invalid in [
        doc("", "empty id", json!({}), None),
        doc("x\n", "control char", json!({}), None),
        doc("m", "array metadata", json!([1]), None),
        doc("d", "dotted key", json!({"a.b": 1}), None),
        doc("v", "nan", json!({}), Some(vec![f32::NAN, 0.0])),
    ] {
        assert!(
            c.upsert(std::slice::from_ref(&invalid)).is_err(),
            "{invalid:?}"
        );
    }
    assert_eq!(c.count(None).unwrap(), 0);
    assert_eq!(c.stats().unwrap().vectors, 0);

    c.upsert(&[
        doc("dup", "first", json!({"v": 1}), None),
        doc("dup", "second", json!({"v": 2}), Some(vec![0.0, 1.0])),
    ])
    .unwrap();
    let d = c.get("dup", true).unwrap().unwrap();
    assert_eq!(d.text.as_deref(), Some("second"));
    assert_eq!(
        d.vector,
        Some(vec![0.0, 1.0]),
        "a vector set later in the batch survives"
    );
    assert_eq!(c.count(Some(&filter(json!({"v": 1})))).unwrap(), 0);
    assert_eq!(c.stats().unwrap().documents, 1);
}

#[test]
fn text_only_collections_reject_vectors() {
    let dir = tempfile::tempdir().unwrap();
    let c = open(dir.path(), 0);
    assert!(
        c.upsert(&[doc("a", "t", json!({}), Some(vec![1.0]))])
            .is_err()
    );
    c.upsert(&[doc("a", "hello world", json!({"n": 1}), None)])
        .unwrap();
    assert!(
        c.search(&SearchRequest {
            vector: Some(vec![1.0]),
            ..SearchRequest::new(1)
        })
        .is_err()
    );
    // No query at all: the filter's matches, in id order.
    let all = c.search(&SearchRequest::new(10)).unwrap();
    assert_eq!(ids(&all), ["a"]);
}

#[test]
fn concurrent_writers_do_not_lose_documents() {
    let dir = tempfile::tempdir().unwrap();
    let c = open(dir.path(), 2);
    std::thread::scope(|s| {
        for t in 0..4 {
            let c = &c;
            s.spawn(move || {
                for i in 0..25 {
                    let angle = (t * 25 + i) as f32;
                    c.upsert(&[doc(
                        &format!("t{t}-{i}"),
                        "concurrent text",
                        json!({"t": t}),
                        Some(vec![angle.cos(), angle.sin()]),
                    )])
                    .unwrap();
                }
            });
        }
    });
    assert_eq!(c.count(None).unwrap(), 100);
    assert_eq!(c.stats().unwrap().vectors, 100);
    assert_eq!(c.count(Some(&filter(json!({"t": 2})))).unwrap(), 25);
}

#[test]
fn search_requests_parse_from_json() {
    let req = SearchRequest::from_json(&json!({
        "k": 7, "text": "hi", "filter": {"a": 1}, "ef": 99,
        "fusion": {"alpha": 0.25}, "mmr": 0.5, "candidates": 30,
        "min_score": 0.1, "include": ["vector"]
    }))
    .unwrap();
    assert_eq!(req.k, 7);
    assert_eq!(req.text.as_deref(), Some("hi"));
    assert_eq!(req.filter, Some(filter(json!({"a": 1}))));
    assert_eq!(req.fusion, Fusion::Weighted { alpha: 0.25 });
    assert_eq!(
        (req.ef, req.candidates, req.mmr),
        (Some(99), Some(30), Some(0.5))
    );
    assert!(req.include_vector && !req.include_text && !req.include_metadata);
    assert_eq!(
        SearchRequest::from_json(&json!({"fusion": {"rrf": 10}}))
            .unwrap()
            .fusion,
        Fusion::Rrf { k: 10.0 }
    );
    for bad in [
        json!([]),
        json!({"k": -1}),
        json!({"bogus": 1}),
        json!({"fusion": "max"}),
        json!({"include": ["everything"]}),
        json!({"filter": {"$xor": []}}),
    ] {
        assert!(SearchRequest::from_json(&bad).is_err(), "{bad}");
    }
}
