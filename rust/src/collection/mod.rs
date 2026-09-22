//! Document collections: vectors, metadata and full text in one store.
//!
//! A [`Collection`] is the retrieval layer of a local-first AI app. Each
//! [`Document`] has an id, optional text, optional JSON metadata and an
//! optional embedding, and one [`Collection::search`] call can combine:
//!
//! * **vector similarity** — HNSW k-NN over the embeddings;
//! * **full-text relevance** — BM25 over an inverted index of the text;
//! * **metadata filters** — MongoDB-style predicates served from an index
//!   ([`filter`]);
//! * **hybrid fusion** of the two rankings, by Reciprocal Rank Fusion or a
//!   weighted blend of normalised scores;
//! * **MMR diversification**, so the top hits are not near-duplicates of each
//!   other — which matters when they are pasted into an LLM prompt.
//!
//! # Storage
//!
//! A collection is a directory holding a PhoenixDB key/value database
//! (`docs.pdb`: documents, metadata index, postings, statistics — all updated
//! transactionally) and, when it has embeddings, a vector index
//! (`vectors.pvec`). Writes stage every key/value change in one transaction,
//! write and `fsync` the vectors, and only then commit, so a committed
//! document always has its vector. A crash between the vector write and the
//! commit can leave an orphaned vector, which the next open removes; a crash
//! while *replacing* a document can leave its new embedding next to its old
//! text until the (unacknowledged) upsert is retried.
//!
//! Within one [`Collection::upsert`] batch a repeated id is applied in order,
//! so the last occurrence wins.
//!
//! # Example
//!
//! ```no_run
//! use phoenixdb::collection::{Collection, CollectionOptions, Document, Filter, SearchRequest};
//! use serde_json::json;
//!
//! # fn main() -> phoenixdb::Result<()> {
//! let c = Collection::open("kb", CollectionOptions { dim: 3, ..Default::default() })?;
//! c.upsert(&[Document {
//!     id: "doc-1".into(),
//!     text: Some("PhoenixDB is an embedded database".into()),
//!     metadata: json!({"lang": "en", "year": 2025}),
//!     vector: Some(vec![0.1, 0.9, 0.2]),
//! }])?;
//! let hits = c.search(&SearchRequest {
//!     vector: Some(vec![0.1, 0.8, 0.3]),
//!     text: Some("embedded database".into()),
//!     filter: Some(Filter::parse(&json!({"lang": "en", "year": {"$gte": 2020}}))?),
//!     ..SearchRequest::new(5)
//! })?;
//! println!("{}", hits[0].id);
//! # Ok(())
//! # }
//! ```

pub mod filter;
pub mod text;

use crate::error::{Error, Result};
use crate::vector::distance::{dot, norm};
use crate::vector::{HnswParams, MAX_ID_LEN, Metric, VectorEngine, VectorOptions};
use crate::{Database, Options, prefix_successor};
pub use filter::Filter;
use filter::{Flattened, RangeOp, Scalar};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{BTreeSet, HashMap};
use std::ops::Bound;
use std::path::{Path, PathBuf};

// ---- key layout (inside docs.pdb) -------------------------------------------
const CFG_KEY: &[u8] = b"\x10cfg";
const STATS_KEY: &[u8] = b"\x10stats";
const OPEN_KEY: &[u8] = b"\x10open";
const DOC: u8 = 0x11; // DOC id -> DocRecord
const META: u8 = 0x12; // META field 0 value id -> []
const TERM: u8 = 0x13; // TERM term 0 id -> (tf, doc_len)
const IDS: u8 = 0x15; // IDS id -> []

/// Index-key tag of a presence entry (`META field 0 PRESENT id`); scalar
/// encodings use tags 1..=4, so presence entries sort before every value.
const PRESENT: u8 = 0x00;

/// Most metadata values indexed per document.
pub const MAX_INDEXED_VALUES: usize = 1024;
/// Most candidates a search fetches from each retriever.
pub const MAX_CANDIDATES: usize = 50_000;
/// Largest `k` a search accepts.
pub const MAX_K: usize = 10_000;
/// Largest document record (text + metadata), in bytes.
pub const MAX_DOCUMENT_BYTES: usize = 8 * 1024 * 1024;

/// How a collection is laid out. Fixed at creation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CollectionOptions {
    /// Embedding dimensionality; `0` for a collection without vectors (text
    /// and metadata search only). When opening an existing collection, `0`
    /// accepts whatever it was created with.
    pub dim: usize,
    /// Vector similarity metric.
    pub metric: Metric,
    /// HNSW graph parameters.
    pub hnsw: HnswParams,
    /// Maintain the BM25 full-text index.
    pub text_index: bool,
    /// `fsync` every write (the default). Turning it off trades power-loss
    /// durability of the latest writes for much faster bulk ingest.
    pub sync_on_write: bool,
}

impl Default for CollectionOptions {
    fn default() -> Self {
        CollectionOptions {
            dim: 0,
            metric: Metric::Cosine,
            hnsw: HnswParams::default(),
            text_index: true,
            sync_on_write: true,
        }
    }
}

/// Persisted layout.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct Config {
    version: u32,
    dim: usize,
    metric: u8,
    text_index: bool,
}

/// Collection-wide counters for BM25 and diagnostics.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
struct Counters {
    docs: u64,
    text_docs: u64,
    text_terms: u64,
}

/// A stored document.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct DocRecord {
    metadata: String,
    text: Option<String>,
    has_vector: bool,
    text_len: u32,
}

/// One document.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Document {
    /// Unique id, 1..=128 bytes of UTF-8 without control characters.
    pub id: String,
    /// Text for full-text search (and for returning to the caller).
    pub text: Option<String>,
    /// JSON object (or `null`) for filtering and returning.
    pub metadata: Value,
    /// Embedding, exactly `dim` finite floats.
    pub vector: Option<Vec<f32>>,
}

/// How vector and text rankings are combined.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Fusion {
    /// Reciprocal Rank Fusion: `Σ 1 / (k + rank)`. Robust — it needs no score
    /// calibration between retrievers — and the usual default.
    Rrf {
        /// Rank damping constant (60 in the original paper).
        k: f32,
    },
    /// `alpha * vector + (1 - alpha) * text` over min-max-normalised scores.
    Weighted {
        /// Weight of the vector score, in `[0, 1]`.
        alpha: f32,
    },
}

impl Default for Fusion {
    fn default() -> Self {
        Fusion::Rrf { k: 60.0 }
    }
}

/// A search.
#[derive(Debug, Clone, PartialEq)]
pub struct SearchRequest {
    /// Query embedding for vector search.
    pub vector: Option<Vec<f32>>,
    /// Query text for BM25 search.
    pub text: Option<String>,
    /// Metadata filter; only matching documents are returned.
    pub filter: Option<Filter>,
    /// Number of results.
    pub k: usize,
    /// HNSW beam width override.
    pub ef: Option<usize>,
    /// How vector and text rankings combine when both are given.
    pub fusion: Fusion,
    /// MMR trade-off `λ` in `[0, 1]`: 1 = pure relevance, lower values favour
    /// diversity. Needs embeddings.
    pub mmr: Option<f32>,
    /// Candidates fetched from each retriever before fusion/MMR
    /// (default `max(4k, 20)`).
    pub candidates: Option<usize>,
    /// Drop results scoring below this.
    pub min_score: Option<f32>,
    /// Return each hit's text.
    pub include_text: bool,
    /// Return each hit's metadata.
    pub include_metadata: bool,
    /// Return each hit's embedding.
    pub include_vector: bool,
}

impl SearchRequest {
    /// A request for `k` results with text and metadata included.
    #[must_use]
    pub fn new(k: usize) -> Self {
        SearchRequest {
            vector: None,
            text: None,
            filter: None,
            k,
            ef: None,
            fusion: Fusion::default(),
            mmr: None,
            candidates: None,
            min_score: None,
            include_text: true,
            include_metadata: true,
            include_vector: false,
        }
    }

    /// Parses the JSON form used by the C ABI (the vector travels
    /// separately, as binary):
    ///
    /// ```json
    /// {"k": 10, "text": "…", "filter": {…}, "ef": 64,
    ///  "fusion": "rrf" | {"rrf": 60} | {"alpha": 0.5},
    ///  "mmr": 0.5, "candidates": 40, "min_score": 0.1,
    ///  "include": ["text", "metadata", "vector"]}
    /// ```
    pub fn from_json(value: &Value) -> Result<Self> {
        let Value::Object(map) = value else {
            return Err(Error::invalid("a search request must be a JSON object"));
        };
        let mut req = SearchRequest::new(10);
        for (key, v) in map {
            match key.as_str() {
                "k" => req.k = json_usize(v, "k")?,
                "ef" => req.ef = Some(json_usize(v, "ef")?),
                "candidates" => req.candidates = Some(json_usize(v, "candidates")?),
                "text" => {
                    req.text = match v {
                        Value::Null => None,
                        Value::String(s) => Some(s.clone()),
                        _ => return Err(Error::invalid("`text` must be a string")),
                    }
                }
                "filter" => {
                    req.filter = match v {
                        Value::Null => None,
                        other => Some(Filter::parse(other)?),
                    }
                }
                "mmr" => req.mmr = Some(json_f32(v, "mmr")?),
                "min_score" => req.min_score = Some(json_f32(v, "min_score")?),
                "fusion" => {
                    req.fusion = match v {
                        Value::String(s) if s == "rrf" => Fusion::default(),
                        Value::Object(o) if o.contains_key("rrf") => Fusion::Rrf {
                            k: json_f32(&o["rrf"], "fusion.rrf")?,
                        },
                        Value::Object(o) if o.contains_key("alpha") => Fusion::Weighted {
                            alpha: json_f32(&o["alpha"], "fusion.alpha")?,
                        },
                        _ => {
                            return Err(Error::invalid(
                                "`fusion` must be \"rrf\", {\"rrf\": k} or {\"alpha\": a}",
                            ));
                        }
                    }
                }
                "include" => {
                    let Value::Array(items) = v else {
                        return Err(Error::invalid("`include` must be an array"));
                    };
                    req.include_text = false;
                    req.include_metadata = false;
                    for item in items {
                        match item.as_str() {
                            Some("text") => req.include_text = true,
                            Some("metadata") => req.include_metadata = true,
                            Some("vector") => req.include_vector = true,
                            _ => {
                                return Err(Error::invalid(
                                    "`include` entries are \"text\", \"metadata\" or \"vector\"",
                                ));
                            }
                        }
                    }
                }
                other => return Err(Error::invalid(format!("unknown search option `{other}`"))),
            }
        }
        Ok(req)
    }
}

fn json_usize(v: &Value, what: &str) -> Result<usize> {
    v.as_u64()
        .and_then(|n| usize::try_from(n).ok())
        .ok_or_else(|| Error::invalid(format!("`{what}` must be a non-negative integer")))
}

fn json_f32(v: &Value, what: &str) -> Result<f32> {
    v.as_f64()
        .filter(|f| f.is_finite())
        .map(|f| f as f32)
        .ok_or_else(|| Error::invalid(format!("`{what}` must be a finite number")))
}

/// One search result.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Hit {
    /// Document id.
    pub id: String,
    /// Final score, higher is better.
    pub score: f32,
    /// Vector similarity (metric score) when vector search ran.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vector_score: Option<f32>,
    /// Raw metric distance when vector search ran.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub distance: Option<f32>,
    /// BM25 score when text search ran and matched.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text_score: Option<f32>,
    /// Document text, if requested.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    /// Document metadata, if requested.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<Value>,
    /// Document embedding, if requested.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vector: Option<Vec<f32>>,
}

/// Collection statistics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct CollectionStats {
    /// Documents stored.
    pub documents: u64,
    /// Documents with text.
    pub text_documents: u64,
    /// Documents with an embedding.
    pub vectors: u64,
    /// Embedding dimensionality (0 = none).
    pub dim: usize,
    /// Orphaned vectors removed and missing vectors found by the last
    /// recovery (non-zero only after a crash).
    pub repaired: u64,
}

/// A collection of documents; see the module docs.
pub struct Collection {
    dir: PathBuf,
    db: Database,
    vectors: Option<VectorEngine>,
    config: Config,
    metric: Metric,
    sync_on_write: bool,
    repaired: u64,
    crashed: bool,
}

fn key(tag: u8, parts: &[&[u8]]) -> Vec<u8> {
    let mut k = vec![tag];
    for p in parts {
        k.extend_from_slice(p);
    }
    k
}

fn meta_prefix(field: &str) -> Vec<u8> {
    key(META, &[field.as_bytes(), &[0]])
}

fn term_prefix(term: &str) -> Vec<u8> {
    key(TERM, &[term.as_bytes(), &[0]])
}

/// Length of the encoded scalar at the start of `bytes` (see
/// [`Scalar::encode`]), so the id that follows can be recovered.
fn encoded_len(bytes: &[u8]) -> Option<usize> {
    match *bytes.first()? {
        0x01 => Some(1),
        0x02 => Some(2),
        0x03 => Some(9),
        0x04 => {
            let mut i = 1;
            while i + 1 < bytes.len() {
                if bytes[i] == 0 {
                    match bytes[i + 1] {
                        0x01 => return Some(i + 2),
                        0xFF => i += 2,
                        _ => return None,
                    }
                } else {
                    i += 1;
                }
            }
            None
        }
        _ => None,
    }
}

/// Validates a document id.
pub fn validate_id(id: &str) -> Result<()> {
    if id.is_empty() || id.len() > MAX_ID_LEN {
        return Err(Error::invalid(format!(
            "document id must be 1..={MAX_ID_LEN} bytes, got {}",
            id.len()
        )));
    }
    if id.chars().any(char::is_control) {
        return Err(Error::invalid(
            "document id must not contain control characters",
        ));
    }
    Ok(())
}

type IdSet = BTreeSet<String>;

impl Collection {
    /// Opens (creating if necessary) the collection in directory `dir`.
    ///
    /// An existing collection's layout wins over `options` where they can be
    /// reconciled (`dim == 0` opens whatever exists); a conflicting `dim` or
    /// metric is an error rather than a silent reinterpretation.
    pub fn open(dir: impl AsRef<Path>, options: CollectionOptions) -> Result<Self> {
        let dir = dir.as_ref().to_path_buf();
        std::fs::create_dir_all(&dir)?;
        let db = Database::open(
            dir.join("docs.pdb"),
            Options {
                sync_on_commit: options.sync_on_write,
                ..Options::default()
            },
        )?;

        let config = match db.get_auto(CFG_KEY) {
            Ok(bytes) => {
                let stored: Config = serde_json::from_slice(&bytes)
                    .map_err(|e| Error::corrupt(format!("collection config: {e}")))?;
                if options.dim != 0 && options.dim != stored.dim {
                    return Err(Error::invalid(format!(
                        "collection at {} has {}-dimensional vectors, not {}",
                        dir.display(),
                        stored.dim,
                        options.dim
                    )));
                }
                if stored.dim > 0
                    && options.dim != 0
                    && Metric::from_u8(stored.metric)? != options.metric
                {
                    return Err(Error::invalid(format!(
                        "collection at {} uses the {} metric",
                        dir.display(),
                        Metric::from_u8(stored.metric)?.name()
                    )));
                }
                stored
            }
            Err(Error::NotFound) => {
                if options.dim > 0 {
                    crate::vector::distance::validate_dim(options.dim)?;
                }
                let config = Config {
                    version: 1,
                    dim: options.dim,
                    metric: options.metric.as_u8(),
                    text_index: options.text_index,
                };
                let bytes = serde_json::to_vec(&config)
                    .map_err(|e| Error::invalid(format!("collection config: {e}")))?;
                db.put_auto(CFG_KEY, &bytes)?;
                config
            }
            Err(e) => return Err(e),
        };
        let metric = Metric::from_u8(config.metric)?;
        let vectors = if config.dim > 0 {
            Some(VectorEngine::open(
                dir.join("vectors.pvec"),
                config.dim,
                metric,
                VectorOptions {
                    hnsw: options.hnsw,
                    ..VectorOptions::default()
                },
            )?)
        } else {
            None
        };

        let mut collection = Collection {
            dir,
            db,
            vectors,
            config,
            metric,
            sync_on_write: options.sync_on_write,
            repaired: 0,
            crashed: false,
        };
        // An "open" marker surviving from last time means the previous
        // session never closed cleanly: reconcile the two stores.
        if collection.db.get_auto(OPEN_KEY).is_ok() {
            collection.repaired = collection.reconcile()?;
        }
        collection.db.put_auto(OPEN_KEY, &[1])?;
        Ok(collection)
    }

    /// Directory holding the collection.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.dir
    }

    /// Embedding dimensionality (0 when the collection has no vectors).
    #[must_use]
    pub fn dim(&self) -> usize {
        self.config.dim
    }

    /// Vector metric.
    #[must_use]
    pub fn metric(&self) -> Metric {
        self.metric
    }

    /// Removes vectors whose document never committed; counts documents
    /// whose vector was lost. Returns how many problems were found.
    fn reconcile(&self) -> Result<u64> {
        let Some(engine) = &self.vectors else {
            return Ok(0);
        };
        let mut repaired = 0u64;
        for id in engine.ids() {
            if self.db.get_auto(&key(IDS, &[id.as_bytes()])).is_err() {
                engine.remove(&id)?;
                repaired += 1;
            }
        }
        if repaired > 0 {
            engine.flush()?;
        }
        let txn = self.db.begin(true)?;
        let ids = self.all_ids(txn);
        let _ = self.db.rollback(txn);
        for id in ids? {
            if let Some(record) = self.record_auto(&id)?
                && record.has_vector
                && !engine.contains(&id)
            {
                repaired += 1;
            }
        }
        Ok(repaired)
    }

    // ---- reads ------------------------------------------------------------

    fn record_in(&self, txn: u64, id: &str) -> Result<Option<DocRecord>> {
        match self.db.get(txn, &key(DOC, &[id.as_bytes()])) {
            Ok(bytes) => Ok(Some(bincode::deserialize(&bytes)?)),
            Err(Error::NotFound) => Ok(None),
            Err(e) => Err(e),
        }
    }

    fn record_auto(&self, id: &str) -> Result<Option<DocRecord>> {
        match self.db.get_auto(&key(DOC, &[id.as_bytes()])) {
            Ok(bytes) => Ok(Some(bincode::deserialize(&bytes)?)),
            Err(Error::NotFound) => Ok(None),
            Err(e) => Err(e),
        }
    }

    fn counters(&self, txn: u64) -> Result<Counters> {
        match self.db.get(txn, STATS_KEY) {
            Ok(bytes) => Ok(bincode::deserialize(&bytes)?),
            Err(Error::NotFound) => Ok(Counters::default()),
            Err(e) => Err(e),
        }
    }

    /// Scans keys under `prefix` in `txn`, handing each key's suffix to `f`.
    fn scan_suffixes(
        &self,
        txn: u64,
        lo: Bound<&[u8]>,
        hi: Bound<&[u8]>,
        strip: usize,
        mut f: impl FnMut(&[u8], &[u8]) -> Result<()>,
    ) -> Result<()> {
        self.db.scan_txn(txn, lo, hi, |k, v| {
            if k.len() >= strip {
                f(&k[strip..], &v)?;
            }
            Ok(true)
        })
    }

    fn prefix_scan(
        &self,
        txn: u64,
        prefix: &[u8],
        f: impl FnMut(&[u8], &[u8]) -> Result<()>,
    ) -> Result<()> {
        let end = prefix_successor(prefix);
        let hi = end.as_deref().map_or(Bound::Unbounded, Bound::Excluded);
        self.scan_suffixes(txn, Bound::Included(prefix), hi, prefix.len(), f)
    }

    fn all_ids(&self, txn: u64) -> Result<IdSet> {
        let mut ids = IdSet::new();
        self.prefix_scan(txn, &[IDS], |suffix, _| {
            ids.insert(String::from_utf8_lossy(suffix).into_owned());
            Ok(())
        })?;
        Ok(ids)
    }

    /// Ids of documents whose `field` holds an indexed value in `(lo, hi)`,
    /// where the bounds are full index keys.
    fn ids_in_range(
        &self,
        txn: u64,
        field: &str,
        lo: Bound<&[u8]>,
        hi: Bound<&[u8]>,
    ) -> Result<IdSet> {
        let base = meta_prefix(field);
        let mut ids = IdSet::new();
        self.scan_suffixes(txn, lo, hi, base.len(), |suffix, _| {
            let n = encoded_len(suffix)
                .ok_or_else(|| Error::corrupt("malformed metadata index key"))?;
            ids.insert(String::from_utf8_lossy(&suffix[n..]).into_owned());
            Ok(())
        })?;
        Ok(ids)
    }

    fn ids_eq(&self, txn: u64, field: &str, value: &Scalar) -> Result<IdSet> {
        let mut prefix = meta_prefix(field);
        prefix.extend_from_slice(&value.encode());
        let end = prefix_successor(&prefix);
        let hi = end.as_deref().map_or(Bound::Unbounded, Bound::Excluded);
        self.ids_in_range(txn, field, Bound::Included(&prefix), hi)
    }

    /// Every id in the collection, loaded at most once per evaluation.
    fn universe<'a>(&self, txn: u64, all: &'a mut Option<IdSet>) -> Result<&'a IdSet> {
        if all.is_none() {
            *all = Some(self.all_ids(txn)?);
        }
        Ok(all.as_ref().expect("just filled"))
    }

    /// Evaluates `filter` to the set of matching ids, from the index.
    ///
    /// Agrees with [`Filter::matches`] by construction: equality, ranges and
    /// `$in` read value entries, `$exists` reads presence entries, and the
    /// negations are complements within the id set.
    fn eval(&self, txn: u64, filter: &Filter, all: &mut Option<IdSet>) -> Result<IdSet> {
        let complement = |this: &Self, hit: IdSet, all: &mut Option<IdSet>| -> Result<IdSet> {
            Ok(this.universe(txn, all)?.difference(&hit).cloned().collect())
        };
        Ok(match filter {
            Filter::All => self.universe(txn, all)?.clone(),
            Filter::Eq(field, v) => self.ids_eq(txn, field, v)?,
            Filter::In(field, vs) => {
                let mut out = IdSet::new();
                for v in vs {
                    out.extend(self.ids_eq(txn, field, v)?);
                }
                out
            }
            Filter::Range(field, op, v) => {
                let base = meta_prefix(field);
                let mut at = base.clone();
                at.extend_from_slice(&v.encode());
                let after = prefix_successor(&at).unwrap_or_else(|| at.clone());
                let mut type_lo = base.clone();
                type_lo.push(v.type_tag());
                let mut type_hi = base;
                type_hi.push(v.type_tag() + 1);
                let (lo, hi): (Bound<&[u8]>, Bound<&[u8]>) = match op {
                    RangeOp::Gt => (Bound::Included(&after), Bound::Excluded(&type_hi)),
                    RangeOp::Gte => (Bound::Included(&at), Bound::Excluded(&type_hi)),
                    RangeOp::Lt => (Bound::Included(&type_lo), Bound::Excluded(&at)),
                    RangeOp::Lte => (Bound::Included(&type_lo), Bound::Excluded(&after)),
                };
                self.ids_in_range(txn, field, lo, hi)?
            }
            Filter::Exists(field, true) => self.ids_present(txn, field)?,
            Filter::Exists(field, false) => {
                let present = self.ids_present(txn, field)?;
                complement(self, present, all)?
            }
            Filter::Ne(field, v) => {
                let eq = self.ids_eq(txn, field, v)?;
                complement(self, eq, all)?
            }
            Filter::Nin(field, vs) => {
                let hit = self.eval(txn, &Filter::In(field.clone(), vs.clone()), all)?;
                complement(self, hit, all)?
            }
            Filter::And(fs) => {
                let mut acc: Option<IdSet> = None;
                for f in fs {
                    let set = self.eval(txn, f, all)?;
                    let next = match acc {
                        None => set,
                        Some(prev) => prev.intersection(&set).cloned().collect(),
                    };
                    let empty = next.is_empty();
                    acc = Some(next);
                    if empty {
                        break;
                    }
                }
                acc.unwrap_or_default()
            }
            Filter::Or(fs) => {
                let mut acc = IdSet::new();
                for f in fs {
                    acc.extend(self.eval(txn, f, all)?);
                }
                acc
            }
            Filter::Not(f) => {
                let hit = self.eval(txn, f, all)?;
                complement(self, hit, all)?
            }
        })
    }

    /// Ids of documents where `field` is present.
    fn ids_present(&self, txn: u64, field: &str) -> Result<IdSet> {
        let mut prefix = meta_prefix(field);
        prefix.push(PRESENT);
        let mut ids = IdSet::new();
        self.prefix_scan(txn, &prefix, |suffix, _| {
            ids.insert(String::from_utf8_lossy(suffix).into_owned());
            Ok(())
        })?;
        Ok(ids)
    }

    /// Matching ids, or `None` for "everything" (no filter).
    fn filter_ids(&self, txn: u64, filter: Option<&Filter>) -> Result<Option<IdSet>> {
        match filter {
            None | Some(Filter::All) => Ok(None),
            Some(f) => Ok(Some(self.eval(txn, f, &mut None)?)),
        }
    }

    /// The document with `id`, with its embedding when `with_vector`.
    pub fn get(&self, id: &str, with_vector: bool) -> Result<Option<Document>> {
        let Some(record) = self.record_auto(id)? else {
            return Ok(None);
        };
        let vector = match (&self.vectors, with_vector && record.has_vector) {
            (Some(engine), true) => engine.get(id).ok(),
            _ => None,
        };
        Ok(Some(Document {
            id: id.to_string(),
            text: record.text,
            metadata: parse_metadata(&record.metadata),
            vector,
        }))
    }

    /// Number of documents, or of those matching `filter`.
    pub fn count(&self, filter: Option<&Filter>) -> Result<u64> {
        let txn = self.db.begin(true)?;
        let result = (|| match self.filter_ids(txn, filter)? {
            None => Ok(self.counters(txn)?.docs),
            Some(ids) => Ok(ids.len() as u64),
        })();
        let _ = self.db.rollback(txn);
        result
    }

    /// Documents matching `filter`, in id order, after `after` (exclusive),
    /// at most `limit` (0 = no limit). Vectors are not included.
    pub fn list(
        &self,
        filter: Option<&Filter>,
        limit: usize,
        after: Option<&str>,
    ) -> Result<Vec<Document>> {
        let txn = self.db.begin(true)?;
        let result = (|| {
            let ids = match self.filter_ids(txn, filter)? {
                Some(ids) => ids,
                None => self.all_ids(txn)?,
            };
            let mut out = Vec::new();
            let range = match after {
                Some(a) => ids.range::<str, _>((Bound::Excluded(a), Bound::Unbounded)),
                None => ids.range::<str, _>(..),
            };
            for id in range {
                if limit != 0 && out.len() >= limit {
                    break;
                }
                if let Some(record) = self.record_in(txn, id)? {
                    out.push(Document {
                        id: id.clone(),
                        text: record.text,
                        metadata: parse_metadata(&record.metadata),
                        vector: None,
                    });
                }
            }
            Ok(out)
        })();
        let _ = self.db.rollback(txn);
        result
    }

    /// Statistics.
    pub fn stats(&self) -> Result<CollectionStats> {
        let txn = self.db.begin(true)?;
        let c = self.counters(txn);
        let _ = self.db.rollback(txn);
        let c = c?;
        Ok(CollectionStats {
            documents: c.docs,
            text_documents: c.text_docs,
            vectors: self.vectors.as_ref().map_or(0, |v| v.len() as u64),
            dim: self.config.dim,
            repaired: self.repaired,
        })
    }

    // ---- writes -------------------------------------------------------------

    /// Inserts or replaces documents, atomically: every document lands or
    /// none does.
    pub fn upsert(&self, docs: &[Document]) -> Result<()> {
        if docs.is_empty() {
            return Ok(());
        }
        // Validate everything before touching storage.
        let mut prepared = Vec::with_capacity(docs.len());
        for doc in docs {
            validate_id(&doc.id)?;
            match &doc.metadata {
                Value::Null | Value::Object(_) => {}
                _ => {
                    return Err(Error::invalid(format!(
                        "document `{}`: metadata must be a JSON object or null",
                        doc.id
                    )));
                }
            }
            let flat = filter::flatten(&doc.metadata, MAX_INDEXED_VALUES)?;
            if let Some(v) = &doc.vector {
                match &self.vectors {
                    Some(_) => crate::vector::distance::validate_vector(v, self.config.dim)?,
                    None => {
                        return Err(Error::invalid(
                            "this collection has no vectors (it was created with dim 0)",
                        ));
                    }
                }
            }
            let metadata = if doc.metadata.is_null() {
                String::new()
            } else {
                doc.metadata.to_string()
            };
            let (terms, text_len) = match (&doc.text, self.config.text_index) {
                (Some(t), true) => text::term_frequencies(t),
                _ => (HashMap::new(), 0),
            };
            let record = DocRecord {
                metadata,
                text: doc.text.clone(),
                has_vector: doc.vector.is_some(),
                text_len,
            };
            let encoded = bincode::serialize(&record)?;
            if encoded.len() > MAX_DOCUMENT_BYTES {
                return Err(Error::invalid(format!(
                    "document `{}` is {} bytes; the limit is {MAX_DOCUMENT_BYTES}",
                    doc.id,
                    encoded.len()
                )));
            }
            prepared.push((doc, flat, terms, record, encoded));
        }

        let _turn = self.db.serialize_writes();
        let txn = self.db.begin(false)?;
        let staged = (|| -> Result<Vec<String>> {
            let mut counters = self.counters(txn)?;
            let mut vectorless = Vec::new();
            for (doc, flat, terms, record, encoded) in &prepared {
                if let Some(old) = self.record_in(txn, &doc.id)? {
                    self.unindex(txn, &doc.id, &old, &mut counters)?;
                    if old.has_vector && !record.has_vector {
                        vectorless.push(doc.id.clone());
                    }
                }
                let id = doc.id.as_bytes();
                self.db.insert(txn, &key(DOC, &[id]), encoded)?;
                self.db.insert(txn, &key(IDS, &[id]), &[])?;
                for k in index_keys(flat, id) {
                    self.db.insert(txn, &k, &[])?;
                }
                for (term, tf) in terms {
                    let mut k = term_prefix(term);
                    k.extend_from_slice(id);
                    let mut posting = [0u8; 8];
                    posting[..4].copy_from_slice(&tf.to_le_bytes());
                    posting[4..].copy_from_slice(&record.text_len.to_le_bytes());
                    self.db.insert(txn, &k, &posting)?;
                }
                counters.docs += 1;
                if record.text.is_some() {
                    counters.text_docs += 1;
                    counters.text_terms += u64::from(record.text_len);
                }
            }
            self.db
                .insert(txn, STATS_KEY, &bincode::serialize(&counters)?)?;
            Ok(vectorless)
        })();
        let mut vectorless = match staged {
            Ok(v) => v,
            Err(e) => {
                let _ = self.db.rollback(txn);
                return Err(e);
            }
        };

        // Vectors go to disk before the documents commit, so a committed
        // document never lacks its vector.
        if let Some(engine) = &self.vectors {
            let items: Vec<(&str, &[f32])> = prepared
                .iter()
                .filter_map(|(doc, ..)| doc.vector.as_deref().map(|v| (doc.id.as_str(), v)))
                .collect();
            let written = engine.insert_many(&items).and_then(|()| {
                if self.sync_on_write {
                    engine.flush()
                } else {
                    Ok(())
                }
            });
            if let Err(e) = written {
                let _ = self.db.rollback(txn);
                return Err(e);
            }
        }
        if let Err(e) = self.db.commit(txn) {
            let _ = self.db.rollback(txn);
            return Err(e);
        }
        // Only ids whose *final* version in this batch has no vector.
        let last: HashMap<&str, bool> = docs
            .iter()
            .map(|d| (d.id.as_str(), d.vector.is_some()))
            .collect();
        vectorless.retain(|id| last.get(id.as_str()) == Some(&false));
        if let Some(engine) = &self.vectors {
            for id in vectorless {
                let _ = engine.remove(&id);
            }
        }
        Ok(())
    }

    /// Removes a document's index entries and counts (not its record).
    fn unindex(&self, txn: u64, id: &str, old: &DocRecord, counters: &mut Counters) -> Result<()> {
        let idb = id.as_bytes();
        // The stored metadata passed `flatten` when it was written, so this
        // reproduces exactly the keys that were indexed.
        let flat = filter::flatten(&parse_metadata(&old.metadata), MAX_INDEXED_VALUES)?;
        for k in index_keys(&flat, idb) {
            remove_if_present(&self.db, txn, &k)?;
        }
        if let (Some(t), true) = (&old.text, self.config.text_index) {
            let (terms, _) = text::term_frequencies(t);
            for term in terms.keys() {
                let mut k = term_prefix(term);
                k.extend_from_slice(idb);
                remove_if_present(&self.db, txn, &k)?;
            }
            counters.text_docs = counters.text_docs.saturating_sub(1);
            counters.text_terms = counters.text_terms.saturating_sub(u64::from(old.text_len));
        }
        counters.docs = counters.docs.saturating_sub(1);
        Ok(())
    }

    /// Deletes documents by id; returns how many existed.
    pub fn delete(&self, ids: &[&str]) -> Result<u64> {
        let _turn = self.db.serialize_writes();
        let txn = self.db.begin(false)?;
        let staged = (|| -> Result<Vec<String>> {
            let mut counters = self.counters(txn)?;
            let mut removed = Vec::new();
            for id in ids {
                let Some(old) = self.record_in(txn, id)? else {
                    continue;
                };
                self.unindex(txn, id, &old, &mut counters)?;
                remove_if_present(&self.db, txn, &key(DOC, &[id.as_bytes()]))?;
                remove_if_present(&self.db, txn, &key(IDS, &[id.as_bytes()]))?;
                removed.push((*id).to_string());
            }
            self.db
                .insert(txn, STATS_KEY, &bincode::serialize(&counters)?)?;
            Ok(removed)
        })();
        let removed = match staged {
            Ok(r) => r,
            Err(e) => {
                let _ = self.db.rollback(txn);
                return Err(e);
            }
        };
        if let Err(e) = self.db.commit(txn) {
            let _ = self.db.rollback(txn);
            return Err(e);
        }
        // After the commit: a crash here leaves an orphaned vector, which
        // recovery removes.
        if let Some(engine) = &self.vectors {
            for id in &removed {
                let _ = engine.remove(id);
            }
        }
        Ok(removed.len() as u64)
    }

    // ---- search -------------------------------------------------------------

    /// Runs a search; see [`SearchRequest`].
    ///
    /// * vector only — nearest neighbours by the collection's metric;
    /// * text only — BM25;
    /// * both — the two rankings fused by [`Fusion`];
    /// * neither — the filter's matches (or everything), in id order.
    pub fn search(&self, req: &SearchRequest) -> Result<Vec<Hit>> {
        if req.k == 0 {
            return Ok(Vec::new());
        }
        if req.k > MAX_K {
            return Err(Error::invalid(format!("k must be at most {MAX_K}")));
        }
        if let Some(l) = req.mmr
            && !(0.0..=1.0).contains(&l)
        {
            return Err(Error::invalid("mmr lambda must be in [0, 1]"));
        }
        if let Some(v) = &req.vector {
            if self.vectors.is_none() {
                return Err(Error::invalid("this collection has no vectors"));
            }
            crate::vector::distance::validate_vector(v, self.config.dim)?;
        }
        let text_query = req.text.as_deref().filter(|t| !t.trim().is_empty());
        let fetch = req
            .candidates
            .unwrap_or(req.k * 4)
            .max(req.k)
            .clamp(20, MAX_CANDIDATES);

        let txn = self.db.begin(true)?;
        let result = self.search_in(txn, req, text_query, fetch);
        let _ = self.db.rollback(txn);
        result
    }

    fn search_in(
        &self,
        txn: u64,
        req: &SearchRequest,
        text_query: Option<&str>,
        fetch: usize,
    ) -> Result<Vec<Hit>> {
        let allowed = self.filter_ids(txn, req.filter.as_ref())?;
        if allowed.as_ref().is_some_and(BTreeSet::is_empty) {
            return Ok(Vec::new());
        }

        // Candidate generation.
        let vector_hits: Vec<(String, f32, f32)> = match (&req.vector, &self.vectors) {
            (Some(query), Some(engine)) => {
                let matches = match &allowed {
                    Some(ids) => {
                        let refs: Vec<&str> = ids.iter().map(String::as_str).collect();
                        engine.search_ids(query, fetch, req.ef, &refs)?
                    }
                    None => engine.search(query, fetch, req.ef)?,
                };
                matches
                    .into_iter()
                    .map(|m| (m.id, m.distance, m.score))
                    .collect()
            }
            _ => Vec::new(),
        };
        let text_hits = match text_query {
            Some(q) if self.config.text_index => self.bm25(txn, q, allowed.as_ref(), fetch)?,
            _ => Vec::new(),
        };

        // Scoring.
        let mut scored: Vec<Hit> = if req.vector.is_none() && text_query.is_none() {
            let ids: Vec<String> = match allowed {
                Some(ids) => ids.into_iter().take(req.k).collect(),
                None => self.all_ids(txn)?.into_iter().take(req.k).collect(),
            };
            ids.into_iter().map(|id| blank_hit(id, 1.0)).collect()
        } else {
            fuse(
                &vector_hits,
                &text_hits,
                req.fusion,
                req.vector.is_some(),
                text_query.is_some(),
            )
        };

        if let (Some(lambda), Some(engine)) = (req.mmr, &self.vectors) {
            scored = mmr(engine, scored, req.k, lambda);
        } else {
            scored.truncate(req.k);
        }
        if let Some(min) = req.min_score {
            scored.retain(|h| h.score >= min);
        }

        for hit in &mut scored {
            if (req.include_text || req.include_metadata)
                && let Some(record) = self.record_in(txn, &hit.id)?
            {
                if req.include_text {
                    hit.text = record.text;
                }
                if req.include_metadata {
                    hit.metadata = Some(parse_metadata(&record.metadata));
                }
            }
            if req.include_vector
                && let Some(engine) = &self.vectors
            {
                hit.vector = engine.get(&hit.id).ok();
            }
        }
        Ok(scored)
    }

    /// BM25 over the inverted index: the best `n` `(id, score)` pairs.
    fn bm25(
        &self,
        txn: u64,
        query: &str,
        allowed: Option<&IdSet>,
        n: usize,
    ) -> Result<Vec<(String, f32)>> {
        let counters = self.counters(txn)?;
        if counters.text_docs == 0 {
            return Ok(Vec::new());
        }
        let avg_len = counters.text_terms as f32 / counters.text_docs as f32;
        let mut terms = text::tokenize(query);
        terms.sort();
        terms.dedup();
        let mut scores: HashMap<String, f32> = HashMap::new();
        for term in terms {
            let mut postings: Vec<(String, u32, u32)> = Vec::new();
            self.prefix_scan(txn, &term_prefix(&term), |suffix, value| {
                if value.len() == 8 {
                    let tf = u32::from_le_bytes(value[..4].try_into().expect("4 bytes"));
                    let len = u32::from_le_bytes(value[4..].try_into().expect("4 bytes"));
                    postings.push((String::from_utf8_lossy(suffix).into_owned(), tf, len));
                }
                Ok(())
            })?;
            if postings.is_empty() {
                continue;
            }
            let idf = text::idf(counters.text_docs, postings.len() as u64);
            for (id, tf, len) in postings {
                if allowed.is_some_and(|a| !a.contains(&id)) {
                    continue;
                }
                *scores.entry(id).or_insert(0.0) += text::bm25(idf, tf, len, avg_len);
            }
        }
        let mut ranked: Vec<(String, f32)> = scores.into_iter().collect();
        ranked.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        ranked.truncate(n);
        Ok(ranked)
    }

    /// Syncs the vectors and checkpoints the documents.
    pub fn flush(&self) -> Result<()> {
        if let Some(engine) = &self.vectors {
            engine.save(None)?;
        }
        self.db.checkpoint()
    }
}

impl Collection {
    /// Drops the collection as if the process had died: nothing is flushed
    /// and the unclean-shutdown marker stays. Intended for recovery tests.
    #[doc(hidden)]
    pub fn simulate_crash(mut self) {
        self.crashed = true;
        self.db.crashed = true;
        if let Some(engine) = &mut self.vectors {
            engine.crashed = true;
        }
    }

    /// Test hook: writes a vector with no document, as a crash between the
    /// vector write and the commit would.
    #[doc(hidden)]
    pub fn inject_orphan_vector(&self, id: &str, vector: &[f32]) -> Result<()> {
        match &self.vectors {
            Some(engine) => engine.insert(id, vector),
            None => Err(Error::invalid("this collection has no vectors")),
        }
    }
}

impl Drop for Collection {
    fn drop(&mut self) {
        if self.crashed {
            return;
        }
        // A clean close: the next open need not reconcile. The vectors are
        // synced first so the marker never outlives unsynced data; the engine
        // writes its graph snapshot when it drops. Best effort — a failure
        // only means the next open does a little extra work.
        if let Some(engine) = &self.vectors
            && engine.flush().is_err()
        {
            return;
        }
        let _ = self.db.delete_auto(OPEN_KEY);
    }
}

impl std::fmt::Debug for Collection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Collection")
            .field("dir", &self.dir)
            .field("dim", &self.config.dim)
            .finish()
    }
}

/// Metadata index keys for one document.
fn index_keys(flat: &Flattened, id: &[u8]) -> Vec<Vec<u8>> {
    let mut keys = Vec::with_capacity(flat.values.len() + flat.present.len());
    for (field, value) in &flat.values {
        let mut k = meta_prefix(field);
        k.extend_from_slice(&value.encode());
        k.extend_from_slice(id);
        keys.push(k);
    }
    for field in &flat.present {
        let mut k = meta_prefix(field);
        k.push(PRESENT);
        k.extend_from_slice(id);
        keys.push(k);
    }
    keys
}

fn remove_if_present(db: &Database, txn: u64, key: &[u8]) -> Result<()> {
    match db.delete(txn, key) {
        Ok(()) | Err(Error::NotFound) => Ok(()),
        Err(e) => Err(e),
    }
}

fn parse_metadata(text: &str) -> Value {
    if text.is_empty() {
        Value::Null
    } else {
        serde_json::from_str(text).unwrap_or(Value::Null)
    }
}

fn blank_hit(id: String, score: f32) -> Hit {
    Hit {
        id,
        score,
        vector_score: None,
        distance: None,
        text_score: None,
        text: None,
        metadata: None,
        vector: None,
    }
}

/// Combines the rankings into scored hits, best first.
fn fuse(
    vector: &[(String, f32, f32)],
    text: &[(String, f32)],
    fusion: Fusion,
    use_vector: bool,
    use_text: bool,
) -> Vec<Hit> {
    let mut hits: HashMap<String, Hit> = HashMap::new();
    for (id, distance, score) in vector {
        let h = hits
            .entry(id.clone())
            .or_insert_with(|| blank_hit(id.clone(), 0.0));
        h.distance = Some(*distance);
        h.vector_score = Some(*score);
    }
    for (id, score) in text {
        let h = hits
            .entry(id.clone())
            .or_insert_with(|| blank_hit(id.clone(), 0.0));
        h.text_score = Some(*score);
    }
    match (use_vector, use_text) {
        (true, false) => {
            for h in hits.values_mut() {
                h.score = h.vector_score.unwrap_or(f32::MIN);
            }
        }
        (false, true) => {
            for h in hits.values_mut() {
                h.score = h.text_score.unwrap_or(0.0);
            }
        }
        _ => match fusion {
            Fusion::Rrf { k } => {
                let k = k.max(1.0);
                for (rank, (id, ..)) in vector.iter().enumerate() {
                    hits.get_mut(id).expect("inserted").score += 1.0 / (k + rank as f32 + 1.0);
                }
                for (rank, (id, _)) in text.iter().enumerate() {
                    hits.get_mut(id).expect("inserted").score += 1.0 / (k + rank as f32 + 1.0);
                }
            }
            Fusion::Weighted { alpha } => {
                let alpha = alpha.clamp(0.0, 1.0);
                let (vmin, vmax) = min_max(vector.iter().map(|(_, _, s)| *s));
                let (tmin, tmax) = min_max(text.iter().map(|(_, s)| *s));
                for h in hits.values_mut() {
                    let v = h.vector_score.map_or(0.0, |s| normalise(s, vmin, vmax));
                    let t = h.text_score.map_or(0.0, |s| normalise(s, tmin, tmax));
                    h.score = alpha * v + (1.0 - alpha) * t;
                }
            }
        },
    }
    let mut out: Vec<Hit> = hits.into_values().collect();
    out.sort_by(|a, b| b.score.total_cmp(&a.score).then_with(|| a.id.cmp(&b.id)));
    out
}

fn min_max(scores: impl Iterator<Item = f32>) -> (f32, f32) {
    scores.fold((f32::INFINITY, f32::NEG_INFINITY), |(lo, hi), s| {
        (lo.min(s), hi.max(s))
    })
}

fn normalise(s: f32, lo: f32, hi: f32) -> f32 {
    if hi > lo { (s - lo) / (hi - lo) } else { 1.0 }
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let d = norm(a) * norm(b);
    if d <= f32::MIN_POSITIVE {
        0.0
    } else {
        dot(a, b) / d
    }
}

/// Maximal Marginal Relevance: greedily picks `k` hits maximising
/// `λ·relevance − (1−λ)·max_similarity_to_already_picked`.
///
/// Relevance is the fused score normalised to `[0, 1]`, so MMR works for text
/// and hybrid rankings too; redundancy is cosine similarity of embeddings
/// (documents without one are never considered redundant).
fn mmr(engine: &VectorEngine, candidates: Vec<Hit>, k: usize, lambda: f32) -> Vec<Hit> {
    let (lo, hi) = min_max(candidates.iter().map(|h| h.score));
    let mut pool: Vec<(Hit, Option<Vec<f32>>, f32)> = candidates
        .into_iter()
        .map(|h| {
            let v = engine.get(&h.id).ok();
            let rel = normalise(h.score, lo, hi);
            (h, v, rel)
        })
        .collect();
    let mut picked: Vec<(Hit, Option<Vec<f32>>)> = Vec::with_capacity(k);
    while picked.len() < k && !pool.is_empty() {
        let mut best = 0usize;
        let mut best_value = f32::NEG_INFINITY;
        for (i, (_, v, rel)) in pool.iter().enumerate() {
            let redundancy = match v {
                Some(v) => picked
                    .iter()
                    .filter_map(|(_, p)| p.as_ref().map(|p| cosine(v, p)))
                    .fold(0.0f32, f32::max),
                None => 0.0,
            };
            let value = lambda * rel - (1.0 - lambda) * redundancy;
            if value > best_value {
                best_value = value;
                best = i;
            }
        }
        let (hit, v, _) = pool.remove(best);
        picked.push((hit, v));
    }
    picked.into_iter().map(|(h, _)| h).collect()
}

#[cfg(test)]
mod tests;
