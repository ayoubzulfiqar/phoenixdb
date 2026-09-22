//! Text analysis for full-text search: tokenizer, stemmer and BM25 scoring.
//!
//! # Tokenizer
//!
//! * Lower-cases with full Unicode case mapping.
//! * Splits on anything that is not alphanumeric, so it works for every
//!   space-separated script (Latin, Cyrillic, Greek, Arabic, Devanagari, ...).
//! * Han, Hiragana, Katakana and Hangul are written without spaces, so runs of
//!   those characters are indexed as overlapping **bigrams** (and a lone
//!   character as a unigram) — the standard dictionary-free approach, which
//!   makes CJK text searchable at all.
//! * Drops a small set of English stopwords and applies a conservative plural
//!   stemmer (the Harman "S" stemmer), so `databases` matches `database`
//!   without the aggressive conflations of Porter-style stemming.
//!
//! The same analysis runs at index and query time, which is all BM25 needs.

use std::collections::HashMap;

/// BM25 term-frequency saturation.
pub const BM25_K1: f32 = 1.2;
/// BM25 length normalisation.
pub const BM25_B: f32 = 0.75;

/// Longest token indexed, in characters. Longer runs are usually hashes,
/// base64 or URLs, which only bloat the index.
const MAX_TOKEN_CHARS: usize = 40;

const STOPWORDS: &[&str] = &[
    "a", "an", "and", "are", "as", "at", "be", "been", "but", "by", "for", "from", "had", "has",
    "have", "he", "her", "his", "i", "if", "in", "into", "is", "it", "its", "of", "on", "or",
    "our", "she", "so", "that", "the", "their", "them", "then", "there", "these", "they", "this",
    "those", "to", "was", "we", "were", "which", "while", "will", "with", "you", "your",
];

/// True for scripts written without spaces between words.
fn is_cjk(c: char) -> bool {
    matches!(c as u32,
        0x3040..=0x30FF   // Hiragana, Katakana
        | 0x3400..=0x4DBF // CJK Extension A
        | 0x4E00..=0x9FFF // CJK Unified Ideographs
        | 0xF900..=0xFAFF // CJK Compatibility Ideographs
        | 0xAC00..=0xD7AF // Hangul syllables
        | 0x20000..=0x2FFFF // CJK Extensions B..F
    )
}

/// Conservative English plural stemmer (Harman 1991).
fn stem(word: &str) -> String {
    let n = word.len();
    if n > 4 && word.ends_with("ies") && !word.ends_with("eies") && !word.ends_with("aies") {
        return format!("{}y", &word[..n - 3]);
    }
    if n > 3
        && word.ends_with("es")
        && !(word.ends_with("aes") || word.ends_with("ees") || word.ends_with("oes"))
    {
        return word[..n - 1].to_string();
    }
    if n > 3 && word.ends_with('s') && !(word.ends_with("us") || word.ends_with("ss")) {
        return word[..n - 1].to_string();
    }
    word.to_string()
}

/// Splits `text` into index terms.
#[must_use]
pub fn tokenize(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut word = String::new();
    let mut cjk: Vec<char> = Vec::new();

    let flush_word = |word: &mut String, out: &mut Vec<String>| {
        if !word.is_empty() {
            let chars = word.chars().count();
            if chars <= MAX_TOKEN_CHARS && !STOPWORDS.contains(&word.as_str()) {
                out.push(stem(word));
            }
            word.clear();
        }
    };
    let flush_cjk = |cjk: &mut Vec<char>, out: &mut Vec<String>| {
        match cjk.len() {
            0 => {}
            1 => out.push(cjk[0].to_string()),
            _ => {
                for pair in cjk.windows(2) {
                    out.push(pair.iter().collect());
                }
            }
        }
        cjk.clear();
    };

    for c in text.chars() {
        if is_cjk(c) {
            flush_word(&mut word, &mut out);
            cjk.push(c);
        } else if c.is_alphanumeric() {
            flush_cjk(&mut cjk, &mut out);
            word.extend(c.to_lowercase());
        } else {
            flush_word(&mut word, &mut out);
            flush_cjk(&mut cjk, &mut out);
        }
    }
    flush_word(&mut word, &mut out);
    flush_cjk(&mut cjk, &mut out);
    out
}

/// Term frequencies of `text`, plus its length in terms.
#[must_use]
pub fn term_frequencies(text: &str) -> (HashMap<String, u32>, u32) {
    let tokens = tokenize(text);
    let len = tokens.len() as u32;
    let mut tf: HashMap<String, u32> = HashMap::new();
    for t in tokens {
        *tf.entry(t).or_insert(0) += 1;
    }
    (tf, len)
}

/// Inverse document frequency (the BM25+ smoothed form, never negative).
#[must_use]
pub fn idf(docs: u64, doc_freq: u64) -> f32 {
    let n = docs as f32;
    let df = doc_freq as f32;
    (1.0 + (n - df + 0.5) / (df + 0.5)).ln()
}

/// BM25 contribution of one term to one document.
#[must_use]
pub fn bm25(idf: f32, tf: u32, doc_len: u32, avg_doc_len: f32) -> f32 {
    let tf = tf as f32;
    let norm = if avg_doc_len > 0.0 {
        doc_len as f32 / avg_doc_len
    } else {
        1.0
    };
    idf * (tf * (BM25_K1 + 1.0)) / (tf + BM25_K1 * (1.0 - BM25_B + BM25_B * norm))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lowercases_splits_and_drops_stopwords() {
        assert_eq!(
            tokenize("The Quick, brown FOX — jumps!"),
            ["quick", "brown", "fox", "jump"]
        );
    }

    #[test]
    fn plurals_are_stemmed_conservatively() {
        assert_eq!(
            tokenize("databases queries boxes"),
            ["database", "query", "boxe"]
        );
        assert_eq!(tokenize("class status bus"), ["class", "status", "bus"]);
    }

    #[test]
    fn cjk_is_indexed_as_bigrams() {
        assert_eq!(tokenize("東京都"), ["東京", "京都"]);
        assert_eq!(tokenize("猫"), ["猫"]);
        assert_eq!(
            tokenize("rust 数据库 fast"),
            ["rust", "数据", "据库", "fast"]
        );
    }

    #[test]
    fn other_scripts_and_numbers_work() {
        assert_eq!(tokenize("Привет мир 2024"), ["привет", "мир", "2024"]);
        assert_eq!(tokenize("café naïve"), ["café", "naïve"]);
    }

    #[test]
    fn absurdly_long_tokens_are_skipped() {
        let long = "x".repeat(200);
        assert_eq!(tokenize(&format!("keep {long} this")), ["keep"]);
    }

    #[test]
    fn bm25_rewards_rarity_and_saturates_frequency() {
        let rare = idf(1000, 2);
        let common = idf(1000, 800);
        assert!(rare > common && common > 0.0);
        let once = bm25(rare, 1, 100, 100.0);
        let many = bm25(rare, 50, 100, 100.0);
        assert!(many > once && many < once * (BM25_K1 + 1.0) + 1e-3);
        // Shorter documents score higher for the same term frequency.
        assert!(bm25(rare, 1, 10, 100.0) > bm25(rare, 1, 1000, 100.0));
    }
}
