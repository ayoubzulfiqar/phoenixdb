//! Bloom filters: O(1) negative lookups for SSTables.
//!
//! Every SSTable carries a filter over the full key set it contains. A point
//! lookup consults the filter *before* touching the disk, so a key that is not
//! present costs one cache-resident bit-probe instead of an index binary search
//! plus a block read.
//!
//! # Parameters
//!
//! For `n` keys and `m` bits the optimal number of hash functions is
//! `k = (m/n) · ln 2`, giving a false-positive rate of roughly
//! `(1 - e^(-kn/m))^k`. At the default 10 bits/key that is `k = 7` and a ~0.8 %
//! false-positive rate. False positives only cost a wasted block read; **false
//! negatives are impossible**, which is what makes the filter safe to consult
//! before the real index.
//!
//! # Hashing
//!
//! Kirsch-Mitzenmacher double hashing: two independent 64-bit hashes are
//! derived from one pass over the key and combined as `h1 + i·h2`, which is
//! provably as good as `k` independent hashes for filter purposes and costs a
//! single pass instead of `k`.

use crate::error::{Error, Result};

/// Bits allocated per key when no explicit budget is given.
pub const DEFAULT_BITS_PER_KEY: u32 = 10;

/// Serialised header size: `num_bits` (8) + `num_hashes` (4).
const HEADER_LEN: usize = 12;

/// Hard cap on filter size (128 MiB of bits) so a corrupt length cannot drive
/// an unbounded allocation during decode.
const MAX_BITS: u64 = 1 << 30;

/// A classic Bloom filter over byte-string keys.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BloomFilter {
    /// Bit array, packed 64 bits per word.
    words: Vec<u64>,
    /// Number of addressable bits (`words.len() * 64`, rounded).
    num_bits: u64,
    /// Number of hash probes per key.
    num_hashes: u32,
}

impl BloomFilter {
    /// Builds an empty filter sized for `expected_keys` at `bits_per_key`.
    ///
    /// `expected_keys` is only a sizing hint: inserting more keys than
    /// estimated raises the false-positive rate but never breaks correctness.
    #[must_use]
    pub fn with_capacity(expected_keys: usize, bits_per_key: u32) -> Self {
        let bpk = bits_per_key.clamp(1, 64) as u64;
        let n = expected_keys.max(1) as u64;
        // Round up to a whole word and keep at least one word.
        let num_bits = (n.saturating_mul(bpk))
            .clamp(64, MAX_BITS)
            .next_multiple_of(64);
        let num_hashes = Self::optimal_hashes(bpk);
        BloomFilter {
            words: vec![0u64; (num_bits / 64) as usize],
            num_bits,
            num_hashes,
        }
    }

    /// `k = (m/n)·ln2`, clamped to a sane probe count.
    #[must_use]
    fn optimal_hashes(bits_per_key: u64) -> u32 {
        let k = (bits_per_key as f64 * std::f64::consts::LN_2).round() as u32;
        k.clamp(1, 30)
    }

    /// Number of bits in the filter.
    #[must_use]
    pub fn num_bits(&self) -> u64 {
        self.num_bits
    }

    /// Number of hash probes performed per key.
    #[must_use]
    pub fn num_hashes(&self) -> u32 {
        self.num_hashes
    }

    /// Heap footprint of the bit array in bytes.
    #[must_use]
    pub fn size_bytes(&self) -> usize {
        self.words.len() * 8
    }

    /// Records `key` as present.
    pub fn insert(&mut self, key: &[u8]) {
        let (h1, h2) = Self::hash_pair(key);
        for i in 0..self.num_hashes as u64 {
            let bit = h1.wrapping_add(i.wrapping_mul(h2)) % self.num_bits;
            self.words[(bit / 64) as usize] |= 1u64 << (bit % 64);
        }
    }

    /// Returns `false` only when `key` is **definitely absent**.
    ///
    /// A `true` result means "probably present" — the caller must still consult
    /// the real index.
    #[must_use]
    pub fn contains(&self, key: &[u8]) -> bool {
        let (h1, h2) = Self::hash_pair(key);
        for i in 0..self.num_hashes as u64 {
            let bit = h1.wrapping_add(i.wrapping_mul(h2)) % self.num_bits;
            if self.words[(bit / 64) as usize] & (1u64 << (bit % 64)) == 0 {
                return false;
            }
        }
        true
    }

    /// Estimated false-positive rate given `inserted` keys.
    #[must_use]
    pub fn estimated_fp_rate(&self, inserted: usize) -> f64 {
        let m = self.num_bits as f64;
        let n = inserted as f64;
        let k = self.num_hashes as f64;
        (1.0 - (-k * n / m).exp()).powf(k)
    }

    /// Serialises the filter: `[num_bits u64][num_hashes u32][words…]`.
    #[must_use]
    pub fn encode(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(HEADER_LEN + self.words.len() * 8);
        out.extend_from_slice(&self.num_bits.to_le_bytes());
        out.extend_from_slice(&self.num_hashes.to_le_bytes());
        for w in &self.words {
            out.extend_from_slice(&w.to_le_bytes());
        }
        out
    }

    /// Parses a filter produced by [`BloomFilter::encode`].
    ///
    /// Every field is bounds-checked: a corrupt or hostile buffer yields
    /// [`Error::Corruption`] rather than a panic or a huge allocation.
    pub fn decode(bytes: &[u8]) -> Result<Self> {
        if bytes.len() < HEADER_LEN {
            return Err(Error::corrupt(format!(
                "bloom filter is {} bytes, need at least {HEADER_LEN}",
                bytes.len()
            )));
        }
        let num_bits = u64::from_le_bytes(
            bytes[0..8]
                .try_into()
                .map_err(|_| Error::corrupt("bloom header truncated"))?,
        );
        let num_hashes = u32::from_le_bytes(
            bytes[8..12]
                .try_into()
                .map_err(|_| Error::corrupt("bloom header truncated"))?,
        );
        if num_bits == 0 || num_bits > MAX_BITS || num_bits % 64 != 0 {
            return Err(Error::corrupt(format!(
                "bloom filter declares an implausible bit count of {num_bits}"
            )));
        }
        if num_hashes == 0 || num_hashes > 30 {
            return Err(Error::corrupt(format!(
                "bloom filter declares {num_hashes} hash probes"
            )));
        }
        let word_count = (num_bits / 64) as usize;
        let expected = HEADER_LEN + word_count * 8;
        if bytes.len() != expected {
            return Err(Error::corrupt(format!(
                "bloom filter is {} bytes, header implies {expected}",
                bytes.len()
            )));
        }
        let mut words = Vec::with_capacity(word_count);
        for i in 0..word_count {
            let off = HEADER_LEN + i * 8;
            words.push(u64::from_le_bytes(
                bytes[off..off + 8]
                    .try_into()
                    .map_err(|_| Error::corrupt("bloom word truncated"))?,
            ));
        }
        Ok(BloomFilter {
            words,
            num_bits,
            num_hashes,
        })
    }

    /// Two independent 64-bit hashes from one pass over `key`.
    ///
    /// FNV-1a supplies the raw mixing; a SplitMix64 finaliser removes the poor
    /// avalanche behaviour FNV exhibits in its low bits. `h2` is forced odd so
    /// that `h1 + i·h2` visits distinct residues modulo a power-of-two-derived
    /// bit count.
    #[inline]
    fn hash_pair(key: &[u8]) -> (u64, u64) {
        const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
        const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

        let mut a = FNV_OFFSET;
        let mut b = FNV_OFFSET ^ 0x9e37_79b9_7f4a_7c15;
        for &byte in key {
            a ^= byte as u64;
            a = a.wrapping_mul(FNV_PRIME);
            b = b.rotate_left(7) ^ (byte as u64);
            b = b.wrapping_mul(FNV_PRIME);
        }
        (splitmix64(a), splitmix64(b) | 1)
    }
}

/// SplitMix64 finaliser — a bijective avalanche mixer.
#[inline]
fn splitmix64(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_false_negatives_ever() {
        let keys: Vec<Vec<u8>> = (0..5_000u32)
            .map(|i| format!("key-{i:08}").into_bytes())
            .collect();
        let mut f = BloomFilter::with_capacity(keys.len(), DEFAULT_BITS_PER_KEY);
        for k in &keys {
            f.insert(k);
        }
        for k in &keys {
            assert!(f.contains(k), "false negative on {k:?} — filter is unsound");
        }
    }

    #[test]
    fn false_positive_rate_is_near_theory() {
        let n = 10_000usize;
        let mut f = BloomFilter::with_capacity(n, DEFAULT_BITS_PER_KEY);
        for i in 0..n {
            f.insert(format!("present-{i:08}").as_bytes());
        }
        let probes = 20_000;
        let mut hits = 0;
        for i in 0..probes {
            if f.contains(format!("absent-{i:08}").as_bytes()) {
                hits += 1;
            }
        }
        let observed = hits as f64 / probes as f64;
        // Theory says ~0.8 % at 10 bits/key; allow generous slack for hash luck.
        assert!(
            observed < 0.05,
            "false-positive rate {observed:.4} is far above the ~0.008 target"
        );
    }

    #[test]
    fn empty_filter_rejects_everything() {
        let f = BloomFilter::with_capacity(1000, DEFAULT_BITS_PER_KEY);
        assert!(!f.contains(b"anything"));
    }

    #[test]
    fn encode_decode_roundtrip() {
        let mut f = BloomFilter::with_capacity(500, 12);
        for i in 0..500u32 {
            f.insert(&i.to_le_bytes());
        }
        let bytes = f.encode();
        let back = BloomFilter::decode(&bytes).unwrap();
        assert_eq!(f, back);
        for i in 0..500u32 {
            assert!(back.contains(&i.to_le_bytes()));
        }
    }

    #[test]
    fn decode_rejects_truncated_and_hostile_input() {
        assert!(BloomFilter::decode(&[]).is_err());
        assert!(BloomFilter::decode(&[0u8; 8]).is_err());

        let mut f = BloomFilter::with_capacity(64, 10);
        f.insert(b"x");
        let mut bytes = f.encode();
        // Claim an absurd bit count.
        bytes[0..8].copy_from_slice(&u64::MAX.to_le_bytes());
        assert!(BloomFilter::decode(&bytes).is_err());

        let mut bytes = f.encode();
        bytes.truncate(bytes.len() - 1);
        assert!(BloomFilter::decode(&bytes).is_err());

        let mut bytes = f.encode();
        bytes[8..12].copy_from_slice(&0u32.to_le_bytes()); // zero probes
        assert!(BloomFilter::decode(&bytes).is_err());
    }

    #[test]
    fn sizing_is_sane() {
        let f = BloomFilter::with_capacity(1000, 10);
        assert_eq!(f.num_bits(), 10_048); // 10_000 rounded up to a word boundary
        assert_eq!(f.num_hashes(), 7);
        assert!(f.estimated_fp_rate(1000) < 0.02);
    }

    #[test]
    fn zero_expected_keys_still_builds() {
        let mut f = BloomFilter::with_capacity(0, 10);
        f.insert(b"k");
        assert!(f.contains(b"k"));
    }
}
