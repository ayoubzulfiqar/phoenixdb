//! Transparent AES-256-GCM encryption at rest.
//!
//! # Crate choice
//!
//! The design called for `ring`. That crate needs a C toolchain, which breaks
//! cross-compilation to Android/iOS from a machine without one (and fails
//! outright on this build host). `aes-gcm` from RustCrypto is pure Rust,
//! implements the same AEAD, and uses AES-NI at runtime when available.
//!
//! # Order of operations
//!
//! The requirement is that page writes are encrypted *before* checksumming and
//! reads are decrypted *after* checksum verification:
//!
//! ```text
//!   write: plaintext ──encrypt──▶ ciphertext ──CRC──▶ disk
//!   read:  disk ──verify CRC──▶ ciphertext ──decrypt──▶ plaintext
//! ```
//!
//! This ordering means a corrupt page is rejected by the cheap CRC check before
//! the expensive AEAD runs, and a tampered page fails the GCM authentication
//! tag even if an attacker recomputed the CRC.
//!
//! # Nonce discipline
//!
//! GCM is catastrophically broken by nonce reuse under the same key: two
//! messages sharing a nonce leak their XOR and allow tag forgery. Nonces here
//! are **deterministic per page and version**: `page_id` (4 bytes) plus a
//! monotonically increasing `version` (8 bytes). The caller must bump the
//! version on every rewrite of a page — [`PageCipher::encrypt_page`] takes it
//! explicitly rather than hiding a counter, so the discipline is visible and
//! testable. [`NonceTracker`] enforces it in debug builds and tests.
//!
//! # Key handling
//!
//! [`EncryptionKey`] zeroes its bytes on drop via
//! [`secure_zero`](crate::security::secure_zero) so a key does not linger in
//! freed heap memory.

use crate::error::{Error, Result};
use crate::security::secure_zero;
use aes_gcm::aead::{Aead, KeyInit, Payload};
use aes_gcm::{Aes256Gcm, Nonce};
use std::collections::HashMap;

/// Length of an AES-256 key in bytes.
pub const KEY_LEN: usize = 32;

/// Length of the GCM nonce in bytes (96 bits, the standard choice).
pub const NONCE_LEN: usize = 12;

/// Length of the GCM authentication tag appended to every ciphertext.
pub const TAG_LEN: usize = 16;

/// Bytes of overhead encryption adds to a payload.
pub const CIPHERTEXT_OVERHEAD: usize = TAG_LEN;

/// A 256-bit encryption key that is wiped from memory when dropped.
#[derive(Clone)]
pub struct EncryptionKey {
    bytes: [u8; KEY_LEN],
}

impl EncryptionKey {
    /// Wraps raw key material.
    #[must_use]
    pub fn from_bytes(bytes: [u8; KEY_LEN]) -> Self {
        EncryptionKey { bytes }
    }

    /// Derives a key from a passphrase.
    ///
    /// # Security
    ///
    /// This is **not** a password-based KDF: it is a fast domain-separated hash
    /// suitable only for turning an already-high-entropy secret into a key. For
    /// user-chosen passwords use Argon2id or scrypt and pass the result to
    /// [`EncryptionKey::from_bytes`]. Named to make the weakness obvious at the
    /// call site.
    #[must_use]
    pub fn derive_insecure_from_passphrase(passphrase: &[u8], salt: &[u8]) -> Self {
        // Four independently-seeded FNV-1a lanes, each finalised with
        // SplitMix64, produce the 32 bytes. Adequate for deriving a key from a
        // random secret; not a substitute for a real KDF.
        let mut key = [0u8; KEY_LEN];
        for (lane, chunk) in key.chunks_mut(8).enumerate() {
            let mut h: u64 = 0xcbf2_9ce4_8422_2325 ^ (lane as u64).wrapping_mul(0x9E37_79B9);
            for &b in b"phoenixdb-kdf-v1" {
                h ^= b as u64;
                h = h.wrapping_mul(0x0000_0100_0000_01b3);
            }
            for &b in salt {
                h ^= b as u64;
                h = h.wrapping_mul(0x0000_0100_0000_01b3);
            }
            for &b in passphrase {
                h ^= b as u64;
                h = h.wrapping_mul(0x0000_0100_0000_01b3);
            }
            // SplitMix64 finaliser for avalanche.
            let mut z = h.wrapping_add(0x9E37_79B9_7F4A_7C15);
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^= z >> 31;
            chunk.copy_from_slice(&z.to_le_bytes());
        }
        EncryptionKey { bytes: key }
    }

    /// Borrows the raw key bytes.
    #[must_use]
    pub fn as_bytes(&self) -> &[u8; KEY_LEN] {
        &self.bytes
    }
}

impl Drop for EncryptionKey {
    fn drop(&mut self) {
        secure_zero(&mut self.bytes);
    }
}

impl std::fmt::Debug for EncryptionKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Never print key material, not even truncated.
        f.write_str("EncryptionKey(<redacted>)")
    }
}

/// Builds the deterministic nonce for `(page_id, version)`.
///
/// Layout: `[page_id u32 LE][version u64 LE]` = 12 bytes exactly.
#[must_use]
pub fn page_nonce(page_id: u32, version: u64) -> [u8; NONCE_LEN] {
    let mut nonce = [0u8; NONCE_LEN];
    nonce[0..4].copy_from_slice(&page_id.to_le_bytes());
    nonce[4..12].copy_from_slice(&version.to_le_bytes());
    nonce
}

/// Encrypts and decrypts page payloads with AES-256-GCM.
pub struct PageCipher {
    cipher: Aes256Gcm,
}

impl PageCipher {
    /// Creates a cipher from `key`.
    pub fn new(key: &EncryptionKey) -> Result<Self> {
        let cipher = Aes256Gcm::new_from_slice(key.as_bytes())
            .map_err(|_| Error::invalid("AES-256-GCM rejected the key length"))?;
        Ok(PageCipher { cipher })
    }

    /// Encrypts `plaintext` for `page_id` at `version`.
    ///
    /// `page_id` is bound as additional authenticated data, so a ciphertext
    /// moved to a different page slot fails authentication instead of decrypting
    /// to valid-looking bytes. The returned buffer is
    /// `plaintext.len() + TAG_LEN` bytes.
    pub fn encrypt_page(&self, page_id: u32, version: u64, plaintext: &[u8]) -> Result<Vec<u8>> {
        let nonce_bytes = page_nonce(page_id, version);
        let nonce = Nonce::from_slice(&nonce_bytes);
        let aad = page_id.to_le_bytes();
        self.cipher
            .encrypt(
                nonce,
                Payload {
                    msg: plaintext,
                    aad: &aad,
                },
            )
            .map_err(|_| Error::corrupt("AES-GCM encryption failed"))
    }

    /// Decrypts `ciphertext` for `page_id` at `version`.
    ///
    /// Returns [`Error::Corruption`] when the authentication tag does not
    /// verify — which covers tampering, a wrong key, a wrong page id, and a
    /// wrong version. Authentication failure is never distinguished from
    /// corruption, so an attacker learns nothing from the error.
    pub fn decrypt_page(&self, page_id: u32, version: u64, ciphertext: &[u8]) -> Result<Vec<u8>> {
        if ciphertext.len() < TAG_LEN {
            return Err(Error::corrupt(format!(
                "ciphertext for page {page_id} is {} bytes, shorter than the {TAG_LEN}-byte tag",
                ciphertext.len()
            )));
        }
        let nonce_bytes = page_nonce(page_id, version);
        let nonce = Nonce::from_slice(&nonce_bytes);
        let aad = page_id.to_le_bytes();
        self.cipher
            .decrypt(
                nonce,
                Payload {
                    msg: ciphertext,
                    aad: &aad,
                },
            )
            .map_err(|_| {
                Error::corrupt(format!(
                    "page {page_id} failed AES-GCM authentication (tampered, wrong key, \
                     or wrong version)"
                ))
            })
    }
}

impl std::fmt::Debug for PageCipher {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("PageCipher(AES-256-GCM)")
    }
}

/// Debug-time guard against catastrophic GCM nonce reuse.
///
/// Records the highest version seen per page and rejects any attempt to reuse
/// or go backwards. Intended for tests and debug builds; the production write
/// path derives versions from the monotonically increasing page LSN.
#[derive(Debug, Default)]
pub struct NonceTracker {
    seen: HashMap<u32, u64>,
}

impl NonceTracker {
    /// Creates an empty tracker.
    #[must_use]
    pub fn new() -> Self {
        NonceTracker {
            seen: HashMap::new(),
        }
    }

    /// Records `(page_id, version)`, or fails if that nonce was already used.
    pub fn record(&mut self, page_id: u32, version: u64) -> Result<()> {
        match self.seen.get(&page_id) {
            Some(&previous) if version <= previous => Err(Error::invalid(format!(
                "GCM nonce reuse on page {page_id}: version {version} <= previous {previous}"
            ))),
            _ => {
                self.seen.insert(page_id, version);
                Ok(())
            }
        }
    }

    /// Highest version recorded for `page_id`.
    #[must_use]
    pub fn version_of(&self, page_id: u32) -> Option<u64> {
        self.seen.get(&page_id).copied()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_key() -> EncryptionKey {
        EncryptionKey::from_bytes([0x42; KEY_LEN])
    }

    #[test]
    fn roundtrip_recovers_the_plaintext() {
        let cipher = PageCipher::new(&test_key()).unwrap();
        let plaintext = b"the quick brown fox jumps over the lazy dog";
        let ct = cipher.encrypt_page(7, 1, plaintext).unwrap();
        assert_ne!(
            &ct[..plaintext.len()],
            plaintext,
            "output must not be plaintext"
        );
        assert_eq!(ct.len(), plaintext.len() + TAG_LEN);

        let pt = cipher.decrypt_page(7, 1, &ct).unwrap();
        assert_eq!(pt, plaintext);
    }

    #[test]
    fn tampering_with_any_byte_fails_authentication() {
        let cipher = PageCipher::new(&test_key()).unwrap();
        let ct = cipher.encrypt_page(1, 1, b"sensitive data").unwrap();
        for i in 0..ct.len() {
            let mut bad = ct.clone();
            bad[i] ^= 0x01;
            assert!(
                cipher.decrypt_page(1, 1, &bad).is_err(),
                "flipping byte {i} must fail the GCM tag"
            );
        }
    }

    #[test]
    fn wrong_key_cannot_decrypt() {
        let a = PageCipher::new(&EncryptionKey::from_bytes([1; KEY_LEN])).unwrap();
        let b = PageCipher::new(&EncryptionKey::from_bytes([2; KEY_LEN])).unwrap();
        let ct = a.encrypt_page(1, 1, b"secret").unwrap();
        assert!(matches!(
            b.decrypt_page(1, 1, &ct),
            Err(Error::Corruption(_))
        ));
    }

    #[test]
    fn page_id_is_authenticated_so_pages_cannot_be_swapped() {
        // Relocating a ciphertext to another page slot must not decrypt: this
        // is what stops an attacker shuffling pages inside the file.
        let cipher = PageCipher::new(&test_key()).unwrap();
        let ct = cipher.encrypt_page(10, 5, b"page ten contents").unwrap();
        assert!(
            cipher.decrypt_page(11, 5, &ct).is_err(),
            "page id must be bound"
        );
        assert!(cipher.decrypt_page(10, 5, &ct).is_ok());
    }

    #[test]
    fn version_is_bound_so_old_pages_cannot_be_replayed() {
        // Rollback attack: substituting an older version of the same page.
        let cipher = PageCipher::new(&test_key()).unwrap();
        let old = cipher.encrypt_page(3, 1, b"balance=100").unwrap();
        let new = cipher.encrypt_page(3, 2, b"balance=000").unwrap();
        assert_ne!(old, new);
        // The reader expects version 2; replaying version 1 must fail.
        assert!(cipher.decrypt_page(3, 2, &old).is_err());
        assert_eq!(cipher.decrypt_page(3, 1, &old).unwrap(), b"balance=100");
    }

    #[test]
    fn same_plaintext_different_versions_gives_different_ciphertext() {
        let cipher = PageCipher::new(&test_key()).unwrap();
        let a = cipher.encrypt_page(1, 1, b"identical").unwrap();
        let b = cipher.encrypt_page(1, 2, b"identical").unwrap();
        assert_ne!(a, b, "nonce must vary with version or GCM is broken");
    }

    #[test]
    fn nonce_is_unique_per_page_and_version() {
        assert_ne!(page_nonce(1, 1), page_nonce(2, 1));
        assert_ne!(page_nonce(1, 1), page_nonce(1, 2));
        assert_eq!(page_nonce(1, 1), page_nonce(1, 1));
        assert_eq!(page_nonce(0, 0).len(), NONCE_LEN);
    }

    #[test]
    fn nonce_tracker_rejects_reuse_and_rollback() {
        let mut t = NonceTracker::new();
        t.record(1, 1).unwrap();
        t.record(1, 2).unwrap();
        assert!(t.record(1, 2).is_err(), "exact reuse must be rejected");
        assert!(t.record(1, 1).is_err(), "going backwards must be rejected");
        t.record(2, 1).unwrap(); // different page is fine
        assert_eq!(t.version_of(1), Some(2));
        assert_eq!(t.version_of(99), None);
    }

    #[test]
    fn truncated_ciphertext_is_rejected() {
        let cipher = PageCipher::new(&test_key()).unwrap();
        assert!(cipher.decrypt_page(1, 1, &[]).is_err());
        assert!(cipher.decrypt_page(1, 1, &[0u8; TAG_LEN - 1]).is_err());
    }

    #[test]
    fn empty_plaintext_roundtrips() {
        let cipher = PageCipher::new(&test_key()).unwrap();
        let ct = cipher.encrypt_page(1, 1, b"").unwrap();
        assert_eq!(ct.len(), TAG_LEN, "empty message is just the tag");
        assert_eq!(cipher.decrypt_page(1, 1, &ct).unwrap(), b"");
    }

    #[test]
    fn full_page_sized_payload_roundtrips() {
        let cipher = PageCipher::new(&test_key()).unwrap();
        let page = vec![0xABu8; crate::page::PAGE_SIZE];
        let ct = cipher.encrypt_page(42, 9, &page).unwrap();
        assert_eq!(ct.len(), crate::page::PAGE_SIZE + TAG_LEN);
        assert_eq!(cipher.decrypt_page(42, 9, &ct).unwrap(), page);
    }

    #[test]
    fn passphrase_derivation_is_deterministic_and_salt_sensitive() {
        let a = EncryptionKey::derive_insecure_from_passphrase(b"hunter2", b"salt-a");
        let b = EncryptionKey::derive_insecure_from_passphrase(b"hunter2", b"salt-a");
        let c = EncryptionKey::derive_insecure_from_passphrase(b"hunter2", b"salt-b");
        let d = EncryptionKey::derive_insecure_from_passphrase(b"hunter3", b"salt-a");
        assert_eq!(a.as_bytes(), b.as_bytes());
        assert_ne!(a.as_bytes(), c.as_bytes(), "salt must change the key");
        assert_ne!(a.as_bytes(), d.as_bytes(), "passphrase must change the key");
        assert_ne!(a.as_bytes(), &[0u8; KEY_LEN], "key must not be all zeroes");
    }

    #[test]
    fn debug_never_leaks_key_material() {
        let key = EncryptionKey::from_bytes([0xAB; KEY_LEN]);
        let rendered = format!("{key:?}");
        assert!(rendered.contains("redacted"));
        assert!(!rendered.contains("ab"), "key bytes must not appear");
        assert!(!rendered.contains("171"));
    }
}
