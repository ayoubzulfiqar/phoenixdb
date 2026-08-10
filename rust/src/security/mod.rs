//! Security primitives for the FFI boundary.
//!
//! Three concerns live here:
//!
//! 1. **Constant-time comparison** — used for handle magic tags and any
//!    sensitive internal identifier, so that a caller cannot learn the value of
//!    a tag by timing repeated probes.
//! 2. **Pointer / length validation** — every byte slice reconstructed from a
//!    raw C pointer goes through [`slice_from_parts`], which refuses null
//!    pointers, over-long lengths and (where possible) misaligned inputs
//!    *before* any dereference occurs.
//! 3. **Handle tagging** — [`HandleTag`] embeds a magic value in every
//!    heap-allocated object handed to C so use-after-free and wild pointers are
//!    caught with high probability instead of being dereferenced.

use crate::error::{Error, Result};

pub mod audit;
pub mod rbac;

#[cfg(feature = "encryption")]
pub mod encryption;

pub use audit::{AuditLog, AuditRecord, Outcome};
pub use rbac::{AccessControl, Permission, Principal, Role};

#[cfg(feature = "encryption")]
pub use encryption::{EncryptionKey, PageCipher};

/// Maximum key length accepted at the FFI boundary (1 MiB).
///
/// Note: the *storage engine* imposes a much smaller structural limit
/// ([`crate::page::MAX_KEY_SIZE`]) because keys must be comparable in-page.
/// Keys between the two limits are rejected with `InvalidArgument`, never by
/// dereferencing an oversized buffer.
pub const MAX_KEY_LEN: usize = 1024 * 1024;

/// Maximum value length accepted at the FFI boundary (10 MiB).
pub const MAX_VALUE_LEN: usize = 10 * 1024 * 1024;

/// Compares two byte slices in time independent of their *contents*.
///
/// Only the length is allowed to short-circuit: length is not secret in this
/// codebase (it is carried in the clear in every page header), while the bytes
/// themselves may be. The accumulator folds every byte difference with a
/// bitwise XOR so that no early exit is possible for equal-length inputs.
#[inline]
#[must_use]
pub fn ct_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut diff: u8 = 0;
    for (x, y) in a.iter().zip(b.iter()) {
        // `u8::eq` semantics expressed as a branch-free XOR mask.
        diff |= x ^ y;
    }
    ct_is_zero_u8(diff)
}

/// Branch-free `value == 0` for a byte.
#[inline]
#[must_use]
pub fn ct_is_zero_u8(value: u8) -> bool {
    // Spread any set bit into the high bit, then invert.
    let v = value as u32;
    let folded = (v | v.wrapping_neg()) >> 31; // 0 when v == 0, else 1
    (folded ^ 1) == 1
}

/// Constant-time equality for 64-bit identifiers (transaction ids, magic tags).
#[inline]
#[must_use]
pub fn ct_eq_u64(a: u64, b: u64) -> bool {
    let diff = a ^ b;
    let folded = (diff | diff.wrapping_neg()) >> 63; // 0 when equal
    (folded ^ 1) == 1
}

/// Overwrites a buffer with zeroes using volatile writes so the compiler cannot
/// elide the clear for a soon-to-be-dropped allocation.
#[inline]
pub fn secure_zero(buf: &mut [u8]) {
    for byte in buf.iter_mut() {
        // SAFETY: `byte` is a valid, uniquely borrowed, properly aligned `u8`.
        unsafe { std::ptr::write_volatile(byte, 0) };
    }
    std::sync::atomic::compiler_fence(std::sync::atomic::Ordering::SeqCst);
}

/// Validates a key length against the FFI limit.
pub fn validate_key_len(len: usize) -> Result<()> {
    if len == 0 {
        return Err(Error::invalid("key must not be empty"));
    }
    if len > MAX_KEY_LEN {
        return Err(Error::invalid(format!(
            "key length {len} exceeds FFI limit of {MAX_KEY_LEN} bytes"
        )));
    }
    Ok(())
}

/// Validates a value length against the FFI limit.
pub fn validate_value_len(len: usize) -> Result<()> {
    if len > MAX_VALUE_LEN {
        return Err(Error::invalid(format!(
            "value length {len} exceeds FFI limit of {MAX_VALUE_LEN} bytes"
        )));
    }
    Ok(())
}

/// Rebuilds a `&[u8]` from a C pointer/length pair after full validation.
///
/// Returns `InvalidArgument` (status `-2`) for a null pointer, an over-long
/// length, or a pointer/length pair that would wrap the address space. The
/// pointer is **never** dereferenced before all checks pass.
///
/// # Safety
/// The caller guarantees that, when `ptr` is non-null, `ptr..ptr + len` is a
/// single allocated object that stays immutable and live for `'a`.
pub unsafe fn slice_from_parts<'a>(ptr: *const u8, len: usize, max: usize) -> Result<&'a [u8]> {
    if len > max {
        return Err(Error::invalid(format!(
            "buffer length {len} exceeds limit of {max} bytes"
        )));
    }
    if len == 0 {
        // A zero-length slice never dereferences the pointer; normalise it so a
        // null pointer with length 0 is still a valid empty buffer.
        return Ok(&[]);
    }
    if ptr.is_null() {
        return Err(Error::invalid("null pointer with non-zero length"));
    }
    if (ptr as usize).checked_add(len).is_none() {
        return Err(Error::invalid(
            "pointer + length overflows the address space",
        ));
    }
    // SAFETY: non-null, non-wrapping, length-checked; validity is the caller's
    // documented obligation.
    Ok(unsafe { std::slice::from_raw_parts(ptr, len) })
}

/// A 64-bit magic tag embedded in every heap object exposed to C.
///
/// The tag is verified in constant time on every entry point. A mismatch means
/// the pointer is stale (use-after-free), foreign, or corrupted, and the call
/// is rejected with `InvalidArgument` instead of dereferencing further.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(transparent)]
pub struct HandleTag(u64);

impl HandleTag {
    /// "PHNXDB\0\x01" — chosen to be unlikely in freed or uninitialised memory.
    pub const MAGIC: u64 = 0x5048_4E58_4442_0001;

    /// Creates a live tag.
    #[inline]
    #[must_use]
    pub const fn new() -> Self {
        HandleTag(Self::MAGIC)
    }

    /// Verifies the tag in constant time.
    #[inline]
    #[must_use]
    pub fn is_valid(&self) -> bool {
        ct_eq_u64(self.0, Self::MAGIC)
    }

    /// Poisons the tag on close so a later use-after-free is detected.
    #[inline]
    pub fn poison(&mut self) {
        // SAFETY: `self.0` is a valid, uniquely borrowed `u64`.
        unsafe { std::ptr::write_volatile(&mut self.0, 0xDEAD_BEEF_DEAD_BEEF) };
    }
}

impl Default for HandleTag {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ct_eq_matches_semantics() {
        assert!(ct_eq(b"", b""));
        assert!(ct_eq(b"phoenix", b"phoenix"));
        assert!(!ct_eq(b"phoenix", b"phoeniy"));
        assert!(!ct_eq(b"phoenix", b"phoenixx"));
        assert!(!ct_eq(b"\x00", b"\x80"));
    }

    #[test]
    fn ct_is_zero_is_exact() {
        assert!(ct_is_zero_u8(0));
        for v in 1..=255u8 {
            assert!(!ct_is_zero_u8(v), "u8 {v} reported as zero");
        }
    }

    #[test]
    fn ct_eq_u64_is_exact() {
        assert!(ct_eq_u64(0, 0));
        assert!(ct_eq_u64(u64::MAX, u64::MAX));
        assert!(!ct_eq_u64(0, 1));
        assert!(!ct_eq_u64(1 << 63, 0));
    }

    #[test]
    fn handle_tag_roundtrip() {
        let mut tag = HandleTag::new();
        assert!(tag.is_valid());
        tag.poison();
        assert!(!tag.is_valid());
    }

    #[test]
    fn null_pointer_is_rejected_before_deref() {
        let r = unsafe { slice_from_parts(std::ptr::null(), 16, MAX_KEY_LEN) };
        assert!(matches!(r, Err(Error::InvalidArgument(_))));
    }

    #[test]
    fn oversized_length_is_rejected() {
        let data = [0u8; 4];
        let r = unsafe { slice_from_parts(data.as_ptr(), MAX_KEY_LEN + 1, MAX_KEY_LEN) };
        assert!(matches!(r, Err(Error::InvalidArgument(_))));
    }

    #[test]
    fn secure_zero_clears() {
        let mut buf = [1u8, 2, 3, 4];
        secure_zero(&mut buf);
        assert_eq!(buf, [0, 0, 0, 0]);
    }
}
