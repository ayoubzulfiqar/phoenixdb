//! Distance kernels for vector similarity search.
//!
//! # Why three code paths
//!
//! The inner loop of a k-NN search is a dot product or a squared-difference
//! sum over `dim` floats, executed thousands of times per query. Getting it
//! wrong costs an order of magnitude, so this module provides:
//!
//! 1. **A portable, auto-vectorising kernel.** Eight independent accumulators
//!    over `chunks_exact(8)` give LLVM a shape it reliably lowers to `vaddps`
//!    / `vfmadd` on x86 and `fmla.4s` on AArch64. On aarch64 this *is* the
//!    NEON path — NEON is baseline in every `aarch64-*` target spec, so no
//!    feature flag or runtime probe is needed.
//! 2. **An explicit NEON kernel** on aarch64, compiled unconditionally because
//!    `target_feature = "neon"` is always on for that architecture.
//! 3. **An explicit AVX2 + FMA kernel** on x86_64, selected by *runtime*
//!    detection.
//!
//! # Why AVX2 is dispatched at runtime rather than via `RUSTFLAGS`
//!
//! PhoenixDB ships prebuilt binaries inside the published package. Compiling
//! the whole crate with `-C target-feature=+avx2` would make it `SIGILL` on
//! every pre-Haswell x86_64 CPU — a crash on the user's machine, not a build
//! error we would ever see. `#[target_feature]` on a single leaf function plus
//! `is_x86_feature_detected!` gives the same throughput on capable hardware
//! and a correct fallback everywhere else. ARM64 keeps `+neon` in `RUSTFLAGS`
//! because it is unconditionally part of the AArch64 baseline.
//!
//! Every kernel here is numerically identical up to floating-point
//! associativity: the accumulator layouts differ, so results may differ in the
//! last ULP or two. Tests assert against a scalar reference with an epsilon.

use crate::error::{Error, Result};
use serde::{Deserialize, Serialize};

/// Largest dimensionality accepted by the engine.
///
/// Bounded so a corrupt or hostile `dim` cannot drive an enormous allocation
/// before any data is read. 65 536 is comfortably above every production
/// embedding model in use (OpenAI 3-large is 3 072, Cohere v3 is 1 024).
pub const MAX_DIM: usize = 65_536;

/// Similarity metric used to order neighbours.
///
/// The discriminants are part of the C ABI: `phoenix_vector_init` takes this
/// as a `uint8_t` and they must never be renumbered.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Metric {
    /// Angular distance, `1 - cos(a, b)`, in `[0, 2]`.
    ///
    /// Scale-invariant, and the right default for text embeddings, which are
    /// trained with a cosine objective.
    Cosine = 0,
    /// Straight-line (L2) distance, in `[0, inf)`.
    ///
    /// The graph orders candidates by the *squared* distance — monotonic in
    /// L2, and one square root cheaper per comparison — and takes the square
    /// root once per returned result in [`Metric::finalize`].
    Euclidean = 1,
    /// Negated inner product, `-(a . b)`.
    ///
    /// Negated so that, like the other two metrics, smaller is more similar.
    /// Unlike them it is unbounded and not a true metric; use it only with
    /// embeddings whose magnitude is meaningful (e.g. MIPS recommenders).
    DotProduct = 2,
}

impl Metric {
    /// Decodes the wire representation used by the C ABI.
    pub fn from_u8(value: u8) -> Result<Self> {
        match value {
            0 => Ok(Metric::Cosine),
            1 => Ok(Metric::Euclidean),
            2 => Ok(Metric::DotProduct),
            other => Err(Error::invalid(format!(
                "unknown metric {other}: expected 0 (cosine), 1 (euclidean) or 2 (dot product)"
            ))),
        }
    }

    /// Encodes this metric for the C ABI.
    #[inline]
    #[must_use]
    pub fn as_u8(self) -> u8 {
        self as u8
    }

    /// Lower-case name, used in diagnostics and the Dart layer.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Metric::Cosine => "cosine",
            Metric::Euclidean => "euclidean",
            Metric::DotProduct => "dot_product",
        }
    }

    /// Whether this metric needs the cached L2 norm of each stored vector.
    ///
    /// Only cosine does. Caching the norm turns a cosine comparison into one
    /// dot product plus one multiply, instead of three dot products.
    #[inline]
    #[must_use]
    pub fn uses_norm(self) -> bool {
        matches!(self, Metric::Cosine)
    }

    /// Converts an ordering distance into the value reported to callers.
    ///
    /// The graph works exclusively in ordering distances (squared L2 for
    /// [`Metric::Euclidean`]); this is applied exactly once, to the `k`
    /// results that actually leave the engine.
    #[inline]
    #[must_use]
    pub fn finalize(self, ordering_distance: f32) -> f32 {
        match self {
            Metric::Euclidean => ordering_distance.max(0.0).sqrt(),
            Metric::Cosine | Metric::DotProduct => ordering_distance,
        }
    }

    /// Converts a finalized distance into a "higher is better" score.
    ///
    /// * cosine — the cosine similarity itself, in `[-1, 1]`
    /// * euclidean — `1 / (1 + d)`, in `(0, 1]`
    /// * dot product — the raw inner product
    #[inline]
    #[must_use]
    pub fn score(self, distance: f32) -> f32 {
        match self {
            Metric::Cosine => 1.0 - distance,
            Metric::Euclidean => 1.0 / (1.0 + distance.max(0.0)),
            Metric::DotProduct => -distance,
        }
    }
}

/// The ordering distance between a query and a stored vector.
///
/// `query_norm` and `stored_norm` are the cached L2 norms; they are ignored
/// for metrics where [`Metric::uses_norm`] is false, so callers may pass any
/// value (conventionally `0.0`) there.
///
/// Smaller is always more similar, for every metric.
#[inline]
#[must_use]
pub fn ordering_distance(
    metric: Metric,
    query: &[f32],
    query_norm: f32,
    stored: &[f32],
    stored_norm: f32,
) -> f32 {
    debug_assert_eq!(query.len(), stored.len(), "dimension mismatch");
    match metric {
        Metric::Cosine => {
            let denominator = query_norm * stored_norm;
            if denominator <= f32::MIN_POSITIVE {
                // A zero vector has no direction. Reporting the maximum cosine
                // distance keeps the ordering total and avoids a NaN, which
                // would poison every comparison in the heap.
                return 1.0;
            }
            let similarity = dot(query, stored) / denominator;
            // Guard against similarity drifting outside [-1, 1] from rounding.
            1.0 - similarity.clamp(-1.0, 1.0)
        }
        Metric::Euclidean => squared_l2(query, stored),
        Metric::DotProduct => -dot(query, stored),
    }
}

/// The L2 norm of `v`, used to cache the denominator of the cosine metric.
#[inline]
#[must_use]
pub fn norm(v: &[f32]) -> f32 {
    dot(v, v).max(0.0).sqrt()
}

// ---------------------------------------------------------------------------
// Kernel dispatch
// ---------------------------------------------------------------------------

/// Which kernel family this CPU can execute.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Backend {
    /// Portable, auto-vectorised loops. Also the NEON path on AArch64.
    Portable,
    /// x86_64 with AVX2 and FMA.
    #[cfg(target_arch = "x86_64")]
    Avx2Fma,
}

/// Resolved once per process; `is_x86_feature_detected!` is cheap but not free,
/// and the answer cannot change while the process runs.
static BACKEND: std::sync::OnceLock<Backend> = std::sync::OnceLock::new();

#[inline]
fn backend() -> Backend {
    *BACKEND.get_or_init(detect_backend)
}

fn detect_backend() -> Backend {
    #[cfg(target_arch = "x86_64")]
    {
        // FMA is checked as well as AVX2: the kernel uses `_mm256_fmadd_ps`,
        // and the two feature bits are independent even though every shipping
        // CPU with AVX2 also has FMA3.
        if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
        {
            return Backend::Avx2Fma;
        }
    }
    Backend::Portable
}

/// Name of the kernel selected for this CPU. Exposed for diagnostics.
#[must_use]
pub fn active_kernel() -> &'static str {
    match backend() {
        #[cfg(target_arch = "x86_64")]
        Backend::Avx2Fma => "avx2+fma",
        Backend::Portable => {
            if cfg!(target_arch = "aarch64") {
                "neon"
            } else {
                "portable"
            }
        }
    }
}

/// Inner product of two equal-length slices.
///
/// Falls back to the portable kernel when the lengths differ, which cannot
/// happen through the engine (dimensions are validated on the way in) but must
/// not be undefined behaviour if it ever did.
#[inline]
#[must_use]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len().min(b.len());
    let (a, b) = (&a[..len], &b[..len]);
    match backend() {
        #[cfg(target_arch = "x86_64")]
        // SAFETY: `Backend::Avx2Fma` is only produced by `detect_backend`
        // after `is_x86_feature_detected!` confirmed both features, and the
        // slices are equal-length by the truncation above.
        Backend::Avx2Fma => unsafe { dot_avx2(a, b) },
        Backend::Portable => dot_portable(a, b),
    }
}

/// Squared Euclidean distance between two equal-length slices.
#[inline]
#[must_use]
pub fn squared_l2(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len().min(b.len());
    let (a, b) = (&a[..len], &b[..len]);
    match backend() {
        #[cfg(target_arch = "x86_64")]
        // SAFETY: see `dot`.
        Backend::Avx2Fma => unsafe { squared_l2_avx2(a, b) },
        Backend::Portable => squared_l2_portable(a, b),
    }
}

// ---------------------------------------------------------------------------
// Portable kernels (auto-vectorised; the NEON path on AArch64)
// ---------------------------------------------------------------------------

/// Lanes per accumulator block.
///
/// Eight matches one AVX2 register and two NEON `float32x4_t` registers, and
/// gives the scheduler eight independent dependency chains to hide FMA
/// latency with.
const LANES: usize = 8;

#[inline]
fn dot_portable(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = [0f32; LANES];
    let mut a_chunks = a.chunks_exact(LANES);
    let mut b_chunks = b.chunks_exact(LANES);
    for (x, y) in a_chunks.by_ref().zip(b_chunks.by_ref()) {
        for ((slot, &xv), &yv) in acc.iter_mut().zip(x.iter()).zip(y.iter()) {
            *slot = xv.mul_add(yv, *slot);
        }
    }
    let mut total: f32 = acc.iter().sum();
    for (&x, &y) in a_chunks.remainder().iter().zip(b_chunks.remainder()) {
        total = x.mul_add(y, total);
    }
    total
}

#[inline]
fn squared_l2_portable(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = [0f32; LANES];
    let mut a_chunks = a.chunks_exact(LANES);
    let mut b_chunks = b.chunks_exact(LANES);
    for (x, y) in a_chunks.by_ref().zip(b_chunks.by_ref()) {
        for ((slot, &xv), &yv) in acc.iter_mut().zip(x.iter()).zip(y.iter()) {
            let d = xv - yv;
            *slot = d.mul_add(d, *slot);
        }
    }
    let mut total: f32 = acc.iter().sum();
    for (&x, &y) in a_chunks.remainder().iter().zip(b_chunks.remainder()) {
        let d = x - y;
        total = d.mul_add(d, total);
    }
    total
}

// ---------------------------------------------------------------------------
// x86_64: AVX2 + FMA
// ---------------------------------------------------------------------------

#[cfg(target_arch = "x86_64")]
mod avx2 {
    use std::arch::x86_64::{
        __m256, _mm_add_ps, _mm_add_ss, _mm_cvtss_f32, _mm_movehl_ps, _mm_shuffle_ps,
        _mm256_add_ps, _mm256_castps256_ps128, _mm256_extractf128_ps, _mm256_fmadd_ps,
        _mm256_loadu_ps, _mm256_setzero_ps, _mm256_sub_ps,
    };

    /// Horizontal sum of eight lanes.
    ///
    /// # Safety
    /// Requires AVX (for the 256-bit register) on the executing CPU, which
    /// `#[target_feature]` makes the caller's obligation.
    #[inline]
    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn horizontal_sum(v: __m256) -> f32 {
        // Every intrinsic here is a register-only operation with no memory
        // operand, so under `#[target_feature(enable = "avx2")]` they are all
        // safe to call directly.
        let high = _mm256_extractf128_ps(v, 1);
        let low = _mm256_castps256_ps128(v);
        let sum = _mm_add_ps(high, low);
        let shuffled = _mm_movehl_ps(sum, sum);
        let sum = _mm_add_ps(sum, shuffled);
        let shuffled = _mm_shuffle_ps(sum, sum, 0x55);
        _mm_cvtss_f32(_mm_add_ss(sum, shuffled))
    }

    /// Four-way-unrolled AVX2 dot product.
    ///
    /// # Safety
    /// `a` and `b` must have the same length, and the CPU must support AVX2
    /// and FMA.
    #[target_feature(enable = "avx2,fma")]
    pub(super) unsafe fn dot(a: &[f32], b: &[f32]) -> f32 {
        // SAFETY: all loads below are bounds-derived from `len` and use the
        // unaligned form, so no alignment precondition applies.
        unsafe {
            let len = a.len();
            let (mut pa, mut pb) = (a.as_ptr(), b.as_ptr());
            let mut acc0 = _mm256_setzero_ps();
            let mut acc1 = _mm256_setzero_ps();
            let mut acc2 = _mm256_setzero_ps();
            let mut acc3 = _mm256_setzero_ps();

            let mut i = 0usize;
            while i + 32 <= len {
                acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(pa), _mm256_loadu_ps(pb), acc0);
                acc1 =
                    _mm256_fmadd_ps(_mm256_loadu_ps(pa.add(8)), _mm256_loadu_ps(pb.add(8)), acc1);
                acc2 = _mm256_fmadd_ps(
                    _mm256_loadu_ps(pa.add(16)),
                    _mm256_loadu_ps(pb.add(16)),
                    acc2,
                );
                acc3 = _mm256_fmadd_ps(
                    _mm256_loadu_ps(pa.add(24)),
                    _mm256_loadu_ps(pb.add(24)),
                    acc3,
                );
                pa = pa.add(32);
                pb = pb.add(32);
                i += 32;
            }
            while i + 8 <= len {
                acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(pa), _mm256_loadu_ps(pb), acc0);
                pa = pa.add(8);
                pb = pb.add(8);
                i += 8;
            }

            let packed = _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3));
            let mut total = horizontal_sum(packed);
            while i < len {
                total = (*pa).mul_add(*pb, total);
                pa = pa.add(1);
                pb = pb.add(1);
                i += 1;
            }
            total
        }
    }

    /// Four-way-unrolled AVX2 squared L2 distance.
    ///
    /// # Safety
    /// `a` and `b` must have the same length, and the CPU must support AVX2
    /// and FMA.
    #[target_feature(enable = "avx2,fma")]
    pub(super) unsafe fn squared_l2(a: &[f32], b: &[f32]) -> f32 {
        // SAFETY: as in `dot` — unaligned loads bounded by `len`.
        unsafe {
            let len = a.len();
            let (mut pa, mut pb) = (a.as_ptr(), b.as_ptr());
            let mut acc0 = _mm256_setzero_ps();
            let mut acc1 = _mm256_setzero_ps();
            let mut acc2 = _mm256_setzero_ps();
            let mut acc3 = _mm256_setzero_ps();

            let mut i = 0usize;
            while i + 32 <= len {
                let d0 = _mm256_sub_ps(_mm256_loadu_ps(pa), _mm256_loadu_ps(pb));
                let d1 = _mm256_sub_ps(_mm256_loadu_ps(pa.add(8)), _mm256_loadu_ps(pb.add(8)));
                let d2 = _mm256_sub_ps(_mm256_loadu_ps(pa.add(16)), _mm256_loadu_ps(pb.add(16)));
                let d3 = _mm256_sub_ps(_mm256_loadu_ps(pa.add(24)), _mm256_loadu_ps(pb.add(24)));
                acc0 = _mm256_fmadd_ps(d0, d0, acc0);
                acc1 = _mm256_fmadd_ps(d1, d1, acc1);
                acc2 = _mm256_fmadd_ps(d2, d2, acc2);
                acc3 = _mm256_fmadd_ps(d3, d3, acc3);
                pa = pa.add(32);
                pb = pb.add(32);
                i += 32;
            }
            while i + 8 <= len {
                let d = _mm256_sub_ps(_mm256_loadu_ps(pa), _mm256_loadu_ps(pb));
                acc0 = _mm256_fmadd_ps(d, d, acc0);
                pa = pa.add(8);
                pb = pb.add(8);
                i += 8;
            }

            let packed = _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3));
            let mut total = horizontal_sum(packed);
            while i < len {
                let d = *pa - *pb;
                total = d.mul_add(d, total);
                pa = pa.add(1);
                pb = pb.add(1);
                i += 1;
            }
            total
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_avx2(a: &[f32], b: &[f32]) -> f32 {
    // SAFETY: `#[target_feature]` on this function makes AVX2 and FMA the
    // caller's obligation, which the dispatcher discharges with
    // `is_x86_feature_detected!`; the slices are equal-length by construction.
    unsafe { avx2::dot(a, b) }
}

#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx2,fma")]
unsafe fn squared_l2_avx2(a: &[f32], b: &[f32]) -> f32 {
    // SAFETY: see `dot_avx2`.
    unsafe { avx2::squared_l2(a, b) }
}

// ---------------------------------------------------------------------------
// AArch64: NEON
// ---------------------------------------------------------------------------
//
// NEON is mandatory in the AArch64 base architecture, so these are compiled
// unconditionally on that target and need no runtime probe. They are used
// through `dot_portable` / `squared_l2_portable`, which the compiler lowers to
// the same instruction sequence; the explicit versions exist so the intent is
// checked by the type system rather than assumed from codegen.

#[cfg(target_arch = "aarch64")]
mod neon {
    use std::arch::aarch64::{
        float32x4_t, vaddq_f32, vaddvq_f32, vdupq_n_f32, vfmaq_f32, vsubq_f32,
    };

    /// NEON dot product over 16-float blocks.
    ///
    /// # Safety
    /// `a` and `b` must have the same length. NEON itself is baseline on
    /// AArch64, so no feature check is required.
    pub(super) unsafe fn dot(a: &[f32], b: &[f32]) -> f32 {
        // SAFETY: pointer arithmetic stays within `len` on both slices, and
        // `read_unaligned` imposes no alignment requirement.
        unsafe {
            let len = a.len();
            let (mut pa, mut pb) = (a.as_ptr(), b.as_ptr());
            let mut acc0 = vdupq_n_f32(0.0);
            let mut acc1 = vdupq_n_f32(0.0);
            let mut acc2 = vdupq_n_f32(0.0);
            let mut acc3 = vdupq_n_f32(0.0);

            let mut i = 0usize;
            while i + 16 <= len {
                acc0 = vfmaq_f32(acc0, load(pa), load(pb));
                acc1 = vfmaq_f32(acc1, load(pa.add(4)), load(pb.add(4)));
                acc2 = vfmaq_f32(acc2, load(pa.add(8)), load(pb.add(8)));
                acc3 = vfmaq_f32(acc3, load(pa.add(12)), load(pb.add(12)));
                pa = pa.add(16);
                pb = pb.add(16);
                i += 16;
            }
            while i + 4 <= len {
                acc0 = vfmaq_f32(acc0, load(pa), load(pb));
                pa = pa.add(4);
                pb = pb.add(4);
                i += 4;
            }
            let packed = vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3));
            let mut total = vaddvq_f32(packed);
            while i < len {
                total = (*pa).mul_add(*pb, total);
                pa = pa.add(1);
                pb = pb.add(1);
                i += 1;
            }
            total
        }
    }

    /// NEON squared L2 distance over 16-float blocks.
    ///
    /// # Safety
    /// `a` and `b` must have the same length.
    pub(super) unsafe fn squared_l2(a: &[f32], b: &[f32]) -> f32 {
        // SAFETY: as in `dot`.
        unsafe {
            let len = a.len();
            let (mut pa, mut pb) = (a.as_ptr(), b.as_ptr());
            let mut acc0 = vdupq_n_f32(0.0);
            let mut acc1 = vdupq_n_f32(0.0);

            let mut i = 0usize;
            while i + 8 <= len {
                let d0 = vsubq_f32(load(pa), load(pb));
                let d1 = vsubq_f32(load(pa.add(4)), load(pb.add(4)));
                acc0 = vfmaq_f32(acc0, d0, d0);
                acc1 = vfmaq_f32(acc1, d1, d1);
                pa = pa.add(8);
                pb = pb.add(8);
                i += 8;
            }
            while i + 4 <= len {
                let d = vsubq_f32(load(pa), load(pb));
                acc0 = vfmaq_f32(acc0, d, d);
                pa = pa.add(4);
                pb = pb.add(4);
                i += 4;
            }
            let mut total = vaddvq_f32(vaddq_f32(acc0, acc1));
            while i < len {
                let d = *pa - *pb;
                total = d.mul_add(d, total);
                pa = pa.add(1);
                pb = pb.add(1);
                i += 1;
            }
            total
        }
    }

    /// Unaligned 4-lane load.
    ///
    /// # Safety
    /// `p` must be readable for four `f32`s.
    #[inline]
    unsafe fn load(p: *const f32) -> float32x4_t {
        // SAFETY: the caller guarantees four readable floats; the transmute
        // from `[f32; 4]` to `float32x4_t` is the documented representation.
        unsafe {
            std::mem::transmute::<[f32; 4], float32x4_t>(p.cast::<[f32; 4]>().read_unaligned())
        }
    }
}

/// NEON dot product. Present on AArch64 only; see the module docs.
#[cfg(target_arch = "aarch64")]
#[inline]
#[must_use]
pub fn dot_neon(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len().min(b.len());
    // SAFETY: slices truncated to a common length; NEON is baseline here.
    unsafe { neon::dot(&a[..len], &b[..len]) }
}

/// NEON squared L2 distance. Present on AArch64 only; see the module docs.
#[cfg(target_arch = "aarch64")]
#[inline]
#[must_use]
pub fn squared_l2_neon(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len().min(b.len());
    // SAFETY: slices truncated to a common length; NEON is baseline here.
    unsafe { neon::squared_l2(&a[..len], &b[..len]) }
}

/// Validates a dimensionality before it is used to size an allocation.
pub fn validate_dim(dim: usize) -> Result<()> {
    if dim == 0 {
        return Err(Error::invalid("vector dimension must be greater than zero"));
    }
    if dim > MAX_DIM {
        return Err(Error::invalid(format!(
            "vector dimension {dim} exceeds the limit of {MAX_DIM}"
        )));
    }
    Ok(())
}

/// Rejects a vector whose length does not match the index, or that contains a
/// non-finite component.
///
/// NaN is refused rather than tolerated: a single NaN makes every comparison
/// against it false, which silently corrupts the ordering of the graph's
/// priority queues and produces wrong results for *other* queries.
pub fn validate_vector(vector: &[f32], dim: usize) -> Result<()> {
    if vector.len() != dim {
        return Err(Error::invalid(format!(
            "vector has {} dimension(s), index expects {dim}",
            vector.len()
        )));
    }
    if let Some(position) = vector.iter().position(|v| !v.is_finite()) {
        return Err(Error::invalid(format!(
            "vector component {position} is not finite ({})",
            vector[position]
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Straight-line reference implementation the kernels are checked against.
    fn reference_dot(a: &[f32], b: &[f32]) -> f64 {
        a.iter()
            .zip(b)
            .map(|(x, y)| f64::from(*x) * f64::from(*y))
            .sum()
    }

    fn reference_l2(a: &[f32], b: &[f32]) -> f64 {
        a.iter()
            .zip(b)
            .map(|(x, y)| {
                let d = f64::from(*x) - f64::from(*y);
                d * d
            })
            .sum()
    }

    /// Deterministic pseudo-random vectors, so a failure is reproducible.
    fn sample(dim: usize, seed: u64) -> Vec<f32> {
        let mut state = seed | 1;
        (0..dim)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                ((state >> 40) as f32 / 8_388_608.0) - 1.0
            })
            .collect()
    }

    #[test]
    fn kernels_match_the_reference_at_every_length() {
        // Lengths chosen to exercise every tail: below one block, exactly one
        // block, unrolled multiples, and awkward remainders.
        for dim in [
            1usize, 2, 3, 7, 8, 9, 15, 16, 31, 32, 33, 63, 64, 127, 384, 768, 1536,
        ] {
            let a = sample(dim, 0x9E37_79B9_7F4A_7C15);
            let b = sample(dim, 0xBF58_476D_1CE4_E5B9);

            let expected_dot = reference_dot(&a, &b);
            let got_dot = f64::from(dot(&a, &b));
            let tolerance = 1e-4 * (dim as f64).sqrt().max(1.0);
            assert!(
                (got_dot - expected_dot).abs() < tolerance,
                "dot mismatch at dim {dim}: {got_dot} vs {expected_dot}"
            );

            let expected_l2 = reference_l2(&a, &b);
            let got_l2 = f64::from(squared_l2(&a, &b));
            assert!(
                (got_l2 - expected_l2).abs() < tolerance,
                "l2 mismatch at dim {dim}: {got_l2} vs {expected_l2}"
            );
        }
    }

    #[test]
    fn portable_and_dispatched_kernels_agree() {
        // On a machine with AVX2 this compares two genuinely different code
        // paths; without it, it is a cheap tautology that still guards the
        // slicing logic.
        for dim in [8usize, 100, 512, 1000] {
            let a = sample(dim, 11);
            let b = sample(dim, 22);
            assert!((dot(&a, &b) - dot_portable(&a, &b)).abs() < 1e-3);
            assert!((squared_l2(&a, &b) - squared_l2_portable(&a, &b)).abs() < 1e-3);
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn neon_kernels_agree_with_portable() {
        for dim in [4usize, 5, 16, 17, 128, 769] {
            let a = sample(dim, 33);
            let b = sample(dim, 44);
            assert!((dot_neon(&a, &b) - dot_portable(&a, &b)).abs() < 1e-3);
            assert!((squared_l2_neon(&a, &b) - squared_l2_portable(&a, &b)).abs() < 1e-3);
        }
    }

    #[test]
    fn cosine_distance_is_exact_for_known_angles() {
        let a = vec![1.0f32, 0.0, 0.0];
        let identical = vec![1.0f32, 0.0, 0.0];
        let orthogonal = vec![0.0f32, 1.0, 0.0];
        let opposite = vec![-1.0f32, 0.0, 0.0];
        let scaled = vec![7.5f32, 0.0, 0.0];

        let na = norm(&a);
        let d = |v: &Vec<f32>| ordering_distance(Metric::Cosine, &a, na, v, norm(v));

        assert!(d(&identical).abs() < 1e-6, "identical vectors: distance 0");
        assert!(
            (d(&orthogonal) - 1.0).abs() < 1e-6,
            "orthogonal: distance 1"
        );
        assert!((d(&opposite) - 2.0).abs() < 1e-6, "opposite: distance 2");
        assert!(
            d(&scaled).abs() < 1e-6,
            "cosine must ignore magnitude entirely"
        );
    }

    #[test]
    fn cosine_at_forty_five_degrees() {
        let a = vec![1.0f32, 0.0];
        let b = vec![1.0f32, 1.0];
        let distance = ordering_distance(Metric::Cosine, &a, norm(&a), &b, norm(&b));
        let expected = 1.0 - std::f32::consts::FRAC_1_SQRT_2;
        assert!(
            (distance - expected).abs() < 1e-6,
            "45 degrees should be {expected}, got {distance}"
        );
    }

    #[test]
    fn zero_vector_never_produces_nan() {
        let a = vec![0.0f32; 8];
        let b = vec![1.0f32; 8];
        let distance = ordering_distance(Metric::Cosine, &a, norm(&a), &b, norm(&b));
        assert!(distance.is_finite(), "got {distance}");
        assert!((distance - 1.0).abs() < 1e-6);
    }

    #[test]
    fn euclidean_and_dot_orderings() {
        let a = vec![3.0f32, 4.0];
        let b = vec![0.0f32, 0.0];
        let squared = ordering_distance(Metric::Euclidean, &a, 0.0, &b, 0.0);
        assert!((squared - 25.0).abs() < 1e-5);
        assert!((Metric::Euclidean.finalize(squared) - 5.0).abs() < 1e-5);

        let c = vec![1.0f32, 2.0];
        let d = vec![3.0f32, 4.0];
        // 1*3 + 2*4 = 11, negated so that "smaller is nearer" holds.
        assert!((ordering_distance(Metric::DotProduct, &c, 0.0, &d, 0.0) + 11.0).abs() < 1e-5);
    }

    #[test]
    fn metric_codes_round_trip() {
        for metric in [Metric::Cosine, Metric::Euclidean, Metric::DotProduct] {
            assert_eq!(Metric::from_u8(metric.as_u8()).unwrap(), metric);
        }
        assert!(Metric::from_u8(3).is_err());
        assert!(Metric::from_u8(255).is_err());
    }

    #[test]
    fn dimension_validation_rejects_zero_and_overflow() {
        assert!(validate_dim(0).is_err());
        assert!(validate_dim(1).is_ok());
        assert!(validate_dim(MAX_DIM).is_ok());
        assert!(validate_dim(MAX_DIM + 1).is_err());
    }

    #[test]
    fn vector_validation_rejects_wrong_length_and_non_finite() {
        assert!(validate_vector(&[1.0, 2.0, 3.0], 3).is_ok());
        assert!(validate_vector(&[1.0, 2.0], 3).is_err());
        assert!(validate_vector(&[1.0, 2.0, 3.0, 4.0], 3).is_err());
        assert!(validate_vector(&[1.0, f32::NAN, 3.0], 3).is_err());
        assert!(validate_vector(&[1.0, f32::INFINITY, 3.0], 3).is_err());
    }

    #[test]
    fn scores_are_monotonic_in_distance() {
        for metric in [Metric::Cosine, Metric::Euclidean, Metric::DotProduct] {
            let near = metric.score(0.1);
            let far = metric.score(0.9);
            assert!(
                near > far,
                "{}: score must decrease as distance grows",
                metric.name()
            );
        }
    }
}
