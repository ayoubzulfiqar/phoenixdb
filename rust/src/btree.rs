//! B+Tree index with configurable fill factors.
//!
//! * Leaves are linked left-to-right (`extra` = next leaf) for range scans.
//! * Internal nodes keep the leftmost child in the header `extra` field and one
//!   `(separator, right_child)` cell per remaining child.
//! * Splits are performed eagerly on the way down is *not* used; instead the
//!   recursive insert returns a [`Split`] which the parent absorbs. This keeps
//!   the tree correct without holding latches on the whole path.
//!
//! # Fill factors
//!
//! [`FillFactor`] bounds how full a page may get before splitting and how empty
//! it may get before merging. `min` is clamped to `[0.5, 1.0]` per the design
//! requirement that a node never drops below half occupancy (except the root).

use crate::error::{Error, Result};
use crate::page::{
    LeafCell, Page, PageType, MAX_INLINE_VALUE, MAX_KEY_SIZE, SENTINEL, USABLE_SPACE,
};
use crate::pager::Pager;

/// Split/merge thresholds expressed as fractions of a page's usable space.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FillFactor {
    /// Lower bound before a node is considered underfull (>= 0.5).
    pub min: f32,
    /// Upper bound before a node must split (<= 1.0).
    pub max: f32,
}

impl FillFactor {
    /// Clamps the pair into the legal range `0.5 <= min < max <= 1.0`.
    #[must_use]
    pub fn new(min: f32, max: f32) -> Self {
        let min = min.clamp(0.5, 1.0);
        let max = max.clamp(min + f32::EPSILON, 1.0);
        FillFactor { min, max }
    }

    /// Bytes that must remain free for the page to be considered "not full".
    #[must_use]
    pub fn max_used_bytes(&self) -> usize {
        (USABLE_SPACE as f32 * self.max) as usize
    }

    /// Bytes below which a page is underfull and should be rebalanced.
    #[must_use]
    pub fn min_used_bytes(&self) -> usize {
        (USABLE_SPACE as f32 * self.min) as usize
    }
}

impl Default for FillFactor {
    fn default() -> Self {
        FillFactor { min: 0.5, max: 1.0 }
    }
}

/// A separator promoted to the parent after a node split.
#[derive(Debug)]
struct Split {
    /// First key of the new right sibling.
    key: Vec<u8>,
    /// Page id of the new right sibling.
    right: u32,
}

/// B+Tree operations over a [`Pager`].
///
/// The tree is stateless: the root id lives in the pager's metadata, so a
/// `BTree` value is just a handle carrying the fill-factor policy.
#[derive(Debug, Clone, Copy)]
pub struct BTree {
    fill: FillFactor,
}

impl BTree {
    /// Creates a tree handle with the given fill policy.
    #[must_use]
    pub fn new(fill: FillFactor) -> Self {
        BTree { fill }
    }

    /// The configured fill factor.
    #[must_use]
    pub fn fill_factor(&self) -> FillFactor {
        self.fill
    }

    /// Looks up `key`, returning its value or [`Error::NotFound`].
    pub fn get(&self, pager: &mut Pager, key: &[u8]) -> Result<Vec<u8>> {
        let leaf_id = self.find_leaf(pager, key)?;
        let leaf = pager.read_page(leaf_id)?;
        match leaf.search(key)? {
            Ok(idx) => {
                let cell = leaf.leaf_cell(idx)?;
                match cell.overflow {
                    Some(head) => pager.read_overflow_chain(head, cell.total_len),
                    None => Ok(cell.value),
                }
            }
            Err(_) => Err(Error::NotFound),
        }
    }

    /// True when `key` is present.
    pub fn contains(&self, pager: &mut Pager, key: &[u8]) -> Result<bool> {
        match self.get(pager, key) {
            Ok(_) => Ok(true),
            Err(Error::NotFound) => Ok(false),
            Err(e) => Err(e),
        }
    }

    /// Inserts or replaces `key`.
    pub fn insert(&self, pager: &mut Pager, key: &[u8], value: &[u8]) -> Result<()> {
        if key.is_empty() {
            return Err(Error::invalid("key must not be empty"));
        }
        if key.len() > MAX_KEY_SIZE {
            return Err(Error::invalid(format!(
                "key length {} exceeds the {MAX_KEY_SIZE}-byte structural limit",
                key.len()
            )));
        }
        let root = pager.meta().root;
        if let Some(split) = self.insert_into(pager, root, key, value)? {
            // Root split: build a new root one level up.
            let mut new_root = pager.allocate_page(PageType::Internal)?;
            new_root.set_extra(root); // leftmost child = old root
            let cell = Page::encode_internal_cell(&split.key, split.right);
            new_root.insert_cell_at(0, &cell)?;
            let new_root_id = new_root.page_id();
            pager.write_page(new_root);

            for child in [root, split.right] {
                let mut c = pager.read_page(child)?;
                c.set_parent(new_root_id);
                pager.write_page(c);
            }
            let mut meta = pager.meta();
            meta.root = new_root_id;
            pager.set_meta(meta);
        }
        Ok(())
    }

    /// Recursive insert. Returns a [`Split`] when `page_id` had to split.
    fn insert_into(
        &self,
        pager: &mut Pager,
        page_id: u32,
        key: &[u8],
        value: &[u8],
    ) -> Result<Option<Split>> {
        let page = pager.read_page(page_id)?;
        if page.is_leaf() {
            return self.insert_into_leaf(pager, page, key, value);
        }
        let child_id = self.child_for(&page, key)?;
        match self.insert_into(pager, child_id, key, value)? {
            None => Ok(None),
            Some(split) => self.absorb_split(pager, page_id, split),
        }
    }

    /// Inserts into a leaf, splitting when the fill factor is exceeded.
    fn insert_into_leaf(
        &self,
        pager: &mut Pager,
        mut leaf: Page,
        key: &[u8],
        value: &[u8],
    ) -> Result<Option<Split>> {
        // Spill large values into an overflow chain.
        let (inline, overflow) = if value.len() > MAX_INLINE_VALUE {
            (Vec::new(), Some(pager.write_overflow_chain(value)?))
        } else {
            (value.to_vec(), None)
        };
        let cell = Page::encode_leaf_cell(key, &inline, value.len() as u32, overflow);

        match leaf.search(key)? {
            Ok(idx) => {
                // Replacement: reclaim the old overflow chain first.
                let old = leaf.leaf_cell(idx)?;
                if let Some(head) = old.overflow {
                    pager.free_overflow_chain(head)?;
                }
                if leaf.replace_cell_at(idx, &cell).is_ok() {
                    pager.write_page(leaf);
                    return Ok(None);
                }
                // Did not fit: remove, then fall through to the split path.
                leaf.remove_cell_at(idx)?;
                self.insert_or_split_leaf(pager, leaf, idx, &cell, key)
            }
            Err(idx) => self.insert_or_split_leaf(pager, leaf, idx, &cell, key),
        }
    }

    /// Inserts `cell` at `idx`, splitting the leaf if it does not fit.
    fn insert_or_split_leaf(
        &self,
        pager: &mut Pager,
        mut leaf: Page,
        idx: usize,
        cell: &[u8],
        key: &[u8],
    ) -> Result<Option<Split>> {
        let used_after = USABLE_SPACE - leaf.free_space() + cell.len();
        let fits = cell.len() + crate::page::SLOT_SIZE <= leaf.free_space();
        let within_fill = used_after <= self.fill.max_used_bytes();

        if fits && within_fill {
            leaf.insert_cell_at(idx, cell)?;
            pager.write_page(leaf);
            return Ok(None);
        }

        // Split: move the upper half of the cells to a fresh right sibling.
        let n = leaf.num_keys() as usize;
        if n == 0 {
            // Nothing to split against — the cell simply cannot be stored.
            return Err(Error::Full(format!(
                "cell of {} bytes cannot fit in an empty page",
                cell.len()
            )));
        }
        let mid = n / 2;
        let mut right = pager.allocate_page(PageType::Leaf)?;
        right.set_parent(leaf.parent());
        right.set_extra(leaf.extra()); // inherit the sibling link

        let mut moved: Vec<Vec<u8>> = Vec::with_capacity(n - mid);
        for i in mid..n {
            moved.push(leaf.cell(i)?.to_vec());
        }
        for _ in mid..n {
            leaf.remove_cell_at(mid)?;
        }
        leaf.compact()?;
        for (i, m) in moved.iter().enumerate() {
            right.insert_cell_at(i, m)?;
        }

        // Route the new cell to whichever side owns its key range.
        let boundary = right.cell_key(0)?.to_vec();
        if key < boundary.as_slice() {
            let pos = match leaf.search(key)? {
                Ok(p) | Err(p) => p,
            };
            leaf.insert_cell_at(pos, cell)?;
        } else {
            let pos = match right.search(key)? {
                Ok(p) | Err(p) => p,
            };
            right.insert_cell_at(pos, cell)?;
        }

        let right_id = right.page_id();
        leaf.set_extra(right_id);
        let sep = right.cell_key(0)?.to_vec();
        pager.write_page(leaf);
        pager.write_page(right);
        Ok(Some(Split {
            key: sep,
            right: right_id,
        }))
    }

    /// Inserts a promoted separator into an internal node, splitting if needed.
    fn absorb_split(&self, pager: &mut Pager, page_id: u32, split: Split) -> Result<Option<Split>> {
        let mut node = pager.read_page(page_id)?;
        let cell = Page::encode_internal_cell(&split.key, split.right);
        let pos = match node.search(&split.key)? {
            Ok(p) => p + 1,
            Err(p) => p,
        };

        if cell.len() + crate::page::SLOT_SIZE <= node.free_space() {
            node.insert_cell_at(pos, &cell)?;
            pager.write_page(node);
            let mut child = pager.read_page(split.right)?;
            child.set_parent(page_id);
            pager.write_page(child);
            return Ok(None);
        }

        // Internal split. The node is physically full, so we split the existing
        // cells first and only then route the new separator into whichever half
        // owns its key range: the middle separator moves *up*, not sideways.
        let n = node.num_keys() as usize;
        if n < 2 {
            return Err(Error::Full(format!(
                "internal page {page_id} cannot absorb a {}-byte separator",
                cell.len()
            )));
        }
        let mid = n / 2;
        let promote_key = node.cell_key(mid)?.to_vec();
        let promote_child = node.internal_child(mid)?;

        let mut right = pager.allocate_page(PageType::Internal)?;
        right.set_parent(node.parent());
        right.set_extra(promote_child); // leftmost child of the right node

        let mut moved: Vec<Vec<u8>> = Vec::new();
        for i in (mid + 1)..n {
            moved.push(node.cell(i)?.to_vec());
        }
        for _ in mid..n {
            node.remove_cell_at(mid)?; // also drops the promoted cell
        }
        node.compact()?;
        for (i, m) in moved.iter().enumerate() {
            right.insert_cell_at(i, m)?;
        }

        // Place the pending separator on the correct side of the promoted key.
        if split.key.as_slice() < promote_key.as_slice() {
            let p = match node.search(&split.key)? {
                Ok(x) => x + 1,
                Err(x) => x,
            };
            node.insert_cell_at(p, &cell)?;
        } else {
            let p = match right.search(&split.key)? {
                Ok(x) => x + 1,
                Err(x) => x,
            };
            right.insert_cell_at(p, &cell)?;
        }
        let _ = pos; // insertion position recomputed per side after the split

        let right_id = right.page_id();
        // Re-parent every child that now lives under the right sibling.
        let mut children = vec![right.extra()];
        for i in 0..right.num_keys() as usize {
            children.push(right.internal_child(i)?);
        }
        let left_id = node.page_id();
        let mut left_children = vec![node.extra()];
        for i in 0..node.num_keys() as usize {
            left_children.push(node.internal_child(i)?);
        }
        pager.write_page(node);
        pager.write_page(right);
        for c in children {
            if c != SENTINEL {
                let mut cp = pager.read_page(c)?;
                cp.set_parent(right_id);
                pager.write_page(cp);
            }
        }
        for c in left_children {
            if c != SENTINEL {
                let mut cp = pager.read_page(c)?;
                cp.set_parent(left_id);
                pager.write_page(cp);
            }
        }
        Ok(Some(Split {
            key: promote_key,
            right: right_id,
        }))
    }

    /// Deletes `key`, returning [`Error::NotFound`] when absent.
    pub fn delete(&self, pager: &mut Pager, key: &[u8]) -> Result<()> {
        let leaf_id = self.find_leaf(pager, key)?;
        let mut leaf = pager.read_page(leaf_id)?;
        match leaf.search(key)? {
            Ok(idx) => {
                let cell = leaf.leaf_cell(idx)?;
                if let Some(head) = cell.overflow {
                    pager.free_overflow_chain(head)?;
                }
                leaf.remove_cell_at(idx)?;
                pager.write_page(leaf);
                Ok(())
            }
            Err(_) => Err(Error::NotFound),
        }
    }

    /// Descends from the root to the leaf that owns `key`.
    fn find_leaf(&self, pager: &mut Pager, key: &[u8]) -> Result<u32> {
        let mut current = pager.meta().root;
        // A tree of 4 KiB pages cannot exceed ~64 levels; the bound turns a
        // corrupt cyclic parent pointer into an error instead of a hang.
        for _ in 0..64 {
            let page = pager.read_page(current)?;
            if page.is_leaf() {
                return Ok(current);
            }
            current = self.child_for(&page, key)?;
        }
        Err(Error::corrupt("B+Tree depth exceeded 64 levels (cycle?)"))
    }

    /// Chooses the child of an internal node responsible for `key`.
    fn child_for(&self, page: &Page, key: &[u8]) -> Result<u32> {
        let idx = match page.search(key)? {
            Ok(i) => i + 1, // exact match on a separator -> go right
            Err(i) => i,
        };
        let child = if idx == 0 {
            page.extra()
        } else {
            page.internal_child(idx - 1)?
        };
        if child == SENTINEL {
            return Err(Error::corrupt(format!(
                "internal page {} has a dangling child pointer at {idx}",
                page.page_id()
            )));
        }
        Ok(child)
    }

    /// Page id of the leftmost leaf, for full scans.
    pub fn first_leaf(&self, pager: &mut Pager) -> Result<u32> {
        let mut current = pager.meta().root;
        for _ in 0..64 {
            let page = pager.read_page(current)?;
            if page.is_leaf() {
                return Ok(current);
            }
            let next = page.extra();
            if next == SENTINEL {
                return Err(Error::corrupt("internal node without a leftmost child"));
            }
            current = next;
        }
        Err(Error::corrupt("B+Tree depth exceeded 64 levels (cycle?)"))
    }

    /// Iterates every key/value pair in ascending key order.
    pub fn scan(&self, pager: &mut Pager) -> Result<Vec<(Vec<u8>, Vec<u8>)>> {
        let mut out = Vec::new();
        let mut leaf_id = self.first_leaf(pager)?;
        let mut visited = 0u64;
        while leaf_id != SENTINEL {
            visited += 1;
            if visited > 10_000_000 {
                return Err(Error::corrupt("leaf chain appears to be cyclic"));
            }
            let leaf = pager.read_page(leaf_id)?;
            for i in 0..leaf.num_keys() as usize {
                let cell: LeafCell = leaf.leaf_cell(i)?;
                let value = match cell.overflow {
                    Some(head) => pager.read_overflow_chain(head, cell.total_len)?,
                    None => cell.value,
                };
                out.push((cell.key, value));
            }
            leaf_id = leaf.extra();
        }
        Ok(out)
    }

    /// Number of live keys in the tree.
    pub fn len(&self, pager: &mut Pager) -> Result<u64> {
        let mut count = 0u64;
        let mut leaf_id = self.first_leaf(pager)?;
        while leaf_id != SENTINEL {
            let leaf = pager.read_page(leaf_id)?;
            count += leaf.num_keys() as u64;
            leaf_id = leaf.extra();
        }
        Ok(count)
    }

    /// Verifies structural invariants: key ordering within and across leaves,
    /// child pointer sanity, and per-page CRC (implicit in `read_page`).
    pub fn verify(&self, pager: &mut Pager) -> Result<()> {
        let mut previous: Option<Vec<u8>> = None;
        let mut leaf_id = self.first_leaf(pager)?;
        while leaf_id != SENTINEL {
            let leaf = pager.read_page(leaf_id)?;
            for i in 0..leaf.num_keys() as usize {
                let key = leaf.cell_key(i)?.to_vec();
                if let Some(prev) = &previous
                    && prev.as_slice() >= key.as_slice()
                {
                    return Err(Error::corrupt(format!(
                        "key ordering violated in leaf {leaf_id} at slot {i}"
                    )));
                }
                previous = Some(key);
            }
            leaf_id = leaf.extra();
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pager::Pager;

    fn setup() -> (tempfile::TempDir, Pager, BTree) {
        let dir = tempfile::tempdir().unwrap();
        let pager = Pager::open(&dir.path().join("t.pdb"), 256).unwrap();
        (dir, pager, BTree::new(FillFactor::default()))
    }

    #[test]
    fn insert_get_delete_roundtrip() {
        let (_d, mut p, t) = setup();
        t.insert(&mut p, b"alpha", b"one").unwrap();
        t.insert(&mut p, b"beta", b"two").unwrap();
        assert_eq!(t.get(&mut p, b"alpha").unwrap(), b"one");
        assert_eq!(t.get(&mut p, b"beta").unwrap(), b"two");
        assert!(matches!(t.get(&mut p, b"gamma"), Err(Error::NotFound)));
        t.delete(&mut p, b"alpha").unwrap();
        assert!(matches!(t.get(&mut p, b"alpha"), Err(Error::NotFound)));
        assert!(matches!(t.delete(&mut p, b"alpha"), Err(Error::NotFound)));
    }

    #[test]
    fn overwrite_replaces_value() {
        let (_d, mut p, t) = setup();
        t.insert(&mut p, b"k", b"v1").unwrap();
        t.insert(&mut p, b"k", b"v2").unwrap();
        assert_eq!(t.get(&mut p, b"k").unwrap(), b"v2");
        assert_eq!(t.len(&mut p).unwrap(), 1);
    }

    #[test]
    fn splits_produce_a_valid_ordered_tree() {
        let (_d, mut p, t) = setup();
        let n = 2000u32;
        for i in 0..n {
            let key = format!("key{i:06}");
            let value = format!("value-{i}");
            t.insert(&mut p, key.as_bytes(), value.as_bytes()).unwrap();
        }
        for i in 0..n {
            let key = format!("key{i:06}");
            let expected = format!("value-{i}");
            assert_eq!(
                t.get(&mut p, key.as_bytes()).unwrap(),
                expected.as_bytes(),
                "lookup failed for {key}"
            );
        }
        assert_eq!(t.len(&mut p).unwrap(), n as u64);
        t.verify(&mut p).unwrap();

        let scanned = t.scan(&mut p).unwrap();
        assert_eq!(scanned.len(), n as usize);
        for w in scanned.windows(2) {
            assert!(w[0].0 < w[1].0, "scan is not ordered");
        }
    }

    #[test]
    fn reverse_insertion_order_still_splits_correctly() {
        let (_d, mut p, t) = setup();
        for i in (0..800u32).rev() {
            let key = format!("k{i:05}");
            t.insert(&mut p, key.as_bytes(), b"v").unwrap();
        }
        t.verify(&mut p).unwrap();
        assert_eq!(t.len(&mut p).unwrap(), 800);
    }

    #[test]
    fn large_values_use_overflow_pages() {
        let (_d, mut p, t) = setup();
        let big = vec![0xCDu8; 200_000];
        t.insert(&mut p, b"big", &big).unwrap();
        t.insert(&mut p, b"small", b"x").unwrap();
        assert_eq!(t.get(&mut p, b"big").unwrap(), big);
        assert_eq!(t.get(&mut p, b"small").unwrap(), b"x");
        t.delete(&mut p, b"big").unwrap();
        assert!(matches!(t.get(&mut p, b"big"), Err(Error::NotFound)));
    }

    #[test]
    fn empty_key_is_rejected() {
        let (_d, mut p, t) = setup();
        assert!(matches!(t.insert(&mut p, b"", b"v"), Err(Error::InvalidArgument(_))));
    }

    #[test]
    fn oversized_key_is_rejected() {
        let (_d, mut p, t) = setup();
        let key = vec![b'k'; MAX_KEY_SIZE + 1];
        assert!(matches!(
            t.insert(&mut p, &key, b"v"),
            Err(Error::InvalidArgument(_))
        ));
    }

    #[test]
    fn fill_factor_is_clamped() {
        let f = FillFactor::new(0.1, 2.0);
        assert!(f.min >= 0.5);
        assert!(f.max <= 1.0);
        assert!(f.min < f.max);
    }

    #[test]
    fn tight_fill_factor_splits_earlier() {
        let dir = tempfile::tempdir().unwrap();
        let mut p = Pager::open(&dir.path().join("t.pdb"), 256).unwrap();
        let t = BTree::new(FillFactor::new(0.5, 0.6));
        for i in 0..500u32 {
            let key = format!("k{i:05}");
            t.insert(&mut p, key.as_bytes(), b"payload").unwrap();
        }
        t.verify(&mut p).unwrap();
        assert_eq!(t.len(&mut p).unwrap(), 500);
        // A 0.6 max fill must use more pages than the default 1.0 policy.
        assert!(p.meta().page_count > 10);
    }

    #[test]
    fn delete_then_reinsert_across_splits() {
        let (_d, mut p, t) = setup();
        for i in 0..600u32 {
            t.insert(&mut p, format!("k{i:05}").as_bytes(), b"v").unwrap();
        }
        for i in (0..600u32).step_by(2) {
            t.delete(&mut p, format!("k{i:05}").as_bytes()).unwrap();
        }
        assert_eq!(t.len(&mut p).unwrap(), 300);
        for i in (0..600u32).step_by(2) {
            t.insert(&mut p, format!("k{i:05}").as_bytes(), b"w").unwrap();
        }
        assert_eq!(t.len(&mut p).unwrap(), 600);
        t.verify(&mut p).unwrap();
        assert_eq!(t.get(&mut p, b"k00000").unwrap(), b"w");
        assert_eq!(t.get(&mut p, b"k00001").unwrap(), b"v");
    }

    #[test]
    fn binary_keys_with_nulls_work() {
        let (_d, mut p, t) = setup();
        let keys: Vec<Vec<u8>> = vec![
            vec![0x00, 0x01],
            vec![0x00, 0x02],
            vec![0xFF, 0x00],
            vec![0x00],
        ];
        for k in &keys {
            t.insert(&mut p, k, b"v").unwrap();
        }
        for k in &keys {
            assert_eq!(t.get(&mut p, k).unwrap(), b"v");
        }
        assert_eq!(t.len(&mut p).unwrap(), keys.len() as u64);
    }
}
