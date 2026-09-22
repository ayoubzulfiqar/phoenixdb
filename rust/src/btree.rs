//! B+Tree index with configurable fill factors.
//!
//! * Leaves are linked left-to-right (`extra` = next leaf) for range scans.
//! * Internal nodes keep the leftmost child in the header `extra` field and one
//!   `(separator, right_child)` cell per remaining child. A key equal to a
//!   separator belongs to the separator's right child.
//! * Inserts recurse to the leaf and return a [`Split`] that the parent
//!   absorbs, so no latch is ever held on a whole path.
//! * Splits are **byte-balanced**: the cells (including the incoming one) are
//!   divided where both halves fit a page, never by count alone, so a page of
//!   mixed small and large cells always splits successfully.
//! * Deletes reclaim pages: a leaf that becomes empty is unlinked from the
//!   leaf chain and its parent, internal nodes left without children go with
//!   it, and a root with a single child collapses into that child.
//! * Parent pointers in page headers are not maintained (navigation always
//!   descends from the root); the field is reserved.
//!
//! # Fill factors
//!
//! [`FillFactor`] bounds how full a leaf may get before it splits. `min` is
//! clamped to `[0.5, 1.0]` and reported by [`BTree::check`] as the
//! under-occupancy threshold.

use crate::error::{Error, Result};
use crate::page::{
    LeafCell, MAX_INLINE_VALUE, MAX_KEY_SIZE, PAGE_HEADER_SIZE, PAGE_SIZE, Page, PageType,
    SENTINEL, SLOT_SIZE, USABLE_SPACE,
};
use crate::pager::Pager;
use std::collections::HashSet;
use std::ops::Bound;

/// Largest leaf cell stored inline. Longer values spill into an overflow
/// chain, which keeps every leaf cell at or below
/// `max(MAX_INLINE_CELL, 8 + MAX_KEY_SIZE + 4)` bytes — small enough that any
/// three cells fit one page, which is what guarantees a split can always
/// place every cell.
pub const MAX_INLINE_CELL: usize = USABLE_SPACE / 4 - SLOT_SIZE;

/// Maximum tree height. A 4 KiB-page tree cannot exceed this for any file a
/// 32-bit page id can address; deeper means a cyclic child pointer.
const MAX_DEPTH: usize = 64;

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

    /// Bytes a page may use before it is considered full.
    #[must_use]
    pub fn max_used_bytes(&self) -> usize {
        (USABLE_SPACE as f32 * self.max) as usize
    }

    /// Bytes below which a page is underfull.
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

/// Bounds a walk along the leaf chain.
///
/// A healthy chain visits each leaf once, so it can never be longer than the
/// number of pages in the file. A longer walk means a sibling link points
/// backwards (corruption, or two handles writing one file) and the walk would
/// otherwise never end — and a scan collecting results would grow without
/// bound until the OS kills the process.
struct LeafChainGuard {
    remaining: u64,
}

impl LeafChainGuard {
    fn new(pager: &Pager) -> Self {
        LeafChainGuard {
            remaining: u64::from(pager.meta().page_count),
        }
    }

    fn step(&mut self) -> Result<()> {
        if self.remaining == 0 {
            return Err(Error::corrupt(
                "leaf chain is longer than the file has pages (cyclic sibling link)",
            ));
        }
        self.remaining -= 1;
        Ok(())
    }
}

/// Result of a full structural check, see [`BTree::check`].
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct TreeReport {
    /// Levels from the root to the leaves (1 for a lone root leaf).
    pub depth: u32,
    /// Keys stored in the tree.
    pub keys: u64,
    /// Leaf pages reachable from the root.
    pub leaf_pages: u32,
    /// Internal pages reachable from the root.
    pub internal_pages: u32,
    /// Overflow pages reachable from leaf cells.
    pub overflow_pages: u32,
    /// Pages on the free list.
    pub free_pages: u32,
    /// Allocated pages that are neither reachable nor free — space leaked by a
    /// crash between allocation and checkpoint. Harmless, reclaimable by a
    /// rebuild.
    pub unreachable_pages: u32,
    /// Non-root leaves below the fill factor's `min` occupancy.
    pub underfull_leaves: u32,
}

/// B+Tree operations over a [`Pager`].
///
/// The tree is stateless: the root id lives in the pager's metadata, so a
/// `BTree` value is just a handle carrying the fill-factor policy. Lookups and
/// scans need only `&Pager`, so they can run under a shared lock.
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
    pub fn get(&self, pager: &Pager, key: &[u8]) -> Result<Vec<u8>> {
        let leaf_id = self.find_leaf(pager, key)?;
        let leaf = pager.read_page(leaf_id)?;
        match leaf.search(key)? {
            Ok(idx) => Self::cell_value(pager, leaf.leaf_cell(idx)?),
            Err(_) => Err(Error::NotFound),
        }
    }

    /// True when `key` is present. Does not read overflow values.
    pub fn contains(&self, pager: &Pager, key: &[u8]) -> Result<bool> {
        let leaf_id = self.find_leaf(pager, key)?;
        Ok(pager.read_page(leaf_id)?.search(key)?.is_ok())
    }

    fn cell_value(pager: &Pager, cell: LeafCell) -> Result<Vec<u8>> {
        match cell.overflow {
            Some(head) => pager.read_overflow_chain(head, cell.total_len),
            None => Ok(cell.value),
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
        let value_len = u32::try_from(value.len())
            .map_err(|_| Error::invalid("value length does not fit in 32 bits"))?;
        let root = pager.meta().root;
        if let Some(split) = self.insert_into(pager, root, key, value, value_len, 0)? {
            // Root split: build a new root one level up.
            let mut new_root = pager.allocate_page(PageType::Internal)?;
            new_root.set_extra(root); // leftmost child = old root
            let cell = Page::encode_internal_cell(&split.key, split.right);
            new_root.insert_cell_at(0, &cell)?;
            let new_root_id = new_root.page_id();
            pager.write_page(new_root);
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
        value_len: u32,
        depth: usize,
    ) -> Result<Option<Split>> {
        if depth > MAX_DEPTH {
            return Err(Error::corrupt("B+Tree depth exceeded 64 levels (cycle?)"));
        }
        let page = pager.read_page(page_id)?;
        if page.is_leaf() {
            return self.insert_into_leaf(pager, page, key, value, value_len);
        }
        let child_id = self.child_for(&page, key)?;
        match self.insert_into(pager, child_id, key, value, value_len, depth + 1)? {
            None => Ok(None),
            Some(split) => self.absorb_split(pager, page, split),
        }
    }

    /// Inserts into a leaf, splitting when the fill factor is exceeded.
    fn insert_into_leaf(
        &self,
        pager: &mut Pager,
        mut leaf: Page,
        key: &[u8],
        value: &[u8],
        value_len: u32,
    ) -> Result<Option<Split>> {
        let position = leaf.search(key)?;
        if let Ok(idx) = position {
            // Replacement: release the old value's overflow chain and drop the
            // old cell; the new cell then takes the ordinary insert path.
            let old = leaf.leaf_cell(idx)?;
            if let Some(head) = old.overflow {
                pager.free_overflow_chain(head)?;
            }
            leaf.remove_cell_at(idx)?;
        }
        let idx = match position {
            Ok(i) | Err(i) => i,
        };

        // Spill large values into an overflow chain.
        let inline =
            value.len() <= MAX_INLINE_VALUE && 8 + key.len() + value.len() <= MAX_INLINE_CELL;
        let cell = if inline {
            Page::encode_leaf_cell(key, value, value_len, None)
        } else {
            let head = pager.write_overflow_chain(value)?;
            Page::encode_leaf_cell(key, &[], value_len, Some(head))
        };

        // Removed cells leave holes that `free_space` does not count; compact
        // before judging either limit so holes never force a split.
        let need = cell.len() + SLOT_SIZE;
        let max_used = self.fill.max_used_bytes();
        if need > leaf.free_space() || USABLE_SPACE - leaf.free_space() + need > max_used {
            leaf.compact()?;
        }
        let fits = need <= leaf.free_space();
        let within_fill = USABLE_SPACE - leaf.free_space() + need <= max_used;
        if fits && (within_fill || leaf.num_keys() == 0) {
            leaf.insert_cell_at(idx, &cell)?;
            pager.write_page(leaf);
            return Ok(None);
        }
        self.split_leaf(pager, leaf, idx, cell)
    }

    /// Splits `leaf` around the incoming `cell` (destined for slot `idx`).
    fn split_leaf(
        &self,
        pager: &mut Pager,
        mut leaf: Page,
        idx: usize,
        cell: Vec<u8>,
    ) -> Result<Option<Split>> {
        let n = leaf.num_keys() as usize;
        let mut cells: Vec<Vec<u8>> = Vec::with_capacity(n + 1);
        for i in 0..n {
            cells.push(leaf.cell(i)?.to_vec());
        }
        cells.insert(idx, cell);
        // Appending past the end of the rightmost leaf (monotonic keys: ids,
        // timestamps): keep this page full and start a fresh one, instead of
        // leaving every leaf of an append-only workload half empty.
        let appending = idx == n && n > 0 && leaf.extra() == SENTINEL;
        let at = if appending {
            Some(n)
        } else {
            split_point(&cells, false)
        }
        .ok_or_else(|| {
            Error::Full(format!(
                "leaf {} cannot be split to place a new cell",
                leaf.page_id()
            ))
        })?;

        let mut right = pager.allocate_page(PageType::Leaf)?;
        right.set_extra(leaf.extra()); // inherit the sibling link
        leaf.clear_cells();
        for (i, c) in cells[..at].iter().enumerate() {
            leaf.insert_cell_at(i, c)?;
        }
        for (i, c) in cells[at..].iter().enumerate() {
            right.insert_cell_at(i, c)?;
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
    fn absorb_split(
        &self,
        pager: &mut Pager,
        mut node: Page,
        split: Split,
    ) -> Result<Option<Split>> {
        let cell = Page::encode_internal_cell(&split.key, split.right);
        let pos = match node.search(&split.key)? {
            Ok(p) => p + 1,
            Err(p) => p,
        };
        let fits = cell.len() + SLOT_SIZE <= node.free_space() || {
            node.compact()?;
            cell.len() + SLOT_SIZE <= node.free_space()
        };
        if fits {
            node.insert_cell_at(pos, &cell)?;
            pager.write_page(node);
            return Ok(None);
        }

        // Internal split over the full ordered cell list (new one included):
        // cells[..m] stay, cells[m] moves up, cells[m + 1..] go right with
        // cells[m]'s child as the right node's leftmost child.
        let n = node.num_keys() as usize;
        let mut cells: Vec<Vec<u8>> = Vec::with_capacity(n + 1);
        for i in 0..n {
            cells.push(node.cell(i)?.to_vec());
        }
        cells.insert(pos, cell);
        // Same append optimisation as for leaves: a separator arriving at the
        // right edge moves up by itself and the node stays full.
        let m = if pos == n {
            Some(n)
        } else {
            split_point(&cells, true)
        }
        .ok_or_else(|| {
            Error::Full(format!(
                "internal page {} cannot be split to place a separator",
                node.page_id()
            ))
        })?;
        let (promote_key, promote_child) = decode_internal_cell(&cells[m])?;

        let mut right = pager.allocate_page(PageType::Internal)?;
        right.set_extra(promote_child);
        node.clear_cells();
        for (i, c) in cells[..m].iter().enumerate() {
            node.insert_cell_at(i, c)?;
        }
        for (i, c) in cells[m + 1..].iter().enumerate() {
            right.insert_cell_at(i, c)?;
        }
        let right_id = right.page_id();
        pager.write_page(node);
        pager.write_page(right);
        Ok(Some(Split {
            key: promote_key,
            right: right_id,
        }))
    }

    /// Deletes `key`, returning [`Error::NotFound`] when absent.
    ///
    /// A leaf left empty is removed from the tree and returned to the free
    /// list, together with any internal node that loses its last child.
    pub fn delete(&self, pager: &mut Pager, key: &[u8]) -> Result<()> {
        // Descend, remembering (internal page, child slot taken) per level.
        // Slot 0 is the leftmost child (`extra`), slot i + 1 is cell i's child.
        let mut path: Vec<(u32, usize)> = Vec::new();
        let mut current = pager.meta().root;
        let mut leaf = loop {
            if path.len() > MAX_DEPTH {
                return Err(Error::corrupt("B+Tree depth exceeded 64 levels (cycle?)"));
            }
            let page = pager.read_page(current)?;
            if page.is_leaf() {
                break page;
            }
            let slot = child_slot_for(&page, key)?;
            path.push((current, slot));
            current = child_at(&page, slot)?;
        };
        let leaf_id = current;
        let idx = match leaf.search(key)? {
            Ok(idx) => idx,
            Err(_) => return Err(Error::NotFound),
        };
        if let Some(head) = leaf.leaf_cell(idx)?.overflow {
            pager.free_overflow_chain(head)?;
        }
        leaf.remove_cell_at(idx)?;
        if leaf.num_keys() > 0 || path.is_empty() {
            pager.write_page(leaf);
            return Ok(());
        }
        self.remove_empty_leaf(pager, leaf, leaf_id, path)
    }

    /// Unlinks the (now empty) `leaf` reached through `path`.
    fn remove_empty_leaf(
        &self,
        pager: &mut Pager,
        leaf: Page,
        leaf_id: u32,
        mut path: Vec<(u32, usize)>,
    ) -> Result<()> {
        // Internal nodes directly above the leaf that have no other child go
        // away with it. `anchor` is the first ancestor that keeps a child.
        let mut orphaned: Vec<u32> = Vec::new();
        while let Some(&(pid, _)) = path.last() {
            if pager.read_page(pid)?.num_keys() == 0 {
                orphaned.push(pid);
                path.pop();
            } else {
                break;
            }
        }
        let Some(&(anchor_id, slot)) = path.last() else {
            // This leaf was the tree's only leaf: it becomes an empty root.
            pager.write_page(leaf);
            let mut meta = pager.meta();
            meta.root = leaf_id;
            pager.set_meta(meta);
            for pid in orphaned {
                pager.free_page(pid)?;
            }
            return Ok(());
        };

        // Keep the leaf chain intact: the predecessor now skips this leaf.
        if let Some(prev_id) = self.previous_leaf(pager, &path)? {
            let mut prev = pager.read_page(prev_id)?;
            prev.set_extra(leaf.extra());
            pager.write_page(prev);
        }

        let mut anchor = pager.read_page(anchor_id)?;
        if slot == 0 {
            let first = anchor.internal_child(0)?;
            anchor.set_extra(first);
            anchor.remove_cell_at(0)?;
        } else {
            anchor.remove_cell_at(slot - 1)?;
        }
        pager.write_page(anchor);
        pager.free_page(leaf_id)?;
        for pid in orphaned {
            pager.free_page(pid)?;
        }

        // A root left with a single child hands the tree to that child.
        loop {
            let root_id = pager.meta().root;
            let root = pager.read_page(root_id)?;
            if root.is_leaf() || root.num_keys() > 0 {
                break;
            }
            let mut meta = pager.meta();
            meta.root = root.extra();
            pager.set_meta(meta);
            pager.free_page(root_id)?;
        }
        Ok(())
    }

    /// The leaf immediately before the subtree reached through `path`, found
    /// by stepping one child left at the deepest level where that is possible
    /// and then descending along rightmost children.
    fn previous_leaf(&self, pager: &Pager, path: &[(u32, usize)]) -> Result<Option<u32>> {
        let Some(&(pid, slot)) = path.iter().rev().find(|(_, slot)| *slot > 0) else {
            return Ok(None); // the leftmost leaf has no predecessor
        };
        let mut current = child_at(&pager.read_page(pid)?, slot - 1)?;
        for _ in 0..=MAX_DEPTH {
            let page = pager.read_page(current)?;
            if page.is_leaf() {
                return Ok(Some(current));
            }
            current = child_at(&page, page.num_keys() as usize)?;
        }
        Err(Error::corrupt("B+Tree depth exceeded 64 levels (cycle?)"))
    }

    /// Descends from the root to the leaf that owns `key`.
    fn find_leaf(&self, pager: &Pager, key: &[u8]) -> Result<u32> {
        let mut current = pager.meta().root;
        // The bound turns a corrupt cyclic child pointer into an error
        // instead of a hang.
        for _ in 0..=MAX_DEPTH {
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
        child_at(page, child_slot_for(page, key)?)
    }

    /// Page id of the leftmost leaf, for full scans.
    pub fn first_leaf(&self, pager: &Pager) -> Result<u32> {
        let mut current = pager.meta().root;
        for _ in 0..=MAX_DEPTH {
            let page = pager.read_page(current)?;
            if page.is_leaf() {
                return Ok(current);
            }
            current = child_at(&page, 0)?;
        }
        Err(Error::corrupt("B+Tree depth exceeded 64 levels (cycle?)"))
    }

    /// Iterates every key/value pair in ascending key order.
    pub fn scan(&self, pager: &Pager) -> Result<Vec<(Vec<u8>, Vec<u8>)>> {
        let mut out = Vec::new();
        self.scan_iter(pager, |item| {
            out.push(item);
            Ok(())
        })?;
        Ok(out)
    }

    /// Calls `f` for every key/value pair in ascending key order.
    ///
    /// This is the streaming primitive: it avoids materializing the full result
    /// set when the caller can process entries incrementally.
    pub fn scan_iter<F>(&self, pager: &Pager, mut f: F) -> Result<()>
    where
        F: FnMut((Vec<u8>, Vec<u8>)) -> Result<()>,
    {
        self.range_iter(pager, Bound::Unbounded, Bound::Unbounded, |k, v| {
            f((k, v))?;
            Ok(true)
        })
    }

    /// Calls `f` for every pair with a key inside `(lo, hi)`, ascending.
    ///
    /// `f` returns `Ok(true)` to continue and `Ok(false)` to stop early; values
    /// past the upper bound (and past an early stop) are never read, so a
    /// bounded scan costs only the pages it touches.
    pub fn range_iter<F>(
        &self,
        pager: &Pager,
        lo: Bound<&[u8]>,
        hi: Bound<&[u8]>,
        mut f: F,
    ) -> Result<()>
    where
        F: FnMut(Vec<u8>, Vec<u8>) -> Result<bool>,
    {
        let mut leaf_id = match lo {
            Bound::Unbounded => self.first_leaf(pager)?,
            Bound::Included(k) | Bound::Excluded(k) => self.find_leaf(pager, k)?,
        };
        let mut guard = LeafChainGuard::new(pager);
        let mut first = true;
        while leaf_id != SENTINEL {
            guard.step()?;
            let leaf = pager.read_page(leaf_id)?;
            let n = leaf.num_keys() as usize;
            let start = if first {
                match lo {
                    Bound::Unbounded => 0,
                    Bound::Included(k) => match leaf.search(k)? {
                        Ok(i) | Err(i) => i,
                    },
                    Bound::Excluded(k) => match leaf.search(k)? {
                        Ok(i) => i + 1,
                        Err(i) => i,
                    },
                }
            } else {
                0
            };
            first = false;
            for i in start..n {
                let key = leaf.cell_key(i)?;
                let past_end = match hi {
                    Bound::Unbounded => false,
                    Bound::Included(h) => key > h,
                    Bound::Excluded(h) => key >= h,
                };
                if past_end {
                    return Ok(());
                }
                let cell = leaf.leaf_cell(i)?;
                let key = cell.key.clone();
                let value = Self::cell_value(pager, cell)?;
                if !f(key, value)? {
                    return Ok(());
                }
            }
            leaf_id = leaf.extra();
        }
        Ok(())
    }

    /// Number of live keys in the tree.
    pub fn len(&self, pager: &Pager) -> Result<u64> {
        let mut count = 0u64;
        let mut leaf_id = self.first_leaf(pager)?;
        let mut guard = LeafChainGuard::new(pager);
        while leaf_id != SENTINEL {
            guard.step()?;
            let leaf = pager.read_page(leaf_id)?;
            count += leaf.num_keys() as u64;
            leaf_id = leaf.extra();
        }
        Ok(count)
    }

    /// True when the tree holds no keys.
    pub fn is_empty(&self, pager: &Pager) -> Result<bool> {
        Ok(self.len(pager)? == 0)
    }

    /// Verifies the tree; see [`BTree::check`].
    pub fn verify(&self, pager: &Pager) -> Result<()> {
        self.check(pager).map(|_| ())
    }

    /// Full structural check of the tree and the free list.
    ///
    /// Every page is CRC-checked on read. On top of that:
    ///
    /// * keys are strictly ascending within a page and inside the range the
    ///   parent's separators assign to it;
    /// * every leaf sits at the same depth;
    /// * no page is reachable twice (no cycles, no shared subtrees);
    /// * the leaf chain visits exactly the leaves of an in-order walk;
    /// * every overflow chain is well-formed and yields the declared length;
    /// * the free list holds only free pages that the tree does not use.
    pub fn check(&self, pager: &Pager) -> Result<TreeReport> {
        let meta = pager.meta();
        let mut state = CheckState {
            seen: HashSet::new(),
            leaves: Vec::new(),
            leaf_depth: None,
            report: TreeReport::default(),
            page_count: meta.page_count,
        };
        self.check_node(pager, meta.root, None, None, 1, &mut state)?;

        // The sibling chain must be exactly the in-order sequence of leaves.
        let mut chain_id = self.first_leaf(pager)?;
        let mut guard = LeafChainGuard::new(pager);
        let mut position = 0usize;
        while chain_id != SENTINEL {
            guard.step()?;
            if state.leaves.get(position) != Some(&chain_id) {
                return Err(Error::corrupt(format!(
                    "leaf chain reaches page {chain_id} at position {position}, \
                     but the tree order expects {:?}",
                    state.leaves.get(position)
                )));
            }
            position += 1;
            chain_id = pager.read_page(chain_id)?.extra();
        }
        if position != state.leaves.len() {
            return Err(Error::corrupt(format!(
                "leaf chain ends after {position} of {} leaves",
                state.leaves.len()
            )));
        }

        // The free list must be free pages the tree does not use.
        let mut free_id = meta.free_list;
        while free_id != SENTINEL {
            if free_id == 0 || free_id >= meta.page_count {
                return Err(Error::corrupt(format!(
                    "free list points outside the file (page {free_id})"
                )));
            }
            if !state.seen.insert(free_id) {
                return Err(Error::corrupt(format!(
                    "page {free_id} is both in use and on the free list (or the list cycles)"
                )));
            }
            let page = pager.read_page(free_id)?;
            if page.page_type()? != PageType::Free {
                return Err(Error::corrupt(format!(
                    "free list holds page {free_id}, a {:?} page",
                    page.page_type()?
                )));
            }
            state.report.free_pages += 1;
            free_id = page.extra();
        }

        // Page 0 is the meta page; everything else is tree, free, or leaked.
        let accounted = state.seen.len() as u32 + 1;
        state.report.unreachable_pages = meta.page_count.saturating_sub(accounted);
        state.report.depth = state.leaf_depth.unwrap_or(1);
        Ok(state.report)
    }

    fn check_node(
        &self,
        pager: &Pager,
        id: u32,
        lo: Option<&[u8]>,
        hi: Option<&[u8]>,
        depth: u32,
        state: &mut CheckState,
    ) -> Result<()> {
        if depth as usize > MAX_DEPTH {
            return Err(Error::corrupt("B+Tree depth exceeded 64 levels (cycle?)"));
        }
        if id == 0 || id >= state.page_count {
            return Err(Error::corrupt(format!(
                "tree references page {id}, outside 1..{}",
                state.page_count
            )));
        }
        if !state.seen.insert(id) {
            return Err(Error::corrupt(format!("page {id} is reachable twice")));
        }
        let page = pager.read_page(id)?;
        let n = page.num_keys() as usize;
        let mut previous: Option<&[u8]> = None;
        for i in 0..n {
            let key = page.cell_key(i)?;
            if previous.is_some_and(|p| p >= key) {
                return Err(Error::corrupt(format!(
                    "key ordering violated in page {id} at slot {i}"
                )));
            }
            if lo.is_some_and(|l| key < l) || hi.is_some_and(|h| key >= h) {
                return Err(Error::corrupt(format!(
                    "key in page {id} slot {i} lies outside the range its parent assigns"
                )));
            }
            previous = Some(key);
        }

        match page.page_type()? {
            PageType::Leaf => {
                match state.leaf_depth {
                    None => state.leaf_depth = Some(depth),
                    Some(d) if d != depth => {
                        return Err(Error::corrupt(format!(
                            "leaf {id} at depth {depth}, other leaves at depth {d}"
                        )));
                    }
                    Some(_) => {}
                }
                state.leaves.push(id);
                state.report.leaf_pages += 1;
                state.report.keys += n as u64;
                let used = USABLE_SPACE - page.free_space();
                if depth > 1 && used < self.fill.min_used_bytes() {
                    state.report.underfull_leaves += 1;
                }
                for i in 0..n {
                    let cell = page.leaf_cell(i)?;
                    if let Some(head) = cell.overflow {
                        state.report.overflow_pages +=
                            check_overflow(pager, head, cell.total_len, state)?;
                    }
                }
            }
            PageType::Internal => {
                state.report.internal_pages += 1;
                let mut child_lo = lo;
                for slot in 0..=n {
                    let child_hi = if slot < n {
                        Some(page.cell_key(slot)?)
                    } else {
                        hi
                    };
                    let child = child_at(&page, slot)?;
                    self.check_node(pager, child, child_lo, child_hi, depth + 1, state)?;
                    child_lo = child_hi;
                }
            }
            other => {
                return Err(Error::corrupt(format!(
                    "tree references page {id}, a {other:?} page"
                )));
            }
        }
        Ok(())
    }
}

struct CheckState {
    seen: HashSet<u32>,
    leaves: Vec<u32>,
    leaf_depth: Option<u32>,
    report: TreeReport,
    page_count: u32,
}

/// Walks an overflow chain for [`BTree::check`], returning its page count.
fn check_overflow(pager: &Pager, head: u32, total_len: u32, state: &mut CheckState) -> Result<u32> {
    let mut cursor = head;
    let mut pages = 0u32;
    let mut bytes = 0u64;
    while cursor != SENTINEL {
        if cursor == 0 || cursor >= state.page_count {
            return Err(Error::corrupt(format!(
                "overflow chain points outside the file (page {cursor})"
            )));
        }
        if !state.seen.insert(cursor) {
            return Err(Error::corrupt(format!(
                "overflow page {cursor} is reachable twice"
            )));
        }
        let page = pager.read_page(cursor)?;
        bytes += page.read_overflow()?.len() as u64;
        pages += 1;
        cursor = page.extra();
    }
    if bytes != u64::from(total_len) {
        return Err(Error::corrupt(format!(
            "overflow chain at {head} holds {bytes} bytes, cell declares {total_len}"
        )));
    }
    Ok(pages)
}

/// Child slot of an internal node for `key`: 0 is the leftmost child, `i + 1`
/// is the child of cell `i`. A key equal to a separator goes right.
fn child_slot_for(page: &Page, key: &[u8]) -> Result<usize> {
    Ok(match page.search(key)? {
        Ok(i) => i + 1,
        Err(i) => i,
    })
}

/// Page id stored in child `slot` of an internal node.
fn child_at(page: &Page, slot: usize) -> Result<u32> {
    let child = if slot == 0 {
        page.extra()
    } else {
        page.internal_child(slot - 1)?
    };
    if child == SENTINEL {
        return Err(Error::corrupt(format!(
            "internal page {} has a dangling child pointer at slot {slot}",
            page.page_id()
        )));
    }
    Ok(child)
}

fn decode_internal_cell(cell: &[u8]) -> Result<(Vec<u8>, u32)> {
    if cell.len() < 6 {
        return Err(Error::corrupt("internal cell too short"));
    }
    let key_len = u16::from_le_bytes([cell[0], cell[1]]) as usize;
    if 6 + key_len > cell.len() {
        return Err(Error::corrupt("internal cell key escapes the cell"));
    }
    let child = u32::from_le_bytes([cell[2], cell[3], cell[4], cell[5]]);
    Ok((cell[6..6 + key_len].to_vec(), child))
}

/// Picks where to divide an ordered cell list so that both halves fit a page,
/// with the byte volumes as even as possible.
///
/// For a leaf (`promote == false`) the result `s` means `cells[..s]` and
/// `cells[s..]`, both non-empty. For an internal node the result `m` means
/// `cells[..m]` stay, `cells[m]` moves to the parent and `cells[m + 1..]` go
/// right. `None` only if no division fits, which the cell-size limits rule out.
fn split_point(cells: &[Vec<u8>], promote: bool) -> Option<usize> {
    const CAPACITY: usize = PAGE_SIZE - PAGE_HEADER_SIZE;
    let sizes: Vec<usize> = cells.iter().map(|c| c.len() + SLOT_SIZE).collect();
    let total: usize = sizes.iter().sum();
    let mut best: Option<(usize, usize)> = None; // (imbalance, index)
    let mut left = 0usize;
    for (i, &size) in sizes.iter().enumerate() {
        let (l, r, candidate) = if promote {
            (left, total - left - size, i)
        } else {
            (left + size, total - left - size, i + 1)
        };
        left += size;
        let valid = if promote {
            true
        } else {
            candidate < cells.len() // both halves non-empty
        };
        if valid && l <= CAPACITY && r <= CAPACITY {
            let imbalance = l.abs_diff(r);
            if best.is_none_or(|(b, _)| imbalance < b) {
                best = Some((imbalance, candidate));
            }
        }
    }
    best.map(|(_, i)| i)
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
        assert_eq!(t.get(&p, b"alpha").unwrap(), b"one");
        assert_eq!(t.get(&p, b"beta").unwrap(), b"two");
        assert!(matches!(t.get(&p, b"gamma"), Err(Error::NotFound)));
        t.delete(&mut p, b"alpha").unwrap();
        assert!(matches!(t.get(&p, b"alpha"), Err(Error::NotFound)));
        assert!(matches!(t.delete(&mut p, b"alpha"), Err(Error::NotFound)));
    }

    #[test]
    fn overwrite_replaces_value() {
        let (_d, mut p, t) = setup();
        t.insert(&mut p, b"k", b"v1").unwrap();
        t.insert(&mut p, b"k", b"v2").unwrap();
        assert_eq!(t.get(&p, b"k").unwrap(), b"v2");
        assert_eq!(t.len(&p).unwrap(), 1);
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
                t.get(&p, key.as_bytes()).unwrap(),
                expected.as_bytes(),
                "lookup failed for {key}"
            );
        }
        assert_eq!(t.len(&p).unwrap(), n as u64);
        t.verify(&p).unwrap();

        let scanned = t.scan(&p).unwrap();
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
        t.verify(&p).unwrap();
        assert_eq!(t.len(&p).unwrap(), 800);
    }

    #[test]
    fn large_values_use_overflow_pages() {
        let (_d, mut p, t) = setup();
        let big = vec![0xCDu8; 200_000];
        t.insert(&mut p, b"big", &big).unwrap();
        t.insert(&mut p, b"small", b"x").unwrap();
        assert_eq!(t.get(&p, b"big").unwrap(), big);
        assert_eq!(t.get(&p, b"small").unwrap(), b"x");
        t.delete(&mut p, b"big").unwrap();
        assert!(matches!(t.get(&p, b"big"), Err(Error::NotFound)));
    }

    #[test]
    fn empty_key_is_rejected() {
        let (_d, mut p, t) = setup();
        assert!(matches!(
            t.insert(&mut p, b"", b"v"),
            Err(Error::InvalidArgument(_))
        ));
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
        t.verify(&p).unwrap();
        assert_eq!(t.len(&p).unwrap(), 500);
        // A 0.6 max fill must use more pages than the default 1.0 policy.
        let (_d2, mut dense, default_tree) = setup();
        for i in 0..500u32 {
            let key = format!("k{i:05}");
            default_tree
                .insert(&mut dense, key.as_bytes(), b"payload")
                .unwrap();
        }
        assert!(
            p.meta().page_count > dense.meta().page_count,
            "0.6 fill used {} pages, 1.0 fill used {}",
            p.meta().page_count,
            dense.meta().page_count
        );
        t.check(&p).unwrap();
    }

    #[test]
    fn delete_then_reinsert_across_splits() {
        let (_d, mut p, t) = setup();
        for i in 0..600u32 {
            t.insert(&mut p, format!("k{i:05}").as_bytes(), b"v")
                .unwrap();
        }
        for i in (0..600u32).step_by(2) {
            t.delete(&mut p, format!("k{i:05}").as_bytes()).unwrap();
        }
        assert_eq!(t.len(&p).unwrap(), 300);
        for i in (0..600u32).step_by(2) {
            t.insert(&mut p, format!("k{i:05}").as_bytes(), b"w")
                .unwrap();
        }
        assert_eq!(t.len(&p).unwrap(), 600);
        t.verify(&p).unwrap();
        assert_eq!(t.get(&p, b"k00000").unwrap(), b"w");
        assert_eq!(t.get(&p, b"k00001").unwrap(), b"v");
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
            assert_eq!(t.get(&p, k).unwrap(), b"v");
        }
        assert_eq!(t.len(&p).unwrap(), keys.len() as u64);
    }

    #[test]
    fn cyclic_leaf_chain_is_an_error_not_a_hang() {
        let (_d, mut p, t) = setup();
        for i in 0..400u32 {
            t.insert(&mut p, format!("k{i:05}").as_bytes(), &[0u8; 64])
                .unwrap();
        }
        // Point the first leaf's sibling link back at itself.
        let first = t.first_leaf(&p).unwrap();
        let mut leaf = p.read_page(first).unwrap();
        leaf.set_extra(first);
        p.write_page(leaf);

        let mut seen = 0usize;
        let err = t
            .scan_iter(&p, |_| {
                seen += 1;
                Ok(())
            })
            .unwrap_err();
        assert!(matches!(err, Error::Corruption(_)), "got {err:?}");
        assert!(
            seen < 100_000,
            "walk must stop near the page count, saw {seen}"
        );
        assert!(matches!(t.len(&p), Err(Error::Corruption(_))));
        assert!(matches!(t.verify(&p), Err(Error::Corruption(_))));
    }

    /// Tiny deterministic PRNG so the model tests need no extra dependency.
    struct XorShift(u64);
    impl XorShift {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
        fn below(&mut self, n: u64) -> u64 {
            self.next() % n
        }
    }

    #[test]
    fn growing_a_value_in_a_full_leaf_keeps_its_neighbours() {
        // Regression: a replacement that no longer fit removed the old cell,
        // failed, and then removed slot `idx` again — deleting the next key.
        let (_d, mut p, t) = setup();
        t.insert(&mut p, b"k0", b"x").unwrap();
        for i in 1..4u32 {
            t.insert(&mut p, format!("k{i}").as_bytes(), &[1u8; 1000])
                .unwrap();
        }
        t.insert(&mut p, b"k0", &[2u8; 1024]).unwrap();
        let keys: Vec<Vec<u8>> = t.scan(&p).unwrap().into_iter().map(|(k, _)| k).collect();
        assert_eq!(
            keys,
            vec![
                b"k0".to_vec(),
                b"k1".to_vec(),
                b"k2".to_vec(),
                b"k3".to_vec()
            ]
        );
        assert_eq!(t.get(&p, b"k0").unwrap(), vec![2u8; 1024]);
        t.check(&p).unwrap();
    }

    #[test]
    fn maximum_size_keys_and_values_always_split() {
        // Regression: count-based splits could leave the incoming cell with
        // no room, failing the insert (and every later checkpoint) with Full.
        let (_d, mut p, t) = setup();
        let key = |c: u8, n: usize| {
            let mut k = vec![c; n];
            k[0] = c;
            k
        };
        t.insert(&mut p, &key(b'a', MAX_KEY_SIZE), &[0u8; 1024])
            .unwrap();
        t.insert(&mut p, &key(b'c', MAX_KEY_SIZE), &[0u8; 8])
            .unwrap();
        t.insert(&mut p, &key(b'b', MAX_KEY_SIZE), &[0u8; 1024])
            .unwrap();
        let mut rng = XorShift(7);
        for i in 0..300u32 {
            let len = 1 + rng.below(MAX_KEY_SIZE as u64) as usize;
            let mut k = vec![0u8; len];
            for b in &mut k {
                *b = rng.next() as u8;
            }
            k[0] = (i % 251) as u8;
            let v = vec![1u8; rng.below(2048) as usize];
            t.insert(&mut p, &k, &v).unwrap();
        }
        let report = t.check(&p).unwrap();
        assert!(report.depth >= 2);
    }

    #[test]
    fn deleting_everything_returns_pages_to_the_free_list() {
        let (_d, mut p, t) = setup();
        for i in 0..3000u32 {
            t.insert(&mut p, format!("key{i:06}").as_bytes(), &[9u8; 40])
                .unwrap();
        }
        let before = t.check(&p).unwrap();
        assert!(before.depth >= 2, "want a multi-level tree, got {before:?}");
        for i in 0..3000u32 {
            t.delete(&mut p, format!("key{i:06}").as_bytes()).unwrap();
        }
        let after = t.check(&p).unwrap();
        assert_eq!(after.keys, 0);
        assert_eq!(after.depth, 1, "the root must collapse back to one leaf");
        assert_eq!(after.leaf_pages, 1);
        assert_eq!(after.internal_pages, 0);
        assert_eq!(
            after.free_pages,
            before.leaf_pages + before.internal_pages - 1,
            "every page but the surviving root leaf is reusable"
        );
        // Freed pages are recycled rather than growing the file.
        let page_count = p.meta().page_count;
        for i in 0..3000u32 {
            t.insert(&mut p, format!("key{i:06}").as_bytes(), &[9u8; 40])
                .unwrap();
        }
        assert_eq!(p.meta().page_count, page_count);
        t.check(&p).unwrap();
    }

    #[test]
    fn deleting_a_middle_range_keeps_the_leaf_chain_consistent() {
        let (_d, mut p, t) = setup();
        for i in 0..2000u32 {
            t.insert(&mut p, format!("k{i:05}").as_bytes(), &[1u8; 64])
                .unwrap();
        }
        for i in 500..1500u32 {
            t.delete(&mut p, format!("k{i:05}").as_bytes()).unwrap();
        }
        let report = t.check(&p).unwrap();
        assert_eq!(report.keys, 1000);
        assert!(report.free_pages > 0);
        let keys: Vec<Vec<u8>> = t.scan(&p).unwrap().into_iter().map(|(k, _)| k).collect();
        assert_eq!(keys.len(), 1000);
        assert_eq!(keys[499], b"k00499");
        assert_eq!(keys[500], b"k01500");
    }

    #[test]
    fn range_iter_honours_bounds_and_early_stop() {
        let (_d, mut p, t) = setup();
        for i in 0..1000u32 {
            t.insert(
                &mut p,
                format!("k{i:04}").as_bytes(),
                format!("{i}").as_bytes(),
            )
            .unwrap();
        }
        let collect = |lo: Bound<&[u8]>, hi: Bound<&[u8]>| {
            let mut out = Vec::new();
            t.range_iter(&p, lo, hi, |k, _| {
                out.push(String::from_utf8(k).unwrap());
                Ok(true)
            })
            .unwrap();
            out
        };
        let r = collect(Bound::Included(b"k0100"), Bound::Excluded(b"k0105"));
        assert_eq!(r, ["k0100", "k0101", "k0102", "k0103", "k0104"]);
        let r = collect(Bound::Excluded(b"k0100"), Bound::Included(b"k0102"));
        assert_eq!(r, ["k0101", "k0102"]);
        let r = collect(Bound::Included(b"k0998x"), Bound::Unbounded);
        assert_eq!(r, ["k0999"]);
        let r = collect(Bound::Included(b"a"), Bound::Excluded(b"k0002"));
        assert_eq!(r, ["k0000", "k0001"]);
        assert!(collect(Bound::Included(b"z"), Bound::Unbounded).is_empty());

        let mut seen = 0;
        t.range_iter(&p, Bound::Unbounded, Bound::Unbounded, |_, _| {
            seen += 1;
            Ok(seen < 10)
        })
        .unwrap();
        assert_eq!(seen, 10, "returning false stops the walk");
    }

    #[test]
    fn random_operations_match_a_model() {
        use std::collections::BTreeMap;
        for (seed, fill) in [
            (1u64, FillFactor::default()),
            (99, FillFactor::new(0.5, 0.7)),
        ] {
            let dir = tempfile::tempdir().unwrap();
            let mut p = Pager::open(&dir.path().join("t.pdb"), 64).unwrap();
            let t = BTree::new(fill);
            let mut model: BTreeMap<Vec<u8>, Vec<u8>> = BTreeMap::new();
            let mut rng = XorShift(seed);
            for round in 0..20 {
                for _ in 0..400 {
                    let key = format!("{:05}", rng.below(3000)).into_bytes();
                    match rng.below(10) {
                        0..=5 => {
                            // Mix inline, near-limit and overflow values.
                            let len = match rng.below(4) {
                                0 => rng.below(16),
                                1 => 900 + rng.below(200),
                                2 => rng.below(300),
                                _ => 2000 + rng.below(9000),
                            } as usize;
                            let value = vec![(rng.next() & 0xFF) as u8; len];
                            t.insert(&mut p, &key, &value).unwrap();
                            model.insert(key, value);
                        }
                        _ => {
                            let expected = model.remove(&key).is_some();
                            match t.delete(&mut p, &key) {
                                Ok(()) => assert!(expected),
                                Err(Error::NotFound) => assert!(!expected),
                                Err(e) => panic!("delete failed: {e:?}"),
                            }
                        }
                    }
                }
                if round % 5 == 4 {
                    p.flush().unwrap();
                }
                let report = t.check(&p).unwrap();
                assert_eq!(report.keys, model.len() as u64, "seed {seed} round {round}");
            }
            let scanned = t.scan(&p).unwrap();
            let expected: Vec<(Vec<u8>, Vec<u8>)> = model.into_iter().collect();
            assert_eq!(scanned, expected, "seed {seed}");
        }
    }

    #[test]
    fn check_detects_a_key_outside_its_parent_range() {
        let (_d, mut p, t) = setup();
        for i in 0..400u32 {
            t.insert(&mut p, format!("k{i:04}").as_bytes(), &[0u8; 64])
                .unwrap();
        }
        // Rewrite the last leaf's first key to something smaller than its
        // separator: ordering inside the page still holds, the range does not.
        let mut leaf_id = t.first_leaf(&p).unwrap();
        loop {
            let leaf = p.read_page(leaf_id).unwrap();
            if leaf.extra() == SENTINEL {
                break;
            }
            leaf_id = leaf.extra();
        }
        let mut leaf = p.read_page(leaf_id).unwrap();
        let cell = Page::encode_leaf_cell(b"a", &[0u8; 64], 64, None);
        leaf.replace_cell_at(0, &cell).unwrap();
        p.write_page(leaf);
        assert!(matches!(t.check(&p), Err(Error::Corruption(_))));
    }

    #[test]
    fn sequential_inserts_fill_leaves_instead_of_halving_them() {
        let (_d, mut p, t) = setup();
        for i in 0..20_000u32 {
            t.insert(&mut p, format!("key{i:08}").as_bytes(), &[9u8; 40])
                .unwrap();
        }
        let report = t.check(&p).unwrap();
        assert!(report.depth >= 3, "{report:?}");
        assert!(
            report.underfull_leaves <= 1,
            "append-only load should pack leaves: {report:?}"
        );
        // Random order still balances.
        let (_d2, mut p2, t2) = setup();
        let mut rng = XorShift(3);
        for _ in 0..5_000 {
            let k = format!("key{:08}", rng.below(1_000_000));
            t2.insert(&mut p2, k.as_bytes(), &[9u8; 40]).unwrap();
        }
        t2.check(&p2).unwrap();
    }
}
