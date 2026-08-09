//! Cross-platform read-only memory mapping for zero-copy page reads.
//!
//! Reads are served straight out of the mapping (no `read(2)`, no copy into a
//! user buffer beyond the caller's own `Page`). Writes deliberately go through
//! ordinary positional file writes plus an explicit `fsync`, because a dirty
//! `MAP_SHARED` page has no ordering guarantee with respect to the WAL until
//! `msync` returns — mixing the two makes crash analysis far simpler.
//!
//! The mapping is re-established whenever the file grows (see
//! [`Mmap::remap`]). Callers must hold the pager lock across `remap`, since any
//! outstanding slice borrows are invalidated.

use crate::error::{Error, Result};
use std::fs::File;

#[cfg(unix)]
use std::os::unix::io::AsRawFd;
#[cfg(windows)]
use std::os::windows::io::AsRawHandle;

/// A read-only view of a file region.
///
/// `len == 0` represents "no mapping" and is always safe to hold.
pub struct Mmap {
    ptr: *mut u8,
    len: usize,
    #[cfg(windows)]
    handle: *mut std::ffi::c_void,
}

// SAFETY: the mapping is read-only for its whole lifetime and the pager
// serialises `remap`/`unmap` behind a write lock, so shared references to the
// bytes can safely cross threads.
unsafe impl Send for Mmap {}
unsafe impl Sync for Mmap {}

impl Mmap {
    /// An empty mapping.
    #[must_use]
    pub fn empty() -> Self {
        Mmap {
            ptr: std::ptr::null_mut(),
            len: 0,
            #[cfg(windows)]
            handle: std::ptr::null_mut(),
        }
    }

    /// Maps the first `len` bytes of `file` read-only.
    ///
    /// A `len` of zero yields [`Mmap::empty`].
    pub fn map(file: &File, len: usize) -> Result<Self> {
        if len == 0 {
            return Ok(Mmap::empty());
        }
        #[cfg(unix)]
        {
            // SAFETY: fd is valid for the duration of the call; a null hint lets
            // the kernel choose the address; failure is checked below.
            let ptr = unsafe {
                libc::mmap(
                    std::ptr::null_mut(),
                    len,
                    libc::PROT_READ,
                    libc::MAP_SHARED,
                    file.as_raw_fd(),
                    0,
                )
            };
            if ptr == libc::MAP_FAILED {
                return Err(Error::Io(std::io::Error::last_os_error()));
            }
            Ok(Mmap {
                ptr: ptr.cast::<u8>(),
                len,
            })
        }
        #[cfg(windows)]
        {
            use windows_sys::Win32::System::Memory::{
                MapViewOfFile, FILE_MAP_READ, PAGE_READONLY,
            };
            let high = (len >> 32) as u32;
            let low = (len & 0xFFFF_FFFF) as u32;
            // SAFETY: handle comes from a live `File`; sizes are split correctly.
            let mapping = unsafe {
                windows_sys::Win32::System::Memory::CreateFileMappingW(
                    file.as_raw_handle() as _,
                    std::ptr::null(),
                    PAGE_READONLY,
                    high,
                    low,
                    std::ptr::null(),
                )
            };
            if mapping.is_null() {
                return Err(Error::Io(std::io::Error::last_os_error()));
            }
            // SAFETY: `mapping` is a valid section handle just created above.
            let view = unsafe { MapViewOfFile(mapping, FILE_MAP_READ, 0, 0, len) };
            if view.Value.is_null() {
                // SAFETY: closing the section handle we own after a failed map.
                unsafe { windows_sys::Win32::Foundation::CloseHandle(mapping) };
                return Err(Error::Io(std::io::Error::last_os_error()));
            }
            Ok(Mmap {
                ptr: view.Value.cast::<u8>(),
                len,
                handle: mapping,
            })
        }
    }

    /// Length of the mapping in bytes.
    #[must_use]
    pub fn len(&self) -> usize {
        self.len
    }

    /// True when nothing is mapped.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Borrows `len` bytes at `offset`, or `None` when the range is outside the
    /// mapping. Never panics and never reads past the end.
    #[must_use]
    pub fn slice(&self, offset: usize, len: usize) -> Option<&[u8]> {
        let end = offset.checked_add(len)?;
        if end > self.len || self.ptr.is_null() {
            return None;
        }
        // SAFETY: bounds checked above; the region stays mapped for `&self` and
        // the pager holds a lock preventing concurrent `remap`.
        Some(unsafe { std::slice::from_raw_parts(self.ptr.add(offset), len) })
    }

    /// Unmaps and remaps at the new length. Invalidates all outstanding slices.
    pub fn remap(&mut self, file: &File, new_len: usize) -> Result<()> {
        let fresh = Mmap::map(file, new_len)?;
        *self = fresh; // dropping the old mapping unmaps it
        Ok(())
    }
}

impl Drop for Mmap {
    fn drop(&mut self) {
        if self.ptr.is_null() || self.len == 0 {
            return;
        }
        #[cfg(unix)]
        {
            // SAFETY: `ptr`/`len` are exactly what `mmap` returned.
            unsafe { libc::munmap(self.ptr.cast::<libc::c_void>(), self.len) };
        }
        #[cfg(windows)]
        {
            use windows_sys::Win32::Foundation::CloseHandle;
            use windows_sys::Win32::System::Memory::{UnmapViewOfFile, MEMORY_MAPPED_VIEW_ADDRESS};
            // SAFETY: address came from `MapViewOfFile`; handle from `CreateFileMappingW`.
            unsafe {
                UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS {
                    Value: self.ptr.cast(),
                });
                if !self.handle.is_null() {
                    CloseHandle(self.handle as _);
                }
            }
        }
        self.ptr = std::ptr::null_mut();
        self.len = 0;
    }
}

impl std::fmt::Debug for Mmap {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Mmap").field("len", &self.len).finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn maps_and_reads_back() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("m.bin");
        let mut f = File::create(&path).unwrap();
        f.write_all(&[9u8; 8192]).unwrap();
        f.sync_all().unwrap();
        drop(f);

        let f = File::open(&path).unwrap();
        let m = Mmap::map(&f, 8192).unwrap();
        assert_eq!(m.len(), 8192);
        assert_eq!(m.slice(0, 4).unwrap(), &[9, 9, 9, 9]);
        assert_eq!(m.slice(4096, 4096).unwrap().len(), 4096);
        assert!(m.slice(8192, 1).is_none(), "out-of-range read must be None");
        assert!(m.slice(usize::MAX, 1).is_none(), "overflow must be None");
    }

    #[test]
    fn empty_mapping_is_safe() {
        let m = Mmap::empty();
        assert!(m.is_empty());
        assert!(m.slice(0, 0).is_none() || m.slice(0, 0) == Some(&[][..]));
    }
}
