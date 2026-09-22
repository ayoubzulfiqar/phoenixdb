//! Small filesystem primitives shared by the storage files.
//!
//! Positional I/O (never moves a shared cursor, so it works through `&File`),
//! exclusive advisory locks, and directory `fsync` for durable renames.

use crate::error::{Error, Result};
use std::fs::File;
use std::path::Path;

/// Reads exactly `buf.len()` bytes at `offset`.
pub(crate) fn read_at(file: &File, buf: &mut [u8], offset: u64) -> Result<()> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::FileExt;
        file.read_exact_at(buf, offset)?;
    }
    #[cfg(windows)]
    {
        use std::os::windows::fs::FileExt;
        let mut done = 0usize;
        while done < buf.len() {
            let n = file.seek_read(&mut buf[done..], offset + done as u64)?;
            if n == 0 {
                return Err(Error::Io(std::io::ErrorKind::UnexpectedEof.into()));
            }
            done += n;
        }
    }
    #[cfg(not(any(unix, windows)))]
    {
        use std::io::{Read, Seek, SeekFrom};
        let mut f = file;
        f.seek(SeekFrom::Start(offset))?;
        f.read_exact(buf)?;
    }
    Ok(())
}

/// Writes all of `buf` at `offset`.
pub(crate) fn write_at(file: &File, buf: &[u8], offset: u64) -> Result<()> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::FileExt;
        file.write_all_at(buf, offset)?;
    }
    #[cfg(windows)]
    {
        use std::os::windows::fs::FileExt;
        let mut done = 0usize;
        while done < buf.len() {
            let n = file.seek_write(&buf[done..], offset + done as u64)?;
            if n == 0 {
                return Err(Error::Io(std::io::ErrorKind::WriteZero.into()));
            }
            done += n;
        }
    }
    #[cfg(not(any(unix, windows)))]
    {
        use std::io::{Seek, SeekFrom, Write};
        let mut f = file;
        f.seek(SeekFrom::Start(offset))?;
        f.write_all(buf)?;
    }
    Ok(())
}

/// Takes an exclusive advisory lock on `file` for as long as it stays open;
/// the OS releases it when the file is closed.
///
/// Two independent engines over one file each keep their own view of it and
/// overwrite each other's data, so a second opener must get a clean
/// [`Error::Busy`] instead. The lock covers other processes as well as a
/// second handle in this process. Filesystems that cannot lock at all (some
/// network mounts) are allowed through rather than made unusable.
pub(crate) fn lock_exclusive(file: &File, path: &Path) -> Result<()> {
    match file.try_lock() {
        Ok(()) => Ok(()),
        Err(std::fs::TryLockError::WouldBlock) => Err(Error::Busy(format!(
            "{} is already open by another handle or process",
            path.display()
        ))),
        Err(std::fs::TryLockError::Error(e)) if e.kind() == std::io::ErrorKind::Unsupported => {
            Ok(())
        }
        Err(std::fs::TryLockError::Error(e)) => Err(Error::Io(e)),
    }
}

/// Makes a rename or file creation in `path`'s directory durable.
///
/// POSIX only guarantees that a directory entry survives power loss once the
/// directory itself is `fsync`ed. Best effort: some platforms (Windows, a few
/// Android filesystems) cannot open a directory for syncing, and the rename
/// is still atomic there.
pub(crate) fn sync_parent_dir(path: &Path) {
    #[cfg(unix)]
    {
        if let Some(dir) = path.parent() {
            let dir = if dir.as_os_str().is_empty() {
                Path::new(".")
            } else {
                dir
            };
            if let Ok(d) = File::open(dir) {
                let _ = d.sync_all();
            }
        }
    }
    #[cfg(not(unix))]
    {
        let _ = path;
    }
}
