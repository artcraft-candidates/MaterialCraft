// SPDX-License-Identifier: MIT OR Apache-2.0
//! File writing for documents and exports. With the `craft` feature every write goes through a staged,
//! flushed, atomic replace, so a crash or a full disk never leaves a half-written file.

use std::{io, path::Path};

/// Write `bytes` to `path`, replacing any existing file as a whole.
pub fn write_file(path: &Path, bytes: &[u8]) -> io::Result<()> {
    #[cfg(feature = "craft")]
    {
        craft_atomic::atomic_write(path, bytes)
    }
    #[cfg(not(feature = "craft"))]
    {
        std::fs::write(path, bytes)
    }
}
