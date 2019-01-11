use std::fs::File;
use std::io::{self, Read, Seek, SeekFrom};

use indicatif::{ProgressBar, ProgressStyle};

pub struct FileProgress {
    inner: File,
    progress: ProgressBar,
}

/// A progress bar that implements the `Read` trait.
///
/// This wrapper of `indicatif`'s `ProgressBar` updates progress based on the
/// current offset within the file.
impl FileProgress {
    pub fn new(file: File) -> io::Result<Self> {
        let metadata = file.metadata()?;
        let progress = ProgressBar::new(metadata.len());
        progress.set_style(
            ProgressStyle::default_bar()
                .template("{bar:40} {bytes}/{total_bytes} ETA: {eta_precise}"),
        );

        Ok(FileProgress {
            inner: file,
            progress,
        })
    }
}

impl Read for FileProgress {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        let n_read = self.inner.read(buf)?;
        let pos = self.inner.seek(SeekFrom::Current(0))?;
        self.progress.set_position(pos);
        Ok(n_read)
    }
}

impl Drop for FileProgress {
    fn drop(&mut self) {
        self.progress.finish();
    }
}
