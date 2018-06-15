use std::fs::File;

use failure::{Error, ResultExt};
use memmap::{Mmap, MmapOptions};

/// Get thread-specific data.
///
/// This function will return a memory map of the corpus data. The initial
/// starting position for the given thread is also returned. This starting
/// Position will always be the beginning of a sentence.
pub fn thread_data(f: &File, thread: usize, n_threads: usize) -> Result<(Mmap, usize), Error> {
    assert!(
        thread < n_threads,
        "Thread {} out of index [0, {})",
        thread,
        n_threads
    );

    let size = f.metadata().context("Cannot get file metadata")?.len();
    let chunk_size = size as usize / n_threads;

    let mmap = unsafe { MmapOptions::new().map(&f)? };

    let mut start = thread * chunk_size;

    // Scan forward to next newline.
    if thread != 0 {
        while start < mmap.len() {
            let next = mmap[start];
            start += 1;
            if next == b'\n' {
                break;
            }
        }
    }

    Ok((mmap, start))
}

#[cfg(test)]
mod tests {
    use std::fs::File;

    use super::thread_data;

    static CHUNKING_TEST_DATA: &str =
        "a b c\nd e f\ng h i\nj k l\nm n o\np q r\ns t u\nv w x\ny z\n";

    #[test]
    fn thread_data_test() {
        let f = File::open("testdata/chunking.txt").unwrap();

        let (mmap, start) = thread_data(&f, 0, 3).unwrap();
        assert_eq!(
            &*mmap,
            CHUNKING_TEST_DATA.as_bytes(),
            "Memory mapping is incorrect"
        );
        assert_eq!(start, 0, "Incorrect start index");

        let (mmap, start) = thread_data(&f, 1, 3).unwrap();
        assert_eq!(
            &*mmap,
            CHUNKING_TEST_DATA.as_bytes(),
            "Memory mapping is incorrect"
        );
        assert_eq!(start, 18, "Incorrect start index");

        let (mmap, start) = thread_data(&f, 2, 3).unwrap();
        assert_eq!(
            &*mmap,
            CHUNKING_TEST_DATA.as_bytes(),
            "Memory mapping is incorrect"
        );
        assert_eq!(start, 36, "Incorrect start index");
    }

    #[should_panic]
    #[test]
    fn thread_data_out_of_bounds_test() {
        let f = File::open("testdata/chunking.txt").unwrap();
        let _ = thread_data(&f, 3, 3).unwrap();
    }
}
