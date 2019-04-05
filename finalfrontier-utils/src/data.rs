use std::fs::File;

use failure::{Error, ResultExt};
use memmap::{Mmap, MmapOptions};

/// Get thread-specific data.
///
/// This function will return a memory map of the corpus data. The initial
/// starting position for the given thread is also returned. This starting
/// Position will always be the beginning of a sentence.
pub fn thread_data_text(f: &File, thread: usize, n_threads: usize) -> Result<(Mmap, usize), Error> {
    assert!(
        thread < n_threads,
        "Thread {} out of index [0, {})",
        thread,
        n_threads
    );

    let size = f.metadata().context("Cannot get file metadata")?.len();
    let chunk_size = size as usize / n_threads;

    let mmap = unsafe { MmapOptions::new().map(&f)? };

    if thread == 0 {
        return Ok((mmap, 0));
    }

    let mut start = thread * chunk_size;
    while start < mmap.len() {
        let next = mmap[start];
        start += 1;
        if next == b'\n' {
            break;
        }
    }

    Ok((mmap, start))
}

/// Get thread-specific data for a CONLLX-Corpus.
///
/// This function will return a memory map of the corpus data. The initial
/// starting position for the given thread is also returned. This starting
/// Position will always be the beginning of a sentence.
pub fn thread_data_conllx(
    f: &File,
    thread: usize,
    n_threads: usize,
) -> Result<(Mmap, usize), Error> {
    assert!(
        thread < n_threads,
        "Thread {} out of index [0, {})",
        thread,
        n_threads
    );

    let size = f.metadata().context("Cannot get file metadata")?.len();
    let chunk_size = size as usize / n_threads;

    let mmap = unsafe { MmapOptions::new().map(&f)? };

    if thread == 0 {
        return Ok((mmap, 0));
    }

    let mut start = thread * chunk_size;
    while start < mmap.len() - 1 {
        let next = mmap[start];
        start += 1;
        if next == b'\n' {
            if mmap[start] == b'\n' {
                start += 1;
                break;
            }
        }
    }

    Ok((mmap, start))
}

#[cfg(test)]
mod tests {
    use std::fs::File;

    use super::{thread_data_conllx, thread_data_text};

    static CHUNKING_TEST_DATA: &str =
        "a b c\nd e f\ng h i\nj k l\nm n o\np q r\ns t u\nv w x\ny z\n";

    static CHUNKING_TEST_DATA_DEPS: &str =
        "a b c\nd e f\n\ng h i\nj k l\n\nm n o\np q r\n\ns t u\nv w x\ny z\n";

    #[test]
    fn thread_data_test() {
        let f = File::open("testdata/chunking.txt").unwrap();

        let (mmap, start) = thread_data_text(&f, 0, 3).unwrap();
        assert_eq!(
            &*mmap,
            CHUNKING_TEST_DATA.as_bytes(),
            "Memory mapping is incorrect"
        );
        assert_eq!(start, 0, "Incorrect start index");

        let (mmap, start) = thread_data_text(&f, 1, 3).unwrap();
        assert_eq!(
            &*mmap,
            CHUNKING_TEST_DATA.as_bytes(),
            "Memory mapping is incorrect"
        );
        assert_eq!(start, 18, "Incorrect start index");

        let (mmap, start) = thread_data_text(&f, 2, 3).unwrap();
        assert_eq!(
            &*mmap,
            CHUNKING_TEST_DATA.as_bytes(),
            "Memory mapping is incorrect"
        );
        assert_eq!(start, 36, "Incorrect start index");
    }

    #[test]
    fn deps_thread_data_test() {
        // file size is 55 bytes
        // starts scanning at index 19
        // first double linebreak is at 26
        // second at 39
        let f = File::open("testdata/dep_chunking.txt").unwrap();
        let (mmap, start) = thread_data_conllx(&f, 0, 3).unwrap();
        assert_eq!(
            &*mmap,
            CHUNKING_TEST_DATA_DEPS.as_bytes(),
            "Memory mapping is incorrect"
        );
        assert_eq!(start, 0, "Incorrect start index");

        let (mmap, start) = thread_data_conllx(&f, 1, 3).unwrap();
        assert_eq!(
            &*mmap,
            CHUNKING_TEST_DATA_DEPS.as_bytes(),
            "Memory mapping is incorrect"
        );
        assert_eq!(start, 26, "Incorrect start index");

        let (mmap, start) = thread_data_conllx(&f, 2, 3).unwrap();
        assert_eq!(
            &*mmap,
            CHUNKING_TEST_DATA_DEPS.as_bytes(),
            "Memory mapping is incorrect"
        );
        assert_eq!(start, 39, "Incorrect start index");
    }

    #[should_panic]
    #[test]
    fn thread_data_out_of_bounds_test() {
        let f = File::open("testdata/chunking.txt").unwrap();
        let _ = thread_data_conllx(&f, 3, 3).unwrap();
    }
}
