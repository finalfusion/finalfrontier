use std::cmp;

/// Iterator over n-grams in a sequence.
///
/// N-grams provides an iterator over the n-grams in a sentence between a
/// minimum and maximum length.
///
/// **Warning:** no guarantee is provided with regard to the iteration
/// order. The iterator only guarantees that all n-grams are produced.
pub struct NGrams<'a, T>
where
    T: 'a,
{
    max_n: usize,
    min_n: usize,
    seq: &'a [T],
    ngram: &'a [T],
}

impl<'a, T> NGrams<'a, T> {
    /// Create a new n-ngram iterator.
    ///
    /// The iterator will create n-ngrams of length *[min_n, max_n]*
    pub fn new(seq: &'a [T], min_n: usize, max_n: usize) -> Self {
        assert!(min_n != 0, "The minimum n-gram length cannot be zero.");
        assert!(
            min_n <= max_n,
            "The maximum length should be equal to or greater than the minimum length."
        );

        let upper = cmp::min(max_n, seq.len());

        NGrams {
            min_n,
            max_n,
            seq,
            ngram: &seq[..upper],
        }
    }
}

impl<'a, T> Iterator for NGrams<'a, T> {
    type Item = &'a [T];

    fn next(&mut self) -> Option<Self::Item> {
        if self.ngram.len() < self.min_n {
            if self.seq.len() <= self.min_n {
                return None;
            }

            self.seq = &self.seq[1..];

            let upper = cmp::min(self.max_n, self.seq.len());
            self.ngram = &self.seq[..upper];
        }

        let ngram = self.ngram;

        self.ngram = &self.ngram[..self.ngram.len() - 1];

        Some(ngram)
    }
}

#[cfg(test)]
mod tests {
    use super::NGrams;

    #[test]
    fn ngrams_test() {
        let hello_chars: Vec<_> = "hellö world".chars().collect();
        let mut hello_check: Vec<&[char]> = vec![
            &['h'],
            &['h', 'e'],
            &['h', 'e', 'l'],
            &['e'],
            &['e', 'l'],
            &['e', 'l', 'l'],
            &['l'],
            &['l', 'l'],
            &['l', 'l', 'ö'],
            &['l'],
            &['l', 'ö'],
            &['l', 'ö', ' '],
            &['ö'],
            &['ö', ' '],
            &['ö', ' ', 'w'],
            &[' '],
            &[' ', 'w'],
            &[' ', 'w', 'o'],
            &['w'],
            &['w', 'o'],
            &['w', 'o', 'r'],
            &['o'],
            &['o', 'r'],
            &['o', 'r', 'l'],
            &['r'],
            &['r', 'l'],
            &['r', 'l', 'd'],
            &['l'],
            &['l', 'd'],
            &['d'],
        ];

        hello_check.sort();

        let mut hello_ngrams: Vec<_> = NGrams::new(&hello_chars, 1, 3).collect();
        hello_ngrams.sort();

        assert_eq!(hello_check, hello_ngrams);
    }

    #[test]
    fn ngrams_23_test() {
        let hello_chars: Vec<_> = "hello world".chars().collect();
        let mut hello_check: Vec<&[char]> = vec![
            &['h', 'e'],
            &['h', 'e', 'l'],
            &['e', 'l'],
            &['e', 'l', 'l'],
            &['l', 'l'],
            &['l', 'l', 'o'],
            &['l', 'o'],
            &['l', 'o', ' '],
            &['o', ' '],
            &['o', ' ', 'w'],
            &[' ', 'w'],
            &[' ', 'w', 'o'],
            &['w', 'o'],
            &['w', 'o', 'r'],
            &['o', 'r'],
            &['o', 'r', 'l'],
            &['r', 'l'],
            &['r', 'l', 'd'],
            &['l', 'd'],
        ];
        hello_check.sort();

        let mut hello_ngrams: Vec<_> = NGrams::new(&hello_chars, 2, 3).collect();
        hello_ngrams.sort();

        assert_eq!(hello_check, hello_ngrams);
    }

    #[test]
    fn empty_ngram_test() {
        let check: &[&[char]] = &[];
        assert_eq!(NGrams::<char>::new(&[], 1, 3).collect::<Vec<_>>(), check);
    }

    #[test]
    #[should_panic]
    fn incorrect_min_n_test() {
        NGrams::<char>::new(&[], 0, 3);
    }

    #[test]
    #[should_panic]
    fn incorrect_max_n_test() {
        NGrams::<char>::new(&[], 2, 1);
    }
}
