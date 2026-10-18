//! Learning BPE merges from a corpus.
//!
//! Training works on pre-tokenizer pieces, not raw text: count how often each distinct piece
//! occurs, start every piece as a list of byte tokens, then repeatedly find the most frequent
//! adjacent pair of tokens (weighted by piece count), give it a new id, and merge it everywhere.
//! Ties go to the smaller pair of ids, so both trainers below produce the same merges.
//! [`train_first_seen`] breaks ties the way minbpe does instead (the pair seen first in the
//! text wins), which reproduces minbpe's merges exactly.

use fancy_regex::Regex;
use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};
use std::cmp::Reverse;
use std::collections::BinaryHeap;

use crate::Rank;

pub type Pair = (Rank, Rank);

/// One learned merge: `pair` becomes the new token `id`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Merge {
    pub pair: Pair,
    pub id: Rank,
    /// How many times the pair occurred when it was chosen.
    pub count: u64,
}

/// Counts the distinct pieces the pre-tokenizer produces across all documents.
pub fn count_chunks<'t>(
    regex: &Regex,
    docs: impl IntoIterator<Item = &'t str>,
) -> HashMap<&'t str, u64> {
    let mut counts: HashMap<&str, u64> = HashMap::default();
    for doc in docs {
        for m in regex.find_iter(doc) {
            *counts
                .entry(m.expect("regex backtrack limit").as_str())
                .or_default() += 1;
        }
    }
    counts
}

/// Like [`count_chunks`], but keeps the pieces in the order they first appear, which
/// [`train_first_seen`] needs. With no regex each document is one piece (minbpe's Basic mode).
pub fn count_chunks_ordered<'t>(
    regex: Option<&Regex>,
    docs: impl IntoIterator<Item = &'t str>,
) -> Vec<(&'t str, u64)> {
    let mut index: HashMap<&str, usize> = HashMap::default();
    let mut out: Vec<(&str, u64)> = Vec::new();
    let mut add = |piece: &'t str| match index.get(piece) {
        Some(&i) => out[i].1 += 1,
        None => {
            index.insert(piece, out.len());
            out.push((piece, 1));
        }
    };
    for doc in docs {
        match regex {
            Some(re) => {
                for m in re.find_iter(doc) {
                    add(m.expect("regex backtrack limit").as_str());
                }
            }
            None if !doc.is_empty() => add(doc),
            None => {}
        }
    }
    out
}

/// minbpe's trainer, tie rule included: recount every pair before each merge, and among the
/// most frequent pairs take the one that appears first in the text. That is what Python's
/// `max(stats, key=stats.get)` does, since a dict keeps insertion order. Pieces must be in
/// first-appearance order, as [`count_chunks_ordered`] returns them.
pub fn train_first_seen(pieces: &[(&str, u64)], merges: usize) -> Vec<Merge> {
    let mut words: Vec<(Vec<Rank>, u64)> = pieces
        .iter()
        .map(|(p, c)| (p.bytes().map(Rank::from).collect(), *c))
        .collect();
    let mut out = Vec::with_capacity(merges);
    for step in 0..merges {
        // pair -> (count, order of first sighting)
        let mut counts: HashMap<Pair, (u64, usize)> = HashMap::default();
        let mut seen = 0usize;
        for (word, count) in &words {
            for w in word.windows(2) {
                let e = counts.entry((w[0], w[1])).or_insert_with(|| {
                    seen += 1;
                    (0, seen)
                });
                e.0 += count;
            }
        }
        let Some((&pair, &(count, _))) = counts
            .iter()
            .max_by(|a, b| a.1 .0.cmp(&b.1 .0).then(b.1 .1.cmp(&a.1 .1)))
        else {
            break;
        };
        let id = 256 + step as Rank;
        for (word, _) in words.iter_mut() {
            if word.windows(2).any(|w| (w[0], w[1]) == pair) {
                *word = merge_word(word, pair, id);
            }
        }
        out.push(Merge { pair, id, count });
    }
    out
}

/// One line of training output in minbpe's format, for `--verbose`:
/// `merge 1/256: (101, 32) -> 256 (b'e ') had 2981 occurrences`.
pub fn describe_merge(m: &Merge, total: usize, vocab: &[Vec<u8>]) -> String {
    format!(
        "merge {}/{}: ({}, {}) -> {} ({}) had {} occurrences",
        m.id - 255,
        total,
        m.pair.0,
        m.pair.1,
        m.id,
        py_bytes(&vocab[m.id as usize]),
        m.count
    )
}

/// Python's `repr()` of a bytes object.
pub fn py_bytes(b: &[u8]) -> String {
    let quote = if b.contains(&b'\'') && !b.contains(&b'"') { '"' } else { '\'' };
    let mut s = String::from("b");
    s.push(quote);
    for &c in b {
        match c {
            b'\\' => s.push_str("\\\\"),
            b'\n' => s.push_str("\\n"),
            b'\r' => s.push_str("\\r"),
            b'\t' => s.push_str("\\t"),
            _ if c as char == quote => {
                s.push('\\');
                s.push(quote);
            }
            0x20..=0x7e => s.push(c as char),
            _ => s.push_str(&format!("\\x{c:02x}")),
        }
    }
    s.push(quote);
    s
}

/// Byte strings of the tokens in id order: 0..256 are single bytes, then one per merge.
pub fn token_list(merges: &[Merge]) -> Vec<Vec<u8>> {
    let mut bytes: Vec<Vec<u8>> = (0..=255u8).map(|b| vec![b]).collect();
    for m in merges {
        let mut joined = bytes[m.pair.0 as usize].clone();
        joined.extend_from_slice(&bytes[m.pair.1 as usize]);
        bytes.push(joined);
    }
    bytes
}

/// Turns piece counts into (byte tokens, count) pairs in a fixed order, so training is
/// reproducible regardless of hash map iteration order.
fn words(chunks: &HashMap<&str, u64>) -> Vec<(Vec<Rank>, u64)> {
    let mut words: Vec<(Vec<Rank>, u64)> = chunks
        .iter()
        .map(|(piece, &count)| (piece.bytes().map(Rank::from).collect(), count))
        .collect();
    words.sort_unstable();
    words
}

fn merge_word(word: &[Rank], pair: Pair, id: Rank) -> Vec<Rank> {
    let mut out = Vec::with_capacity(word.len());
    let mut i = 0;
    while i < word.len() {
        if i + 1 < word.len() && (word[i], word[i + 1]) == pair {
            out.push(id);
            i += 2;
        } else {
            out.push(word[i]);
            i += 1;
        }
    }
    out
}

/// Better count wins; on a tie, the smaller pair wins.
fn better(a: (u64, Pair), b: (u64, Pair)) -> bool {
    a.0 > b.0 || (a.0 == b.0 && a.1 < b.1)
}

/// The textbook trainer: recount every pair in the corpus before each merge.
/// Simple and obviously correct, and the reference the fast trainer is tested against.
pub fn train_naive(chunks: &HashMap<&str, u64>, merges: usize) -> Vec<Merge> {
    let mut words = words(chunks);
    let mut out = Vec::with_capacity(merges);
    for step in 0..merges {
        let mut counts: HashMap<Pair, u64> = HashMap::default();
        for (word, count) in &words {
            for w in word.windows(2) {
                *counts.entry((w[0], w[1])).or_default() += count;
            }
        }
        let mut best: Option<(u64, Pair)> = None;
        for (&pair, &count) in &counts {
            if best.is_none_or(|b| better((count, pair), b)) {
                best = Some((count, pair));
            }
        }
        let Some((count, pair)) = best else { break };
        let id = 256 + step as Rank;
        for (word, _) in words.iter_mut() {
            if word.windows(2).any(|w| (w[0], w[1]) == pair) {
                *word = merge_word(word, pair, id);
            }
        }
        out.push(Merge { pair, id, count });
    }
    out
}

/// The fast trainer: keep pair counts, an index from each pair to the words containing it,
/// and a max-heap of pairs by count. A merge only touches the words that contain the chosen
/// pair, adjusting the counts of the pairs around it. Heap entries go stale when a count
/// changes; a popped entry is used only if it still matches the current count.
pub fn train_incremental(chunks: &HashMap<&str, u64>, merges: usize) -> Vec<Merge> {
    let mut words = words(chunks);
    let mut counts: HashMap<Pair, u64> = HashMap::default();
    let mut index: HashMap<Pair, Vec<u32>> = HashMap::default();
    for (i, (word, count)) in words.iter().enumerate() {
        for w in word.windows(2) {
            let pair = (w[0], w[1]);
            *counts.entry(pair).or_default() += count;
            let list = index.entry(pair).or_default();
            if list.last() != Some(&(i as u32)) {
                list.push(i as u32);
            }
        }
    }
    let mut heap: BinaryHeap<(u64, Reverse<Pair>)> =
        counts.iter().map(|(&p, &c)| (c, Reverse(p))).collect();
    let mut out = Vec::with_capacity(merges);
    let mut seen: Vec<u32> = vec![u32::MAX; words.len()];

    for step in 0..merges {
        // Pop until an entry matches its pair's current count.
        let (count, pair) = loop {
            let Some((count, Reverse(pair))) = heap.pop() else {
                return out;
            };
            if count > 0 && counts.get(&pair) == Some(&count) {
                break (count, pair);
            }
        };
        let id = 256 + step as Rank;
        let mut changed: HashSet<Pair> = HashSet::default();
        for i in index.remove(&pair).unwrap_or_default() {
            let i = i as usize;
            if seen[i] == step as u32 {
                continue;
            }
            seen[i] = step as u32;
            let (word, weight) = &mut words[i];
            if !word.windows(2).any(|w| (w[0], w[1]) == pair) {
                continue;
            }
            for w in word.windows(2) {
                let p = (w[0], w[1]);
                let c = counts.get_mut(&p).expect("counted pair");
                *c -= *weight;
                changed.insert(p);
            }
            *word = merge_word(word, pair, id);
            for w in word.windows(2) {
                let p = (w[0], w[1]);
                *counts.entry(p).or_default() += *weight;
                changed.insert(p);
                // Pairs without the new token were already indexed for this word.
                if p.0 == id || p.1 == id {
                    let list = index.entry(p).or_default();
                    if list.last() != Some(&(i as u32)) {
                        list.push(i as u32);
                    }
                }
            }
        }
        changed.remove(&pair);
        counts.remove(&pair);
        for p in changed {
            match counts.get(&p) {
                Some(&0) => {
                    counts.remove(&p);
                }
                Some(&c) => heap.push((c, Reverse(p))),
                None => {}
            }
        }
        out.push(Merge { pair, id, count });
    }
    out
}

/// The byte string of every token: ids 0..256 are single bytes, then one per merge.
pub fn vocabulary(merges: &[Merge]) -> HashMap<Vec<u8>, Rank> {
    token_list(merges)
        .into_iter()
        .enumerate()
        .map(|(i, b)| (b, i as Rank))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Tokenizer, GPT4_PATTERN};

    const TEXT: &str = "the cat sat on the mat. the cat ate the rat! aaaa aaa aa a \
                        The Cat? 2024, 2025 and 12345 — café café naïve";

    #[test]
    fn first_merge_is_the_most_frequent_pair() {
        let re = Regex::new(GPT4_PATTERN).unwrap();
        let chunks = count_chunks(&re, ["abab ab", "xy"]);
        let merges = train_naive(&chunks, 1);
        assert_eq!(merges[0].pair, (b'a' as Rank, b'b' as Rank));
        assert_eq!(merges[0].count, 3);
    }

    #[test]
    fn repeated_letters_merge_left_to_right() {
        assert_eq!(merge_word(&[1, 1, 1], (1, 1), 9), vec![9, 1]);
        assert_eq!(merge_word(&[1, 1, 1, 1], (1, 1), 9), vec![9, 9]);
    }

    #[test]
    fn fast_trainer_matches_naive_trainer() {
        let re = Regex::new(GPT4_PATTERN).unwrap();
        let chunks = count_chunks(&re, [TEXT, TEXT, "the theme of the thesis"]);
        let naive = train_naive(&chunks, 60);
        let fast = train_incremental(&chunks, 60);
        assert!(!naive.is_empty());
        assert_eq!(naive, fast);
    }

    #[test]
    fn trained_tokenizer_round_trips_and_compresses() {
        let re = Regex::new(GPT4_PATTERN).unwrap();
        let chunks = count_chunks(&re, [TEXT]);
        let merges = train_incremental(&chunks, 40);
        let t = Tokenizer::new(vocabulary(&merges), GPT4_PATTERN).unwrap();
        let ids = t.encode(TEXT);
        assert_eq!(t.decode(&ids), TEXT.as_bytes());
        assert!(ids.len() < TEXT.len());
        // A word the merges fully cover becomes one token.
        assert_eq!(t.encode(" cat").len(), 1);
    }
}
