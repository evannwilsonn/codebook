//! Encoding and decoding with a ranked vocabulary, plus tiktoken's rank-file format.

use base64::{engine::general_purpose::STANDARD, Engine};
use fancy_regex::Regex;
use rustc_hash::FxHashMap as HashMap;

/// A token id. In a BPE vocabulary the id doubles as the merge priority: lower merges first.
pub type Rank = u32;

pub struct Tokenizer {
    encoder: HashMap<Vec<u8>, Rank>,
    decoder: HashMap<Rank, Vec<u8>>,
    regex: Regex,
    /// Split with the hand-written GPT-4 scanner instead of the regex (same pieces, faster).
    scan: bool,
}

impl Tokenizer {
    /// Builds a tokenizer from byte-string ranks and a pre-tokenizer pattern. Every single byte
    /// must have a rank, so that any text can be encoded.
    pub fn new(encoder: HashMap<Vec<u8>, Rank>, pattern: &str) -> Result<Self, String> {
        for b in 0..=255u8 {
            if !encoder.contains_key(&vec![b]) {
                return Err(format!("byte {b} has no rank"));
            }
        }
        let decoder: HashMap<Rank, Vec<u8>> =
            encoder.iter().map(|(k, &v)| (v, k.clone())).collect();
        if decoder.len() != encoder.len() {
            return Err("two byte strings share a rank".into());
        }
        let regex = Regex::new(pattern).map_err(|e| e.to_string())?;
        let scan = pattern == crate::GPT4_PATTERN;
        Ok(Self {
            encoder,
            decoder,
            regex,
            scan,
        })
    }

    /// Reads a `.tiktoken` file: one `<base64 bytes> <rank>` pair per line.
    pub fn from_tiktoken(text: &str, pattern: &str) -> Result<Self, String> {
        let mut encoder = HashMap::default();
        for (n, line) in text.lines().enumerate() {
            if line.is_empty() {
                continue;
            }
            let (token, rank) = line
                .split_once(' ')
                .ok_or(format!("line {}: expected two fields", n + 1))?;
            let bytes = STANDARD
                .decode(token)
                .map_err(|e| format!("line {}: {e}", n + 1))?;
            let rank: Rank = rank
                .trim()
                .parse()
                .map_err(|e| format!("line {}: {e}", n + 1))?;
            encoder.insert(bytes, rank);
        }
        Self::new(encoder, pattern)
    }

    /// Writes the vocabulary in `.tiktoken` format, in rank order.
    pub fn to_tiktoken(&self) -> String {
        let mut ranks: Vec<(&Vec<u8>, &Rank)> = self.encoder.iter().collect();
        ranks.sort_by_key(|&(_, r)| *r);
        let mut out = String::new();
        for (bytes, rank) in ranks {
            out.push_str(&STANDARD.encode(bytes));
            out.push(' ');
            out.push_str(&rank.to_string());
            out.push('\n');
        }
        out
    }

    pub fn vocab_size(&self) -> usize {
        self.encoder.len()
    }

    pub fn token_bytes(&self, id: Rank) -> Option<&[u8]> {
        self.decoder.get(&id).map(|v| v.as_slice())
    }

    /// Always split with the regex, even for the GPT-4 pattern. Used to measure the scanner.
    pub fn with_regex_split(mut self) -> Self {
        self.scan = false;
        self
    }

    /// The pieces the pre-tokenizer splits `text` into. Tokens never cross a piece boundary.
    pub fn pieces<'t>(&'t self, text: &'t str) -> Box<dyn Iterator<Item = &'t str> + 't> {
        if self.scan {
            Box::new(crate::scan::gpt4_pieces(text))
        } else {
            Box::new(
                self.regex
                    .find_iter(text)
                    .map(|m| m.expect("regex backtrack limit").as_str()),
            )
        }
    }

    /// Encodes text as tiktoken's `encode_ordinary` does: special tokens are treated as text.
    pub fn encode(&self, text: &str) -> Vec<Rank> {
        let mut out = Vec::with_capacity(text.len() / 4);
        for piece in self.pieces(text) {
            self.encode_piece(piece.as_bytes(), &mut out);
        }
        out
    }

    /// Like [`encode`](Self::encode), but remembers the tokens of every piece it has seen.
    /// Natural text repeats the same words constantly, so most pieces skip the merge loop.
    pub fn encode_cached(&self, text: &str, cache: &mut HashMap<String, Vec<Rank>>) -> Vec<Rank> {
        let mut out = Vec::with_capacity(text.len() / 4);
        for piece in self.pieces(text) {
            if let Some(ids) = cache.get(piece) {
                out.extend_from_slice(ids);
            } else {
                let start = out.len();
                self.encode_piece(piece.as_bytes(), &mut out);
                cache.insert(piece.to_owned(), out[start..].to_vec());
            }
        }
        out
    }

    fn encode_piece(&self, piece: &[u8], out: &mut Vec<Rank>) {
        if let Some(&rank) = self.encoder.get(piece) {
            out.push(rank);
            return;
        }
        let bounds = merge_bounds(&self.encoder, piece);
        for w in bounds.windows(2) {
            out.push(self.encoder[&piece[w[0]..w[1]]]);
        }
    }

    /// Concatenates the bytes of each token. The result is valid UTF-8 only when the ids are
    /// a whole encoding; a single token can hold part of a multi-byte character.
    pub fn decode(&self, ids: &[Rank]) -> Vec<u8> {
        let mut out = Vec::with_capacity(ids.len() * 4);
        for id in ids {
            out.extend_from_slice(&self.decoder[id]);
        }
        out
    }
}

/// Splits `piece` into tokens by BPE and returns the token boundaries (byte offsets, starting
/// at 0 and ending at `piece.len()`).
///
/// It starts from single bytes and repeatedly merges the adjacent pair whose joined bytes have
/// the lowest rank, until no adjacent pair is in the vocabulary. `parts[i]` holds the start of
/// the i-th current token and the rank of the pair it starts; after a merge only the two
/// neighbouring pair ranks change. This is the same procedure as tiktoken's `byte_pair_merge`.
pub fn merge_bounds(ranks: &HashMap<Vec<u8>, Rank>, piece: &[u8]) -> Vec<usize> {
    const NONE: Rank = Rank::MAX;
    // The rank of the token pair starting at part i, i.e. the bytes from part i to part i + 2.
    let pair_rank = |parts: &[(usize, Rank)], i: usize| -> Rank {
        if i + 2 < parts.len() {
            *ranks
                .get(&piece[parts[i].0..parts[i + 2].0])
                .unwrap_or(&NONE)
        } else {
            NONE
        }
    };

    let mut parts: Vec<(usize, Rank)> = (0..=piece.len()).map(|i| (i, NONE)).collect();
    for i in 0..parts.len() {
        parts[i].1 = pair_rank(&parts, i);
    }
    loop {
        let mut best = (NONE, 0);
        for (i, &(_, rank)) in parts.iter().enumerate() {
            if rank < best.0 {
                best = (rank, i);
            }
        }
        if best.0 == NONE {
            break;
        }
        let i = best.1;
        parts.remove(i + 1);
        parts[i].1 = pair_rank(&parts, i);
        if i > 0 {
            parts[i - 1].1 = pair_rank(&parts, i - 1);
        }
    }
    parts.into_iter().map(|(start, _)| start).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::GPT4_PATTERN;

    fn bytes_vocab() -> HashMap<Vec<u8>, Rank> {
        (0..=255u8).map(|b| (vec![b], b as Rank)).collect()
    }

    #[test]
    fn merges_lowest_rank_first() {
        let mut v = bytes_vocab();
        v.insert(b"ab".to_vec(), 256);
        v.insert(b"bc".to_vec(), 257);
        v.insert(b"abc".to_vec(), 258);
        // "ab" outranks "bc", so a|b|c -> ab|c -> abc.
        assert_eq!(merge_bounds(&v, b"abc"), vec![0, 3]);
        // Without "abc", it stops at ab|c.
        v.remove(b"abc".as_slice());
        assert_eq!(merge_bounds(&v, b"abc"), vec![0, 2, 3]);
    }

    #[test]
    fn round_trips_any_text_with_bytes_only() {
        let t = Tokenizer::new(bytes_vocab(), GPT4_PATTERN).unwrap();
        let text = "héllo, wörld 🎬 12345\n\n  tabs\tand spaces   ";
        let ids = t.encode(text);
        assert_eq!(ids.len(), text.len());
        assert_eq!(t.decode(&ids), text.as_bytes());
    }

    #[test]
    fn pre_tokenizer_splits_like_gpt4() {
        let t = Tokenizer::new(bytes_vocab(), GPT4_PATTERN).unwrap();
        let pieces: Vec<&str> = t.pieces("I'll pay 12345 dollars!!  ok").collect();
        assert_eq!(
            pieces,
            ["I", "'ll", " pay", " ", "123", "45", " dollars", "!!", " ", " ok"]
        );
    }

    #[test]
    fn tiktoken_format_round_trips() {
        let mut v = bytes_vocab();
        v.insert(b" the".to_vec(), 256);
        let t = Tokenizer::new(v, GPT4_PATTERN).unwrap();
        let again = Tokenizer::from_tiktoken(&t.to_tiktoken(), GPT4_PATTERN).unwrap();
        assert_eq!(again.encode(" the cat"), t.encode(" the cat"));
        assert_eq!(again.vocab_size(), 257);
    }

    #[test]
    fn cache_gives_the_same_ids() {
        let mut v = bytes_vocab();
        v.insert(b"ca".to_vec(), 256);
        v.insert(b" ca".to_vec(), 257);
        let t = Tokenizer::new(v, GPT4_PATTERN).unwrap();
        let mut cache = HashMap::default();
        let text = "cat cat cat scat";
        assert_eq!(t.encode_cached(text, &mut cache), t.encode(text));
        assert_eq!(t.encode_cached(text, &mut cache), t.encode(text));
    }
}
