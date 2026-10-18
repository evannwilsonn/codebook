//! Encoding and decoding with a ranked vocabulary, plus tiktoken's rank-file format.

use base64::{engine::general_purpose::STANDARD, Engine};
use fancy_regex::Regex;
use rustc_hash::FxHashMap as HashMap;

/// A token id. In a BPE vocabulary the id doubles as the merge priority: lower merges first.
pub type Rank = u32;

pub struct Tokenizer {
    encoder: HashMap<Vec<u8>, Rank>,
    decoder: HashMap<Rank, Vec<u8>>,
    pattern: String,
    /// The pre-tokenizer. `None` is minbpe's BasicTokenizer: the whole text is one piece.
    regex: Option<Regex>,
    /// Split with the hand-written GPT-4 scanner instead of the regex (same pieces, faster).
    scan: bool,
    /// Special tokens in the order they were registered, and a regex that finds any of them.
    special: Vec<(String, Rank)>,
    special_regex: Option<Regex>,
    /// The merge that made each token from id 256 on, when known (training or a minbpe model).
    merges: Option<Vec<(Rank, Rank)>>,
}

/// Which special tokens [`Tokenizer::encode_with_special`] may produce, as in minbpe and
/// tiktoken: `All`, `None` (special strings are encoded as ordinary text), `NoneRaise` (the
/// default in tiktoken: it is an error for the text to contain one), or a subset.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AllowedSpecial {
    All,
    None,
    NoneRaise,
    Set(Vec<String>),
}

/// Pieces longer than this are merged a whole round at a time (minbpe's way) instead of one
/// pair at a time (tiktoken's way). The result is the same; only the cost differs.
const LONG_PIECE: usize = 1024;

impl Tokenizer {
    /// Builds a tokenizer from byte-string ranks and a pre-tokenizer pattern. Every single byte
    /// must have a rank, so that any text can be encoded. An empty pattern means no
    /// pre-tokenizer at all ([`crate::NO_SPLIT`], minbpe's BasicTokenizer).
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
        let regex = if pattern.is_empty() {
            None
        } else {
            Some(Regex::new(pattern).map_err(|e| e.to_string())?)
        };
        let scan = pattern == crate::GPT4_PATTERN;
        Ok(Self {
            encoder,
            decoder,
            pattern: pattern.to_owned(),
            regex,
            scan,
            special: Vec::new(),
            special_regex: None,
            merges: None,
        })
    }

    /// Records the merge list the vocabulary came from (training output, or a minbpe model).
    pub fn with_merges(mut self, merges: Vec<(Rank, Rank)>) -> Self {
        self.merges = Some(merges);
        self
    }

    /// The merges that build each token from id 256 on, if they were recorded.
    pub fn merges(&self) -> Option<&[(Rank, Rank)]> {
        self.merges.as_deref()
    }

    pub fn pattern(&self) -> &str {
        &self.pattern
    }

    /// Adds special tokens: strings that become one id each and never take part in merges,
    /// like `<|endoftext|>`. Their ids must not collide with ordinary tokens.
    pub fn register_special_tokens(&mut self, tokens: &[(&str, Rank)]) -> Result<(), String> {
        for &(s, id) in tokens {
            if s.is_empty() {
                return Err("a special token can't be empty".into());
            }
            if self.decoder.contains_key(&id) || self.special.iter().any(|(t, i)| t == s || *i == id) {
                return Err(format!("special token {s:?} with id {id} collides with an existing token"));
            }
            self.special.push((s.to_owned(), id));
        }
        // Longest first, so that a special token that is a prefix of another can't win.
        let mut alts: Vec<&str> = self.special.iter().map(|(s, _)| s.as_str()).collect();
        alts.sort_by_key(|s| std::cmp::Reverse(s.len()));
        let alts: Vec<String> = alts.iter().map(|s| fancy_regex::escape(s).into_owned()).collect();
        self.special_regex = Some(Regex::new(&alts.join("|")).map_err(|e| e.to_string())?);
        Ok(())
    }

    /// The special tokens, in registration order.
    pub fn special_tokens(&self) -> &[(String, Rank)] {
        &self.special
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

    /// Ordinary tokens plus special tokens.
    pub fn vocab_size(&self) -> usize {
        self.encoder.len() + self.special.len()
    }

    /// The ranked byte strings, without special tokens.
    pub fn ranks(&self) -> &HashMap<Vec<u8>, Rank> {
        &self.encoder
    }

    pub fn token_bytes(&self, id: Rank) -> Option<&[u8]> {
        self.decoder.get(&id).map(|v| v.as_slice()).or_else(|| {
            self.special
                .iter()
                .find(|(_, i)| *i == id)
                .map(|(s, _)| s.as_bytes())
        })
    }

    /// Always split with the regex, even for the GPT-4 pattern. Used to measure the scanner.
    pub fn with_regex_split(mut self) -> Self {
        self.scan = false;
        self
    }

    /// The pieces the pre-tokenizer splits `text` into. Tokens never cross a piece boundary.
    pub fn pieces<'t>(&'t self, text: &'t str) -> Box<dyn Iterator<Item = &'t str> + 't> {
        match &self.regex {
            _ if self.scan => Box::new(crate::scan::gpt4_pieces(text)),
            Some(regex) => Box::new(
                regex
                    .find_iter(text)
                    .map(|m| m.expect("regex backtrack limit").as_str()),
            ),
            None => Box::new((!text.is_empty()).then_some(text).into_iter()),
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

    /// Encodes text that may contain special tokens, following minbpe's `encode`. Text between
    /// the allowed special tokens is encoded as ordinary text.
    pub fn encode_with_special(&self, text: &str, allowed: &AllowedSpecial) -> Result<Vec<Rank>, String> {
        let allowed: Vec<&(String, Rank)> = match allowed {
            AllowedSpecial::All => self.special.iter().collect(),
            AllowedSpecial::None => Vec::new(),
            AllowedSpecial::NoneRaise => {
                if let Some((s, _)) = self.special.iter().find(|(s, _)| text.contains(s.as_str())) {
                    return Err(format!("the text contains the special token {s:?}, which isn't allowed"));
                }
                Vec::new()
            }
            AllowedSpecial::Set(names) => {
                for n in names {
                    if !self.special.iter().any(|(s, _)| s == n) {
                        return Err(format!("{n:?} is not a special token of this tokenizer"));
                    }
                }
                self.special.iter().filter(|(s, _)| names.contains(s)).collect()
            }
        };
        if allowed.is_empty() {
            return Ok(self.encode(text));
        }
        let mut alts: Vec<&str> = allowed.iter().map(|(s, _)| s.as_str()).collect();
        alts.sort_by_key(|s| std::cmp::Reverse(s.len()));
        let alts: Vec<String> = alts.iter().map(|s| fancy_regex::escape(s).into_owned()).collect();
        let re = Regex::new(&alts.join("|")).map_err(|e| e.to_string())?;
        let mut out = Vec::new();
        let mut last = 0;
        for m in re.find_iter(text) {
            let m = m.map_err(|e| e.to_string())?;
            out.extend(self.encode(&text[last..m.start()]));
            out.push(allowed.iter().find(|(s, _)| s == m.as_str()).expect("matched a special").1);
            last = m.end();
        }
        out.extend(self.encode(&text[last..]));
        Ok(out)
    }

    fn encode_piece(&self, piece: &[u8], out: &mut Vec<Rank>) {
        if let Some(&rank) = self.encoder.get(piece) {
            out.push(rank);
            return;
        }
        if piece.len() > LONG_PIECE {
            out.extend(merge_rounds(&self.encoder, piece));
            return;
        }
        let bounds = merge_bounds(&self.encoder, piece);
        for w in bounds.windows(2) {
            out.push(self.encoder[&piece[w[0]..w[1]]]);
        }
    }

    /// Concatenates the bytes of each token. The result is valid UTF-8 only when the ids are
    /// a whole encoding; a single token can hold part of a multi-byte character.
    /// Panics on an id that isn't in the vocabulary; see [`try_decode`](Self::try_decode).
    pub fn decode(&self, ids: &[Rank]) -> Vec<u8> {
        self.try_decode(ids).expect("unknown token id")
    }

    /// Like [`decode`](Self::decode), but an unknown id is an error rather than a panic.
    pub fn try_decode(&self, ids: &[Rank]) -> Result<Vec<u8>, String> {
        let mut out = Vec::with_capacity(ids.len() * 4);
        for &id in ids {
            out.extend_from_slice(self.token_bytes(id).ok_or(format!("{id} is not a token id"))?);
        }
        Ok(out)
    }

    /// Decodes to text, replacing bytes that aren't valid UTF-8 with U+FFFD, as minbpe does with
    /// `errors="replace"`. Decoding token 128 alone, half of a two-byte character, gives "\u{fffd}".
    pub fn decode_lossy(&self, ids: &[Rank]) -> Result<String, String> {
        Ok(String::from_utf8_lossy(&self.try_decode(ids)?).into_owned())
    }
}

/// Encodes a piece minbpe's way: each round, find the lowest-rank adjacent pair and merge every
/// occurrence of it, left to right. A merge can only create pairs whose token was added later,
/// with a higher rank, so this gives the same tokens as [`merge_bounds`], in time proportional
/// to the piece length times the number of rounds rather than to its square.
pub fn merge_rounds(ranks: &HashMap<Vec<u8>, Rank>, piece: &[u8]) -> Vec<Rank> {
    let mut spans: Vec<(usize, usize)> = (0..piece.len()).map(|i| (i, i + 1)).collect();
    loop {
        let mut best = Rank::MAX;
        for w in spans.windows(2) {
            if let Some(&r) = ranks.get(&piece[w[0].0..w[1].1]) {
                best = best.min(r);
            }
        }
        if best == Rank::MAX {
            break;
        }
        let mut next = Vec::with_capacity(spans.len());
        let mut i = 0;
        while i < spans.len() {
            if i + 1 < spans.len() && ranks.get(&piece[spans[i].0..spans[i + 1].1]) == Some(&best) {
                next.push((spans[i].0, spans[i + 1].1));
                i += 2;
            } else {
                next.push(spans[i]);
                i += 1;
            }
        }
        spans = next;
    }
    spans.iter().map(|&(a, b)| ranks[&piece[a..b]]).collect()
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
