//! Karpathy's minbpe file formats, so a tokenizer trained here can be read by minbpe and
//! the other way round.
//!
//! A `.model` file is what minbpe's `load` reads:
//!
//! ```text
//! minbpe v1
//! <split pattern, empty for the Basic tokenizer>
//! <number of special tokens>
//! <special token> <id>        (one line each)
//! <id1> <id2>                 (one line per merge; the merge's id is 256 + its line number)
//! ```
//!
//! A `.vocab` file is for reading only: every token rendered as text, with the two tokens it
//! was merged from (`[e][ ] -> [e ] 256`).

use std::fmt::Write;

use rustc_hash::FxHashMap as HashMap;
use unicode_general_category::{get_general_category, GeneralCategory as G};

use crate::{Rank, Tokenizer};

/// Text for a token's bytes as minbpe shows it: invalid UTF-8 becomes U+FFFD, and control and
/// other invisible characters (Unicode category C) are escaped as `\uXXXX`.
pub fn render_token(bytes: &[u8]) -> String {
    let text = String::from_utf8_lossy(bytes);
    let mut out = String::with_capacity(text.len());
    for ch in text.chars() {
        let c = matches!(
            get_general_category(ch),
            G::Control | G::Format | G::Surrogate | G::PrivateUse | G::Unassigned
        );
        if c {
            write!(out, "\\u{:04x}", ch as u32).unwrap();
        } else {
            out.push(ch);
        }
    }
    out
}

/// The merges must build ids 256, 257, ... in order, starting from the identity byte tokens,
/// which is how minbpe numbers tokens. GPT-4's vocabulary isn't numbered that way.
fn check_minbpe_numbering(t: &Tokenizer) -> Result<&[(Rank, Rank)], String> {
    let merges = t
        .merges()
        .ok_or("this tokenizer has no merge list to save (train it, or load a .model)")?;
    for b in 0..=255u8 {
        if t.ranks().get(&vec![b]) != Some(&(b as Rank)) {
            return Err("minbpe numbers byte b as token b; this vocabulary doesn't".into());
        }
    }
    if t.ranks().len() != 256 + merges.len() {
        return Err("the merge list doesn't cover the vocabulary".into());
    }
    Ok(merges)
}

/// The contents of a minbpe `.model` file.
pub fn to_model(t: &Tokenizer) -> Result<String, String> {
    let merges = check_minbpe_numbering(t)?;
    let mut out = format!("minbpe v1\n{}\n{}\n", t.pattern(), t.special_tokens().len());
    for (s, id) in t.special_tokens() {
        writeln!(out, "{s} {id}").unwrap();
    }
    for (a, b) in merges {
        writeln!(out, "{a} {b}").unwrap();
    }
    Ok(out)
}

/// The contents of a minbpe `.vocab` file: bytes, then merged tokens. Special tokens aren't
/// listed, matching minbpe: its `register_special_tokens` doesn't rebuild the vocabulary that
/// `save` writes from, so a trained tokenizer's .vocab never shows them.
pub fn to_vocab(t: &Tokenizer) -> Result<String, String> {
    let merges = check_minbpe_numbering(t)?;
    let bytes = |id: Rank| t.token_bytes(id).expect("token id");
    let mut out = String::new();
    for id in 0..256 + merges.len() as Rank {
        let s = render_token(bytes(id));
        if id >= 256 {
            let (a, b) = merges[(id - 256) as usize];
            writeln!(out, "[{}][{}] -> [{s}] {id}", render_token(bytes(a)), render_token(bytes(b))).unwrap();
        } else {
            writeln!(out, "[{s}] {id}").unwrap();
        }
    }
    Ok(out)
}

/// Reads a minbpe `.model` file.
pub fn from_model(text: &str) -> Result<Tokenizer, String> {
    let mut lines = text.split('\n');
    let mut next = |what: &str| lines.next().ok_or(format!("model file ends before the {what}"));
    if next("version")?.trim_end_matches('\r') != "minbpe v1" {
        return Err("not a minbpe v1 model file".into());
    }
    let pattern = next("pattern")?.trim_end_matches('\r').to_owned();
    let n: usize = next("special token count")?
        .trim()
        .parse()
        .map_err(|e| format!("special token count: {e}"))?;
    let mut special = Vec::with_capacity(n);
    for _ in 0..n {
        let line = next("special tokens")?.trim_end_matches('\r');
        let (s, id) = line.rsplit_once(' ').ok_or(format!("bad special token line {line:?}"))?;
        let id: Rank = id.parse().map_err(|e| format!("special token id: {e}"))?;
        special.push((s.to_owned(), id));
    }
    let mut merges = Vec::new();
    for line in lines {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let (a, b) = line.split_once(' ').ok_or(format!("bad merge line {line:?}"))?;
        let pair: (Rank, Rank) = (
            a.parse().map_err(|e| format!("merge {line:?}: {e}"))?,
            b.parse().map_err(|e| format!("merge {line:?}: {e}"))?,
        );
        let id = 256 + merges.len() as Rank;
        if pair.0 >= id || pair.1 >= id {
            return Err(format!("merge {id} uses a token that doesn't exist yet"));
        }
        merges.push(pair);
    }
    let mut bytes: Vec<Vec<u8>> = (0..=255u8).map(|b| vec![b]).collect();
    for &(a, b) in &merges {
        let mut joined = bytes[a as usize].clone();
        joined.extend_from_slice(&bytes[b as usize]);
        bytes.push(joined);
    }
    let ranks: HashMap<Vec<u8>, Rank> = bytes.into_iter().enumerate().map(|(i, b)| (b, i as Rank)).collect();
    let mut t = Tokenizer::new(ranks, &pattern)?.with_merges(merges);
    let refs: Vec<(&str, Rank)> = special.iter().map(|(s, id)| (s.as_str(), *id)).collect();
    t.register_special_tokens(&refs)?;
    Ok(t)
}

/// Recovers the merge that made each token of a ranked vocabulary that has no merge list,
/// like GPT-4's, as minbpe's `recover_merges` does: run BPE on the token's own bytes using only
/// tokens ranked below it, which leaves exactly the two tokens it was merged from.
/// Returns (left, right, token) for every multi-byte token, in rank order.
pub fn recover_merges(ranks: &HashMap<Vec<u8>, Rank>) -> Result<Vec<(Rank, Rank, Rank)>, String> {
    let mut tokens: Vec<(&Vec<u8>, Rank)> = ranks.iter().filter(|(b, _)| b.len() > 1).map(|(b, &r)| (b, r)).collect();
    tokens.sort_by_key(|&(_, r)| r);
    let mut out = Vec::with_capacity(tokens.len());
    for (bytes, rank) in tokens {
        let parts = bpe_below(ranks, bytes, rank);
        if parts.len() != 2 {
            return Err(format!("token {rank} doesn't split into two lower-ranked tokens"));
        }
        out.push((ranks[&parts[0]], ranks[&parts[1]], rank));
    }
    Ok(out)
}

/// BPE on `piece` using only tokens ranked below `max_rank`.
fn bpe_below(ranks: &HashMap<Vec<u8>, Rank>, piece: &[u8], max_rank: Rank) -> Vec<Vec<u8>> {
    let mut parts: Vec<Vec<u8>> = piece.iter().map(|&b| vec![b]).collect();
    loop {
        let mut best: Option<(Rank, usize)> = None;
        for i in 0..parts.len().saturating_sub(1) {
            let mut joined = parts[i].clone();
            joined.extend_from_slice(&parts[i + 1]);
            if let Some(&r) = ranks.get(&joined) {
                if r < max_rank && best.is_none_or(|(b, _)| r < b) {
                    best = Some((r, i));
                }
            }
        }
        let Some((_, i)) = best else { break };
        let right = parts.remove(i + 1);
        parts[i].extend_from_slice(&right);
    }
    parts
}

/// minbpe's GPT-4 split pattern. It differs from tiktoken's ([`crate::GPT4_PATTERN`]) in a few
/// possessive quantifiers and in having no `\s++$` branch, so text ending in spaces can split
/// differently. Use it to train exactly as minbpe's RegexTokenizer does.
pub const GPT4_SPLIT_PATTERN: &str = r"'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+";

/// The special tokens of `cl100k_base` (GPT-4), with tiktoken's ids.
pub const CL100K_SPECIAL: [(&str, Rank); 5] = [
    ("<|endoftext|>", 100257),
    ("<|fim_prefix|>", 100258),
    ("<|fim_middle|>", 100259),
    ("<|fim_suffix|>", 100260),
    ("<|endofprompt|>", 100276),
];

/// The special token of `r50k_base` (GPT-2): 256 bytes + 50,000 merges + this one.
pub const R50K_SPECIAL: [(&str, Rank); 1] = [("<|endoftext|>", 50256)];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn renders_like_minbpe() {
        assert_eq!(render_token(b"e "), "e ");
        assert_eq!(render_token(b"\n"), "\\u000a");
        assert_eq!(render_token(&[0xe2, 0x80]), "\u{fffd}");
        assert_eq!(render_token("\u{200b}".as_bytes()), "\\u200b"); // zero-width space is Cf
    }

    #[test]
    fn recovers_merges_of_a_small_vocabulary() {
        let mut ranks: HashMap<Vec<u8>, Rank> = (0..=255u8).map(|b| (vec![b], b as Rank)).collect();
        ranks.insert(b"ab".to_vec(), 256);
        ranks.insert(b"abc".to_vec(), 257);
        assert_eq!(recover_merges(&ranks).unwrap(), vec![(97, 98, 256), (256, 99, 257)]);
    }
}
