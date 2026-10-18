//! Codebook against Karpathy's minbpe and tiktoken, on fixtures made by running them.
//!
//! tests/fixtures/minbpe.json comes from minbpe itself: the merges its BasicTokenizer and
//! RegexTokenizer learn from taylorswift.txt at vocab size 512, some encodings, the Wikipedia
//! example from the lecture, and tiktoken's output on minbpe's special-token test string. The
//! .model and .vocab files are written by minbpe's `save`.
//!
//! The GPT-4 checks need data/ranks/cl100k_base.tiktoken (scripts/get-data.sh) and are skipped
//! without it.

use codebook::encoder::{merge_bounds, merge_rounds};
use codebook::minbpe::{from_model, recover_merges, to_model, to_vocab, CL100K_SPECIAL};
use codebook::train::{count_chunks_ordered, train_first_seen, vocabulary, Merge};
use codebook::{AllowedSpecial, Rank, Tokenizer, GPT4_PATTERN, NO_SPLIT};
use fancy_regex::Regex;
use serde_json::Value;

const DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures");

fn fixture() -> Value {
    serde_json::from_str(&std::fs::read_to_string(format!("{DIR}/minbpe.json")).unwrap()).unwrap()
}

fn ids(v: &Value) -> Vec<Rank> {
    v.as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as Rank).collect()
}

/// minbpe's GPT-4 split pattern, as its saved model files record it. It differs slightly from
/// tiktoken's, which Codebook uses by default.
fn minbpe_gpt4_pattern() -> String {
    let model = std::fs::read_to_string(format!("{DIR}/regex-specials.model")).unwrap();
    model.lines().nth(1).unwrap().to_owned()
}

/// Trains like minbpe: Basic (no split) or Regex (minbpe's GPT-4 pattern), first-seen ties.
fn train(text: &str, pattern: &str, vocab_size: usize) -> (Tokenizer, Vec<Merge>) {
    let regex = (!pattern.is_empty()).then(|| Regex::new(pattern).unwrap());
    let pieces = count_chunks_ordered(regex.as_ref(), [text]);
    let merges = train_first_seen(&pieces, vocab_size - 256);
    let t = Tokenizer::new(vocabulary(&merges), pattern)
        .unwrap()
        .with_merges(merges.iter().map(|m| m.pair).collect());
    (t, merges)
}

fn cl100k() -> Option<Tokenizer> {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/data/ranks/cl100k_base.tiktoken");
    let text = std::fs::read_to_string(path).ok()?;
    let mut t = Tokenizer::from_tiktoken(&text, GPT4_PATTERN).unwrap();
    t.register_special_tokens(&CL100K_SPECIAL).unwrap();
    Some(t)
}

#[test]
fn wikipedia_example_from_the_lecture() {
    // aaabdaaabac, three merges: aa -> Z, Za -> Y, Yb -> X, giving X d X a c.
    let fx = fixture();
    for (name, pattern) in [("basic", NO_SPLIT.to_owned()), ("regex", minbpe_gpt4_pattern())] {
        let (t, _) = train("aaabdaaabac", &pattern, 259);
        assert_eq!(t.encode("aaabdaaabac"), ids(&fx[name]["wikipedia"]), "{name}");
        assert_eq!(t.encode("aaabdaaabac"), [258, 100, 258, 97, 99]);
    }
}

#[test]
fn learns_the_same_merges_as_minbpe_on_taylor_swift() {
    let fx = fixture();
    let text = std::fs::read_to_string(format!("{DIR}/taylorswift.txt")).unwrap();
    for (name, pattern) in [("basic", NO_SPLIT.to_owned()), ("regex", minbpe_gpt4_pattern())] {
        let (t, merges) = train(&text, &pattern, 512);
        let want: Vec<(Rank, Rank)> = fx[name]["merges"]
            .as_array()
            .unwrap()
            .iter()
            .map(|p| (p[0].as_u64().unwrap() as Rank, p[1].as_u64().unwrap() as Rank))
            .collect();
        let got: Vec<(Rank, Rank)> = merges.iter().map(|m| m.pair).collect();
        assert_eq!(got.len(), 256, "{name}");
        assert_eq!(got, want, "{name}: merges differ from minbpe");
        for case in fx[name]["encode"].as_array().unwrap() {
            let s = case["text"].as_str().unwrap();
            assert_eq!(t.encode(s), ids(&case["ids"]), "{name}: encode({s:?})");
            assert_eq!(t.decode(&t.encode(s)), s.as_bytes());
        }
        // The whole training text round-trips, and Basic mode does it as one 185 KB piece.
        assert_eq!(t.decode(&t.encode(&text)), text.as_bytes(), "{name}");
    }
}

#[test]
fn round_trips_minbpe_test_strings() {
    let text = std::fs::read_to_string(format!("{DIR}/taylorswift.txt")).unwrap();
    let (basic, _) = train(&text, NO_SPLIT, 300);
    let (regex, _) = train(&text, &minbpe_gpt4_pattern(), 300);
    for t in [&basic, &regex] {
        for s in ["", "?", "hello world!!!? (안녕하세요!) lol123 😉"] {
            assert_eq!(t.decode(&t.encode(s)), s.as_bytes());
        }
    }
}

#[test]
fn loads_a_model_minbpe_saved_and_encodes_the_same() {
    let fx = fixture();
    let s = fx["gpt4"]["specialsString"].as_str().unwrap();
    let model = std::fs::read_to_string(format!("{DIR}/regex-specials.model")).unwrap();
    let t = from_model(&model).unwrap();
    assert_eq!(t.special_tokens().len(), 5);
    assert_eq!(t.encode_with_special(s, &AllowedSpecial::All).unwrap(), ids(&fx["model"]["regexSpecialsAll"]));
    assert_eq!(t.encode_with_special(s, &AllowedSpecial::None).unwrap(), ids(&fx["model"]["regexSpecialsNone"]));
    let subset = AllowedSpecial::Set(vec!["<|endoftext|>".into()]);
    assert_eq!(t.encode_with_special(s, &subset).unwrap(), ids(&fx["model"]["regexSpecialsSubset"]));
    assert!(t.encode_with_special(s, &AllowedSpecial::NoneRaise).is_err());
    let all = t.encode_with_special(s, &AllowedSpecial::All).unwrap();
    assert_eq!(t.decode(&all), s.as_bytes());

    let basic = from_model(&std::fs::read_to_string(format!("{DIR}/basic-small.model")).unwrap()).unwrap();
    assert_eq!(basic.encode(s), ids(&fx["model"]["basicSmall"]));
}

#[test]
fn writes_model_and_vocab_files_byte_for_byte_like_minbpe() {
    for name in ["regex-specials", "basic-small"] {
        let model = std::fs::read_to_string(format!("{DIR}/{name}.model")).unwrap();
        let vocab = std::fs::read_to_string(format!("{DIR}/{name}.vocab")).unwrap();
        let t = from_model(&model).unwrap();
        assert_eq!(to_model(&t).unwrap(), model, "{name}.model");
        assert_eq!(to_vocab(&t).unwrap(), vocab, "{name}.vocab");
    }
}

#[test]
fn save_and_load_round_trip() {
    let text = std::fs::read_to_string(format!("{DIR}/taylorswift.txt")).unwrap();
    let (mut t, _) = train(&text, &minbpe_gpt4_pattern(), 256 + 64);
    t.register_special_tokens(&[("<|endoftext|>", 100257), ("<|fim_prefix|>", 100258)]).unwrap();
    let again = from_model(&to_model(&t).unwrap()).unwrap();
    let s = "<|endoftext|>Hello world, it's 2024!<|fim_prefix|> ok";
    let a = t.encode_with_special(s, &AllowedSpecial::All).unwrap();
    assert_eq!(again.encode_with_special(s, &AllowedSpecial::All).unwrap(), a);
    assert_eq!(again.decode(&a), s.as_bytes());
}

#[test]
fn decoding_half_a_character_gives_the_replacement_character() {
    // The lecture's example: token 128 alone is a UTF-8 continuation byte.
    let (t, _) = train("hello", NO_SPLIT, 256);
    assert_eq!(t.decode_lossy(&[128]).unwrap(), "\u{fffd}");
    assert!(t.try_decode(&[999]).is_err());
}

#[test]
fn whole_round_merging_matches_pair_by_pair_merging() {
    let text = std::fs::read_to_string(format!("{DIR}/taylorswift.txt")).unwrap();
    let (t, _) = train(&text, NO_SPLIT, 512);
    let ranks = t.ranks();
    let mut seed = 7u64;
    for _ in 0..200 {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let mut start = (seed >> 33) as usize % (text.len() - 3000);
        let mut end = start + 1 + (seed >> 20) as usize % 2500;
        while !text.is_char_boundary(start) {
            start += 1;
        }
        while !text.is_char_boundary(end) {
            end += 1;
        }
        let piece = text[start..end].as_bytes();
        let bounds = merge_bounds(ranks, piece);
        let pairwise: Vec<Rank> = bounds.windows(2).map(|w| ranks[&piece[w[0]..w[1]]]).collect();
        assert_eq!(merge_rounds(ranks, piece), pairwise);
    }
}

#[test]
fn gpt4_special_tokens_match_tiktoken() {
    let Some(t) = cl100k() else { return };
    let fx = fixture();
    let s = fx["gpt4"]["specialsString"].as_str().unwrap();
    assert_eq!(t.encode_with_special(s, &AllowedSpecial::All).unwrap(), ids(&fx["gpt4"]["allAllowed"]));
    assert_eq!(t.encode(s), ids(&fx["gpt4"]["ordinary"]));
    assert!(t.encode_with_special(s, &AllowedSpecial::NoneRaise).is_err());
}

#[test]
fn gpt4_merges_can_be_recovered_and_the_first_is_two_spaces() {
    let Some(t) = cl100k() else { return };
    let merges = recover_merges(t.ranks()).unwrap();
    assert_eq!(merges.len(), t.ranks().len() - 256);
    let (a, b, id) = merges[0];
    assert_eq!(id, 256);
    assert_eq!((t.token_bytes(a).unwrap(), t.token_bytes(b).unwrap()), (&b" "[..], &b" "[..]));
}
