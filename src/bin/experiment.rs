//! The whole evaluation, start to finish. Writes results/experiment.json and the trained
//! vocabulary tokenizers/imdb-32k.tiktoken.
//!
//!   cargo run --release --bin experiment
//!
//! Needs data/corpus/*.txt (scripts/build-corpus.py), data/ranks/*.tiktoken and the tiktoken
//! reference output in bench/out/ (bench/tiktoken_bench.py).

use std::fmt::Write as _;
use std::fs;
use std::time::Instant;

use codebook::train::{count_chunks, train_incremental, train_naive, vocabulary, Merge};
use codebook::{Rank, Tokenizer, GPT2_PATTERN, GPT4_PATTERN};
use fancy_regex::Regex;
use rustc_hash::FxHashMap as HashMap;

const SEP: char = '\x1e';
const VOCAB: usize = 32_768;
const SWEEP: [usize; 8] = [256, 512, 1024, 2048, 4096, 8192, 16_384, 32_768];
const TRAINER_CHECK_MERGES: usize = 1024;
const SPEED_RUNS: usize = 3;

struct Corpus {
    key: &'static str,
    file: &'static str,
    docs: Vec<String>,
    bytes: usize,
}

fn load(key: &'static str, file: &'static str) -> Corpus {
    let text = fs::read_to_string(format!("data/corpus/{file}.txt"))
        .unwrap_or_else(|e| panic!("data/corpus/{file}.txt: {e} (run scripts/build-corpus.py)"));
    let docs: Vec<String> = text.split(SEP).map(str::to_owned).collect();
    let bytes = docs.iter().map(String::len).sum();
    Corpus {
        key,
        file,
        docs,
        bytes,
    }
}

fn read_reference(path: &str) -> Vec<Vec<Rank>> {
    let raw =
        fs::read(path).unwrap_or_else(|e| panic!("{path}: {e} (run bench/tiktoken_bench.py)"));
    let mut words = raw
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]));
    let n = words.next().unwrap() as usize;
    (0..n)
        .map(|_| {
            let len = words.next().unwrap() as usize;
            words.by_ref().take(len).collect()
        })
        .collect()
}

fn median(mut xs: Vec<f64>) -> f64 {
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    xs[xs.len() / 2]
}

/// Token count over a corpus, plus whether every document decodes back to itself.
fn measure(t: &Tokenizer, c: &Corpus) -> (usize, bool) {
    let mut cache = HashMap::default();
    let mut tokens = 0;
    let mut round_trip = true;
    for d in &c.docs {
        let ids = t.encode_cached(d, &mut cache);
        tokens += ids.len();
        round_trip &= t.decode(&ids) == d.as_bytes();
    }
    (tokens, round_trip)
}

/// Median seconds to encode every document, with or without a piece cache.
fn time_encode(t: &Tokenizer, c: &Corpus, cached: bool) -> f64 {
    let runs = (0..SPEED_RUNS)
        .map(|_| {
            let mut cache = HashMap::default();
            let start = Instant::now();
            let mut n = 0;
            for d in &c.docs {
                n += if cached {
                    t.encode_cached(d, &mut cache).len()
                } else {
                    t.encode(d).len()
                };
            }
            std::hint::black_box(n);
            start.elapsed().as_secs_f64()
        })
        .collect();
    median(runs)
}

/// A JSON string body for a token's bytes (invalid UTF-8 shows as U+FFFD).
fn printable(bytes: &[u8]) -> String {
    let s = String::from_utf8_lossy(bytes);
    let mut out = String::new();
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => write!(out, "\\u{:04x}", c as u32).unwrap(),
            c => out.push(c),
        }
    }
    out
}

fn main() {
    fs::create_dir_all("results").unwrap();
    fs::create_dir_all("tokenizers").unwrap();
    let train = load("imdbTrain", "imdb-train");
    let tests = [load("imdbTest", "imdb-test"), load("rust", "rust")];
    println!(
        "train: {} docs, {:.1} MB",
        train.docs.len(),
        train.bytes as f64 / 1e6
    );
    for c in &tests {
        println!(
            "{}: {} docs, {:.1} MB",
            c.key,
            c.docs.len(),
            c.bytes as f64 / 1e6
        );
    }

    // Data: pre-tokenize the training reviews once.
    let regex = Regex::new(GPT4_PATTERN).unwrap();
    let start = Instant::now();
    let chunks = count_chunks(&regex, train.docs.iter().map(String::as_str));
    let chunk_seconds = start.elapsed().as_secs_f64();
    let pieces: u64 = chunks.values().sum();
    println!(
        "pre-tokenized: {pieces} pieces, {} distinct, {chunk_seconds:.1}s",
        chunks.len()
    );

    // Trainers: the naive one is the reference; the fast one must agree merge for merge.
    let start = Instant::now();
    let naive = train_naive(&chunks, TRAINER_CHECK_MERGES);
    let naive_seconds = start.elapsed().as_secs_f64();
    let start = Instant::now();
    let fast = train_incremental(&chunks, TRAINER_CHECK_MERGES);
    let fast_seconds = start.elapsed().as_secs_f64();
    let identical = naive == fast;
    println!("trainers, {TRAINER_CHECK_MERGES} merges: naive {naive_seconds:.2}s, incremental {fast_seconds:.2}s, identical: {identical}");
    assert!(
        identical,
        "the incremental trainer disagrees with the naive one"
    );

    // Model: train the full 32k vocabulary. Smaller vocabularies are prefixes of the same merges.
    let start = Instant::now();
    let merges: Vec<Merge> = train_incremental(&chunks, VOCAB - 256);
    let train_seconds = start.elapsed().as_secs_f64();
    println!("trained {} merges in {train_seconds:.1}s", merges.len());
    let full = Tokenizer::new(vocabulary(&merges), GPT4_PATTERN).unwrap();
    fs::write("tokenizers/imdb-32k.tiktoken", full.to_tiktoken()).unwrap();

    // Evaluation 1: compression on held-out text as the vocabulary grows.
    let mut sweep = Vec::new();
    for &v in &SWEEP {
        let t = Tokenizer::new(vocabulary(&merges[..v - 256]), GPT4_PATTERN).unwrap();
        let mut row = format!("    {{\"vocab\": {v}");
        for c in &tests {
            let (tokens, ok) = measure(&t, c);
            assert!(ok, "round trip failed at vocab {v} on {}", c.key);
            write!(
                row,
                ", \"{}\": {{\"tokens\": {tokens}, \"bytesPerToken\": {:.4}, \"roundTrip\": {ok}}}",
                c.key,
                c.bytes as f64 / tokens as f64
            )
            .unwrap();
            println!(
                "vocab {v:>6} {:>9}: {:.3} bytes/token",
                c.key,
                c.bytes as f64 / tokens as f64
            );
        }
        row.push('}');
        sweep.push(row);
    }

    // Evaluation 2: load OpenAI's vocabularies and match tiktoken token for token. The GPT-4
    // vocabulary is checked twice: splitting with the regex, and with the hand-written scanner.
    // Evaluation 3: speed, single thread, on the same documents tiktoken was timed on.
    let mut reference = Vec::new();
    let mut speed = Vec::new();
    let mut profile = Vec::new();
    for (name, pattern) in [("cl100k_base", GPT4_PATTERN), ("r50k_base", GPT2_PATTERN)] {
        let text = fs::read_to_string(format!("data/ranks/{name}.tiktoken")).expect("rank file");
        let scanned = Tokenizer::from_tiktoken(&text, pattern).unwrap();
        let regexed = Tokenizer::from_tiktoken(&text, pattern)
            .unwrap()
            .with_regex_split();
        let mut variants = vec![("regex", &regexed)];
        if pattern == GPT4_PATTERN {
            variants.push(("scanner", &scanned));
        }
        let mut row = format!(
            "    {{\"name\": \"{name}\", \"vocab\": {}",
            scanned.vocab_size()
        );
        for c in &tests {
            let expected = read_reference(&format!("bench/out/{name}-{}.bin", c.file));
            assert_eq!(expected.len(), c.docs.len());
            let mut fields = Vec::new();
            for (split, t) in &variants {
                let mut tokens = 0;
                let mut mismatched = 0;
                for (d, want) in c.docs.iter().zip(&expected) {
                    let got = t.encode(d);
                    tokens += got.len();
                    if &got != want {
                        mismatched += 1;
                    }
                }
                fields.push(format!(
                    "\"{split}\": {{\"tokens\": {tokens}, \"mismatched\": {mismatched}}}"
                ));
                println!("{name} {split:>7} {:>9}: {:.3} bytes/token, {mismatched} of {} documents differ from tiktoken", c.key, c.bytes as f64 / tokens as f64, c.docs.len());
                let mb = c.bytes as f64 / 1e6;
                let plain = mb / time_encode(t, c, false);
                let cached = mb / time_encode(t, c, true);
                speed.push(format!("    {{\"encoding\": \"{name}\", \"split\": \"{split}\", \"corpus\": \"{}\", \"plainMbPerSecond\": {plain:.2}, \"cachedMbPerSecond\": {cached:.2}}}", c.key));
                println!("    speed: {plain:.2} MB/s plain, {cached:.2} MB/s with piece cache");
            }
            let tokens = c
                .docs
                .iter()
                .map(|d| scanned.encode(d).len())
                .sum::<usize>();
            write!(row, ", \"{}\": {{\"tokens\": {tokens}, \"bytesPerToken\": {:.4}, \"documents\": {}, {}}}", c.key, c.bytes as f64 / tokens as f64, c.docs.len(), fields.join(", ")).unwrap();
        }
        row.push('}');
        reference.push(row);

        // Where the time goes: splitting alone versus the whole encode.
        if pattern == GPT4_PATTERN {
            let c = &tests[0];
            let mb = c.bytes as f64 / 1e6;
            for (split, t) in &variants {
                let split_only = median(
                    (0..SPEED_RUNS)
                        .map(|_| {
                            let start = Instant::now();
                            let n: usize = c.docs.iter().map(|d| t.pieces(d).count()).sum();
                            std::hint::black_box(n);
                            start.elapsed().as_secs_f64()
                        })
                        .collect(),
                );
                let whole = time_encode(t, c, false);
                profile.push(format!("    {{\"split\": \"{split}\", \"splitSeconds\": {split_only:.3}, \"encodeSeconds\": {whole:.3}, \"splitShare\": {:.3}}}", split_only / whole));
                println!("    {split}: splitting {split_only:.2}s of {whole:.2}s encode ({:.0}%) on {mb:.0} MB", 100.0 * split_only / whole);
            }
            // The scanner must produce the regex's pieces on every document, training text included.
            let mut differ = 0;
            let mut docs = 0;
            for c in [&train, &tests[0], &tests[1]] {
                for d in &c.docs {
                    docs += 1;
                    if !scanned.pieces(d).eq(regexed.pieces(d)) {
                        differ += 1;
                    }
                }
            }
            println!("    scanner vs regex pieces: {differ} of {docs} documents differ");
            assert_eq!(differ, 0, "the scanner disagrees with the regex");
            profile.push(format!(
                "    {{\"piecesCheckedDocuments\": {docs}, \"piecesDiffer\": {differ}}}"
            ));
        }
    }

    // The trained vocabulary's own speed, with the scanner.
    for c in &tests {
        let mb = c.bytes as f64 / 1e6;
        let plain = mb / time_encode(&full, c, false);
        let cached = mb / time_encode(&full, c, true);
        speed.push(format!("    {{\"encoding\": \"imdb-32k\", \"split\": \"scanner\", \"corpus\": \"{}\", \"plainMbPerSecond\": {plain:.2}, \"cachedMbPerSecond\": {cached:.2}}}", c.key));
        println!(
            "imdb-32k {:>9}: {plain:.2} MB/s plain, {cached:.2} MB/s with piece cache",
            c.key
        );
    }

    // A few tokens worth looking at: the first merges, and the longest tokens learned.
    let first: Vec<String> = merges
        .iter()
        .take(40)
        .map(|m| {
            format!(
                "    {{\"id\": {}, \"text\": \"{}\", \"count\": {}}}",
                m.id,
                printable(full.token_bytes(m.id).unwrap()),
                m.count
            )
        })
        .collect();
    let mut by_length: Vec<Rank> = (256..VOCAB as Rank).collect();
    by_length.sort_by_key(|&id| (std::cmp::Reverse(full.token_bytes(id).unwrap().len()), id));
    let longest: Vec<String> = by_length
        .iter()
        .take(12)
        .map(|&id| {
            format!(
                "{{\"id\": {id}, \"text\": \"{}\"}}",
                printable(full.token_bytes(id).unwrap())
            )
        })
        .collect();

    let json = format!(
        "{{\n  \"corpus\": {{\n    \"train\": {{\"documents\": {}, \"bytes\": {}, \"pieces\": {pieces}, \"distinctPieces\": {}, \"seconds\": {chunk_seconds:.2}}},\n    \"imdbTest\": {{\"documents\": {}, \"bytes\": {}}},\n    \"rust\": {{\"documents\": {}, \"bytes\": {}}}\n  }},\n  \"trainers\": {{\"merges\": {TRAINER_CHECK_MERGES}, \"naiveSeconds\": {naive_seconds:.2}, \"incrementalSeconds\": {fast_seconds:.3}, \"identical\": {identical}}},\n  \"training\": {{\"vocab\": {VOCAB}, \"merges\": {}, \"seconds\": {train_seconds:.2}}},\n  \"sweep\": [\n{}\n  ],\n  \"reference\": [\n{}\n  ],\n  \"speed\": [\n{}\n  ],\n  \"profile\": [\n{}\n  ],\n  \"firstMerges\": [\n{}\n  ],\n  \"longestTokens\": [{}]\n}}\n",
        train.docs.len(),
        train.bytes,
        chunks.len(),
        tests[0].docs.len(),
        tests[0].bytes,
        tests[1].docs.len(),
        tests[1].bytes,
        merges.len(),
        sweep.join(",\n"),
        reference.join(",\n"),
        speed.join(",\n"),
        profile.join(",\n"),
        first.join(",\n"),
        longest.join(", ")
    );
    fs::write("results/experiment.json", json).unwrap();
    println!("Wrote results/experiment.json and tokenizers/imdb-32k.tiktoken");
}
