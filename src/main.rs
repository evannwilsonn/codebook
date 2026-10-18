//! The `codebook` command-line tool.
//!
//!   codebook train <corpus> --vocab <n> [--pattern gpt4|gpt2] [--naive] [-o out.tiktoken]
//!   codebook encode <vocab> [--pieces] [text...]      (reads stdin when no text is given)
//!   codebook decode <vocab> <id>...
//!   codebook count  <vocab> <file>...
//!
//! <vocab> is `--gpt4` (cl100k_base), `--gpt2` (r50k_base), or `--ranks <file.tiktoken>`
//! with an optional `--pattern gpt4|gpt2` (default gpt4). The OpenAI rank files are read from
//! data/ranks/, or from the directory in CODEBOOK_RANKS.

use std::io::Read;
use std::process::ExitCode;
use std::time::Instant;

use codebook::train::{count_chunks, train_incremental, train_naive, vocabulary};
use codebook::{Rank, Tokenizer, GPT2_PATTERN, GPT4_PATTERN};
use fancy_regex::Regex;

const USAGE: &str = "usage:
  codebook train <corpus> --vocab <n> [--pattern gpt4|gpt2] [--naive] [-o out.tiktoken]
  codebook encode <vocab> [--pieces] [text...]
  codebook decode <vocab> <id>...
  codebook count  <vocab> <file>...
<vocab> is --gpt4, --gpt2, or --ranks <file.tiktoken> [--pattern gpt4|gpt2]";

struct Args {
    positional: Vec<String>,
    flags: Vec<(String, Option<String>)>,
}

impl Args {
    fn parse(raw: impl Iterator<Item = String>) -> Self {
        const TAKES_VALUE: [&str; 4] = ["--vocab", "--pattern", "--ranks", "-o"];
        let mut positional = Vec::new();
        let mut flags = Vec::new();
        let mut raw = raw.peekable();
        while let Some(a) = raw.next() {
            if TAKES_VALUE.contains(&a.as_str()) {
                flags.push((a, raw.next()));
            } else if a.starts_with('-') && a.len() > 1 && a.parse::<f64>().is_err() {
                flags.push((a, None));
            } else {
                positional.push(a);
            }
        }
        Self { positional, flags }
    }
    fn has(&self, name: &str) -> bool {
        self.flags.iter().any(|(f, _)| f == name)
    }
    fn value(&self, name: &str) -> Option<&str> {
        self.flags
            .iter()
            .find(|(f, _)| f == name)
            .and_then(|(_, v)| v.as_deref())
    }
}

fn pattern(args: &Args) -> Result<&'static str, String> {
    match args.value("--pattern").unwrap_or("gpt4") {
        "gpt4" => Ok(GPT4_PATTERN),
        "gpt2" => Ok(GPT2_PATTERN),
        other => Err(format!("unknown pattern {other:?}; use gpt4 or gpt2")),
    }
}

fn load_vocab(args: &Args) -> Result<Tokenizer, String> {
    let dir = std::env::var("CODEBOOK_RANKS").unwrap_or_else(|_| "data/ranks".into());
    let (path, pattern) = if args.has("--gpt4") {
        (format!("{dir}/cl100k_base.tiktoken"), GPT4_PATTERN)
    } else if args.has("--gpt2") {
        (format!("{dir}/r50k_base.tiktoken"), GPT2_PATTERN)
    } else if let Some(p) = args.value("--ranks") {
        (p.to_owned(), pattern(args)?)
    } else {
        return Err("choose a vocabulary: --gpt4, --gpt2 or --ranks <file>".into());
    };
    let text = std::fs::read_to_string(&path).map_err(|e| format!("{path}: {e}"))?;
    Tokenizer::from_tiktoken(&text, pattern)
}

fn train(args: &Args) -> Result<(), String> {
    let path = args.positional.get(1).ok_or("train: give a corpus file")?;
    let vocab: usize = args
        .value("--vocab")
        .ok_or("train: give --vocab <n>")?
        .parse()
        .map_err(|e| format!("--vocab: {e}"))?;
    if vocab < 256 {
        return Err("--vocab must be at least 256 (one token per byte)".into());
    }
    let pattern = pattern(args)?;
    let text = std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?;
    // Documents may be separated by the ASCII record separator; otherwise the file is one document.
    let regex = Regex::new(pattern).map_err(|e| e.to_string())?;
    let start = Instant::now();
    let chunks = count_chunks(&regex, text.split('\x1e'));
    let merges = if args.has("--naive") {
        train_naive(&chunks, vocab - 256)
    } else {
        train_incremental(&chunks, vocab - 256)
    };
    let t = Tokenizer::new(vocabulary(&merges), pattern)?;
    let out = args.value("-o").unwrap_or("vocab.tiktoken");
    std::fs::write(out, t.to_tiktoken()).map_err(|e| format!("{out}: {e}"))?;
    eprintln!(
        "learned {} merges from {} distinct pieces in {:.1}s; wrote {out} ({} tokens)",
        merges.len(),
        chunks.len(),
        start.elapsed().as_secs_f64(),
        t.vocab_size()
    );
    Ok(())
}

fn text_arg(args: &Args) -> Result<String, String> {
    if args.positional.len() > 1 {
        return Ok(args.positional[1..].join(" "));
    }
    let mut s = String::new();
    std::io::stdin()
        .read_to_string(&mut s)
        .map_err(|e| e.to_string())?;
    Ok(s)
}

fn encode(args: &Args) -> Result<(), String> {
    let t = load_vocab(args)?;
    let text = text_arg(args)?;
    let ids = t.encode(&text);
    if args.has("--pieces") {
        for id in &ids {
            println!(
                "{id:>7}  {:?}",
                String::from_utf8_lossy(t.token_bytes(*id).unwrap())
            );
        }
        eprintln!("{} bytes, {} tokens", text.len(), ids.len());
    } else {
        println!(
            "{}",
            ids.iter()
                .map(Rank::to_string)
                .collect::<Vec<_>>()
                .join(" ")
        );
    }
    Ok(())
}

fn decode(args: &Args) -> Result<(), String> {
    let t = load_vocab(args)?;
    let mut ids = Vec::new();
    for s in &args.positional[1..] {
        let id: Rank = s.parse().map_err(|e| format!("{s:?}: {e}"))?;
        if t.token_bytes(id).is_none() {
            return Err(format!("{id} is not in this vocabulary"));
        }
        ids.push(id);
    }
    print!("{}", String::from_utf8_lossy(&t.decode(&ids)));
    Ok(())
}

fn count(args: &Args) -> Result<(), String> {
    let t = load_vocab(args)?;
    if args.positional.len() < 2 {
        return Err("count: give one or more files".into());
    }
    for path in &args.positional[1..] {
        let text = std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?;
        let tokens: usize = text.split('\x1e').map(|d| t.encode(d).len()).sum();
        let bytes = text.len() - text.matches('\x1e').count();
        println!(
            "{path}: {bytes} bytes, {tokens} tokens, {:.3} bytes per token",
            bytes as f64 / tokens.max(1) as f64
        );
    }
    Ok(())
}

fn main() -> ExitCode {
    let args = Args::parse(std::env::args().skip(1));
    let result = match args.positional.first().map(String::as_str) {
        Some("train") => train(&args),
        Some("encode") => encode(&args),
        Some("decode") => decode(&args),
        Some("count") => count(&args),
        _ => Err(USAGE.into()),
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("{e}");
            ExitCode::FAILURE
        }
    }
}
