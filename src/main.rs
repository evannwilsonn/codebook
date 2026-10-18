//! The `codebook` command-line tool.
//!
//!   codebook train <corpus> --vocab <n> [--pattern gpt4|gpt2|minbpe|none] [--tie smallest|first]
//!                  [--naive] [--verbose] [-o out.tiktoken] [--model <prefix>]
//!   codebook encode <vocab> [--pieces] [--allow-special all|none|none_raise|<tok,tok>] [text...]
//!   codebook decode <vocab> <id>...
//!   codebook count  <vocab> <file>...
//!   codebook merges <vocab> [--limit <n>]
//!
//! <vocab> is `--gpt4` (cl100k_base), `--gpt2` (r50k_base), `--ranks <file.tiktoken>` with an
//! optional `--pattern`, or `--model <file.model>` (a minbpe model file). The OpenAI rank files
//! are read from data/ranks/, or from the directory in CODEBOOK_RANKS.
//!
//! `--tie first --pattern minbpe` (or `--pattern none`) trains exactly as minbpe's
//! RegexTokenizer (or BasicTokenizer) does, and `--verbose` prints its per-merge log.

use std::io::Read;
use std::process::ExitCode;
use std::time::Instant;

use codebook::minbpe::{
    from_model, recover_merges, render_token, to_model, to_vocab, CL100K_SPECIAL,
    GPT4_SPLIT_PATTERN, R50K_SPECIAL,
};
use codebook::train::{
    count_chunks, count_chunks_ordered, describe_merge, token_list, train_first_seen,
    train_incremental, train_naive, vocabulary,
};
use codebook::{AllowedSpecial, Rank, Tokenizer, GPT2_PATTERN, GPT4_PATTERN, NO_SPLIT};
use fancy_regex::Regex;

const USAGE: &str = "usage:
  codebook train <corpus> --vocab <n> [--pattern gpt4|gpt2|minbpe|none] [--tie smallest|first]
                 [--naive] [--verbose] [-o out.tiktoken] [--model <prefix>]
  codebook encode <vocab> [--pieces] [--allow-special all|none|none_raise|<tok,tok>] [text...]
  codebook decode <vocab> <id>...
  codebook count  <vocab> <file>...
  codebook merges <vocab> [--limit <n>]
<vocab> is --gpt4, --gpt2, --ranks <file.tiktoken> [--pattern ...], or --model <file.model>";

struct Args {
    positional: Vec<String>,
    flags: Vec<(String, Option<String>)>,
}

impl Args {
    fn parse(raw: impl Iterator<Item = String>) -> Self {
        const TAKES_VALUE: [&str; 8] = [
            "--vocab",
            "--pattern",
            "--ranks",
            "-o",
            "--model",
            "--tie",
            "--allow-special",
            "--limit",
        ];
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
        "minbpe" => Ok(GPT4_SPLIT_PATTERN),
        "none" => Ok(NO_SPLIT),
        other => Err(format!(
            "unknown pattern {other:?}; use gpt4, gpt2, minbpe or none"
        )),
    }
}

fn load_vocab(args: &Args) -> Result<Tokenizer, String> {
    if let Some(path) = args.value("--model") {
        let text = std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?;
        return from_model(&text);
    }
    let dir = std::env::var("CODEBOOK_RANKS").unwrap_or_else(|_| "data/ranks".into());
    let (path, pattern, special): (String, &str, &[(&str, Rank)]) = if args.has("--gpt4") {
        (format!("{dir}/cl100k_base.tiktoken"), GPT4_PATTERN, &CL100K_SPECIAL)
    } else if args.has("--gpt2") {
        (format!("{dir}/r50k_base.tiktoken"), GPT2_PATTERN, &R50K_SPECIAL)
    } else if let Some(p) = args.value("--ranks") {
        (p.to_owned(), pattern(args)?, &[])
    } else {
        return Err("choose a vocabulary: --gpt4, --gpt2, --ranks <file> or --model <file>".into());
    };
    let text = std::fs::read_to_string(&path).map_err(|e| format!("{path}: {e}"))?;
    let mut t = Tokenizer::from_tiktoken(&text, pattern)?;
    t.register_special_tokens(special)?;
    Ok(t)
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
    let regex = if pattern.is_empty() {
        None
    } else {
        Some(Regex::new(pattern).map_err(|e| e.to_string())?)
    };
    // Documents may be separated by the ASCII record separator; otherwise the file is one document.
    let docs = text.split('\x1e');
    let start = Instant::now();
    let (merges, pieces) = match args.value("--tie").unwrap_or("smallest") {
        // minbpe's rule: among equally frequent pairs, the one seen first in the text wins.
        "first" => {
            let pieces = count_chunks_ordered(regex.as_ref(), docs);
            (train_first_seen(&pieces, vocab - 256), pieces.len())
        }
        "smallest" => {
            let regex = regex.ok_or("--pattern none needs --tie first")?;
            let chunks = count_chunks(&regex, docs);
            let merges = if args.has("--naive") {
                train_naive(&chunks, vocab - 256)
            } else {
                train_incremental(&chunks, vocab - 256)
            };
            (merges, chunks.len())
        }
        other => return Err(format!("unknown tie rule {other:?}; use smallest or first")),
    };
    if args.has("--verbose") {
        let tokens = token_list(&merges);
        for m in &merges {
            println!("{}", describe_merge(m, merges.len(), &tokens));
        }
    }
    let t = Tokenizer::new(vocabulary(&merges), pattern)?
        .with_merges(merges.iter().map(|m| m.pair).collect());
    let mut wrote = Vec::new();
    if let Some(prefix) = args.value("--model") {
        for (ext, body) in [("model", to_model(&t)?), ("vocab", to_vocab(&t)?)] {
            let out = format!("{prefix}.{ext}");
            std::fs::write(&out, body).map_err(|e| format!("{out}: {e}"))?;
            wrote.push(out);
        }
    }
    if args.value("--model").is_none() || args.has("-o") {
        let out = args.value("-o").unwrap_or("vocab.tiktoken").to_owned();
        std::fs::write(&out, t.to_tiktoken()).map_err(|e| format!("{out}: {e}"))?;
        wrote.push(out);
    }
    eprintln!(
        "learned {} merges from {} distinct pieces in {:.1}s; wrote {} ({} tokens)",
        merges.len(),
        pieces,
        start.elapsed().as_secs_f64(),
        wrote.join(", "),
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

fn allowed_special(args: &Args) -> Option<AllowedSpecial> {
    Some(match args.value("--allow-special")? {
        "all" => AllowedSpecial::All,
        "none" => AllowedSpecial::None,
        "none_raise" => AllowedSpecial::NoneRaise,
        list => AllowedSpecial::Set(list.split(',').map(str::to_owned).collect()),
    })
}

fn encode(args: &Args) -> Result<(), String> {
    let t = load_vocab(args)?;
    let text = text_arg(args)?;
    // Without --allow-special, special-token text is encoded as ordinary text, like
    // minbpe's encode_ordinary and tiktoken's default.
    let ids = match allowed_special(args) {
        Some(allowed) => t.encode_with_special(&text, &allowed)?,
        None => t.encode(&text),
    };
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
    let ids = args.positional[1..]
        .iter()
        .map(|s| s.parse::<Rank>().map_err(|e| format!("{s:?}: {e}")))
        .collect::<Result<Vec<_>, _>>()?;
    // Bytes that aren't valid UTF-8 (half a character) print as U+FFFD, as minbpe does.
    print!("{}", t.decode_lossy(&ids)?);
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

/// Prints each merged token with the two tokens it came from, in minbpe's .vocab style. A merge
/// list is used when the vocabulary has one; GPT-2 and GPT-4 ship only ranks, so their merges
/// are recovered the way minbpe's GPT4Tokenizer does.
fn merges(args: &Args) -> Result<(), String> {
    let t = load_vocab(args)?;
    let limit: usize = match args.value("--limit") {
        Some(n) => n.parse().map_err(|e| format!("--limit: {e}"))?,
        None => usize::MAX,
    };
    let list: Vec<(Rank, Rank, Rank)> = match t.merges() {
        Some(m) => m
            .iter()
            .enumerate()
            .map(|(i, &(a, b))| (a, b, 256 + i as Rank))
            .collect(),
        None => recover_merges(t.ranks())?,
    };
    let show = |id: Rank| render_token(t.token_bytes(id).expect("token id"));
    for &(a, b, id) in list.iter().take(limit) {
        println!("[{}][{}] -> [{}] {id}", show(a), show(b), show(id));
    }
    for (s, id) in t.special_tokens() {
        println!("[{s}] {id}  (special)");
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
        Some("merges") => merges(&args),
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
