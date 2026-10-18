//! Codebook: a byte-level BPE tokenizer of the kind GPT models use.
//!
//! Text is first split into pieces by a regex (words, numbers, punctuation runs, whitespace),
//! then each piece is turned into token ids by repeatedly merging the adjacent pair of byte
//! strings with the lowest rank. Training learns that ranked list of merges from a corpus.
//!
//! The encoder reads and writes tiktoken's `.tiktoken` rank files, so it can load OpenAI's
//! `cl100k_base` (GPT-4) and `r50k_base` (GPT-2) vocabularies and be checked against tiktoken.

pub mod encoder;
pub mod minbpe;
pub mod scan;
pub mod train;

#[cfg(target_arch = "wasm32")]
mod wasm;

pub use encoder::{AllowedSpecial, Rank, Tokenizer};

/// The pre-tokenizer pattern of `cl100k_base`, used by GPT-3.5 and GPT-4, copied from tiktoken.
/// Letters may carry one leading non-letter (usually a space), numbers split into runs of at
/// most three digits, and a whitespace run gives its last space to the word that follows.
pub const GPT4_PATTERN: &str = r"'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}++|\p{N}{1,3}+| ?[^\s\p{L}\p{N}]++[\r\n]*+|\s++$|\s*[\r\n]|\s+(?!\S)|\s";

/// The pre-tokenizer pattern of `r50k_base`, used by GPT-2, copied from tiktoken.
pub const GPT2_PATTERN: &str =
    r"'(?:[sdmt]|ll|ve|re)| ?\p{L}++| ?\p{N}++| ?[^\s\p{L}\p{N}]++|\s++$|\s+(?!\S)|\s";

/// No pre-tokenizer: the whole text is one piece, as in minbpe's BasicTokenizer.
pub const NO_SPLIT: &str = "";
