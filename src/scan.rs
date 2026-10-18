//! A hand-written scanner that splits text exactly as the GPT-4 pattern does, without a regex.
//!
//! Profiling showed the regex taking most of the encoding time, so this walks the text once
//! and applies the pattern's eight alternatives in order. The experiment checks that it gives
//! the same pieces as the regex on every test document.
//!
//! ```text
//! 1. '(?i:[sdmt]|ll|ve|re)                 contractions
//! 2. [^\r\n\p{L}\p{N}]?+\p{L}++            a word, with at most one leading non-letter
//! 3. \p{N}{1,3}+                           up to three digits
//! 4.  ?[^\s\p{L}\p{N}]++[\r\n]*+           punctuation, with an optional leading space
//! 5. \s++$                                 whitespace that runs to the end
//! 6. \s*[\r\n]                             whitespace up to its last line break
//! 7. \s+(?!\S)                             whitespace except its last character
//! 8. \s                                    one whitespace character
//! ```

use unicode_general_category::{get_general_category, GeneralCategory as Gc};

fn is_letter(c: char) -> bool {
    if c.is_ascii() {
        return c.is_ascii_alphabetic();
    }
    matches!(
        get_general_category(c),
        Gc::UppercaseLetter
            | Gc::LowercaseLetter
            | Gc::TitlecaseLetter
            | Gc::ModifierLetter
            | Gc::OtherLetter
    )
}

/// `\p{N}`: decimal digits, letter numbers (Ⅻ) and other numbers (½, ²).
fn is_number(c: char) -> bool {
    if c.is_ascii() {
        c.is_ascii_digit()
    } else {
        c.is_numeric()
    }
}

/// `\s` in Unicode mode is the White_Space property, which is what `char::is_whitespace` tests.
fn is_space(c: char) -> bool {
    c.is_whitespace()
}

fn is_newline(c: char) -> bool {
    c == '\r' || c == '\n'
}

/// Byte offset where the run of chars satisfying `f` starting at `from` ends.
fn run_end(s: &str, from: usize, f: impl Fn(char) -> bool) -> usize {
    s[from..]
        .char_indices()
        .find(|&(_, c)| !f(c))
        .map_or(s.len(), |(i, _)| from + i)
}

/// Case-insensitive match against a lowercase ASCII letter, using Unicode simple case folding:
/// the only non-ASCII character that folds to one of s, d, m, t, l, v, r, e is ſ (long s).
fn folds_to(c: char, lower: char) -> bool {
    c.to_ascii_lowercase() == lower || (lower == 's' && c == 'ſ')
}

/// Length in bytes of the piece at the start of `s` (which must not be empty).
fn piece_len(s: &str) -> usize {
    let mut chars = s.chars();
    let c0 = chars.next().expect("non-empty");
    let n0 = c0.len_utf8();

    // 1. Contractions.
    if c0 == '\'' {
        let mut rest = s[1..].chars();
        if let Some(c1) = rest.next() {
            if ['s', 'd', 'm', 't'].iter().any(|&l| folds_to(c1, l)) {
                return 1 + c1.len_utf8();
            }
            if let Some(c2) = rest.next() {
                if [('l', 'l'), ('v', 'e'), ('r', 'e')]
                    .iter()
                    .any(|&(a, b)| folds_to(c1, a) && folds_to(c2, b))
                {
                    return 1 + c1.len_utf8() + c2.len_utf8();
                }
            }
        }
    }

    // 2. A word. The optional prefix is possessive, but taking it can only fail where not
    // taking it would also fail, so a plain check is enough.
    let start = if !is_letter(c0) && !is_number(c0) && !is_newline(c0) {
        n0
    } else {
        0
    };
    let end = run_end(s, start, is_letter);
    if end > start {
        return end;
    }

    // 3. Up to three digits.
    if is_number(c0) {
        return s
            .char_indices()
            .take(3)
            .take_while(|&(_, c)| is_number(c))
            .last()
            .map_or(n0, |(i, c)| i + c.len_utf8());
    }

    // 4. Punctuation, with an optional leading space and any line breaks after it.
    let start = if c0 == ' ' { 1 } else { 0 };
    let end = run_end(s, start, |c| !is_space(c) && !is_letter(c) && !is_number(c));
    if end > start {
        return run_end(s, end, is_newline);
    }

    // Everything left starts with whitespace.
    let ws_end = run_end(s, 0, is_space);
    // 5. Whitespace to the end of the text.
    if ws_end == s.len() {
        return ws_end;
    }
    // 6. Whitespace up to and including its last line break.
    if let Some(i) = s[..ws_end].rfind(is_newline) {
        return i + 1;
    }
    // 7. All but the last whitespace character, so that one can lead the next word.
    let last = s[..ws_end]
        .chars()
        .next_back()
        .expect("whitespace")
        .len_utf8();
    if ws_end - last > 0 {
        return ws_end - last;
    }
    // 8. A single whitespace character.
    n0
}

/// The pieces of `text` under the GPT-4 (cl100k_base) pre-tokenizer.
pub fn gpt4_pieces(text: &str) -> impl Iterator<Item = &str> {
    let mut rest = text;
    std::iter::from_fn(move || {
        if rest.is_empty() {
            return None;
        }
        let n = piece_len(rest);
        let (piece, tail) = rest.split_at(n);
        rest = tail;
        Some(piece)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::GPT4_PATTERN;
    use fancy_regex::Regex;

    #[test]
    fn matches_the_regex_on_tricky_text() {
        let re = Regex::new(GPT4_PATTERN).unwrap();
        let samples = [
            "I'll pay 12345 dollars!!  ok",
            "He's here. THEY'RE gone; we'VE won’t 'tis 'ſ",
            "  leading and trailing  \n\n  ",
            "line one\r\nline two\n\n\tindented\tword",
            "fn main() {\n    let x = vec![1, 2, 3];\n}\n",
            "café naïve Ⅻ ½ ² 漢字テキスト 🎬🎬 a\u{0301}b",
            "!!!hello ...world   \u{00a0}nbsp\u{3000}ideographic",
            "'",
            " ",
            "\n",
            "x",
            "1234567",
            "  \n  x",
        ];
        for s in samples {
            let want: Vec<&str> = re.find_iter(s).map(|m| m.unwrap().as_str()).collect();
            let got: Vec<&str> = gpt4_pieces(s).collect();
            assert_eq!(got, want, "on {s:?}");
        }
    }
}
