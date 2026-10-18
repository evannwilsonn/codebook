# Codebook

A byte-pair-encoding (BPE) tokenizer, the kind GPT models use, written from scratch in Rust. It learns a vocabulary from a corpus, encodes and decodes text, reads OpenAI's `.tiktoken` vocabulary files, and produces the same tokens as [tiktoken](https://github.com/openai/tiktoken) on every test document. It ships as a library crate, a command-line tool and a WebAssembly build that runs the demo in the browser.

Demo: `docs/index.html` (served with GitHub Pages).

Project 8 of my AI project run. Like the others, it follows seven steps: problem, data, baseline, model, evaluation, improvement, deployment.

## 1. Problem

A language model doesn't read characters; it reads token ids from a fixed vocabulary. A tokenizer decides how text becomes those ids. A good one:

- **Is lossless.** Decoding the ids gives back the exact bytes.
- **Compresses.** Fewer tokens per byte means shorter sequences, so cheaper and faster models.
- **Is fast.** Every prompt and every training document goes through it.

The name comes from telegraph code books: companies charged by the word, so businesses swapped common phrases for single code words. BPE does the same with byte strings.

The metric is **bytes per token** on text the tokenizer never saw while learning (higher is better), plus exact agreement with tiktoken and encoding throughput.

## 2. Data

- **Training text:** the 25,000 IMDB training reviews (32.6 MB, Maas et al. 2011), with `<br />` turned into line breaks.
- **Held-out, same kind of text:** the 25,000 IMDB test reviews (31.9 MB).
- **Held-out, different kind of text:** 7.9 MB of Rust source from the standard library (`alloc` and `std`, via `rustup component add rust-src`).
- **Reference vocabularies:** OpenAI's `cl100k_base` (GPT-3.5/GPT-4) and `r50k_base` (GPT-2), downloaded from tiktoken's public URLs.

None of this is committed. `scripts/get-data.sh` and `scripts/build-corpus.py` rebuild it; see [`data/README.md`](data/README.md).

Before training, the text is split into pieces with GPT-4's pre-tokenizer regex (words with their leading space, numbers in runs of at most three digits, punctuation, whitespace). Tokens never cross a piece boundary. The training reviews contain 6.76 million pieces, 123,930 of them distinct.

## 3. Baseline

**Bytes only:** 256 tokens, one per byte, so 1.00 bytes per token on any text. It is lossless and needs no training, but every character costs at least one token.

## 4. Model

Byte-level BPE ([`src/train.rs`](src/train.rs), [`src/encoder.rs`](src/encoder.rs)):

- **Training** starts every piece as a list of bytes, then repeatedly finds the most frequent adjacent pair (weighted by how often the piece occurs), gives it the next id, and merges it everywhere. Ties go to the smaller pair, so training is deterministic.
- **Encoding** splits text into pieces, then merges each piece's bytes in rank order, lowest rank first. This is the same procedure tiktoken uses, so a tiktoken rank file works unchanged.

One run learned 32,512 merges, for a 32,768-token vocabulary. Smaller vocabularies are prefixes of the same merge list, so one run gives the whole sweep below. The trained vocabulary is in `tokenizers/imdb-32k.tiktoken`.

## 5. Evaluation

**Compression, bytes per token on held-out text** (higher is better):

| Vocabulary | Entries | IMDB test reviews | Rust source |
|---|---|---|---|
| Bytes only (baseline) | 256 | 1.00 | 1.00 |
| Codebook | 1,024 | 2.72 | 1.48 |
| Codebook | 4,096 | 3.66 | 1.65 |
| Codebook | 16,384 | 4.33 | 1.89 |
| **Codebook** | **32,768** | **4.54** | **1.96** |
| GPT-2, `r50k_base` | 50,256 | 4.45 | 2.32 |
| GPT-4, `cl100k_base` | 100,256 | 4.52 | 3.77 |

With a third of GPT-4's entries, the IMDB-trained vocabulary is slightly better on movie reviews (7,016,155 tokens vs 7,054,295) and about half as good on code. It never saw indentation, `snake_case` or `::`, so it spells them out a few bytes at a time. A tokenizer is a compression scheme fitted to its training text. Every row decodes back to the original bytes on every document.

**Agreement with tiktoken 0.14.0.** I loaded OpenAI's rank files into this encoder and compared ids with tiktoken's `encode_ordinary`, one document at a time:

| Vocabulary | Test documents | Documents that differ |
|---|---|---|
| `cl100k_base` | 25,795 (25,000 reviews + 795 Rust files) | **0** |
| `r50k_base` | 25,795 | **0** |

**Speed**, single thread, median of three passes over the same documents, `cl100k_base`:

| Encoder | IMDB reviews | Rust source |
|---|---|---|
| tiktoken 0.14.0, called from Python | 15.7 MB/s | 10.6 MB/s |
| Codebook, regex split | 16.1 MB/s | 10.0 MB/s |

Same algorithm, same regex engine (`fancy-regex`), roughly the same speed. tiktoken's time includes building Python lists of ids, so this is not a pure Rust-to-Rust comparison.

**Agreement with minbpe.** After the first version I worked through Andrej Karpathy's lecture *Let's build the GPT Tokenizer* and his reference code, [minbpe](https://github.com/karpathy/minbpe), and checked Codebook against it. Codebook's own trainer breaks ties between equally frequent pairs by taking the smaller pair. minbpe takes the pair it saw first, because Python dicts keep insertion order. `--tie first` reproduces that. [`tests/minbpe.rs`](tests/minbpe.rs) runs on fixtures that [`scripts/minbpe-fixtures.py`](scripts/minbpe-fixtures.py) writes by running minbpe and tiktoken:

| Check | Result |
|---|---|
| Merges learned from minbpe's test text (186 KB), vocab 512, Basic (no split) and Regex (GPT-4 split) | all 256 the same, in the same order, for both |
| The lecture's Wikipedia example, `aaabdaaabac` | `[258, 100, 258, 97, 99]`, as minbpe |
| `--verbose` training log, `.model` and `.vocab` files | byte-for-byte the same as minbpe's `train.py` and `save()` |
| Loading a `.model` that minbpe saved, then encoding with special tokens allowed (`all`, `none`, a subset) | same ids; `none_raise` errors as minbpe does |
| GPT-4 special tokens (`<\|endoftext\|>`, the FIM tokens, `<\|endofprompt\|>`) | same ids as tiktoken with `allowed_special="all"` |
| GPT-4's merge list, recovered from its ranks as minbpe's `GPT4Tokenizer` does | all 100,000 recovered; the first is two spaces |

Training the same way, with every pair recounted before each merge, took 0.2 s (Basic) and 0.1 s (Regex) here, against 8.4 s and 12.7 s for minbpe in Python on the same machine.

Two more things from the lecture are now in the encoder:

- **Merge rounds.** minbpe merges every occurrence of the lowest-ranked pair at once, then looks again. Codebook merges one pair at a time. Both give the same tokens, and a test checks that on 200 random slices of text. For pieces over 1 KB (Basic mode turns a whole file into one piece) the encoder switches to rounds, because merging one pair at a time is quadratic in piece length.
- **Half a character.** A token like 128 is a lone UTF-8 continuation byte. Decoding it alone isn't valid text, so `decode` in the CLI prints U+FFFD for it, as the lecture shows, and `try_decode` reports an unknown id as an error instead of panicking.

## 6. Improvement

**Training.** The naive trainer recounts every pair in the corpus before each merge. It took 5.66 s for the first 1,024 merges. The incremental trainer keeps:

- running pair counts;
- an index from each pair to the words that contain it;
- a max-heap of pairs, with stale entries skipped.

A merge then only touches the words that contain the chosen pair. It produced the **same 1,024 merges in the same order** in 0.23 s, and all 32,512 merges in 0.51 s. The experiment asserts that the two trainers agree.

**Encoding.**
- **Profiling.** With the regex split, 77% of encoding time went to splitting the text into pieces: 1.59 of 2.07 s for the IMDB test set. The merging was the small part.
- **First idea, a piece cache.** Remember the tokens of every piece already seen. It made things slower (14.2 MB/s), because looking up and storing strings costs more than the merges it saves.
- **The fix, a scanner.** [`src/scan.rs`](src/scan.rs) is a hand-written scanner that applies the GPT-4 pattern's eight alternatives in order, in one pass, with no regex.
- **Checking the scanner.** It gives the regex's pieces on all 50,795 documents (training text included). It also still matches tiktoken on every test document, as the table above shows.

Results:

| Encoder, `cl100k_base` | Splitting time (IMDB) | IMDB reviews | Rust source |
|---|---|---|---|
| Codebook, regex split | 1.59 s | 16.1 MB/s | 10.0 MB/s |
| **Codebook, scanner** | **0.14 s** | **72.7 MB/s** | **54.5 MB/s** |

That is about 4.6× tiktoken's throughput on reviews and 5.2× on code, on one thread. The scanner is used automatically whenever the pattern is GPT-4's. The GPT-2 pattern still uses the regex.

## 7. Deployment

**Crate:** `codebook::{Tokenizer, train, minbpe}`, with `Tokenizer::from_tiktoken`, `encode`, `encode_with_special`, `register_special_tokens`, `decode`, `try_decode`, `decode_lossy`, `to_tiktoken`; `train::{count_chunks, train_incremental, train_first_seen, vocabulary}`; and `minbpe::{from_model, to_model, to_vocab, recover_merges}` for minbpe's file formats.

**CLI:**

```sh
cargo build --release
./target/release/codebook train data/corpus/imdb-train.txt --vocab 32768 -o my.tiktoken
./target/release/codebook encode --gpt4 --pieces "Hello there, tokenizer!"
./target/release/codebook encode --gpt4 --allow-special all "<|endoftext|>Hello"
./target/release/codebook decode --gpt4 9906 1070
./target/release/codebook count --ranks tokenizers/imdb-32k.tiktoken data/corpus/imdb-test.txt
./target/release/codebook merges --gpt4 --limit 20        # GPT-4's first merges, recovered

# Train exactly as minbpe's train.py, then use the model it writes
./target/release/codebook train tests/fixtures/taylorswift.txt --vocab 512 \
    --pattern minbpe --tie first --verbose --model regex   # writes regex.model, regex.vocab
./target/release/codebook encode --model regex.model "hello world"
```

`--pattern none` is minbpe's BasicTokenizer (no split). `--pattern minbpe` is its GPT-4 split pattern, which differs slightly from tiktoken's at the ends of lines.

**Browser:**
- The crate compiles to `wasm32-unknown-unknown` with a small C-style interface (`src/wasm.rs`, no bindings crate).
- `docs/index.html` embeds the gzip-compressed wasm module and both vocabularies (1.9 MB in total), and encodes as you type with both this vocabulary and GPT-4's.
- Nothing is sent anywhere.

## Run it

```sh
./scripts/get-data.sh                     # IMDB reviews (or scripts/get-data.ps1 on Windows)
rustup component add rust-src             # the Rust source used as out-of-domain text
python3 scripts/build-corpus.py           # writes data/corpus/*.txt
mkdir -p data/ranks && cd data/ranks && \
  curl -O https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken && \
  curl -O https://openaipublic.blob.core.windows.net/encodings/r50k_base.tiktoken && cd ../..
pip install tiktoken && python3 bench/tiktoken_bench.py   # reference ids and speed
cargo test --release                      # 12 unit tests + 10 minbpe checks
python3 scripts/minbpe-fixtures.py path/to/minbpe   # optional: rebuild tests/fixtures
cargo run --release --bin experiment      # about 80 s; writes results/experiment.json
rustup target add wasm32-unknown-unknown
python3 scripts/build-demo.py             # writes docs/index.html
```

## Layout

```
src/encoder.rs           encode, decode, tiktoken rank files, the merge loop
src/scan.rs              the hand-written GPT-4 pre-tokenizer
src/train.rs             naive, incremental and minbpe-style BPE training
src/minbpe.rs            minbpe's .model/.vocab formats, recovering GPT-4's merges
src/main.rs              the codebook CLI
tests/minbpe.rs          the checks against minbpe and tiktoken (fixtures in tests/fixtures/)
src/wasm.rs              the browser interface
src/bin/experiment.rs    the whole evaluation
bench/tiktoken_bench.py  tiktoken's ids and speed, for comparison
tokenizers/              the trained 32k vocabulary
results/                 every number in this README and on the page
web/template.html        the demo page (design notes in web/DESIGN.md)
```
