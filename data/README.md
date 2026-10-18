# Data

Nothing in this folder is committed. To rebuild it:

1. `./scripts/get-data.sh` (or `scripts/get-data.ps1`) downloads the IMDB Large Movie Review Dataset into `data/aclImdb/`.
2. `rustup component add rust-src` installs the Rust standard library source.
3. `python3 scripts/build-corpus.py` writes `data/corpus/`:
   - `imdb-train.txt`: the 25,000 training reviews. The tokenizer learns from these.
   - `imdb-test.txt`: the 25,000 test reviews, held out.
   - `rust.txt`: 7.9 MB of `alloc` and `std` source, held out, a different kind of text.

   Documents are separated by the ASCII record separator (0x1E).
4. `data/ranks/` holds OpenAI's `cl100k_base.tiktoken` and `r50k_base.tiktoken`. Download them from `https://openaipublic.blob.core.windows.net/encodings/`, as tiktoken does.

**Citation.** Andrew L. Maas, Raymond E. Daly, Peter T. Pham, Dan Huang, Andrew Y. Ng, and Christopher Potts. 2011. Learning Word Vectors for Sentiment Analysis. In *Proceedings of ACL 2011*, 142–150.
