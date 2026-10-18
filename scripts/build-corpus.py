"""Builds the three text files the experiment reads.

  data/corpus/imdb-train.txt  25,000 IMDB training reviews (the tokenizer learns from these)
  data/corpus/imdb-test.txt   25,000 IMDB test reviews (held out, same kind of text)
  data/corpus/rust.txt        Rust standard library source (held out, a different kind of text)

Documents are separated by the ASCII record separator (0x1E). Reviews have their "<br />"
tags turned into line breaks. Rust files come from `rustup component add rust-src`: every .rs
file under library/alloc and library/std, in path order, until the file reaches 8 MB.

    python3 scripts/build-corpus.py
"""
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
IMDB = ROOT / "data" / "aclImdb"
OUT = ROOT / "data" / "corpus"
SEP = "\x1e"
RUST_LIMIT = 8_000_000


def reviews(split):
    docs = []
    for label in ("neg", "pos"):
        for path in sorted((IMDB / split / label).glob("*.txt"), key=lambda p: p.name):
            text = path.read_text(encoding="utf-8").replace("<br />", "\n")
            docs.append(text)
    return docs


def rust_sources():
    sysroot = subprocess.run(["rustc", "--print", "sysroot"], capture_output=True, text=True, check=True).stdout.strip()
    library = pathlib.Path(sysroot) / "lib" / "rustlib" / "src" / "rust" / "library"
    if not library.exists():
        sys.exit("Rust source not found. Run: rustup component add rust-src")
    docs, size = [], 0
    for crate in ("alloc", "std"):
        for path in sorted((library / crate).rglob("*.rs")):
            text = path.read_text(encoding="utf-8")
            if SEP in text:
                continue
            if size + len(text.encode()) > RUST_LIMIT:
                return docs
            docs.append(text)
            size += len(text.encode())
    return docs


def write(name, docs):
    assert all(SEP not in d for d in docs)
    data = SEP.join(docs)
    (OUT / name).write_text(data, encoding="utf-8")
    print(f"{name}: {len(docs):,} documents, {len(data.encode()) / 1e6:.1f} MB")


if __name__ == "__main__":
    if not IMDB.exists():
        sys.exit("IMDB reviews not found. Run scripts/get-data.sh first.")
    OUT.mkdir(parents=True, exist_ok=True)
    write("imdb-train.txt", reviews("train"))
    write("imdb-test.txt", reviews("test"))
    write("rust.txt", rust_sources())
