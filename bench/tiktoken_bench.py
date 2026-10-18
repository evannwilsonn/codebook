"""Encodes the held-out corpora with tiktoken (the reference) and records two things:

  bench/out/<encoding>-<corpus>.bin  the token ids for every document, so the Rust side can
                                     check its own encoder against them token for token
  results/tiktoken.json              tiktoken's encoding speed, for the speed comparison

Each document goes through encode_ordinary (special tokens treated as plain text), one call per
document, single thread. Timing is the median of three passes over the whole corpus.

    python3 bench/tiktoken_bench.py
"""
import json
import pathlib
import statistics
import struct
import time

import tiktoken

ROOT = pathlib.Path(__file__).resolve().parent.parent
CORPUS = ROOT / "data" / "corpus"
OUT = ROOT / "bench" / "out"
ENCODINGS = ["cl100k_base", "r50k_base"]
CORPORA = ["imdb-test", "rust"]
SEP = "\x1e"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    results = {"tiktoken": tiktoken.__version__, "runs": 3, "encodings": {}}
    for name in ENCODINGS:
        enc = tiktoken.get_encoding(name)
        results["encodings"][name] = {}
        for corpus in CORPORA:
            docs = (CORPUS / f"{corpus}.txt").read_text(encoding="utf-8").split(SEP)
            size = sum(len(d.encode()) for d in docs)
            ids = [enc.encode_ordinary(d) for d in docs]  # warm-up pass, also the saved output
            times = []
            for _ in range(results["runs"]):
                start = time.perf_counter()
                for d in docs:
                    enc.encode_ordinary(d)
                times.append(time.perf_counter() - start)
            seconds = statistics.median(times)
            with open(OUT / f"{name}-{corpus}.bin", "wb") as f:
                f.write(struct.pack("<I", len(ids)))
                for doc in ids:
                    f.write(struct.pack("<I", len(doc)))
                    f.write(struct.pack(f"<{len(doc)}I", *doc))
            tokens = sum(len(d) for d in ids)
            results["encodings"][name][corpus] = {
                "documents": len(docs),
                "bytes": size,
                "tokens": tokens,
                "seconds": round(seconds, 4),
                "mbPerSecond": round(size / seconds / 1e6, 2),
            }
            print(f"{name:12} {corpus:10} {tokens:>10,} tokens  {size / tokens:.3f} bytes/token  {size / seconds / 1e6:6.2f} MB/s")
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "tiktoken.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
