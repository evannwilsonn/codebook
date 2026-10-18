"""Builds the browser demo, docs/index.html, from web/template.html.

It compiles the crate to WebAssembly, then embeds three gzip-compressed, base64-encoded blobs
(the wasm module, the IMDB-trained vocabulary and GPT-4's cl100k_base vocabulary) plus the
results files. Every number on the page comes from results/*.json.

    python3 scripts/build-demo.py
"""
import base64
import gzip
import json
import pathlib
import subprocess

ROOT = pathlib.Path(__file__).resolve().parent.parent
WASM = ROOT / "target" / "wasm32-unknown-unknown" / "wasm" / "codebook.wasm"


def packed(path):
    return base64.b64encode(gzip.compress(path.read_bytes(), compresslevel=9, mtime=0)).decode("ascii")


def main():
    subprocess.run(
        ["cargo", "build", "--profile", "wasm", "--lib", "--target", "wasm32-unknown-unknown"],
        cwd=ROOT,
        check=True,
    )
    results = json.loads((ROOT / "results" / "experiment.json").read_text())
    tiktoken = json.loads((ROOT / "results" / "tiktoken.json").read_text())
    page = (ROOT / "web" / "template.html").read_text()
    fills = {
        "{{WASM}}": packed(WASM),
        "{{IMDB}}": packed(ROOT / "tokenizers" / "imdb-32k.tiktoken"),
        "{{CL100K}}": packed(ROOT / "data" / "ranks" / "cl100k_base.tiktoken"),
        "{{RESULTS}}": json.dumps(results, separators=(",", ":"), ensure_ascii=False).replace("</", "<\\/"),
        "{{TIKTOKEN}}": json.dumps(tiktoken, separators=(",", ":")),
    }
    for key, value in fills.items():
        assert page.count(key) == 1, key
        page = page.replace(key, value)
    out = ROOT / "docs" / "index.html"
    out.parent.mkdir(exist_ok=True)
    out.write_text(page)
    print(f"Wrote docs/index.html ({len(page.encode()):,} bytes; wasm {WASM.stat().st_size:,} bytes before gzip)")


if __name__ == "__main__":
    main()
