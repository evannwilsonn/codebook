"""Writes tests/fixtures/ by running Karpathy's minbpe and OpenAI's tiktoken.

    pip install tiktoken regex
    git clone https://github.com/karpathy/minbpe
    python3 scripts/minbpe-fixtures.py path/to/minbpe

tests/minbpe.rs then checks Codebook against these files: the merges minbpe's Basic and Regex
tokenizers learn from minbpe's own test text, their encodings, the lecture's Wikipedia example,
tiktoken's special-token handling, and model/vocab files written by minbpe's save().
"""

import json
import os
import shutil
import sys

minbpe_dir = sys.argv[1]
sys.path.insert(0, minbpe_dir)
from minbpe import BasicTokenizer, RegexTokenizer  # noqa: E402
import tiktoken  # noqa: E402

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tests", "fixtures")
os.makedirs(out, exist_ok=True)
shutil.copy(os.path.join(minbpe_dir, "tests", "taylorswift.txt"), out)
text = open(os.path.join(out, "taylorswift.txt"), encoding="utf-8").read()

fx = {"vocab_size": 512}
samples = ["", "?", "hello world!!!? (안녕하세요!) lol123 😉", "Taylor Swift's 2024 tour"]
for name, cls in [("basic", BasicTokenizer), ("regex", RegexTokenizer)]:
    t = cls()
    t.train(text, 512)
    merges = sorted(t.merges.items(), key=lambda kv: kv[1])
    fx[name] = {
        "merges": [[a, b] for (a, b), _ in merges],
        "encode": [{"text": s, "ids": t.encode(s)} for s in samples],
    }
    w = cls()
    w.train("aaabdaaabac", 256 + 3)  # the lecture's example from Wikipedia
    fx[name]["wikipedia"] = w.encode("aaabdaaabac")

# minbpe's special-token test string, encoded by tiktoken's GPT-4 vocabulary.
specials_string = """
<|endoftext|>Hello world this is one document
<|endoftext|>And this is another document
<|endoftext|><|fim_prefix|>And this one has<|fim_suffix|> tokens.<|fim_middle|> FIM
<|endoftext|>Last document!!! 👋<|endofprompt|>
""".strip()
enc = tiktoken.get_encoding("cl100k_base")
fx["gpt4"] = {
    "specialsString": specials_string,
    "allAllowed": enc.encode(specials_string, allowed_special="all"),
    "ordinary": enc.encode_ordinary(specials_string),
    "specialTokens": dict(enc._special_tokens),
}

# Model files written by minbpe's save(), and how the saved tokenizers encode.
specials = dict(enc._special_tokens)
r = RegexTokenizer()
r.train(text, 256 + 64)
r.register_special_tokens(specials)
b = BasicTokenizer()
b.train(text, 256 + 32)
cwd = os.getcwd()
os.chdir(out)
r.save("regex-specials")
b.save("basic-small")
os.chdir(cwd)
fx["model"] = {
    "regexSpecialsAll": r.encode(specials_string, allowed_special="all"),
    "regexSpecialsNone": r.encode(specials_string, allowed_special="none"),
    "regexSpecialsSubset": r.encode(specials_string, allowed_special={"<|endoftext|>"}),
    "basicSmall": b.encode(specials_string),
}
json.dump(fx, open(os.path.join(out, "minbpe.json"), "w"), indent=0)
print("wrote", os.path.normpath(out))
