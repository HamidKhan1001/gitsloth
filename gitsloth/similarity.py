"""Cheap lexical signal: which changes talk about the same identifiers?  Pure stdlib, no embeddings, no model."""
from __future__ import annotations

import math
import re
from collections import Counter

STOP = set("""the a an and or of to in is it for on as at by be this that with from not if else return self cls none true false null
def class function const let var import export new int str bool list dict set get put len range print else elif try except finally while
""".split())
TOKEN = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}")


def _split(word: str) -> list[str]:
    parts = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", word).replace("_", " ").lower().split()
    return [p for p in parts if len(p) > 2 and p not in STOP]


def tokens(unit) -> Counter:
    """Identifier-ish tokens from changed lines, their function header, and the file name. Whole identifiers count
    double, because `flag_value` is a much stronger link between two changes than the word `value`."""
    text = " ".join(l.decode("utf-8", "replace") for l in unit.old_lines[:60] + unit.new_lines[:60])
    first = unit.desc.splitlines()[0] if unit.desc else unit.path
    text += " " + first
    c: Counter = Counter()
    for w in TOKEN.findall(text):
        wl = w.lower()
        if wl in STOP:
            continue
        if "_" in w or re.search(r"[a-z][A-Z]", w):
            c[wl] += 2
        for p in _split(w):
            c[p] += 1
    return c


def tfidf(units) -> list[dict[str, float]]:
    docs = [tokens(u) for u in units]
    df: Counter = Counter()
    for d in docs:
        df.update(d.keys())
    n = len(docs)
    out = []
    for d in docs:
        v = {t: (1 + math.log(c)) * math.log((1 + n) / (1 + df[t])) for t, c in d.items() if df[t] > 0}
        norm = math.sqrt(sum(x * x for x in v.values())) or 1.0
        out.append({t: x / norm for t, x in v.items()})
    return out


def cosine(a: dict[str, float], b: dict[str, float]) -> float:
    if len(a) > len(b):
        a, b = b, a
    return sum(x * b.get(t, 0.0) for t, x in a.items())


def similarity_matrix(units) -> list[list[float]]:
    v = tfidf(units)
    n = len(units)
    s = [[1.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            s[i][j] = s[j][i] = cosine(v[i], v[j])
    return s
