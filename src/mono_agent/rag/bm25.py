"""Okapi BM25 over chunks, with a tokenizer that works for Ukrainian and English."""

import math
import re
from collections import Counter

STOPWORDS = frozenset(
    {
        # Ukrainian
        "і",
        "й",
        "та",
        "або",
        "але",
        "що",
        "як",
        "це",
        "у",
        "в",
        "на",
        "з",
        "із",
        "до",
        "за",
        "від",
        "по",
        "для",
        "не",
        "чи",
        "то",
        # English
        "a",
        "an",
        "the",
        "of",
        "to",
        "in",
        "on",
        "for",
        "and",
        "or",
        "is",
        "are",
        "be",
        "by",
        "with",
        "what",
        "how",
        "my",
        "me",
        "i",
    }
)


def tokenize(text: str) -> list[str]:
    # shortcut: crude stemming by truncating to 6 characters. Ukrainian is highly inflected
    # (кешбек / кешбеку / кешбеком) and has no snowball stemmer; prefix truncation is a known
    # cheap approximation. Upgrade to a morphological analyser (pymorphy3-uk) if BM25 recall
    # on Ukrainian queries turns out low in the eval.
    words = re.findall(r"\w+", text.lower())
    return [w[:6] for w in words if len(w) > 1 and w not in STOPWORDS]


class BM25:
    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.docs = [Counter(tokenize(d)) for d in docs]
        self.lengths = [sum(c.values()) for c in self.docs]
        self.avg_len = sum(self.lengths) / max(len(self.docs), 1)
        df = Counter(term for c in self.docs for term in c)
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query: str) -> list[float]:
        terms = tokenize(query)
        out = []
        for counts, length in zip(self.docs, self.lengths, strict=True):
            s = 0.0
            for t in terms:
                tf = counts.get(t, 0)
                if tf:
                    norm = tf + self.k1 * (1 - self.b + self.b * length / self.avg_len)
                    s += self.idf[t] * tf * (self.k1 + 1) / norm
            out.append(s)
        return out

    def top(self, query: str, k: int) -> list[tuple[int, float]]:
        ranked = sorted(enumerate(self.scores(query)), key=lambda x: -x[1])
        return [(i, s) for i, s in ranked[:k] if s > 0]
