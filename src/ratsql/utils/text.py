"""Deterministic word tokenization and light normalisation.

We deliberately avoid external NLP toolkits (CoreNLP / NLTK / spaCy) so that
preprocessing is fully reproducible and dependency-free.  The original
RAT-SQL implementation used Stanford CoreNLP lemmas for schema linking; we
use a small rule-based singulariser instead (documented simplification:
irregular plurals such as "people" -> "person" are handled by a short
exception list only).
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

# Words, numbers (with decimal points / thousand separators), words with inner
# apostrophes or hyphens, or any single non-space symbol.
_TOKEN_RE = re.compile(r"\d+(?:[.,]\d+)*|\w+(?:['’\-]\w+)*|[^\w\s]", re.UNICODE)

STOPWORDS = frozenset(
    """a an the of in on at to for from by with about as into like through after over between out against
    during without before under around among is are was were be been being am do does did doing have has had
    having what which who whom whose this that these those i me my we our you your he him his she her it its
    they them their and or but if because until while how when where why all any both each few more most other
    some such no nor not only own same so than too very s t can will just should now list give show find return
    tell display get please there here also 's ’s ? . , ! ; : ( ) " '""".split()
)

# Question words that frequently appear in Spider questions but carry no schema signal.
_IRREGULAR_SINGULAR = {
    "people": "person",
    "men": "man",
    "women": "woman",
    "children": "child",
    "mice": "mouse",
    "feet": "foot",
    "teeth": "tooth",
    "geese": "goose",
    "data": "data",
    "series": "series",
    "species": "species",
    "news": "news",
    "status": "status",
    "address": "address",
    "business": "business",
    "class": "class",
    "gas": "gas",
    "bus": "bus",
    "campus": "campus",
    "is": "is",
    "was": "was",
    "has": "has",
    "this": "this",
    "its": "its",
    "his": "his",
    "us": "us",
    "yes": "yes",
    "analysis": "analysis",
}

_NUMBER_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
    "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "hundred": 100, "thousand": 1000,
    "once": 1, "twice": 2, "single": 1, "first": 1, "second": 2, "third": 3,
}


@dataclass(frozen=True)
class Token:
    text: str
    start: int  # character offset (inclusive)
    end: int  # character offset (exclusive)


def tokenize(text: str) -> list[Token]:
    """Split ``text`` into word / number / symbol tokens with character offsets."""
    return [Token(m.group(0), m.start(), m.end()) for m in _TOKEN_RE.finditer(text)]


def tokenize_words(text: str) -> list[str]:
    return [t.text for t in tokenize(text)]


def strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))


def singularize(word: str) -> str:
    """Very small rule-based English singulariser (lower-case input)."""
    if word in _IRREGULAR_SINGULAR:
        return _IRREGULAR_SINGULAR[word]
    if len(word) <= 3 or not word.isalpha():
        return word
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    if word.endswith(("sses", "shes", "ches", "xes", "zes")):
        return word[:-2]
    if word.endswith(("ss", "us", "is", "os")):
        return word
    if word.endswith("s"):
        return word[:-1]
    return word


def normalize_token(tok: str) -> str:
    """Lower-case, strip accents / possessives, singularise."""
    t = strip_accents(tok.lower())
    t = t.replace("’", "'")
    if t.endswith("'s"):
        t = t[:-2]
    t = t.strip("'")
    return singularize(t) if t else t


def normalize_phrase(text: str) -> list[str]:
    """Tokenise and normalise a schema name or question phrase (drops pure punctuation)."""
    out = []
    for tok in tokenize_words(text.replace("_", " ")):
        n = normalize_token(tok)
        if n and (n[0].isalnum()):
            out.append(n)
    return out


def is_stopword(norm_token: str) -> bool:
    return norm_token in STOPWORDS or not any(c.isalnum() for c in norm_token)


def parse_number(text: str) -> float | None:
    """Parse a numeric token ("3", "3.5", "1,000", "three") into a float, else ``None``."""
    t = text.lower().strip()
    if t in _NUMBER_WORDS:
        return float(_NUMBER_WORDS[t])
    t2 = t.replace(",", "")
    try:
        return float(t2)
    except ValueError:
        return None


def is_number_token(text: str) -> bool:
    t = text.replace(",", "")
    try:
        float(t)
        return True
    except ValueError:
        return False


def format_number(x: float) -> str:
    return str(int(x)) if float(x).is_integer() else repr(float(x))
