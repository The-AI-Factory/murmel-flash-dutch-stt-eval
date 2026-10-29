"""Text normalisation for WER: NFKC, lowercase, all punctuation removed, digits expanded to Dutch number words with
num2words, the fillers uh/um/eh/euh/hmm/hm/ah/uhm dropped."""
from __future__ import annotations

import re
import string
import unicodedata

from num2words import num2words

FILLER_WORDS = {"uh", "um", "eh", "euh", "hmm", "hm", "ah", "uhm"}


def _expand_numbers(text: str) -> str:
    def rep(m: re.Match) -> str:
        try:
            return num2words(int(m.group()), lang="nl")
        except Exception:
            return m.group()
    return re.sub(r"\b\d+\b", rep, text)


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "").lower()
    text = "".join(c for c in text if not unicodedata.category(c).startswith("P") and c not in string.punctuation)
    text = _expand_numbers(text)
    return " ".join(w for w in text.split() if w not in FILLER_WORDS).strip()
