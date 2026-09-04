"""
src/preprocessing.py

Shared text-cleaning logic for the Voice Customer Support Intent Classifier.

IMPORTANT: This exact function is used both when TRAINING the model
(notebooks/model_building.ipynb) and at INFERENCE time (app.py) on live
speech transcripts. Keeping it in one shared module -- instead of copy-pasting
it into both places -- guarantees they can never drift apart.
"""

import re
import string

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


def _ensure_nltk_data():
    for pkg in ["stopwords", "wordnet", "omw-1.4"]:
        try:
            nltk.data.find(f"corpora/{pkg}")
        except LookupError:
            nltk.download(pkg, quiet=True)


_ensure_nltk_data()

STOP_WORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()

# The Bitext dataset uses template placeholders like {{Order Number}},
# {{Account Type}}, {{Refund Amount}}, etc. A real customer speaking to the
# app would never say the literal words "Order Number" -- they'd say an
# actual number. We strip these placeholders out entirely rather than
# treating "order" / "number" as meaningful content words, since keeping
# them would let the model "cheat" by memorizing placeholder text instead
# of learning the surrounding intent language.
PLACEHOLDER_PATTERN = re.compile(r"\{\{.*?\}\}")


def clean_placeholders(text: str) -> str:
    """Removes {{...}} template placeholders from Bitext-style text."""
    return PLACEHOLDER_PATTERN.sub(" ", text)


def preprocess(text: str) -> str:
    """
    Full preprocessing pipeline:
    1. Remove {{placeholder}} tokens
    2. Lowercase
    3. Remove punctuation
    4. Remove digits
    5. Tokenize (simple whitespace split)
    6. Remove stopwords
    7. Lemmatize
    """
    text = clean_placeholders(text)
    text = text.lower()
    text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)
    text = re.sub(r"\d+", " ", text)
    tokens = text.split()
    tokens = [
    LEMMATIZER.lemmatize(w)
    for w in tokens
    if len(w) > 1
]
    return " ".join(tokens)