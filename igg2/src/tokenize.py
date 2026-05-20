"""Regex-based tokeniser. No NLTK dependency."""
import re


STOP_WORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "is", "was", "are", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "must", "can", "not", "this",
    "that", "these", "those", "it", "its", "from", "by", "as", "up",
    "out", "about", "into", "through", "during", "before", "after",
    "between", "each", "no", "nor", "so", "yet", "both", "either",
    "just", "than", "too", "very", "s", "t", "re", "ll", "ve", "d", "m",
}


def tokenize(text: str) -> list:
    """Lowercase, strip non-alphanumerics, split on whitespace."""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    return [t for t in text.split() if t]
