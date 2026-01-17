"""TF-IDF search engine with inverted index and LRU caching."""

from __future__ import annotations

import math
import re
from collections import defaultdict
from functools import lru_cache
from typing import TYPE_CHECKING

from search_api.config import SEARCH_CACHE_SIZE

if TYPE_CHECKING:
    from search_api.models import Message

_TOKEN_PATTERN = re.compile(r"\b[a-z0-9]+\b")

STOP_WORDS: frozenset[str] = frozenset(
    {
        "a",
        "an",
        "the",
        "and",
        "or",
        "but",
        "in",
        "on",
        "at",
        "to",
        "for",
        "of",
        "with",
        "by",
        "from",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "must",
        "shall",
        "can",
        "need",
        "this",
        "that",
        "these",
        "those",
        "i",
        "you",
        "he",
        "she",
        "it",
        "we",
        "they",
        "my",
        "your",
        "his",
        "her",
        "its",
        "our",
        "their",
        "me",
        "him",
        "us",
        "them",
        "what",
        "which",
        "who",
        "whom",
        "when",
        "where",
        "why",
        "how",
        "all",
        "each",
        "every",
        "both",
        "few",
        "more",
        "most",
        "other",
        "some",
        "such",
        "no",
        "not",
        "only",
        "same",
        "so",
        "than",
        "too",
        "very",
        "just",
        "also",
        "now",
        "here",
        "there",
        "then",
        "once",
        "if",
        "as",
    }
)

# Suffix rules for stemming (ordered by length for greedy matching)
_SUFFIX_RULES: tuple[tuple[str, str], ...] = (
    ("ational", "ate"),
    ("tional", "tion"),
    ("enci", "ence"),
    ("anci", "ance"),
    ("izer", "ize"),
    ("isation", "ize"),
    ("ization", "ize"),
    ("ation", "ate"),
    ("ator", "ate"),
    ("alism", "al"),
    ("iveness", "ive"),
    ("fulness", "ful"),
    ("ousness", "ous"),
    ("aliti", "al"),
    ("iviti", "ive"),
    ("biliti", "ble"),
    ("alli", "al"),
    ("entli", "ent"),
    ("eli", "e"),
    ("ousli", "ous"),
    ("lessness", "less"),
    ("ness", ""),
    ("ment", ""),
    ("ings", ""),
    ("ing", ""),
    ("edly", "ed"),
    ("ied", "y"),
    ("ies", "y"),
    ("ed", ""),
    ("ly", ""),
    ("es", ""),
    ("s", ""),
)


def stem(word: str) -> str:
    if len(word) <= 3:
        return word
    for suffix, replacement in _SUFFIX_RULES:
        if word.endswith(suffix) and len(word) - len(suffix) >= 2:
            return word[: -len(suffix)] + replacement
    return word


def tokenize(text: str, *, remove_stopwords: bool = False) -> list[str]:
    tokens = _TOKEN_PATTERN.findall(text.lower())
    if remove_stopwords:
        return [t for t in tokens if t not in STOP_WORDS]
    return tokens


class SearchEngine:
    """Full-text search engine with TF-IDF scoring."""

    __slots__ = ("_doc_lengths", "_doc_term_freq", "_idf", "_index", "_messages", "_version")

    def __init__(self) -> None:
        self._index: dict[str, set[int]] = defaultdict(set)
        self._messages: list[Message] = []
        self._doc_term_freq: dict[int, dict[str, int]] = {}
        self._doc_lengths: dict[int, int] = {}
        self._idf: dict[str, float] = {}
        self._version: int = 0

    def build_index(self, messages: list[Message]) -> None:
        self._index.clear()
        self._doc_term_freq.clear()
        self._doc_lengths.clear()
        self._idf.clear()
        self._messages = messages
        self._version += 1

        for idx, msg in enumerate(messages):
            raw_tokens = tokenize(msg.message, remove_stopwords=True)
            stemmed = [stem(t) for t in raw_tokens]

            # Include user name tokens
            name_tokens = [stem(t) for t in tokenize(msg.user_name)]
            all_tokens = stemmed + name_tokens

            term_freq: dict[str, int] = defaultdict(int)
            for token in all_tokens:
                term_freq[token] += 1
                self._index[token].add(idx)

            # Index raw tokens for exact matching
            for token in raw_tokens:
                self._index[token].add(idx)

            self._doc_term_freq[idx] = dict(term_freq)
            self._doc_lengths[idx] = len(all_tokens) or 1

        n_docs = len(messages)
        if n_docs > 0:
            for term, doc_ids in self._index.items():
                self._idf[term] = math.log(n_docs / len(doc_ids))

    def _score(self, doc_idx: int, query_terms: list[str]) -> float:
        doc_tf = self._doc_term_freq.get(doc_idx, {})
        doc_len = self._doc_lengths.get(doc_idx, 1)
        score = 0.0
        for term in query_terms:
            tf = doc_tf.get(term, 0)
            if tf > 0:
                score += (tf / doc_len) * self._idf.get(term, 0.0)
        return score

    @lru_cache(maxsize=SEARCH_CACHE_SIZE)
    def _cached_search(
        self,
        query: str,
        _version: int,  # Used as cache key for invalidation, not in body
    ) -> tuple[tuple[int, ...], int]:
        tokens = tokenize(query, remove_stopwords=True)
        stemmed = [stem(t) for t in tokens]

        if not stemmed:
            return (), 0

        # AND logic: results must contain all search terms
        result: set[int] | None = None
        for token in stemmed:
            matches = self._index.get(token, set())
            result = matches.copy() if result is None else result & matches

        if not result:
            return (), 0

        scored = [(idx, self._score(idx, stemmed)) for idx in result]
        scored.sort(key=lambda x: (x[1], self._messages[x[0]].timestamp), reverse=True)

        return tuple(idx for idx, _ in scored), len(scored)

    def search(self, query: str, *, skip: int = 0, limit: int = 10) -> tuple[list[Message], int]:
        indices, total = self._cached_search(query, self._version)
        return [self._messages[i] for i in indices[skip : skip + limit]], total

    def clear_cache(self) -> None:
        self._cached_search.cache_clear()


search_engine = SearchEngine()
