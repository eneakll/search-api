"""Tests for the search engine."""

import time
from datetime import UTC, datetime

import pytest

from search_api.models import Message
from search_api.search import STOP_WORDS, SearchEngine, stem, tokenize


@pytest.fixture
def sample_messages() -> list[Message]:
    """Create sample messages for testing."""
    return [
        Message(
            id="1",
            user_id="u1",
            user_name="John Doe",
            timestamp=datetime(2025, 1, 15, 10, 0, 0, tzinfo=UTC),
            message="Book a flight to Paris for next Friday",
        ),
        Message(
            id="2",
            user_id="u2",
            user_name="Jane Smith",
            timestamp=datetime(2025, 1, 14, 10, 0, 0, tzinfo=UTC),
            message="Reserve a table at the French restaurant",
        ),
        Message(
            id="3",
            user_id="u1",
            user_name="John Doe",
            timestamp=datetime(2025, 1, 13, 10, 0, 0, tzinfo=UTC),
            message="Cancel my Paris hotel reservation",
        ),
        Message(
            id="4",
            user_id="u3",
            user_name="Alice Wong",
            timestamp=datetime(2025, 1, 12, 10, 0, 0, tzinfo=UTC),
            message="I need tickets to the opera tonight",
        ),
        Message(
            id="5",
            user_id="u1",
            user_name="John Doe",
            timestamp=datetime(2025, 1, 11, 10, 0, 0, tzinfo=UTC),
            message="Book flights to Paris Paris Paris",
        ),
    ]


@pytest.fixture
def search_engine(sample_messages: list[Message]) -> SearchEngine:
    """Create a search engine with sample data."""
    engine = SearchEngine()
    engine.build_index(sample_messages)
    return engine


class TestSearchEngine:
    """Tests for SearchEngine class."""

    def test_single_word_search(self, search_engine: SearchEngine) -> None:
        """Single word returns matching results."""
        results, total = search_engine.search("paris")
        assert total == 3
        assert all("paris" in r.message.lower() for r in results)

    def test_multi_word_and_logic(self, search_engine: SearchEngine) -> None:
        """Multi-word queries use AND logic."""
        results, total = search_engine.search("paris reservation")
        assert total >= 1
        assert any("reservation" in r.message.lower() for r in results)

    def test_case_insensitive(self, search_engine: SearchEngine) -> None:
        """Search is case insensitive."""
        _, t1 = search_engine.search("PARIS")
        _, t2 = search_engine.search("paris")
        _, t3 = search_engine.search("Paris")
        assert t1 == t2 == t3

    def test_no_results(self, search_engine: SearchEngine) -> None:
        """Non-matching query returns empty."""
        results, total = search_engine.search("xyznonexistent")
        assert total == 0
        assert results == []

    def test_pagination_skip(self, search_engine: SearchEngine) -> None:
        all_results, _ = search_engine.search("flight", limit=10)
        skipped, _ = search_engine.search("flight", skip=1, limit=10)
        assert skipped == all_results[1:]

    def test_pagination_limit(self, search_engine: SearchEngine) -> None:
        results, total = search_engine.search("flight", limit=1)
        assert len(results) == 1
        assert total == 2

    def test_empty_query_returns_none(self, search_engine: SearchEngine) -> None:
        _, total = search_engine.search("")
        assert total == 0

    def test_stopword_only_query_returns_none(self, search_engine: SearchEngine) -> None:
        _, total = search_engine.search("the")
        assert total == 0

    def test_search_user_name(self, search_engine: SearchEngine) -> None:
        """User names are searchable."""
        _, total = search_engine.search("john")
        assert total == 3

    def test_special_characters(self, search_engine: SearchEngine) -> None:
        """Special characters don't break search."""
        results, total = search_engine.search("test@#$%")
        assert total == 0
        assert results == []

    def test_rebuild_clears_index(
        self, search_engine: SearchEngine, sample_messages: list[Message]
    ) -> None:
        """Rebuilding index clears previous data."""
        search_engine.build_index(sample_messages[:2])
        _, total = search_engine.search("john")
        assert total == 1  # Only message 1 has John Doe

        search_engine.build_index(sample_messages)
        _, total = search_engine.search("john")
        assert total == 3  # Messages 1, 3, 5 have John Doe

    def test_tfidf_ranking(self, search_engine: SearchEngine) -> None:
        """TF-IDF ranks higher term frequency first."""
        results, _ = search_engine.search("paris")
        assert results[0].id == "5"

    def test_stemming_matches(self, search_engine: SearchEngine) -> None:
        """Stemming matches word variations."""
        _, t1 = search_engine.search("reservation")
        _, t2 = search_engine.search("reserve")
        assert t1 > 0
        assert t2 > 0

    def test_cache_performance(self, search_engine: SearchEngine) -> None:
        """Cached queries are fast."""
        start = time.perf_counter()
        search_engine.search("paris")
        first = time.perf_counter() - start

        start = time.perf_counter()
        search_engine.search("paris")
        second = time.perf_counter() - start

        assert second <= first * 2

    def test_clear_cache(self, search_engine: SearchEngine) -> None:
        """Cache can be cleared."""
        search_engine.search("paris")
        search_engine.clear_cache()
        _, total = search_engine.search("paris")
        assert total == 3


class TestTokenize:
    """Tests for tokenize function."""

    def test_basic(self) -> None:
        assert tokenize("Hello World") == ["hello", "world"]

    def test_punctuation_removed(self) -> None:
        assert tokenize("Hello, World!") == ["hello", "world"]

    def test_numbers_included(self) -> None:
        tokens = tokenize("Flight 123")
        assert "123" in tokens
        assert "flight" in tokens

    def test_empty_string(self) -> None:
        assert tokenize("") == []

    def test_special_chars_only(self) -> None:
        assert tokenize("@#$%") == []

    def test_stopwords_removed(self) -> None:
        tokens = tokenize("the quick fox", remove_stopwords=True)
        assert "the" not in tokens
        assert "quick" in tokens

    def test_stopwords_kept_default(self) -> None:
        tokens = tokenize("the quick fox")
        assert "the" in tokens


class TestStem:
    """Tests for stem function."""

    def test_suffixes(self) -> None:
        assert stem("running") == "runn"
        assert stem("reservation") == "reservate"

    def test_short_words_unchanged(self) -> None:
        assert stem("the") == "the"
        assert stem("a") == "a"

    def test_plurals(self) -> None:
        assert stem("cats") == "cat"
        assert stem("dogs") == "dog"

    def test_ed_suffix(self) -> None:
        assert stem("walked") == "walk"
        assert stem("jumped") == "jump"


class TestStopWords:
    """Tests for stop words."""

    def test_common_words_present(self) -> None:
        assert "the" in STOP_WORDS
        assert "a" in STOP_WORDS
        assert "is" in STOP_WORDS

    def test_is_frozenset(self) -> None:
        assert isinstance(STOP_WORDS, frozenset)
