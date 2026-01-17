"""Tests for the FastAPI application endpoints."""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from search_api.models import Message

# Mock data for testing
MOCK_MESSAGES = [
    Message(
        id="1",
        user_id="u1",
        user_name="John Doe",
        timestamp=datetime(2025, 1, 15, 10, 0, 0, tzinfo=UTC),
        message="Book a flight to Paris",
    ),
    Message(
        id="2",
        user_id="u2",
        user_name="Jane Smith",
        timestamp=datetime(2025, 1, 14, 10, 0, 0, tzinfo=UTC),
        message="Reserve a dinner table",
    ),
]


@pytest.fixture
def client():
    """Create test client with mocked data store."""
    with (
        patch("search_api.main.data_store") as mock_store,
        patch("search_api.main.search_engine") as mock_engine,
    ):
        # Configure mock data store
        mock_store.is_ready = True
        mock_store.total_messages = len(MOCK_MESSAGES)
        mock_store.last_refresh = datetime(2025, 1, 15, 12, 0, 0, tzinfo=UTC)
        mock_store.messages = MOCK_MESSAGES
        mock_store.refresh = AsyncMock()
        mock_store.start_background_refresh = lambda: None
        mock_store.stop_background_refresh = lambda: None

        # Configure mock search engine
        def mock_search(query, skip=0, limit=10):
            if not query:
                return MOCK_MESSAGES[skip : skip + limit], len(MOCK_MESSAGES)
            filtered = [m for m in MOCK_MESSAGES if query.lower() in m.message.lower()]
            return filtered[skip : skip + limit], len(filtered)

        mock_engine.search = mock_search
        mock_engine.build_index = lambda _: None

        # Import app after mocking
        from search_api.main import app

        with TestClient(app, raise_server_exceptions=False) as test_client:
            yield test_client


class TestSearchEndpoint:
    """Test suite for /search endpoint."""

    def test_search_returns_results(self, client):
        """Test basic search returns results."""
        response = client.get("/search?q=paris")
        assert response.status_code == 200
        data = response.json()
        assert "total" in data
        assert "items" in data
        assert "query" in data
        assert data["query"] == "paris"

    def test_search_pagination(self, client):
        """Test search pagination parameters."""
        response = client.get("/search?q=flight&skip=0&limit=1")
        assert response.status_code == 200
        data = response.json()
        assert len(data["items"]) == 1
        assert data["total"] == 1

    def test_search_requires_query(self, client):
        """Test that query parameter is required."""
        response = client.get("/search")
        assert response.status_code == 422  # Validation error

    def test_search_limit_validation(self, client):
        """Test limit parameter validation."""
        # Too high
        response = client.get("/search?q=test&limit=101")
        assert response.status_code == 422

        # Negative
        response = client.get("/search?q=test&limit=-1")
        assert response.status_code == 422

    def test_search_skip_validation(self, client):
        """Test skip parameter validation."""
        response = client.get("/search?q=test&skip=-1")
        assert response.status_code == 422


class TestHealthEndpoint:
    """Test suite for /health endpoint."""

    def test_health_when_ready(self, client):
        """Test health endpoint when index is ready."""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"

    def test_health_returns_503_when_not_ready(self, client):
        """Test health endpoint when index is not ready."""
        with patch("search_api.main.data_store") as mock_store:
            mock_store.is_ready = False
            response = client.get("/health")
            assert response.status_code == 503


class TestRootEndpoint:
    """Test suite for / endpoint."""

    def test_root_returns_stats(self, client):
        """Test root endpoint returns API stats."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["index_ready"] is True
        assert "stats" in data
        assert data["stats"]["total_messages"] == 2


class TestRefreshEndpoint:
    """Test suite for /refresh endpoint."""

    def test_refresh_triggers_reload(self, client):
        """Test that refresh endpoint triggers data reload."""
        response = client.post("/refresh")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "refreshed"
