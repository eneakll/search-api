"""Tests for the data store module."""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from search_api.data import DataStore
from search_api.models import Message


@pytest.fixture
def data_store():
    """Create a fresh DataStore instance."""
    return DataStore()


class TestDataStore:
    """Tests for DataStore class."""

    def test_initial_state(self, data_store: DataStore) -> None:
        """DataStore starts empty and not ready."""
        assert data_store.messages == []
        assert data_store.last_refresh is None
        assert data_store.is_ready is False
        assert data_store.total_messages == 0

    @pytest.mark.asyncio
    async def test_refresh_success(self, data_store: DataStore) -> None:
        """Refresh populates messages and sets ready state."""
        mock_messages = [
            Message(
                id="1",
                user_id="u1",
                user_name="Test User",
                timestamp=datetime(2025, 1, 15, 10, 0, 0, tzinfo=UTC),
                message="Test message",
            )
        ]

        with patch.object(DataStore, "fetch_all_messages", new_callable=AsyncMock) as mock_fetch:
            mock_fetch.return_value = mock_messages
            await data_store.refresh()

        assert data_store.is_ready is True
        assert data_store.total_messages == 1
        assert data_store.last_refresh is not None

    @pytest.mark.asyncio
    async def test_refresh_failure_when_not_ready_raises(self, data_store: DataStore) -> None:
        """Refresh raises exception if not ready and fetch fails."""
        with patch.object(DataStore, "fetch_all_messages", new_callable=AsyncMock) as mock_fetch:
            mock_fetch.side_effect = httpx.RequestError("Connection failed")
            with pytest.raises(httpx.RequestError):
                await data_store.refresh()

        assert data_store.is_ready is False

    @pytest.mark.asyncio
    async def test_refresh_failure_when_ready_keeps_data(self, data_store: DataStore) -> None:
        """Refresh keeps existing data if already ready and fetch fails."""
        mock_messages = [
            Message(
                id="1",
                user_id="u1",
                user_name="Test User",
                timestamp=datetime(2025, 1, 15, 10, 0, 0, tzinfo=UTC),
                message="Test message",
            )
        ]

        # First successful refresh
        with patch.object(DataStore, "fetch_all_messages", new_callable=AsyncMock) as mock_fetch:
            mock_fetch.return_value = mock_messages
            await data_store.refresh()

        assert data_store.is_ready is True
        original_count = data_store.total_messages

        # Second refresh fails
        with patch.object(DataStore, "fetch_all_messages", new_callable=AsyncMock) as mock_fetch:
            mock_fetch.side_effect = Exception("Network error")
            await data_store.refresh()  # Should not raise

        assert data_store.is_ready is True
        assert data_store.total_messages == original_count

    @pytest.mark.asyncio
    async def test_fetch_page_retry_on_error(self, data_store: DataStore) -> None:
        """Fetch page retries on transient errors."""
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()

        # Fail twice, then succeed
        mock_client.get.side_effect = [
            httpx.RequestError("Timeout"),
            httpx.RequestError("Timeout"),
            mock_response,
        ]

        result = await data_store._fetch_page(mock_client, skip=0, limit=100)
        assert result == mock_response
        assert mock_client.get.call_count == 3

    @pytest.mark.asyncio
    async def test_fetch_page_returns_none_on_404(self, data_store: DataStore) -> None:
        """Fetch page returns None on 404."""
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_response = MagicMock()
        mock_response.status_code = 404

        mock_client.get.return_value = mock_response

        result = await data_store._fetch_page(mock_client, skip=0, limit=100)
        assert result is None

    @pytest.mark.asyncio
    async def test_fetch_page_returns_none_on_429(self, data_store: DataStore) -> None:
        """Fetch page returns None on rate limit."""
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_response = MagicMock()
        mock_response.status_code = 429

        mock_client.get.return_value = mock_response

        result = await data_store._fetch_page(mock_client, skip=0, limit=100)
        assert result is None

    def test_start_background_refresh(self, data_store: DataStore) -> None:
        """Background refresh can be started."""
        with patch("search_api.data.asyncio.create_task") as mock_task:
            mock_task.return_value = MagicMock()
            data_store.start_background_refresh()
            assert mock_task.called

    def test_stop_background_refresh(self, data_store: DataStore) -> None:
        """Background refresh can be stopped."""
        mock_task = MagicMock()
        data_store._refresh_task = mock_task
        data_store.stop_background_refresh()
        mock_task.cancel.assert_called_once()
        assert data_store._refresh_task is None

    def test_stop_background_refresh_when_not_running(self, data_store: DataStore) -> None:
        """Stopping when not running is a no-op."""
        data_store.stop_background_refresh()  # Should not raise
        assert data_store._refresh_task is None
