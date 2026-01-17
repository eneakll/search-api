"""Data fetching and storage with background refresh."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import httpx

from search_api.config import (
    DATA_SOURCE_URL,
    HTTP_MAX_RETRIES,
    HTTP_TIMEOUT,
    MAX_RECORDS,
    REFRESH_INTERVAL_SECONDS,
)
from search_api.models import Message

if TYPE_CHECKING:
    from asyncio import Task

logger = logging.getLogger(__name__)


class DataStore:
    """In-memory data store with incremental sync."""

    __slots__ = ("_last_refresh", "_last_total", "_messages", "_on_refresh", "_ready", "_refresh_task")

    def __init__(self) -> None:
        self._messages: list[Message] = []
        self._last_refresh: datetime | None = None
        self._last_total: int = 0
        self._refresh_task: Task[None] | None = None
        self._ready: bool = False
        self._on_refresh: Callable[[list[Message]], None] | None = None

    def set_on_refresh(self, callback: Callable[[list[Message]], None]) -> None:
        self._on_refresh = callback

    @property
    def messages(self) -> list[Message]:
        return self._messages

    @property
    def last_refresh(self) -> datetime | None:
        return self._last_refresh

    @property
    def is_ready(self) -> bool:
        return self._ready

    @property
    def total_messages(self) -> int:
        return len(self._messages)

    async def _get_remote_total(self, client: httpx.AsyncClient) -> int | None:
        try:
            response = await client.get(DATA_SOURCE_URL, params={"skip": 0, "limit": 1})
            if response.status_code == 200:
                return response.json().get("total", 0)
        except httpx.RequestError:
            pass
        return None

    async def fetch_all_messages(self) -> list[Message]:
        messages: list[Message] = []
        skip = 0
        limit = 100
        total: int | None = None

        headers = {"User-Agent": "SearchAPI/1.0"}
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT, headers=headers) as client:
            while True:
                response = await self._fetch_page(client, skip, limit)
                if response is None:
                    break

                data = response.json()
                items = data.get("items", [])
                if not items:
                    break

                messages.extend(Message.model_validate(item) for item in items)

                if total is None:
                    total = data.get("total", 0)
                    if total > MAX_RECORDS:
                        logger.warning(
                            "Remote has %d records, exceeds MAX_RECORDS=%d. Truncating.",
                            total,
                            MAX_RECORDS,
                        )

                skip += limit
                if skip >= min(total or 0, MAX_RECORDS):
                    break

                await asyncio.sleep(0.05)

        return messages

    async def _fetch_page(
        self, client: httpx.AsyncClient, skip: int, limit: int
    ) -> httpx.Response | None:
        for attempt in range(HTTP_MAX_RETRIES):
            try:
                response = await client.get(DATA_SOURCE_URL, params={"skip": skip, "limit": limit})
                if response.status_code in {401, 403, 404, 429}:
                    logger.warning("Got %d at skip=%d, stopping", response.status_code, skip)
                    return None
                response.raise_for_status()
                return response
            except httpx.HTTPStatusError:
                if attempt == HTTP_MAX_RETRIES - 1:
                    raise
                await asyncio.sleep(1 << attempt)
            except httpx.RequestError as e:
                logger.warning("Request error (attempt %d): %s", attempt + 1, e)
                if attempt == HTTP_MAX_RETRIES - 1:
                    raise
                await asyncio.sleep(1 << attempt)
        return None

    async def refresh(self, force: bool = False) -> bool:
        logger.info("Checking for data changes...")

        try:
            async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
                remote_total = await self._get_remote_total(client)

            if remote_total is None:
                logger.warning("Could not fetch remote total, skipping refresh")
                return False

            # Skip refresh if total unchanged (data likely unchanged)
            if not force and self._ready and remote_total == self._last_total:
                logger.info("No changes detected (total=%d), skipping refresh", remote_total)
                return False

            logger.info("Changes detected (remote=%d, local=%d), refreshing...",
                       remote_total, self._last_total)

            new_messages = await self.fetch_all_messages()
            self._messages = new_messages
            self._last_total = len(new_messages)
            self._last_refresh = datetime.now(UTC)
            self._ready = True
            logger.info("Loaded %d messages", len(new_messages))
            if self._on_refresh:
                self._on_refresh(new_messages)
            return True

        except Exception:
            logger.exception("Failed to refresh data")
            if not self._ready:
                raise
            return False

    async def _background_loop(self) -> None:
        while True:
            await asyncio.sleep(REFRESH_INTERVAL_SECONDS)
            try:
                await self.refresh()
            except Exception:
                logger.exception("Background refresh failed")

    def start_background_refresh(self) -> None:
        if self._refresh_task is None:
            self._refresh_task = asyncio.create_task(self._background_loop())
            logger.info("Background refresh started (interval: %ds)", REFRESH_INTERVAL_SECONDS)

    def stop_background_refresh(self) -> None:
        if self._refresh_task is not None:
            self._refresh_task.cancel()
            self._refresh_task = None
            logger.info("Background refresh stopped")


data_store = DataStore()
