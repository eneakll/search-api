"""FastAPI application entry point."""

from __future__ import annotations

import logging
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import ORJSONResponse

from search_api.config import (
    API_DESCRIPTION,
    API_TITLE,
    API_VERSION,
    DEFAULT_PAGE_SIZE,
    MAX_PAGE_SIZE,
    REFRESH_INTERVAL_SECONDS,
)
from search_api.data import data_store
from search_api.models import HealthResponse, IndexStats, RefreshResponse, SearchResponse
from search_api.search import search_engine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    logger.info("Starting application...")
    data_store.set_on_refresh(search_engine.build_index)
    await data_store.refresh(force=True)
    data_store.start_background_refresh()
    logger.info("Application ready")
    yield
    logger.info("Shutting down...")
    data_store.stop_background_refresh()


app = FastAPI(
    title=API_TITLE,
    description=API_DESCRIPTION,
    version=API_VERSION,
    lifespan=lifespan,
    default_response_class=ORJSONResponse,
)
app.add_middleware(GZipMiddleware, minimum_size=500)


@app.middleware("http")
async def add_response_time(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - start) * 1000
    response.headers["X-Response-Time"] = f"{elapsed_ms:.2f}ms"
    return response


@app.get("/", response_model=HealthResponse)
async def root() -> HealthResponse:
    stats = None
    if data_store.is_ready:
        stats = IndexStats(
            total_messages=data_store.total_messages,
            last_refresh=data_store.last_refresh,
            refresh_interval_seconds=REFRESH_INTERVAL_SECONDS,
        )
    return HealthResponse(status="ok", index_ready=data_store.is_ready, stats=stats)


@app.get("/health")
async def health() -> dict[str, str]:
    if not data_store.is_ready:
        raise HTTPException(status_code=503, detail="Index not ready")
    return {"status": "healthy"}


@app.get("/search", response_model=SearchResponse)
async def search(
    response: Response,
    q: str = Query(..., description="Search query", min_length=1),
    skip: int = Query(0, ge=0, description="Results to skip"),
    limit: int = Query(DEFAULT_PAGE_SIZE, ge=1, le=MAX_PAGE_SIZE, description="Max results"),
) -> SearchResponse:
    if not data_store.is_ready:
        raise HTTPException(status_code=503, detail="Index not ready")

    results, total = search_engine.search(q, skip=skip, limit=limit)

    # Cache for 60s - browsers/CDNs will serve cached responses
    response.headers["Cache-Control"] = "public, max-age=60"

    return SearchResponse(
        total=total,
        items=results,
        query=q,
    )


@app.post("/refresh", response_model=RefreshResponse)
async def refresh() -> RefreshResponse:
    await data_store.refresh(force=True)
    return RefreshResponse(
        status="refreshed",
        total_messages=data_store.total_messages,
        last_refresh=data_store.last_refresh,
    )
