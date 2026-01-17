from datetime import datetime

from pydantic import BaseModel


class Message(BaseModel):
    id: str
    user_id: str
    user_name: str
    timestamp: datetime
    message: str

    model_config = {"frozen": True}


class SearchResponse(BaseModel):
    total: int
    items: list[Message]
    query: str


class IndexStats(BaseModel):
    total_messages: int
    last_refresh: datetime | None
    refresh_interval_seconds: int


class HealthResponse(BaseModel):
    status: str
    index_ready: bool
    stats: IndexStats | None = None


class RefreshResponse(BaseModel):
    status: str
    total_messages: int
    last_refresh: datetime | None
