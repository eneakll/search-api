import os

DATA_SOURCE_URL: str = os.getenv(
    "DATA_SOURCE_URL",
    "https://november7-730026606190.europe-west1.run.app/messages/",
)

REFRESH_INTERVAL_SECONDS: int = int(os.getenv("REFRESH_INTERVAL_SECONDS", "300"))

API_TITLE: str = "Message Search API"
API_DESCRIPTION: str = "High-performance search API with TF-IDF relevance ranking."
API_VERSION: str = "1.0.0"

SEARCH_CACHE_SIZE: int = int(os.getenv("SEARCH_CACHE_SIZE", "1000"))
DEFAULT_PAGE_SIZE: int = 10
MAX_PAGE_SIZE: int = 100

HTTP_TIMEOUT: float = float(os.getenv("HTTP_TIMEOUT", "30.0"))
HTTP_MAX_RETRIES: int = int(os.getenv("HTTP_MAX_RETRIES", "3"))

MAX_RECORDS: int = int(os.getenv("MAX_RECORDS", "50000"))
