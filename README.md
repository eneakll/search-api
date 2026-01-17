# Message Search API

Full-text search over ~1,300 messages with TF-IDF ranking.

**Live:** https://aurora.enea.work

```bash
curl "https://aurora.enea.work/search?q=paris"
```

## Latency

Server-side: **1-4ms** (cached queries faster). Measured via `X-Response-Time` header.

| Location | End-to-end | Meets <100ms |
|----------|------------|--------------|
| US East | 30-50ms | Yes |
| US West | 50-80ms | Yes |
| EU | 100-150ms | No |
| Asia | 200-300ms | No |

Global <100ms would require edge compute (Cloudflare Workers) or multi-region deployment.

## Design

**Index**: In-memory inverted index. Source API has no search, so local indexing required. ~500KB fits easily in memory.

**Ranking**: TF-IDF. BM25 would be better for production; kept simple for take-home scope.

**Stemming**: Naive suffix-stripper. Works well enough for this dataset; production would use Porter/Snowball.

**Caching**: LRU cache keyed by `(query, index_version)`. Invalidates on rebuild.

## Cold Start

1. Fetch messages from source (~2s)
2. Build index (~50ms)
3. Ready

Total: ~3s. `/health` returns 503 until ready. Background refresh every 5 minutes.

## Response Payload

~2KB for 10 results, GZip compresses to ~800 bytes.

Not implemented: field selection, snippet extraction, ETag/304 responses.

## Tracing Latency

**Server-side**: `X-Response-Time` header shows total request processing time.

```bash
curl -sI "https://aurora.enea.work/search?q=test" | grep x-response-time
```

**Breakdown** (not currently instrumented, but would add):
- Query parsing/stemming: <0.1ms
- Index lookup: <0.5ms
- TF-IDF scoring: ~1-2ms (scales with result set size)
- JSON serialization: ~0.5ms

**Network latency**:

```bash
curl -so /dev/null -w "DNS:%{time_namelookup} TCP:%{time_connect} TLS:%{time_appconnect} TTFB:%{time_starttransfer} Total:%{time_total}\n" "https://aurora.enea.work/search?q=paris"
# DNS:0.004 TCP:0.025 TLS:0.065 TTFB:0.087 Total:0.088
```

Production: OpenTelemetry spans for detailed server-side breakdown.

## What's Missing

| Feature | Approach |
|---------|----------|
| Fuzzy matching | Levenshtein automaton |
| Phrase search | Positional index |
| Atomic rebuild | Build new index, swap under lock |
| Auth on /refresh | API key or JWT |

## Running Locally

```bash
uv sync
uv run uvicorn search_api.main:app --reload
uv run pytest -v
```
