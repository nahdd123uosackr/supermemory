"""Self-hosted Supermemory-compatible vector memory API.

Implements the subset of the cloud Supermemory API that Hermes / external
agents use (see supermemory-operations skill api-reference.md):

  GET  /health
  POST /v3/documents            add single document (embed + store)
  POST /v3/documents/batch     batch add
  GET  /v3/documents           list documents (query: containerTag?, limit, page)
  DELETE /v3/documents/{id}    delete one document
  POST /v3/documents/bulk      bulk delete by ids/containerTag
  POST /v3/search               document-chunk search
  POST /v4/search               memory search (memories / hybrid / documents)
  POST /v4/profile              profile + optional search results
  DELETE /v4/memories          forget memory by id/content (soft delete)
  GET  /v3/settings  POST /v3/settings   profile buckets settings

Storage: PostgreSQL + pgvector (documents table + settings table).
Embeddings: OpenAI-compatible /v1/embeddings upstream (new-api).
Auth: Bearer <SUPERMEMORY_API_KEY>, except /health.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import uuid
from typing import Any, Optional

import httpx
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from psycopg_pool import AsyncConnectionPool

VERSION = "1.0.0"

DATABASE_URL = os.environ["DATABASE_URL"]
EMBEDDING_BASE_URL = os.environ.get("EMBEDDING_BASE_URL", "").rstrip("/")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "nvidia/nemotron-3-embed-1b")
EMBEDDING_DIMENSIONS = int(os.environ.get("EMBEDDING_DIMENSIONS", "2048"))
EMBEDDING_API_KEY = os.environ.get("EMBEDDING_API_KEY", "")
API_KEY = os.environ.get("SUPERMEMORY_API_KEY", "")

app = FastAPI(title="supermemory-api", version=VERSION)
pool: Optional[AsyncConnectionPool] = None


# ---------------------------------------------------------------- auth

async def require_auth(authorization: str = Header(default="")) -> None:
    if not API_KEY:
        return
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or token.strip() != API_KEY:
        raise HTTPException(status_code=401, detail="invalid api key")


# ---------------------------------------------------------------- models

class DocAdd(BaseModel):
    content: str
    container_tag: Optional[str] = None
    container_tags: Optional[list[str]] = None
    containerTag: Optional[str] = None
    containerTags: Optional[list[str]] = None
    custom_id: Optional[str] = None
    customId: Optional[str] = None
    metadata: Optional[dict] = None
    task_type: Optional[str] = None
    dreaming: Optional[str] = None
    entity_context: Optional[str] = None
    filepath: Optional[str] = None


class DocBatch(BaseModel):
    documents: list[DocAdd]


class SearchBody(BaseModel):
    q: str
    container_tag: Optional[str] = None
    container_tags: Optional[list[str]] = None
    containerTag: Optional[str] = None
    containerTags: Optional[list[str]] = None
    limit: int = 5
    threshold: Optional[float] = None
    chunk_threshold: Optional[float] = None
    search_mode: Optional[str] = None
    include_full_docs: Optional[bool] = None
    include_summary: Optional[bool] = None
    doc_id: Optional[str] = None
    filepath: Optional[str] = None
    rerank: Optional[bool] = None
    rewrite_query: Optional[bool] = None


class ProfileBody(BaseModel):
    container_tag: Optional[str] = None
    containerTag: Optional[str] = None
    q: Optional[str] = None
    threshold: Optional[float] = None
    include: Optional[list[str]] = None


class ForgetBody(BaseModel):
    container_tag: Optional[str] = None
    containerTag: Optional[str] = None
    id: Optional[str] = None
    content: Optional[str] = None
    ids: Optional[list[str]] = None
    reason: Optional[str] = None


class BulkDeleteBody(BaseModel):
    ids: Optional[list[str]] = None
    container_tag: Optional[str] = None
    containerTag: Optional[str] = None
    container_tags: Optional[list[str]] = None


# ---------------------------------------------------------------- helpers

def tags_of(body: BaseModel, default: str = "default") -> list[str]:
    tags: list[str] = []
    for attr in ("container_tags", "containerTags", "container_tag", "containerTag"):
        v = getattr(body, attr, None)
        if isinstance(v, list):
            tags += [str(t) for t in v if t]
        elif isinstance(v, str) and v:
            tags.append(v)
    return tags or [default]


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime())


async def embed(texts: list[str]) -> list[list[float]]:
    async with httpx.AsyncClient(timeout=90.0) as c:
        r = await c.post(
            f"{EMBEDDING_BASE_URL}/embeddings",
            headers={"Authorization": f"Bearer {EMBEDDING_API_KEY}",
                     "Content-Type": "application/json"},
            json={"model": EMBEDDING_MODEL, "input": texts},
        )
        r.raise_for_status()
        data = r.json().get("data", [])
    vecs = [sorted(d.items()) and d["embedding"] for d in sorted(data, key=lambda d: d.get("index", 0))]
    return vecs


def vec_literal(v: list[float]) -> str:
    return "[" + ",".join(f"{x:.6f}" for x in v) + "]"


async def ensure_schema() -> None:
    async with pool.connection() as conn:
        await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        await conn.execute(
            """CREATE TABLE IF NOT EXISTS documents (
                 id TEXT PRIMARY KEY,
                 content TEXT NOT NULL,
                 metadata JSONB NOT NULL DEFAULT '{}',
                 embedding vector(%s),
                 container_tag TEXT NOT NULL DEFAULT 'default',
                 status TEXT NOT NULL DEFAULT 'processed',
                 created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                 updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
               )""" % EMBEDDING_DIMENSIONS)
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_documents_container_tag ON documents(container_tag)")
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_documents_created_at ON documents(created_at DESC)")
        await conn.execute(
            """CREATE TABLE IF NOT EXISTS settings (
                 id TEXT PRIMARY KEY DEFAULT 'default',
                 data JSONB NOT NULL DEFAULT '{}',
                 updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
               )""")


async def store_doc(content: str, tags: list[str], metadata: dict,
                    custom_id: Optional[str] = None) -> dict:
    vecs = await embed([content])
    if not vecs or not vecs[0]:
        raise HTTPException(status_code=502, detail="embedding upstream returned no vector")
    doc_id = custom_id or str(uuid.uuid4())
    tag = tags[0] if tags else "default"
    async with pool.connection() as conn:
        await conn.execute(
            """INSERT INTO documents (id, content, metadata, embedding, container_tag, status)
               VALUES (%s, %s, %s, %s::vector, %s, 'processed')
               ON CONFLICT (id) DO UPDATE SET content=EXCLUDED.content,
                 metadata=EXCLUDED.metadata, embedding=EXCLUDED.embedding,
                 container_tag=EXCLUDED.container_tag, updated_at=now()""",
            (doc_id, content, json.dumps(metadata or {}), vec_literal(vecs[0]), tag))
    return {"id": doc_id, "status": "processed"}


async def search_docs(q: str, tags: Optional[list[str]], limit: int,
                      threshold: Optional[float], full: bool = False,
                      summary: bool = False) -> list[dict]:
    vecs = await embed([q])
    if not vecs or not vecs[0]:
        raise HTTPException(status_code=502, detail="embedding upstream returned no vector")
    v = vec_literal(vecs[0])
    where = "WHERE status='processed'"
    params: list[Any] = []
    if tags:
        where += " AND container_tag = ANY(%s)"
        params.append(tags)
    # pgvector cosine distance: 1 - similarity. threshold is similarity cutoff.
    having = ""
    if threshold is not None:
        having = f" AND (1 - (embedding <=> '{v}'::vector)) >= %s"
        params.append(float(threshold))
    params = [v] + params
    sql = f"""SELECT id, content, metadata, created_at, updated_at,
                     1 - (embedding <=> %s::vector) AS score
              FROM documents {where}{having}
              ORDER BY embedding <=> %s::vector LIMIT %s"""
    params = params + [v, max(1, min(limit, 50))]
    out = []
    async with pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(sql, params)
            rows = await cur.fetchall()
            cols = [d[0] for d in cur.description]
            for row in rows:
                rec = dict(zip(cols, row))
                chunks = [{"content": (rec["content"] or "")[:2000],
                           "documentId": rec["id"], "score": float(rec["score"])}]
                item: dict[str, Any] = {
                    "chunks": chunks,
                    "createdAt": rec["created_at"].isoformat() if rec["created_at"] else "",
                    "documentId": rec["id"],
                    "metadata": rec["metadata"] or {},
                    "score": float(rec["score"]),
                    "updatedAt": rec["updated_at"].isoformat() if rec["updated_at"] else "",
                }
                if full:
                    item["content"] = rec["content"]
                if summary:
                    item["summary"] = (rec["content"] or "")[:500]
                out.append(item)
    return out


async def search_memories(q: str, tags: Optional[list[str]], limit: int,
                          threshold: Optional[float]) -> list[dict]:
    docs = await search_docs(q, tags, limit, threshold)
    out = []
    for d in docs:
        text = (d.get("content") or
                (d["chunks"][0]["content"] if d.get("chunks") else ""))
        out.append({
            "id": d["documentId"],
            "metadata": d.get("metadata") or {},
            "similarity": d.get("score", 0.0),
            "updatedAt": d.get("updatedAt", ""),
            "memory": text if isinstance(text, str) else "",
        })
    return out


# ---------------------------------------------------------------- routes

@app.on_event("startup")
async def _startup() -> None:
    global pool
    pool = AsyncConnectionPool(DATABASE_URL, min_size=1, max_size=10,
                               open=False, kwargs={"autocommit": True})
    await pool.open()
    await ensure_schema()


@app.on_event("shutdown")
async def _shutdown() -> None:
    if pool is not None:
        await pool.close()


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "version": VERSION}


@app.post("/v3/documents")
async def add_document(body: DocAdd, _: None = Depends(require_auth)) -> dict:
    if not (body.content or "").strip():
        raise HTTPException(status_code=400, detail="content is required")
    return await store_doc(body.content.strip(), tags_of(body),
                           body.metadata or {}, body.custom_id or body.customId)


@app.post("/v3/documents/batch")
async def batch_add(body: DocBatch, _: None = Depends(require_auth)) -> dict:
    results, ok, fail = [], 0, 0
    for d in body.documents:
        try:
            if not (d.content or "").strip():
                raise ValueError("content is required")
            r = await store_doc(d.content.strip(), tags_of(d),
                                d.metadata or {}, d.custom_id or d.customId)
            results.append({"id": r["id"], "status": "done"})
            ok += 1
        except Exception as e:  # per-item failure must not abort the batch
            results.append({"id": "", "status": "error", "error": str(e)[:300]})
            fail += 1
    return {"success": ok, "failed": fail, "results": results}


@app.get("/v3/documents")
async def list_documents(containerTag: Optional[str] = None,
                         container_tag: Optional[str] = None,
                         limit: int = 20, page: int = 1,
                         _: None = Depends(require_auth)) -> dict:
    tag = containerTag or container_tag
    limit = max(1, min(int(limit), 100))
    offset = max(0, (max(1, int(page)) - 1) * limit)
    async with pool.connection() as conn:
        async with conn.cursor() as cur:
            if tag:
                await cur.execute(
                    "SELECT id, content, metadata, container_tag, created_at, updated_at"
                    " FROM documents WHERE container_tag=%s"
                    " ORDER BY created_at DESC LIMIT %s OFFSET %s",
                    (tag, limit, offset))
            else:
                await cur.execute(
                    "SELECT id, content, metadata, container_tag, created_at, updated_at"
                    " FROM documents ORDER BY created_at DESC LIMIT %s OFFSET %s",
                    (limit, offset))
            rows = await cur.fetchall()
    return {"documents": [
        {"id": r[0], "content": r[1], "metadata": r[2] or {},
         "container_tag": r[3],
         "created_at": r[4].isoformat() if r[4] else "",
         "updated_at": r[5].isoformat() if r[5] else ""}
        for r in rows], "page": page, "limit": limit}


@app.delete("/v3/documents/{doc_id}")
async def delete_document(doc_id: str, _: None = Depends(require_auth)) -> dict:
    async with pool.connection() as conn:
        cur = await conn.execute("DELETE FROM documents WHERE id=%s", (doc_id,))
        n = cur.rowcount
    return {"deleted": n, "id": doc_id}


@app.post("/v3/documents/bulk")
async def bulk_delete(body: BulkDeleteBody, _: None = Depends(require_auth)) -> dict:
    tags = tags_of(body, default="")
    tags = [t for t in tags if t]
    async with pool.connection() as conn:
        if body.ids:
            cur = await conn.execute(
                "DELETE FROM documents WHERE id = ANY(%s)", (body.ids,))
        elif tags:
            cur = await conn.execute(
                "DELETE FROM documents WHERE container_tag = ANY(%s)", (tags,))
        else:
            raise HTTPException(status_code=400,
                                detail="ids or container_tag is required")
        n = cur.rowcount
    return {"deleted": n}


@app.post("/v3/search")
async def search_v3(body: SearchBody, _: None = Depends(require_auth)) -> dict:
    t0 = time.time()
    tags = tags_of(body, default="")
    results = await search_docs(body.q, tags or None, body.limit,
                               body.chunk_threshold if body.chunk_threshold is not None
                               else body.threshold,
                               full=bool(body.include_full_docs),
                               summary=bool(body.include_summary))
    ms = (time.time() - t0) * 1000
    return {"results": results, "timing": ms, "total": len(results)}


@app.post("/v4/search")
async def search_v4(body: SearchBody, _: None = Depends(require_auth)) -> dict:
    t0 = time.time()
    tags = tags_of(body, default="")
    mode = (body.search_mode or "memories").lower()
    if mode == "documents":
        docs = await search_docs(body.q, tags or None, body.limit, body.threshold)
        out = [{"id": d["documentId"], "chunk": d["chunks"][0]["content"],
                "similarity": d["score"], "updatedAt": d["updatedAt"],
                "metadata": d.get("metadata") or {}} for d in docs]
    elif mode == "hybrid":
        docs = await search_docs(body.q, tags or None, body.limit, body.threshold,
                                full=True)
        out = []
        for d in docs:
            out.append({"id": d["documentId"], "memory": d.get("content", ""),
                        "similarity": d["score"], "updatedAt": d["updatedAt"],
                        "metadata": d.get("metadata") or {},
                        "chunks": d.get("chunks", [])})
    else:
        out = await search_memories(body.q, tags or None, body.limit, body.threshold)
    ms = (time.time() - t0) * 1000
    return {"results": out, "timing": ms, "total": len(out)}


@app.post("/v4/profile")
async def profile(body: ProfileBody, _: None = Depends(require_auth)) -> dict:
    tag = body.container_tag or body.containerTag or "default"
    async with pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute("SELECT data FROM settings WHERE id=%s", (tag,))
            row = await cur.fetchone()
            buckets = (row[0] if row else {}) or {}
            await cur.execute(
                "SELECT content FROM documents WHERE container_tag=%s"
                " AND status='processed' ORDER BY created_at DESC LIMIT 50", (tag,))
            recent = [(r[0] or "") for r in await cur.fetchall()]
    # naive split: older half = static, newer half = dynamic
    half = max(1, len(recent) // 2)
    resp: dict[str, Any] = {
        "profile": {"static": recent[half:half + 10], "dynamic": recent[:10],
                    **({"buckets": buckets} if buckets else {})},
    }
    if body.q:
        results = await search_memories(body.q, [tag], 5, body.threshold)
        resp["searchResults"] = {"results": results, "timing": 0, "total": len(results)}
    return resp


@app.delete("/v4/memories")
async def forget(body: ForgetBody, _: None = Depends(require_auth)) -> dict:
    tag = body.container_tag or body.containerTag or "default"
    async with pool.connection() as conn:
        if body.ids:
            cur = await conn.execute(
                "UPDATE documents SET status='forgotten' WHERE id = ANY(%s)", (body.ids,))
            return {"forgotten": cur.rowcount}
        if body.id:
            cur = await conn.execute(
                "UPDATE documents SET status='forgotten' WHERE id=%s", (body.id,))
            return {"forgotten": cur.rowcount}
        if body.content:
            cur = await conn.execute(
                "UPDATE documents SET status='forgotten'"
                " WHERE container_tag=%s AND content=%s", (tag, body.content))
            return {"forgotten": cur.rowcount}
        # wipe whole container tag (explicit, matches old API contract)
        cur = await conn.execute(
            "UPDATE documents SET status='forgotten' WHERE container_tag=%s", (tag,))
        return {"forgotten": cur.rowcount}


@app.get("/v3/settings")
async def get_settings(_: None = Depends(require_auth)) -> dict:
    async with pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute("SELECT id, data FROM settings")
            rows = await cur.fetchall()
    return {"settings": {r[0]: r[1] for r in rows}}


@app.post("/v3/settings")
async def update_settings(body: dict, _: None = Depends(require_auth)) -> dict:
    buckets = body.get("profile_buckets") or body.get("buckets") or {}
    tag = body.get("container_tag") or body.get("containerTag") or "default"
    async with pool.connection() as conn:
        await conn.execute(
            "INSERT INTO settings (id, data) VALUES (%s, %s)"
            " ON CONFLICT (id) DO UPDATE SET data=EXCLUDED.data, updated_at=now()",
            (tag, json.dumps(buckets if isinstance(buckets, dict) else {"raw": buckets})))
    return {"ok": True}


@app.exception_handler(HTTPException)
async def _http_exc(_: Any, exc: HTTPException) -> JSONResponse:
    return JSONResponse(status_code=exc.status_code, content={"error": exc.detail})
