"""Long-term semantic memory backed by ChromaDB (persisted under data/chroma).

Embeddings come from the Gemini embedding model (core.llm.embed) and are passed to Chroma
directly, so Chroma never downloads a local embedding model. Only curated memories are
written (see graph.workflow.finalize): user preferences, human corrections, successful
strategies and high-confidence task decisions. Near-duplicates are skipped.
"""

from __future__ import annotations

import threading
from datetime import datetime
from functools import lru_cache

import chromadb
from chromadb.config import Settings as ChromaSettings

from core.config import get_logger, get_settings
from core.llm import embed
from core.schemas import MemoryRecord, MemoryType, new_id, utcnow

log = get_logger("memory")
COLLECTION = "long_term_memory"


class LongTermMemory:
    def __init__(self) -> None:
        settings = get_settings()
        self._lock = threading.Lock()
        self._client = chromadb.PersistentClient(
            path=str(settings.chroma_dir), settings=ChromaSettings(anonymized_telemetry=False)
        )
        self._collection = self._client.get_or_create_collection(
            COLLECTION, metadata={"hnsw:space": "cosine"}, embedding_function=None
        )

    # -- write ---------------------------------------------------------------------------
    def add(self, content: str, memory_type: MemoryType, task_id: str,
            metadata: dict[str, str | int | float | bool] | None = None) -> tuple[MemoryRecord | None, str]:
        """Store a memory unless a near-duplicate exists. Returns (record | None, reason)."""
        settings = get_settings()
        content = content.strip()
        if len(content) < 10:
            return None, "too short to be useful"
        vector = embed(content, is_query=False)
        with self._lock:
            if self._collection.count():
                near = self._collection.query(query_embeddings=[vector], n_results=1, include=["distances"])
                if near["distances"] and near["distances"][0] and near["distances"][0][0] <= settings.memory_dedupe_distance:
                    return None, f"near-duplicate of {near['ids'][0][0]}"
            record = MemoryRecord(
                memory_id=new_id("mem"), content=content, memory_type=memory_type, task_id=task_id,
                metadata=metadata or {}, embedding_dimensions=len(vector),
            )
            meta = {k: v for k, v in record.metadata.items() if v is not None}
            meta.update({"memory_type": memory_type, "task_id": task_id, "timestamp": record.timestamp.isoformat()})
            self._collection.add(ids=[record.memory_id], documents=[content], embeddings=[vector], metadatas=[meta])
        log.info("Stored %s memory %s", memory_type, record.memory_id)
        return record, "stored"

    # -- read ----------------------------------------------------------------------------
    def search(self, query: str, top_k: int | None = None, max_distance: float | None = None) -> list[MemoryRecord]:
        settings = get_settings()
        top_k = top_k or settings.memory_top_k
        max_distance = settings.memory_max_distance if max_distance is None else max_distance
        with self._lock:
            count = self._collection.count()
        if not count or not query.strip():
            return []
        vector = embed(query, is_query=True)
        with self._lock:
            found = self._collection.query(
                query_embeddings=[vector], n_results=min(top_k, count), include=["documents", "metadatas", "distances"]
            )
        records = []
        for mem_id, doc, meta, dist in zip(found["ids"][0], found["documents"][0], found["metadatas"][0],
                                           found["distances"][0]):
            if dist is None or dist > max_distance:
                continue
            records.append(self._to_record(mem_id, doc, meta, distance=round(float(dist), 4)))
        return records

    def list_all(self, limit: int = 500) -> list[MemoryRecord]:
        with self._lock:
            found = self._collection.get(limit=limit, include=["documents", "metadatas", "embeddings"])
        records = []
        embeddings = found.get("embeddings")
        for i, (mem_id, doc, meta) in enumerate(zip(found["ids"], found["documents"], found["metadatas"])):
            dims = len(embeddings[i]) if embeddings is not None and len(embeddings) > i else None
            records.append(self._to_record(mem_id, doc, meta, dims=dims))
        return sorted(records, key=lambda r: r.timestamp, reverse=True)

    def delete(self, memory_id: str) -> None:
        with self._lock:
            self._collection.delete(ids=[memory_id])
        log.info("Deleted memory %s", memory_id)

    def count(self) -> int:
        with self._lock:
            return self._collection.count()

    @staticmethod
    def _to_record(mem_id: str, doc: str, meta: dict | None, *, distance: float | None = None,
                   dims: int | None = None) -> MemoryRecord:
        meta = dict(meta or {})
        memory_type = meta.pop("memory_type", "project_context")
        task_id = str(meta.pop("task_id", ""))
        ts_raw = meta.pop("timestamp", None)
        try:
            timestamp = datetime.fromisoformat(ts_raw) if ts_raw else utcnow()
        except ValueError:
            timestamp = utcnow()
        return MemoryRecord(memory_id=mem_id, content=doc or "", memory_type=memory_type, task_id=task_id,
                            timestamp=timestamp, metadata=meta, distance=distance, embedding_dimensions=dims)


@lru_cache(maxsize=1)
def get_long_term_memory() -> LongTermMemory:
    return LongTermMemory()
