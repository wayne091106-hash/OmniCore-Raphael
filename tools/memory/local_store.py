"""
tools/memory/local_store.py — local fallback memory store

Small JSON-backed memory store used when Qdrant or embedding dependencies are not
available. It intentionally mirrors the MemoryStore methods used by
MemoryManager so the rest of Raphael can keep working during demos.
"""

import json
import re
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from .persona import CATEGORIES


_TOKEN_RE = re.compile(r"[\w@.+-]+", re.UNICODE)


def _tokens(text: str) -> list[str]:
    raw = [t.lower() for t in _TOKEN_RE.findall(text or "") if len(t) > 1]
    cjk = re.findall(r"[\u4e00-\u9fff]{2,}", text or "")
    grams = []
    for chunk in cjk:
        grams.extend(chunk[i:i + 2] for i in range(max(0, len(chunk) - 1)))
    return raw + grams


class LocalMemoryStore:
    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path else Path(__file__).resolve().parent / "local_memory.json"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self._write([])

    def _read(self) -> list[dict]:
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            return []

    def _write(self, rows: list[dict]) -> None:
        self.path.write_text(
            json.dumps(rows, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _fmt(self, row: dict, score=None) -> dict:
        out = {
            "id": row.get("id", ""),
            "memory": row.get("memory", ""),
            "category": row.get("category", "other"),
            "importance": row.get("importance", 3),
            "created_at": row.get("created_at", ""),
            "score": score,
        }
        if row.get("tags"):
            out["tags"] = row["tags"]
        return out

    def store(self, text: str, user_id: str, category: str = "other",
              importance: int = 3, tags: list[str] | None = None) -> str:
        rows = self._read()
        mem_id = str(uuid.uuid4())
        rows.append({
            "id": mem_id,
            "memory": text.strip(),
            "user_id": user_id,
            "category": category if category in CATEGORIES else "other",
            "importance": max(1, min(5, int(importance or 3))),
            "tags": tags or [],
            "created_at": datetime.now(timezone.utc).isoformat(),
        })
        self._write(rows)
        return mem_id

    def search(self, query: str, user_id: str, limit: int = 5) -> list[dict]:
        q = Counter(_tokens(query))
        rows = [r for r in self._read() if r.get("user_id") == user_id]
        if not q:
            rows.sort(key=lambda r: (r.get("importance", 3), r.get("created_at", "")), reverse=True)
            return [self._fmt(r, 0.0) for r in rows[:limit]]

        scored = []
        for row in rows:
            text = " ".join([
                row.get("memory", ""),
                row.get("category", ""),
                " ".join(row.get("tags", [])),
            ])
            mt = Counter(_tokens(text))
            overlap = sum(min(q[t], mt[t]) for t in q)
            substring_hit = 1 if query and query in text else 0
            email_hit = 1 if "@" in query and any("@" in t for t in _tokens(text)) else 0
            score = overlap / max(1, len(q)) + substring_hit + email_hit + (row.get("importance", 3) * 0.02)
            if score > 0:
                scored.append((score, row))

        scored.sort(key=lambda item: (item[0], item[1].get("created_at", "")), reverse=True)
        return [self._fmt(row, round(score, 4)) for score, row in scored[:limit]]

    def filtered_search(self, query: str, user_id: str, category: str | None = None,
                        min_importance: int | None = None, limit: int = 5,
                        use_hybrid: bool = True) -> list[dict]:
        results = self.search(query, user_id, limit=limit * 4)
        if category:
            results = [r for r in results if r.get("category") == category]
        if min_importance is not None:
            results = [r for r in results if r.get("importance", 0) >= min_importance]
        return results[:limit]

    def get_all(self, user_id: str, limit: int = 200) -> list[dict]:
        rows = [r for r in self._read() if r.get("user_id") == user_id]
        rows.sort(key=lambda r: r.get("created_at", ""), reverse=True)
        return [self._fmt(r, None) for r in rows[:limit]]

    def category_breakdown(self, user_id: str) -> dict[str, int]:
        counts = {cat: 0 for cat in CATEGORIES}
        for row in self._read():
            if row.get("user_id") == user_id:
                cat = row.get("category", "other")
                counts[cat if cat in counts else "other"] += 1
        return counts

    def update_importance(self, memory_id: str, importance: int):
        rows = self._read()
        for row in rows:
            if row.get("id") == memory_id:
                row["importance"] = max(1, min(5, int(importance)))
                break
        self._write(rows)

    def delete_by_id(self, memory_id: str):
        self._write([r for r in self._read() if r.get("id") != memory_id])

    def delete_all(self, user_id: str):
        self._write([r for r in self._read() if r.get("user_id") != user_id])
