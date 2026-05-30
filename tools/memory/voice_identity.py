"""
tools/memory/voice_identity.py — account-scoped voiceprint memory.

This is a lightweight local voiceprint implementation. It stores compact audio
feature vectors in Qdrant under a separate collection and filters every query by
memory account. It is not a biometric-grade speaker recognizer, but it gives the
always-on microphone an extra signal before a heavier model is introduced.
"""

from __future__ import annotations

import datetime
import logging
import math
import os
import uuid

import numpy as np
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PayloadSchemaType,
    PointStruct,
    VectorParams,
)

log = logging.getLogger("memory.voice_identity")

COLLECTION = "raphael_voice_identity_memory"
VECTOR_SIZE = 192
SAMPLE_RATE = 16_000


def _now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _pcm16_to_float(pcm_bytes: bytes) -> np.ndarray:
    if not pcm_bytes:
        return np.zeros(0, dtype=np.float32)
    arr = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32)
    if arr.size == 0:
        return arr
    return np.clip(arr / 32768.0, -1.0, 1.0)


def _normalize(vec: np.ndarray) -> list[float]:
    arr = np.asarray(vec, dtype=np.float32).reshape(-1)
    if arr.size < VECTOR_SIZE:
        arr = np.pad(arr, (0, VECTOR_SIZE - arr.size))
    elif arr.size > VECTOR_SIZE:
        arr = arr[:VECTOR_SIZE]
    norm = float(np.linalg.norm(arr))
    if not math.isfinite(norm) or norm <= 1e-8:
        return [0.0] * VECTOR_SIZE
    return (arr / norm).astype(np.float32).tolist()


def voice_embedding_from_pcm(pcm_bytes: bytes, sample_rate: int = SAMPLE_RATE) -> list[float]:
    audio = _pcm16_to_float(pcm_bytes)
    min_samples = int(sample_rate * 0.75)
    if audio.size < min_samples:
        raise ValueError("voice sample too short")

    audio = audio - float(np.mean(audio))
    peak = float(np.max(np.abs(audio))) or 1.0
    audio = np.clip(audio / peak, -1.0, 1.0)

    frame_len = int(sample_rate * 0.025)
    hop = int(sample_rate * 0.010)
    if frame_len <= 0 or hop <= 0 or audio.size < frame_len:
        raise ValueError("voice sample too short")

    frames = []
    window = np.hanning(frame_len).astype(np.float32)
    for start in range(0, audio.size - frame_len + 1, hop):
        frame = audio[start:start + frame_len]
        energy = float(np.mean(frame * frame))
        if energy < 0.0008:
            continue
        spec = np.abs(np.fft.rfft(frame * window, n=512)) ** 2
        spec = np.log1p(spec)
        bands = np.array_split(spec[2:258], 64)
        frames.append([float(np.mean(b)) for b in bands])

    if len(frames) < 6:
        raise ValueError("not enough voiced frames")

    mat = np.asarray(frames, dtype=np.float32)
    mean = mat.mean(axis=0)
    std = mat.std(axis=0)
    if mat.shape[0] > 1:
        delta = np.diff(mat, axis=0)
        dmean = delta.mean(axis=0)
    else:
        dmean = np.zeros_like(mean)
    return _normalize(np.concatenate([mean, std, dmean], axis=0))


class VoiceIdentityMemory:
    """Qdrant-backed voiceprint memory filtered by memory account."""

    def __init__(self, client, *, threshold: float | None = None):
        self.client = client
        self.threshold = threshold if threshold is not None else float(os.environ.get("RAPHAEL_VOICEPRINT_THRESHOLD", "0.82"))
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        existing = {c.name for c in self.client.get_collections().collections}
        if COLLECTION in existing:
            try:
                info = self.client.get_collection(COLLECTION)
                vectors = info.config.params.vectors
                cfg = vectors.get("voice") if isinstance(vectors, dict) else vectors
                if cfg and cfg.size == VECTOR_SIZE:
                    return
            except Exception:
                pass
            log.warning("Voice identity collection format mismatch, recreating.")
            self.client.delete_collection(COLLECTION)

        self.client.create_collection(
            collection_name=COLLECTION,
            vectors_config={
                "voice": VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
            },
        )
        for field, schema in [
            ("user_id", PayloadSchemaType.KEYWORD),
            ("label", PayloadSchemaType.KEYWORD),
        ]:
            try:
                self.client.create_payload_index(COLLECTION, field, schema)
            except Exception:
                pass

    def enroll_pcm(self, pcm_bytes: bytes, *, user_id: str, label: str, source_text: str = "") -> dict:
        try:
            embedding = voice_embedding_from_pcm(pcm_bytes)
        except Exception as e:
            return {"ok": False, "error": f"聲紋樣本不足或不可用: {e}"}

        point_id = str(uuid.uuid4())
        clean_label = (label or "使用者").strip()[:80] or "使用者"
        self.client.upsert(
            collection_name=COLLECTION,
            points=[PointStruct(
                id=point_id,
                vector={"voice": embedding},
                payload={
                    "user_id": user_id,
                    "label": clean_label,
                    "source_text": source_text[:300],
                    "backend": "spectral-lite",
                    "created_at": _now(),
                    "updated_at": "",
                    "access_count": 0,
                },
            )],
        )
        return {"ok": True, "id": point_id, "label": clean_label, "backend": "spectral-lite"}

    def identify_pcm(self, pcm_bytes: bytes, *, user_id: str, threshold: float | None = None) -> dict:
        threshold = self.threshold if threshold is None else float(threshold)
        try:
            embedding = voice_embedding_from_pcm(pcm_bytes)
        except Exception as e:
            return {"ok": False, "error": str(e), "matches": []}

        result = self.client.query_points(
            collection_name=COLLECTION,
            query=embedding,
            using="voice",
            query_filter=Filter(must=[
                FieldCondition(key="user_id", match=MatchValue(value=user_id)),
            ]),
            limit=3,
            with_payload=True,
        )
        matches = []
        for point in result.points:
            score = float(getattr(point, "score", 0.0) or 0.0)
            payload = point.payload or {}
            matched = score >= threshold
            if matched:
                try:
                    self.client.set_payload(
                        collection_name=COLLECTION,
                        payload={
                            "last_heard_at": _now(),
                            "access_count": int(payload.get("access_count", 0) or 0) + 1,
                        },
                        points=[point.id],
                    )
                except Exception:
                    pass
            matches.append({
                "matched": matched,
                "label": payload.get("label", "") if matched else "未知",
                "score": round(score, 4),
                "threshold": threshold,
                "backend": payload.get("backend", "spectral-lite"),
            })
        return {
            "ok": True,
            "matches": [m for m in matches if m["matched"]],
            "candidates": matches,
            "backend": "spectral-lite",
        }

    def count(self, user_id: str) -> int:
        result = self.client.count(
            collection_name=COLLECTION,
            count_filter=Filter(must=[
                FieldCondition(key="user_id", match=MatchValue(value=user_id)),
            ]),
            exact=True,
        )
        return int(result.count)

    def list_identities(self, user_id: str, limit: int = 100) -> list[dict]:
        records, _ = self.client.scroll(
            collection_name=COLLECTION,
            scroll_filter=Filter(must=[
                FieldCondition(key="user_id", match=MatchValue(value=user_id)),
            ]),
            limit=limit,
            with_payload=True,
        )
        rows = []
        for r in records:
            payload = r.payload or {}
            rows.append({
                "id": str(r.id),
                "label": payload.get("label", ""),
                "backend": payload.get("backend", ""),
                "created_at": payload.get("created_at", ""),
                "last_heard_at": payload.get("last_heard_at", ""),
            })
        return rows

    def delete_all(self, user_id: str) -> None:
        self.client.delete(
            collection_name=COLLECTION,
            points_selector=Filter(must=[
                FieldCondition(key="user_id", match=MatchValue(value=user_id)),
            ]),
        )

    def delete_by_id(self, identity_id: str) -> None:
        self.client.delete(collection_name=COLLECTION, points_selector=[identity_id])
