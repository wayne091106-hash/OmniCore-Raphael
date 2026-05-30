"""
tools/memory/visual_identity.py — account-scoped visual identity memory.

This stores face/person visual embeddings in Qdrant separately from text memory.
It intentionally has a dependency-light OpenCV fallback so Raphael keeps working
even when dedicated face-recognition packages are not installed.
"""

from __future__ import annotations

import datetime
import logging
import math
import os
import uuid
from dataclasses import dataclass
from typing import Any

import cv2
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

log = logging.getLogger("memory.visual_identity")

COLLECTION = "raphael_identity_memory"
VECTOR_SIZE = 512


@dataclass
class FaceCandidate:
    box: tuple[int, int, int, int]
    embedding: list[float]
    confidence: float
    backend: str


def _now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _norm_vec(vec: np.ndarray) -> list[float]:
    arr = np.asarray(vec, dtype=np.float32).reshape(-1)
    if arr.size < VECTOR_SIZE:
        arr = np.pad(arr, (0, VECTOR_SIZE - arr.size))
    elif arr.size > VECTOR_SIZE:
        arr = arr[:VECTOR_SIZE]
    norm = float(np.linalg.norm(arr))
    if not math.isfinite(norm) or norm <= 1e-8:
        return [0.0] * VECTOR_SIZE
    return (arr / norm).astype(np.float32).tolist()


def _normalize_box(box: tuple[int, int, int, int], shape: tuple[int, int]) -> dict:
    h, w = shape
    x1, y1, x2, y2 = box
    return {
        "x1": max(0.0, min(1.0, x1 / max(1, w))),
        "y1": max(0.0, min(1.0, y1 / max(1, h))),
        "x2": max(0.0, min(1.0, x2 / max(1, w))),
        "y2": max(0.0, min(1.0, y2 / max(1, h))),
    }


class _OpenCvFaceBackend:
    name = "opencv-dct"

    def __init__(self):
        base = getattr(cv2, "data", None)
        haar_dir = getattr(base, "haarcascades", "") if base is not None else ""
        frontal = os.path.join(haar_dir, "haarcascade_frontalface_default.xml")
        profile = os.path.join(haar_dir, "haarcascade_profileface.xml")
        self._frontal = cv2.CascadeClassifier(frontal)
        self._profile = cv2.CascadeClassifier(profile)

    @staticmethod
    def _embedding(face_bgr: np.ndarray) -> list[float]:
        gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        gray = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA)
        arr = gray.astype(np.float32) / 255.0
        arr = arr - float(arr.mean())
        std = float(arr.std()) or 1.0
        arr = arr / std
        dct = cv2.dct(arr)
        return _norm_vec(dct.reshape(-1))

    @staticmethod
    def _merge_boxes(boxes: list[tuple[int, int, int, int]]) -> list[tuple[int, int, int, int]]:
        if not boxes:
            return []
        rects = []
        for x1, y1, x2, y2 in boxes:
            rects.append([x1, y1, x2 - x1, y2 - y1])
        grouped, _ = cv2.groupRectangles(rects + rects, groupThreshold=1, eps=0.25)
        if len(grouped) == 0:
            return boxes
        return [(int(x), int(y), int(x + w), int(y + h)) for x, y, w, h in grouped]

    def detect(self, frame: np.ndarray) -> list[FaceCandidate]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]
        min_size = max(40, min(w, h) // 12)
        boxes: list[tuple[int, int, int, int]] = []

        for cascade in (self._frontal, self._profile):
            if cascade.empty():
                continue
            found = cascade.detectMultiScale(
                gray,
                scaleFactor=1.08,
                minNeighbors=5,
                minSize=(min_size, min_size),
            )
            for x, y, bw, bh in found:
                boxes.append((int(x), int(y), int(x + bw), int(y + bh)))

        boxes = self._merge_boxes(boxes)
        out: list[FaceCandidate] = []
        for x1, y1, x2, y2 in boxes:
            pad_x = int((x2 - x1) * 0.18)
            pad_y = int((y2 - y1) * 0.22)
            cx1 = max(0, x1 - pad_x)
            cy1 = max(0, y1 - pad_y)
            cx2 = min(w, x2 + pad_x)
            cy2 = min(h, y2 + pad_y)
            crop = frame[cy1:cy2, cx1:cx2]
            if crop.size == 0:
                continue
            area_ratio = ((x2 - x1) * (y2 - y1)) / max(1, w * h)
            confidence = max(0.35, min(0.95, 0.45 + area_ratio * 4.0))
            out.append(FaceCandidate((cx1, cy1, cx2, cy2), self._embedding(crop), confidence, self.name))

        out.sort(key=lambda f: (f.box[2] - f.box[0]) * (f.box[3] - f.box[1]), reverse=True)
        return out


class _InsightFaceBackend:
    name = "insightface"

    def __init__(self):
        from insightface.app import FaceAnalysis

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self._app = FaceAnalysis(name="buffalo_l", providers=providers)
        self._app.prepare(ctx_id=0, det_size=(640, 640))

    def detect(self, frame: np.ndarray) -> list[FaceCandidate]:
        faces = self._app.get(frame)
        out: list[FaceCandidate] = []
        h, w = frame.shape[:2]
        for face in faces:
            x1, y1, x2, y2 = [int(v) for v in face.bbox]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            emb = getattr(face, "normed_embedding", None)
            if emb is None:
                emb = getattr(face, "embedding", None)
            if emb is None:
                continue
            confidence = float(getattr(face, "det_score", 0.9) or 0.9)
            out.append(FaceCandidate((x1, y1, x2, y2), _norm_vec(np.asarray(emb)), confidence, self.name))
        out.sort(key=lambda f: f.confidence, reverse=True)
        return out


class VisualIdentityMemory:
    """Qdrant-backed visual identity memory, filtered by memory account."""

    def __init__(self, client, *, threshold: float | None = None):
        self.client = client
        self.threshold = threshold if threshold is not None else float(os.environ.get("RAPHAEL_IDENTITY_THRESHOLD", "0.74"))
        self._backend = self._make_backend()
        self._ensure_collection()

    @property
    def backend(self) -> str:
        return self._backend.name

    def _make_backend(self):
        pref = os.environ.get("RAPHAEL_FACE_BACKEND", "opencv").strip().lower()
        if pref in {"insightface", "auto"}:
            try:
                return _InsightFaceBackend()
            except Exception as e:
                if pref == "insightface":
                    log.warning("InsightFace unavailable, using OpenCV fallback: %s", e)
                else:
                    log.info("InsightFace not ready, using OpenCV visual identity fallback: %s", e)
        return _OpenCvFaceBackend()

    def _ensure_collection(self) -> None:
        existing = {c.name for c in self.client.get_collections().collections}
        if COLLECTION in existing:
            try:
                info = self.client.get_collection(COLLECTION)
                vectors = info.config.params.vectors
                cfg = vectors.get("face") if isinstance(vectors, dict) else vectors
                if cfg and cfg.size == VECTOR_SIZE:
                    return
            except Exception:
                pass
            log.warning("Visual identity collection format mismatch, recreating.")
            self.client.delete_collection(COLLECTION)

        self.client.create_collection(
            collection_name=COLLECTION,
            vectors_config={
                "face": VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
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

    @staticmethod
    def _frame_from_jpeg(jpeg_bytes: bytes) -> np.ndarray | None:
        if not jpeg_bytes:
            return None
        frame = cv2.imdecode(np.frombuffer(jpeg_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        return frame

    def detect(self, frame: np.ndarray) -> list[FaceCandidate]:
        return self._backend.detect(frame)

    def enroll_jpeg(self, jpeg_bytes: bytes, *, user_id: str, label: str, source_text: str = "") -> dict:
        frame = self._frame_from_jpeg(jpeg_bytes)
        if frame is None:
            return {"ok": False, "error": "沒有可用的鏡頭影像"}
        faces = self.detect(frame)
        if not faces:
            return {"ok": False, "error": "目前畫面沒有偵測到可註冊的人臉", "faces": 0}

        face = faces[0]
        point_id = str(uuid.uuid4())
        clean_label = (label or "使用者").strip()[:80] or "使用者"
        payload = {
            "user_id": user_id,
            "label": clean_label,
            "source_text": source_text[:300],
            "backend": face.backend,
            "confidence": float(face.confidence),
            "created_at": _now(),
            "updated_at": "",
            "access_count": 0,
        }
        self.client.upsert(
            collection_name=COLLECTION,
            points=[PointStruct(
                id=point_id,
                vector={"face": face.embedding},
                payload=payload,
            )],
        )
        return {
            "ok": True,
            "id": point_id,
            "label": clean_label,
            "faces": len(faces),
            "backend": face.backend,
            "box": _normalize_box(face.box, frame.shape[:2]),
            "confidence": round(float(face.confidence), 4),
        }

    def identify_jpeg(self, jpeg_bytes: bytes, *, user_id: str, limit: int = 5, threshold: float | None = None) -> dict:
        frame = self._frame_from_jpeg(jpeg_bytes)
        if frame is None:
            return {"ok": False, "error": "沒有可用的鏡頭影像", "matches": []}
        return self.identify_frame(frame, user_id=user_id, limit=limit, threshold=threshold)

    def identify_frame(self, frame: np.ndarray, *, user_id: str, limit: int = 5, threshold: float | None = None) -> dict:
        threshold = self.threshold if threshold is None else float(threshold)
        faces = self.detect(frame)
        matches: list[dict[str, Any]] = []
        shape = frame.shape[:2]
        for face in faces[:max(1, limit)]:
            try:
                result = self.client.query_points(
                    collection_name=COLLECTION,
                    query=face.embedding,
                    using="face",
                    query_filter=Filter(must=[
                        FieldCondition(key="user_id", match=MatchValue(value=user_id)),
                    ]),
                    limit=3,
                    with_payload=True,
                )
            except Exception as e:
                return {"ok": False, "error": str(e), "matches": matches}

            best = result.points[0] if result.points else None
            score = float(getattr(best, "score", 0.0) or 0.0) if best else 0.0
            payload = best.payload if best else {}
            label = payload.get("label", "") if payload else ""
            matched = bool(best and score >= threshold)
            if matched:
                try:
                    self.client.set_payload(
                        collection_name=COLLECTION,
                        payload={"last_seen_at": _now(), "access_count": int(payload.get("access_count", 0) or 0) + 1},
                        points=[best.id],
                    )
                except Exception:
                    pass
            matches.append({
                "matched": matched,
                "label": label if matched else "未知",
                "score": round(score, 4),
                "threshold": threshold,
                "backend": face.backend,
                "box": _normalize_box(face.box, shape),
            })

        return {
            "ok": True,
            "faces": len(faces),
            "matches": [m for m in matches if m["matched"]],
            "candidates": matches,
            "backend": self.backend,
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
                "last_seen_at": payload.get("last_seen_at", ""),
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
