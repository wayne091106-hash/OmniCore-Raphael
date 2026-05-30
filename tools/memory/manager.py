"""
tools/memory/manager.py — 記憶管理器
═════════════════════════════════════
組合 store（Qdrant 存取）+ persona（角色 prompt）+ bridge（事件通知）。

core.py 用法：
    manager = MemoryManager(bridge, user_id="wayne")

    # 建構 system prompt（含相關記憶）
    prompt = manager.build_context("你之前說過什麼？")

    # 處理 Gemini tool call
    result = manager.execute("store_memory", {"memory": "...", "category": "...", "importance": 3})

    # 判斷是否為記憶工具
    if manager.is_memory_tool("store_memory"):
        ...
"""

import logging
import re
import json
import os
from typing import Optional
from pathlib import Path

from bridge import Bridge, Channel
from .persona import (
    build_system_prompt,
    MEMORY_TOOL_NAMES,
)
from .visual_identity import VisualIdentityMemory
from .voice_identity import VoiceIdentityMemory

log = logging.getLogger("memory.manager")
ACCOUNTS_FILE = Path(__file__).resolve().parent / "memory_accounts.json"
MEMORY_TRIGGER_RE = re.compile(
    r"(之前|上次|最新|進度|記得|記憶|回憶|有沒有說過|偏好|喜歡|專案|待辦|聯絡|信箱|email|gmail|帳號|密碼|帳密|登入|login|password|credential|moodle|lms|classroom|portal|網址|url|入口|網站|平台|教學平台|數位學習|學習平台|課程系統|決策|設定|配置|環境|bug|修復|紀錄)",
    re.IGNORECASE,
)
SECRET_FIELD_RE = re.compile(
    r"(?i)(密碼|password|passwd|pwd|passcode|token|secret|client_secret)\s*(?:[:=：]|為|为|是|is)?\s*([^\s,，;；。]+)"
)


def _redact_secrets(text: str) -> str:
    return SECRET_FIELD_RE.sub(lambda m: f"{m.group(1)}=********", text or "")

try:
    from .store import MemoryStore
except Exception as e:
    MemoryStore = None
    _STORE_IMPORT_ERROR = e
else:
    _STORE_IMPORT_ERROR = None


class MemoryManager:
    """
    記憶管理器。

    責任：
      1. 被動注入：每次使用者開口時，自動搜尋相關記憶注入 system prompt
      2. 工具執行：處理 Gemini 的記憶工具呼叫（store / search / delete 等）
      3. 事件通知：寫入記憶時發佈 Channel.MEMORY_WRITE 給 UI
    """

    def __init__(
        self,
        bridge: Bridge,
        user_id: str = "default",
        qdrant_host: str = "127.0.0.1",
        qdrant_port: int = 6333,
    ):
        self._bridge = bridge
        self._user_id = user_id
        self._store: Optional["MemoryStore"] = None
        self._visual_identity: Optional[VisualIdentityMemory] = None
        self._voice_identity: Optional[VoiceIdentityMemory] = None
        self._backend = "qdrant_local"
        self._closed = False
        backend_pref = os.environ.get("RAPHAEL_MEMORY_BACKEND", "qdrant_local").strip().lower()

        if MemoryStore is None:
            raise RuntimeError(
                "Qdrant 記憶依賴缺失。請執行：python ui.py --install-deps memory --with-torch"
            ) from _STORE_IMPORT_ERROR

        try:
            if backend_pref in {"qdrant_remote", "remote", "server"}:
                self._store = MemoryStore(
                    host=qdrant_host,
                    port=qdrant_port,
                    mode="remote",
                )
                self._backend = "qdrant_remote"
                log.info("MemoryManager 使用遠端 Qdrant (user=%s, %s:%d)", user_id, qdrant_host, qdrant_port)
            else:
                raw_path = os.environ.get("QDRANT_PATH", "tools/memory/qdrant_data")
                qdrant_path = Path(raw_path)
                if not qdrant_path.is_absolute():
                    qdrant_path = Path(__file__).resolve().parents[2] / qdrant_path
                self._store = MemoryStore(mode="local", path=qdrant_path)
                self._backend = "qdrant_local"
                log.info("MemoryManager 使用本機 Qdrant (user=%s, path=%s)", user_id, qdrant_path)
        except Exception as e:
            msg = str(e)
            if "already accessed by another instance" in msg or "AlreadyLocked" in msg:
                raise RuntimeError(
                    "Qdrant local 記憶庫已被另一個 Raphael 程序使用。"
                    "請先關閉舊的 python ui.py 視窗或在工作管理員結束舊 Python 程序，"
                    "再重新啟動；若需要同時開多個程序，請改用 RAPHAEL_MEMORY_BACKEND=qdrant_remote。"
                ) from e
            raise RuntimeError(f"Qdrant 記憶初始化失敗: {e}") from e
        self._visual_identity = VisualIdentityMemory(self._store.client)
        self._voice_identity = VoiceIdentityMemory(self._store.client)
        self._ensure_account(user_id)

    def _read_accounts(self) -> list[str]:
        try:
            rows = json.loads(ACCOUNTS_FILE.read_text(encoding="utf-8"))
            if isinstance(rows, list):
                return sorted({str(x).strip() for x in rows if str(x).strip()})
        except Exception:
            pass
        return []

    def _write_accounts(self, accounts: list[str]) -> None:
        ACCOUNTS_FILE.write_text(
            json.dumps(sorted(set(accounts)), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _ensure_account(self, user_id: str) -> None:
        accounts = self._read_accounts()
        if user_id not in accounts:
            accounts.append(user_id)
            self._write_accounts(accounts)

    @property
    def user_id(self) -> str:
        return self._user_id

    @user_id.setter
    def user_id(self, value: str):
        clean = self._clean_account(value)
        self._user_id = clean
        self._ensure_account(clean)

    @property
    def store(self):
        return self._store

    @property
    def visual_identity(self) -> Optional[VisualIdentityMemory]:
        return self._visual_identity

    @property
    def voice_identity(self) -> Optional[VoiceIdentityMemory]:
        return self._voice_identity

    @property
    def backend(self) -> str:
        return self._backend

    @staticmethod
    def _clean_account(value: str) -> str:
        clean = re.sub(r"[^\w.@+-]+", "_", str(value or "").strip(), flags=re.UNICODE)
        clean = clean.strip("._-")
        return clean or "default"

    def list_accounts(self) -> dict:
        accounts = self._read_accounts()
        if self._user_id not in accounts:
            accounts.append(self._user_id)
            self._write_accounts(accounts)
        return {"accounts": sorted(accounts), "current": self._user_id, "backend": self._backend}

    def create_account(self, user_id: str) -> dict:
        clean = self._clean_account(user_id)
        accounts = self._read_accounts()
        created = clean not in accounts
        if created:
            accounts.append(clean)
            self._write_accounts(accounts)
        self._user_id = clean
        return {"created": created, **self.list_accounts()}

    def select_account(self, user_id: str) -> dict:
        self.user_id = user_id
        return self.list_accounts()

    def delete_account(self, user_id: str) -> dict:
        clean = self._clean_account(user_id)
        accounts = [x for x in self._read_accounts() if x != clean]
        if not accounts:
            accounts = ["default"]
        self._write_accounts(accounts)
        if self._store:
            try:
                self._store.delete_all(clean)
            except Exception as e:
                log.warning("刪除帳號記憶失敗: %s", e)
        if self._visual_identity:
            try:
                self._visual_identity.delete_all(clean)
            except Exception as e:
                log.warning("刪除帳號圖像身份記憶失敗: %s", e)
        if self._voice_identity:
            try:
                self._voice_identity.delete_all(clean)
            except Exception as e:
                log.warning("刪除帳號聲紋記憶失敗: %s", e)
        if self._user_id == clean:
            self._user_id = accounts[0]
        return {"deleted": clean, **self.list_accounts()}

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        store = self._store
        self._store = None
        self._visual_identity = None
        self._voice_identity = None
        if store and hasattr(store, "close"):
            try:
                store.close()
                log.info("記憶庫已關閉 (%s)", self._backend)
            except Exception as e:
                log.warning("記憶庫關閉失敗: %s", e)

    # ── 被動注入 ─────────────────────────────────────────────────────

    def build_context(self, user_message: str, limit: int = 4, min_score: float = 0.3) -> str:
        """
        根據使用者訊息搜尋相關記憶，組合成完整的 system prompt。

        回傳含記憶 context 的完整 prompt 字串。
        """
        relevant = self.retrieve(user_message, limit=limit, min_score=min_score)

        context = ""
        if relevant:
            context = "\n".join(f"- {r['memory']}" for r in relevant)

        return build_system_prompt(context)

    def retrieve(self, query: str, limit: int = 4, min_score: float = 0.15) -> list[dict]:
        if not self._store:
            return []
        try:
            retrieved = self._store.search(query, user_id=self._user_id, limit=limit)
        except Exception as e:
            log.warning("記憶搜尋失敗: %s", e)
            return []
        return [r for r in retrieved if (r.get("score") is None or (r.get("score") or 0) > min_score)]

    def retrieve_filtered(
        self,
        query: str,
        *,
        category: str | None = None,
        min_importance: int | None = None,
        limit: int = 5,
        min_score: float = 0.0,
    ) -> list[dict]:
        if not self._store:
            return []
        try:
            retrieved = self._store.filtered_search(
                query,
                user_id=self._user_id,
                category=category,
                min_importance=min_importance,
                limit=limit,
            )
        except Exception as e:
            log.warning("過濾記憶搜尋失敗: %s", e)
            return []
        return [r for r in retrieved if (r.get("score") is None or (r.get("score") or 0) >= min_score)]

    def should_retrieve_for(self, query: str) -> bool:
        text = (query or "").strip()
        if len(text) < 5:
            return False
        return bool(MEMORY_TRIGGER_RE.search(text))

    # ── 工具判斷 ─────────────────────────────────────────────────────

    @staticmethod
    def is_memory_tool(name: str) -> bool:
        return name in MEMORY_TOOL_NAMES

    # ── 工具執行 ─────────────────────────────────────────────────────

    async def execute(self, name: str, args: dict) -> dict:
        """
        執行記憶工具呼叫，回傳結果 dict。

        core.py 收到 Gemini tool call 時：
            if manager.is_memory_tool(name):
                result = await manager.execute(name, args)
        """
        if not self._store:
            return {"error": "記憶系統離線"}

        if name == "store_memory":
            return await self._store_memory(args)
        elif name == "search_memories":
            return self._search_memories(args)
        elif name == "filtered_search_memories":
            return self._filtered_search(args)
        elif name == "get_all_memories":
            return self._get_all(args)
        elif name == "get_memory_stats":
            return self._get_stats()
        elif name == "update_memory_importance":
            return self._update_importance(args)
        elif name == "delete_memory":
            return self._delete(args)
        elif name == "get_visual_identity_stats":
            return self._visual_stats()
        elif name == "delete_visual_identity":
            return self._delete_visual_identity(args)
        elif name == "get_voice_identity_stats":
            return self._voice_stats()
        elif name == "delete_voice_identity":
            return self._delete_voice_identity(args)
        else:
            return {"error": f"未知記憶工具: {name}"}

    # ── 各工具實作 ───────────────────────────────────────────────────

    async def _store_memory(self, args: dict) -> dict:
        memory = args.get("memory", "")
        category = args.get("category", "other")
        importance = args.get("importance", 3)

        mem_id = self._store.store(
            memory, user_id=self._user_id,
            category=category, importance=importance,
        )

        await self._bridge.publish(Channel.MEMORY_WRITE, {
            "memory": _redact_secrets(memory) if category == "credential" else memory,
            "category": category,
            "importance": importance,
        })

        log.info("記憶寫入: [%s/★%d] %s", category, importance, _redact_secrets(memory)[:40])
        return {"stored": True, "id": mem_id[:8], "backend": self._backend}

    def _search_memories(self, args: dict) -> dict:
        query = args.get("query", "")
        limit = args.get("limit", 5)
        results = self._store.search(query, user_id=self._user_id, limit=limit)
        return {"results": results, "count": len(results)}

    def _filtered_search(self, args: dict) -> dict:
        results = self._store.filtered_search(
            args.get("query", ""),
            user_id=self._user_id,
            category=args.get("category"),
            min_importance=args.get("min_importance"),
            limit=args.get("limit", 5),
        )
        return {"results": results, "count": len(results)}

    def _get_all(self, args: dict) -> dict:
        results = self._store.get_all(self._user_id, limit=args.get("limit", 50))
        return {"results": results, "count": len(results)}

    def _get_stats(self) -> dict:
        breakdown = self._store.category_breakdown(self._user_id)
        total = sum(breakdown.values())
        return {"total": total, "breakdown": breakdown, "backend": self._backend}

    def _update_importance(self, args: dict) -> dict:
        self._store.update_importance(args["memory_id"], args["importance"])
        return {"updated": True}

    def _delete(self, args: dict) -> dict:
        self._store.delete_by_id(args["memory_id"])
        return {"deleted": True}

    def enroll_visual_identity_from_jpeg(self, jpeg_bytes: bytes, label: str, source_text: str = "") -> dict:
        if not self._visual_identity:
            return {"ok": False, "error": "圖像身份記憶尚未初始化"}
        return self._visual_identity.enroll_jpeg(
            jpeg_bytes,
            user_id=self._user_id,
            label=label,
            source_text=source_text,
        )

    def identify_visual_identities(self, jpeg_bytes: bytes, limit: int = 5) -> dict:
        if not self._visual_identity:
            return {"ok": False, "error": "圖像身份記憶尚未初始化", "matches": []}
        return self._visual_identity.identify_jpeg(
            jpeg_bytes,
            user_id=self._user_id,
            limit=limit,
        )

    def detect_visual_candidates(self, jpeg_bytes: bytes, limit: int = 5) -> dict:
        if not self._visual_identity:
            return {"ok": False, "error": "圖像身份記憶尚未初始化", "matches": [], "candidates": []}
        frame = self._visual_identity._frame_from_jpeg(jpeg_bytes)
        if frame is None:
            return {"ok": False, "error": "沒有可用的鏡頭影像", "matches": [], "candidates": []}
        faces = self._visual_identity.detect(frame)
        shape = frame.shape[:2]

        def normalize(box):
            h, w = shape
            x1, y1, x2, y2 = box
            return {
                "x1": max(0.0, min(1.0, x1 / max(1, w))),
                "y1": max(0.0, min(1.0, y1 / max(1, h))),
                "x2": max(0.0, min(1.0, x2 / max(1, w))),
                "y2": max(0.0, min(1.0, y2 / max(1, h))),
            }

        candidates = []
        for face in faces[:max(1, int(limit or 5))]:
            candidates.append({
                "matched": False,
                "label": "未知",
                "score": 0.0,
                "backend": face.backend,
                "box": normalize(face.box),
            })
        return {
            "ok": True,
            "faces": len(faces),
            "matches": [],
            "candidates": candidates,
            "backend": self._visual_identity.backend,
        }

    def _visual_stats(self) -> dict:
        if not self._visual_identity:
            return {"enabled": False, "count": 0}
        return {
            "enabled": True,
            "count": self._visual_identity.count(self._user_id),
            "backend": self._visual_identity.backend,
            "identities": self._visual_identity.list_identities(self._user_id, limit=50),
        }

    def enroll_voice_identity_from_pcm(self, pcm_bytes: bytes, label: str, source_text: str = "") -> dict:
        if not self._voice_identity:
            return {"ok": False, "error": "聲紋記憶尚未初始化"}
        return self._voice_identity.enroll_pcm(
            pcm_bytes,
            user_id=self._user_id,
            label=label,
            source_text=source_text,
        )

    def identify_voice_identity(self, pcm_bytes: bytes) -> dict:
        if not self._voice_identity:
            return {"ok": False, "error": "聲紋記憶尚未初始化", "matches": []}
        return self._voice_identity.identify_pcm(
            pcm_bytes,
            user_id=self._user_id,
        )

    def _voice_stats(self) -> dict:
        if not self._voice_identity:
            return {"enabled": False, "count": 0}
        return {
            "enabled": True,
            "count": self._voice_identity.count(self._user_id),
            "backend": "spectral-lite",
            "identities": self._voice_identity.list_identities(self._user_id, limit=50),
        }

    def _delete_visual_identity(self, args: dict) -> dict:
        if not self._visual_identity:
            return {"deleted": False, "error": "圖像身份記憶尚未初始化"}
        identity_id = str(args.get("identity_id", "")).strip()
        if not identity_id:
            return {"deleted": False, "error": "缺少 identity_id"}
        self._visual_identity.delete_by_id(identity_id)
        return {"deleted": True, "identity_id": identity_id}

    def _delete_voice_identity(self, args: dict) -> dict:
        if not self._voice_identity:
            return {"deleted": False, "error": "聲紋記憶尚未初始化"}
        identity_id = str(args.get("identity_id", "")).strip()
        if not identity_id:
            return {"deleted": False, "error": "缺少 identity_id"}
        self._voice_identity.delete_by_id(identity_id)
        return {"deleted": True, "identity_id": identity_id}

    async def maybe_auto_store(self, user_text: str, ai_text: str = "") -> dict | None:
        """Heuristic safety net for facts the model failed to store by tool call."""
        text = (user_text or "").strip()
        if not text or not self._store:
            return None

        lowered = text.lower()
        should_store = False
        category = "other"
        importance = 3

        if re.search(r"[\w.+-]+@[\w.-]+\.[a-zA-Z]{2,}", text):
            should_store = True
            category = "personal"
            importance = 4
        if any(k in text for k in ("記住", "請記得", "我的", "我是", "我叫", "我喜歡", "我偏好")):
            should_store = True
            if any(k in text for k in ("喜歡", "偏好", "習慣")):
                category = "preference"
            elif any(k in text for k in ("專案", "進度", "版本", "功能", "bug", "修復")):
                category = "project"
            else:
                category = "personal"
        if any(k in lowered for k in ("todo", "deadline", "meeting")) or any(k in text for k in ("待辦", "截止", "會議", "下次")):
            should_store = True
            category = "event"
            importance = max(importance, 4)

        if not should_store:
            return None

        memory = text
        if len(memory) > 240:
            memory = memory[:237] + "..."

        return await self._store_memory({
            "memory": memory,
            "category": category,
            "importance": importance,
        })
