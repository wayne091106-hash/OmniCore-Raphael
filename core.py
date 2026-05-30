"""
core.py — Raphael 核心（大腦）
═══════════════════════════════
從 gemini_live_core.py 拆出 Gemini WebSocket 管理，
加上 bridge 整合 + 半主動邏輯。

責任：
  1. Gemini Live WebSocket 連線管理（setup / recv / reconnect）
  2. 感知資料轉發：AUDIO_IN → Gemini、VIDEO_IN → Gemini（節流 1fps）
  3. Gemini 事件回發：轉錄 / 音訊 / 狀態 → bridge → UI
  4. 半主動邏輯：PROACTIVE → 組裝 context → send_text_realtime → Gemini 開口
  5. 工具路由：記憶工具 → MemoryManager / 簡易工具 → 本地

用法：
    bridge = Bridge()
    core = RaphaelCore(bridge)
    await core.start()
    ...
    await core.stop()
"""

import asyncio
import base64
import collections
import contextlib
import json
import logging
import os
import re
import ssl
import time

from dotenv import load_dotenv

load_dotenv()

import websockets
import websockets.exceptions

from bridge import Bridge, Channel
from tools.function_call.agent import (
    MinimaxToolAgent,
    TOOL_AGENT_TOOLS,
    route_user_request_for_tools,
    tool_route_requires_delegate,
)
from tools.function_call.implementations import site_memory_search
from tools.memory.manager import MemoryManager
from tools.memory.persona import MEMORY_TOOL_DECLS

log = logging.getLogger("core")


_THINKING_MAP = {
    "": None,
    "off": None,
    "none": None,
    "minimal": "minimal",
    "low": "low",
    "med": "medium",
    "medium": "medium",
    "high": "high",
}

# ══════════════════════════════════════════════════════════════════════════════
# Config（.env 統一放根目錄）
# ══════════════════════════════════════════════════════════════════════════════

class Config:
    API_KEY: str = os.environ.get("GEMINI_API_KEY", "")
    MODEL: str = "gemini-3.1-flash-live-preview"

    VOICE: str = "Puck"
    THINKING: str | None = None
    # 語音語言（BCP-47）。半階聯 live 模型（非 native-audio）可用 speechConfig.languageCode 釘住語言，
    # 避免台灣腔中文被自動偵測誤判成韓文/英文。設空字串則交給模型自動偵測。
    # 台灣中文常用：cmn-Hant-TW（北京話/繁體/台灣）；亦可試 zh-TW / cmn-CN。
    SPEECH_LANGUAGE: str = os.environ.get("RAPHAEL_SPEECH_LANGUAGE", "zh-TW")

    MIC_RATE: int = 16_000
    SPK_RATE: int = 24_000

    VIDEO_FPS_TO_GEMINI: float = 1.0
    VISUAL_IDENTITY_SCAN_FPS: float = float(os.environ.get("RAPHAEL_IDENTITY_SCAN_FPS", "3.0"))
    VOICE_IDENTITY_SCAN_GAP: float = float(os.environ.get("RAPHAEL_VOICE_SCAN_GAP", "1.4"))
    MOUTH_SYNC_THRESHOLD: float = float(os.environ.get("RAPHAEL_MOUTH_SYNC_THRESHOLD", "0.045"))
    VISUAL_IDENTITY_SEEN_COOLDOWN: float = float(os.environ.get("RAPHAEL_IDENTITY_SEEN_COOLDOWN", "60"))
    PROACTIVE_MIN_GAP: float = float(os.environ.get("RAPHAEL_PROACTIVE_MIN_GAP", "2.5"))
    PROACTIVE_REPEAT_GAP: float = float(os.environ.get("RAPHAEL_PROACTIVE_REPEAT_GAP", "4"))
    PROACTIVE_AFTER_USER_GRACE: float = float(os.environ.get("RAPHAEL_PROACTIVE_AFTER_USER_GRACE", "2"))
    PROACTIVE_AFTER_ASSISTANT_GRACE: float = float(os.environ.get("RAPHAEL_PROACTIVE_AFTER_ASSISTANT_GRACE", "1.5"))
    PROACTIVE_AUDIO_CONTEXT_WINDOW: float = float(os.environ.get("RAPHAEL_PROACTIVE_AUDIO_CONTEXT_WINDOW", "8"))
    # 認知心跳：閒置這麼久後，定期戳模型自我檢視是否有值得主動說/做的事（純文字、不送畫面）。
    PROACTIVE_HEARTBEAT_SEC: float = float(os.environ.get("RAPHAEL_PROACTIVE_HEARTBEAT_SEC", "30"))

    WS_URL: str = (
        "wss://generativelanguage.googleapis.com/ws/"
        "google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent"
        f"?key={API_KEY}"
    )
    PING_INTERVAL: int = 20
    MAX_TOOL_ROUNDS: int = int(os.environ.get("MAX_TOOL_ROUNDS", "16"))
    AUTO_MEMORY_SAFETY_NET: bool = os.environ.get("RAPHAEL_AUTO_MEMORY_SAFETY_NET", "0").strip().lower() in {"1", "true", "yes"}

    QDRANT_HOST: str = os.environ.get("QDRANT_HOST", "127.0.0.1")
    QDRANT_PORT: int = int(os.environ.get("QDRANT_PORT", "6333"))
    USER_ID: str = os.environ.get("RAPHAEL_USER", "wayne")


# ══════════════════════════════════════════════════════════════════════════════
# Gemini Live 簡易工具（直接跑在 core，不經額外模型）
# ══════════════════════════════════════════════════════════════════════════════

_SIMPLE_TOOL_DECLS = {
    "functionDeclarations": [
        {
            "name": "get_current_time",
            "description": "取得目前的本地時間",
            "parameters": {"type": "OBJECT", "properties": {}, "required": []},
        },
    ]
}

_SIMPLE_TOOL_NAMES = {"get_current_time"}


_TOOL_AGENT_DECLS = {
    "functionDeclarations": [
        {
            "name": "delegate_tool_task",
            "description": (
                "把需要非記憶工具的任務委派給 Minimax 工具代理。"
                "適用於網路搜尋、Gmail、Calendar、Drive、Sheets、檔案、系統、API、計算、網站登入與瀏覽器/電腦操作等工具任務。"
                "Gemini 必須先整理清楚目標、限制、必要上下文與已查到的記憶，再呼叫此工具。"
            ),
            "parameters": {
                "type": "OBJECT",
                "properties": {
                    "task": {
                        "type": "STRING",
                        "description": "交給 Minimax 的完整任務提示，需包含目標、必要資訊、成功標準與輸出格式。",
                    },
                    "memory_context": {
                        "type": "STRING",
                        "description": "Gemini 已查到且與任務相關的記憶。沒有則填空字串。",
                    },
                },
                "required": ["task"],
            },
        }
    ]
}

_TOOL_AGENT_NAMES = {"delegate_tool_task"}


_LOOK_TOOL_DECLS = {
    "functionDeclarations": [
        {
            "name": "look_now",
            "description": (
                "擷取攝影機此刻的畫面並讓你看見它。"
                "你平常不會持續看到鏡頭畫面；當你需要知道現在畫面上有什麼、"
                "或使用者問及眼前的東西時，先呼叫這個工具，下一輪你就能根據畫面回答。"
                "若視覺來源已關閉或沒有畫面，會回傳對應狀態。"
            ),
            "parameters": {"type": "OBJECT", "properties": {}, "required": []},
        }
    ]
}

_LOOK_TOOL_NAMES = {"look_now"}

# 展覽定點 demo 預設：純 VAD 收音，關閉所有「是否在對我說話」的身份偵測閘門。
# 對嘴、聲紋、視覺身份在多人觀眾的現場會抓不到主講人、甚至抓錯人，反而擋掉真正的輸入。
# 這些功能程式碼與 UI 開關都保留，未來做隨身腦內 AI 時可在 UI 重新開啟。
_SECRET_FIELD_RE = re.compile(
    r"(?i)(密碼|password|passwd|pwd|passcode|token|secret|client_secret)\s*(?:[:=：]|為|为|是|is)?\s*([^\s,，;；。]+)"
)


def _redact_secrets_for_ui(value):
    if isinstance(value, str):
        return _SECRET_FIELD_RE.sub(lambda m: f"{m.group(1)}=********", value)
    if isinstance(value, list):
        return [_redact_secrets_for_ui(v) for v in value]
    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            if re.search(r"(?i)(密碼|password|passwd|pwd|passcode|token|secret|client_secret)", str(key)):
                out[key] = "********"
            else:
                out[key] = _redact_secrets_for_ui(item)
        return out
    return value

_DEFAULT_FEATURES = {
    "vision_gate": True,            # 三軌本地偵測（CLIP/差幀/光流）保留，準確且省 token
    "vision_overlay": True,
    "vision_proactive": True,       # 主動性的觸發來源
    "visual_identity": False,       # 現場會認錯人 → 關閉
    "visual_identity_auto_enroll": False,
    "voice_identity": False,        # 多人環境聲紋不可靠 → 關閉
    "voice_identity_auto_enroll": False,
    "mouth_sync": False,            # 定點鏡頭對不到主講人嘴型 → 關閉
    "advanced_voice_gate": False,   # 關閉後 _audio_forward_loop 走純 VAD 路徑
    "computer_tools": True,
    # 全雙工：預設關＝半雙工（AI 說話期間暫停收音，避免喇叭→麥克風回授把 AI 自己的聲音
    # 當成空白的新一輪而打斷自己）。喇叭場景必須關；接耳機/指向麥克風時可在 UI 開啟恢復 barge-in。
    "full_duplex": False,
    # 語音打斷：當偵測到使用者開始說話時，自動送出中斷信號給 Gemini。
    # 環境吵雜時建議關閉，否則會一直誤觸發打斷。
    "voice_interrupt": False,
}

_SMALLTALK_RE = re.compile(
    r"^\s*(你好|您好|嗨|哈囉|哈啰|hello|hi|hey|hi\s+there|早安|午安|晚安|在嗎|在不在)(啊|呀|喔|哦|唷|ㄚ)?[！!。.\s~～]*$",
    re.IGNORECASE,
)

_PUBLIC_WEB_TASK_RE = re.compile(
    r"(網路|線上|公開|搜尋|找一張|圖片|照片|截圖|新聞|維基|天氣|匯率|股票|資料查詢)",
    re.IGNORECASE,
)

_EXPLICIT_MEMORY_FOR_DELEGATE_RE = re.compile(
    r"(之前|上次|記得|記憶|我有沒有說過|偏好|我的|聯絡方式|信箱|email|gmail|帳號|密碼|帳密|登入|login|password|credential|moodle|lms|classroom|portal|網址|url|入口|網站|平台|教學平台|數位學習|學習平台|課程系統|專案|進度)",
    re.IGNORECASE,
)

_LOGIN_OR_SITE_TASK_RE = re.compile(
    r"(登入|login|帳號|密碼|帳密|credential|moodle|lms|classroom|網站|平台|後台|portal|course|課程|作業|教學平台|數位學習|學習平台|課程系統)",
    re.IGNORECASE,
)

_SITE_QUERY_STOPWORDS = {
    "登入", "帳號", "密碼", "帳密", "網站", "平台", "入口", "課程", "作業",
    "請", "幫我", "使用者", "學生", "Google", "google", "login", "course", "portal",
    "Raphael", "已學網站入口", "已知失敗網址", "優先入口", "失敗", "原因",
    "標題", "備註", "教學平台", "數位學習",
    "http", "https", "www",
}

_SITE_SERVICE_TERMS = {
    "moodle", "lms", "classroom", "portal", "login", "sso",
    "教學平台", "數位學習", "學習平台", "後台", "課程系統",
}

_VISUAL_SELF_RE = re.compile(
    r"(這是我|这是我|這個是我|这个是我|記住我|记住我|認得我|认得我|remember\s+me)",
    re.IGNORECASE,
)

_VISUAL_LABEL_RE = re.compile(
    r"(?:這是|这是|這個人是|这个人是|他是|她是|記住這個人叫|记住这个人叫)\s*([\w\u4e00-\u9fff .@+-]{1,40})",
    re.IGNORECASE,
)

_VOICE_SELF_RE = re.compile(
    r"(這是我的聲音|这是我的声音|記住我的聲音|记住我的声音|認得我的聲音|认得我的声音|記住我說話|记住我说话|我的聲紋|我的声纹|remember\s+my\s+voice|voiceprint)",
    re.IGNORECASE,
)

_VOICE_LABEL_RE = re.compile(
    r"(?:這是|这是|這個聲音是|这个声音是|這是\s*)?([\w\u4e00-\u9fff .@+-]{1,40})\s*(?:的聲音|的声音|的聲紋|的声纹)",
    re.IGNORECASE,
)

_PROACTIVE_QUIET_RE = re.compile(
    r"(安靜|先不要說話|不要主動|別主動|不要插話|閉嘴|靜音|quiet|silent)",
    re.IGNORECASE,
)

_PROACTIVE_RESUME_RE = re.compile(
    r"(可以說話|恢復主動|你可以主動|不用安靜|解除安靜|resume proactive|speak again)",
    re.IGNORECASE,
)

_PROACTIVE_WATCH_RE = re.compile(
    r"(幫我看|幫我盯|看著|觀察|注意|有變化.*告訴|看到.*提醒|主動告訴|你看到什麼)",
    re.IGNORECASE,
)

# 使用者文字出現「看眼前畫面」意圖時，自動把當下幀附給 Gemini（取代過去的持續串流）。
_VISUAL_INTENT_RE = re.compile(
    r"(你?看到|你?看得到|看一下|看看|畫面|鏡頭|攝影機|眼前|現在.*前面|這是什麼|這個是什麼|"
    r"我手上|我拿的|我穿|這張|這邊有什麼|前面有什麼|幫我看|描述.*畫面|看得出|認得.*嗎|"
    r"what do you see|look at|on screen|in front of)",
    re.IGNORECASE,
)


def _execute_simple_tool(name: str, args: dict) -> dict:
    if name == "get_current_time":
        return {"time": time.strftime("%Y-%m-%d %H:%M:%S")}
    return {"error": f"未知工具: {name}"}


def _is_smalltalk(text: str) -> bool:
    return bool(_SMALLTALK_RE.match(text or ""))


def _visual_identity_label_from_text(text: str, default_label: str) -> str | None:
    raw = (text or "").strip()
    if not raw:
        return None
    if _VISUAL_SELF_RE.search(raw):
        return default_label or "我"
    m = _VISUAL_LABEL_RE.search(raw)
    if not m:
        return None
    label = re.sub(r"[，,。.!！?？\s]+$", "", m.group(1).strip())
    if label in {"我", "自己"}:
        return default_label or "我"
    return label or None


def _voice_identity_label_from_text(text: str, default_label: str) -> str | None:
    raw = (text or "").strip()
    if not raw:
        return None
    if _VOICE_SELF_RE.search(raw):
        return default_label or "我"
    m = _VOICE_LABEL_RE.search(raw)
    if not m:
        return None
    label = re.sub(r"[，,。.!！?？\s]+$", "", m.group(1).strip())
    if label in {"我", "自己", "目前使用者", "使用者", "我的"}:
        return default_label or "我"
    return label or None


def _is_silent_response(text: str) -> bool:
    """主動事件中，模型認為無須介入亦無須記錄時會回覆 SILENT。"""
    cleaned = re.sub(r"[\s<>\[\]*。.!！?？]+", "", text or "").upper()
    return "SILENT" in cleaned or not cleaned


_OBSERVE_PREFIX_RE = re.compile(r"^[\s\[\(（【]*OBSERVE[\]\)）】:：\-\s]*", re.IGNORECASE)


def _is_observe_response(text: str) -> bool:
    """主動事件中，模型認為『值得記錄但不值得打斷』時會以 OBSERVE 開頭。"""
    return bool(_OBSERVE_PREFIX_RE.match((text or "").lstrip()))


def _strip_observe(text: str) -> str:
    return _OBSERVE_PREFIX_RE.sub("", (text or "").lstrip(), count=1).strip()


def _should_attach_memory_to_delegate(original_message: str, task: str) -> bool:
    original = original_message or ""
    if _EXPLICIT_MEMORY_FOR_DELEGATE_RE.search(original):
        return True
    if _PUBLIC_WEB_TASK_RE.search(original) or _PUBLIC_WEB_TASK_RE.search(task or ""):
        return False
    return False


def _site_memory_context_for_delegate(original_message: str, task: str) -> str:
    text = f"{original_message or ''}\n{task or ''}"
    if not _LOGIN_OR_SITE_TASK_RE.search(text):
        return ""

    sites: dict[str, dict] = {}
    failures: dict[str, dict] = {}
    for query in _site_memory_queries_for_delegate(original_message, task):
        try:
            found = site_memory_search(query, max_results=5)
        except Exception as e:
            log.debug("網站入口記憶搜尋失敗: %s", e)
            continue
        for site in found.get("sites", []) or []:
            url = str(site.get("url") or "").strip()
            if url:
                sites[url] = site
        for failure in found.get("failures", []) or []:
            url = str(failure.get("url") or "").strip()
            if url:
                failures[url] = failure

    if not sites and not failures:
        return ""
    lines = ["【Raphael 已學網站入口】"]
    for site in list(sites.values())[:5]:
        lines.append(
            f"- 優先入口：{site.get('service', '網站')} → {site.get('url', '')}"
            f"；標題：{site.get('title', '')}；備註：{site.get('note', '')}"
        )
    if failures:
        lines.append("【Raphael 已知失敗網址，除非使用者指定否則避免重試】")
        for item in list(failures.values())[:5]:
            lines.append(
                f"- 失敗：{item.get('service', '網站')} → {item.get('url', '')}"
                f"；原因：{item.get('error', '') or item.get('note', '')}"
            )
    return "\n".join(lines)


def _site_memory_queries_for_delegate(original_message: str, task: str, include_full: bool = True) -> list[str]:
    text = f"{original_message or ''}\n{task or ''}".strip()
    if not text:
        return []
    queries = []
    service_terms = _site_service_terms_for_delegate(text)
    queries.extend(service_terms)
    primary_service = service_terms[0] if service_terms else ""

    chunks = sorted(
        _delegate_memory_terms(text),
        key=lambda item: (0 if item.lower() in {s.lower() for s in _SITE_SERVICE_TERMS} else 1, -len(item), item),
    )
    for chunk in chunks:
        token = chunk.strip()
        if not token or token in _SITE_QUERY_STOPWORDS:
            continue
        if primary_service and token.lower() != primary_service.lower():
            queries.append(f"{token} {primary_service}")
        else:
            queries.append(token)

    deduped: list[str] = []
    seen: set[str] = set()
    for query in queries:
        clean = re.sub(r"\s+", " ", str(query or "").strip())
        key = clean.lower()
        if clean and key not in seen:
            seen.add(key)
            deduped.append(clean)
    return deduped[:10]


def _site_service_terms_for_delegate(text: str) -> list[str]:
    found: list[str] = []
    lower = (text or "").lower()
    for term in sorted(_SITE_SERVICE_TERMS, key=len, reverse=True):
        if term.lower() in lower:
            found.append(term)
    for token in re.findall(r"[a-z0-9][a-z0-9._-]{2,}", lower):
        if token in _SITE_SERVICE_TERMS and token not in found:
            found.append(token)
    deduped: list[str] = []
    seen: set[str] = set()
    for item in found:
        key = item.lower()
        if key not in seen:
            seen.add(key)
            deduped.append(item)
    return deduped[:3]


def _delegate_memory_queries(original_message: str, task: str, site_context: str = "") -> list[str]:
    text = f"{original_message or ''}\n{task or ''}".strip()
    queries = [task, original_message]
    if _LOGIN_OR_SITE_TASK_RE.search(text):
        service_terms = _site_memory_queries_for_delegate(text, "", include_full=False)[:8]
        if service_terms:
            queries.append(" ".join(service_terms[:3]) + " 帳號 密碼 登入")
        queries.append("登入 網站 入口 帳密")
        site_text = f"{text}\n{site_context or ''}"
        for query in _site_memory_queries_for_delegate(site_text, "", include_full=False):
            queries.append(f"{query} 帳號 密碼 登入")
            if re.search(r"google|oauth|sso|單一登入|single sign", f"{text}\n{query}", re.IGNORECASE):
                queries.append(f"{query} Google 帳號")

    deduped: list[str] = []
    seen: set[str] = set()
    for query in queries:
        clean = re.sub(r"\s+", " ", str(query or "").strip())
        key = clean.lower()
        if clean and key not in seen:
            seen.add(key)
            deduped.append(clean)
    return deduped[:18]


def _delegate_memory_terms(text: str) -> set[str]:
    terms: set[str] = set()
    raw_text = text or ""
    for raw in re.findall(r"[\u4e00-\u9fff]{2,12}|[a-z0-9][a-z0-9._@/-]{2,}", raw_text, re.IGNORECASE):
        token = raw.strip().lower()
        if not token or token in {s.lower() for s in _SITE_QUERY_STOPWORDS}:
            continue
        if token in {"gmail", "email", "google", "使用者", "任務", "查詢", "最新進度", "進度"}:
            continue
        if re.search(r"[\u4e00-\u9fff]", token) and re.search(r"(我是|使用者|任務|查詢|進度|請|幫|帮|登入|作業|課程|網站|平台|學生|校友)", token):
            continue
        terms.add(token)
    institution_terms: set[str] = set()
    for m in re.finditer(
        r"(?:我是|我就讀|使用者是|就讀|來自|在|到)?([\u4e00-\u9fff]{1,8}?(?:高級中學|高中|中學|大學|國中|國小|學院|學校|公司))",
        raw_text,
    ):
        institution_terms.add(m.group(1).lower())
    for m in re.finditer(r"(?:我是|使用者是)?([\u4e00-\u9fff]{2,8}?)(?:學生|生|校友)", raw_text):
        institution_terms.add(m.group(1).lower())
    for token in institution_terms:
        terms.add(token)
        for suffix, short in (("中學", "中"), ("高中", "高"), ("大學", "大"), ("國中", "中"), ("國小", "小")):
            if token.endswith(suffix) and len(token) > len(suffix):
                terms.add(token[0] + short)
    return terms


def _domains_from_text(text: str) -> set[str]:
    domains: set[str] = set()
    for match in re.findall(r"(?i)(?:https?://)?(?:[\w.+-]+@)?([a-z0-9-]+(?:\.[a-z0-9-]+)+)", text or ""):
        domain = match.strip(".").lower()
        if domain and not domain.startswith("www."):
            domains.add(domain)
        elif domain.startswith("www."):
            domains.add(domain[4:])
    return domains


def _domain_roots(domain: str) -> set[str]:
    labels = [part for part in domain.lower().split(".") if part]
    roots = {domain.lower()} if labels else set()
    for size in (2, 3, 4):
        if len(labels) >= size:
            roots.add(".".join(labels[-size:]))
    return roots


def _has_domain_affinity(left: str, right: str) -> bool:
    left_roots = set()
    for domain in _domains_from_text(left):
        left_roots.update(_domain_roots(domain))
    right_roots = set()
    for domain in _domains_from_text(right):
        right_roots.update(_domain_roots(domain))
    noise = {"gmail.com", "google.com", "googleapis.com", "youtube.com", "youtu.be"}
    left_roots -= noise
    right_roots -= noise
    return bool(left_roots and right_roots and left_roots & right_roots)


def _delegate_memory_relevant(row: dict, original_message: str, task: str, site_context: str = "") -> bool:
    memory = str((row or {}).get("memory") or "")
    if not memory.strip():
        return False
    category = str((row or {}).get("category") or "").lower()
    target_text = f"{original_message or ''}\n{task or ''}\n{site_context or ''}"
    target_terms = _delegate_memory_terms(target_text)
    memory_terms = _delegate_memory_terms(memory)
    overlap = target_terms & memory_terms
    target_is_login = bool(_LOGIN_OR_SITE_TASK_RE.search(target_text))
    domain_affinity = _has_domain_affinity(target_text, memory)
    service_overlap = overlap & (_SITE_SERVICE_TERMS | {"course", "courses", "課程", "作業"})
    has_specific_overlap = len(overlap - (_SITE_SERVICE_TERMS | {"course", "courses", "課程", "作業"})) >= 1

    if category == "credential":
        if target_is_login:
            return domain_affinity or has_specific_overlap or bool(service_overlap and len(overlap) >= 2)
        return False

    if target_is_login and domain_affinity:
        return True
    if target_is_login and service_overlap:
        return has_specific_overlap or len(overlap) >= 2
    if re.search(r"(coding|程式|程式碼|風格|code)", target_text, re.IGNORECASE):
        if re.search(r"(coding|程式|程式碼|風格|code)", memory, re.IGNORECASE):
            return True
    return len(overlap) >= 2


def _filter_delegate_memories(rows: list[dict], original_message: str, task: str, site_context: str = "", limit: int = 14) -> list[dict]:
    filtered: list[dict] = []
    seen_ids: set[str] = set()
    seen_text: set[str] = set()
    for row in rows or []:
        row_id = str(row.get("id") or "")
        memory = str(row.get("memory") or "")
        key = row_id or memory
        if key in seen_ids or memory in seen_text:
            continue
        if _delegate_memory_relevant(row, original_message, task, site_context):
            filtered.append(row)
            seen_ids.add(key)
            seen_text.add(memory)
        if len(filtered) >= limit:
            break
    return filtered


class ProactiveGovernor:
    """Routes vision events to Gemini while keeping quiet-mode and speaking guards."""

    _MODEL_JUDGE_EVENT_TYPES = {
        "vision_identity",
        "vision:fast_burst",
        "vision:object_motion",
        "vision:motion",
        "vision:object",
        "vision:person_enter",
        "person_enter",
        "vision:person_leave",
        "person_leave",
        "vision:semantic",
    }

    def __init__(self):
        self._last_spoke_at = 0.0
        self._last_spoke_by_key: dict[str, float] = {}
        self._quiet_until = 0.0
        self._watch_until = 0.0
        self._min_gap = Config.PROACTIVE_MIN_GAP
        self._repeat_gap = Config.PROACTIVE_REPEAT_GAP

    @property
    def quiet_active(self) -> bool:
        return time.time() < self._quiet_until

    @property
    def watch_active(self) -> bool:
        return time.time() < self._watch_until

    def note_user_text(self, text: str) -> str:
        now = time.time()
        raw = text or ""
        notes = []
        if _PROACTIVE_RESUME_RE.search(raw):
            self._quiet_until = 0.0
            notes.append("主動回應已恢復")
        elif _PROACTIVE_QUIET_RE.search(raw):
            self._quiet_until = now + 300.0
            notes.append("已進入 5 分鐘安靜模式")
        if _PROACTIVE_WATCH_RE.search(raw):
            self._watch_until = now + 180.0
            notes.append("已進入 3 分鐘觀察模式")
        return "；".join(notes)

    def _event_info(self, payload) -> dict:
        detail = ""
        metrics: dict = {}
        boxes = []
        if isinstance(payload, dict):
            event_type = str(payload.get("type") or payload.get("reason") or "vision:event")
            label = str(payload.get("label") or "")
            score = payload.get("score")
            detail = str(payload.get("detail") or "")
            boxes = payload.get("boxes") or payload.get("feedback_boxes") or []
            metrics = payload.get("metrics") or {}
        else:
            event_type = str(payload)
            label = ""
            score = None

        extra = {"detail": detail, "boxes": boxes, "metrics": metrics}
        if event_type == "vision_identity":
            desc = f"辨識到已記住的人：{label or '某人'}"
            return {"type": event_type, "key": f"{event_type}:{label}", "description": desc, "severity": 3.0, "score": score, **extra}
        if event_type == "vision:fast_burst":
            return {"type": event_type, "key": event_type, "description": "偵測到快速移動的物體", "severity": 2.7, "score": score, **extra}
        if event_type in {"vision:object_motion", "vision:motion", "vision:object"}:
            return {"type": event_type, "key": event_type, "description": "偵測到物體進入或移動畫面", "severity": 2.1, "score": score, **extra}
        if event_type in {"vision:person_enter", "person_enter"}:
            return {"type": event_type, "key": event_type, "description": "有人進入畫面", "severity": 2.6, "score": score, **extra}
        if event_type in {"vision:person_leave", "person_leave"}:
            return {"type": event_type, "key": event_type, "description": "有人離開畫面", "severity": 2.3, "score": score, **extra}
        if event_type == "vision:semantic":
            return {"type": event_type, "key": event_type, "description": "場景語義發生變化", "severity": 1.6, "score": score, **extra}
        return {"type": event_type, "key": event_type, "description": f"環境變化：{event_type}", "severity": 1.8, "score": score, **extra}

    def decide(
        self,
        payload,
        *,
        user_busy: bool,
        last_user_activity: float,
        last_assistant_activity: float,
        voice_active: bool = False,
        recent_voice: bool = False,
        visual_attention: float = 0.0,
        proactive_blocked: bool = False,
    ) -> dict:
        now = time.time()
        info = self._event_info(payload)
        watch = now < self._watch_until
        model_judge_event = info.get("type") in self._MODEL_JUDGE_EVENT_TYPES

        if proactive_blocked:
            return {"action": "silent", "reason": "session_blocked", **info}
        if now < self._quiet_until:
            return {"action": "silent", "reason": "quiet_mode", **info}
        if user_busy:
            return {"action": "defer", "reason": "voice_active", "defer_ms": 900, "voice_context": True, **info}

        user_gap = now - last_user_activity if last_user_activity else 9999.0
        assistant_gap = now - last_assistant_activity if last_assistant_activity else 9999.0
        if user_gap < Config.PROACTIVE_AFTER_USER_GRACE and not (watch or recent_voice or model_judge_event):
            return {"action": "log", "reason": "recent_user_turn", **info}
        if assistant_gap < Config.PROACTIVE_AFTER_ASSISTANT_GRACE and not (watch or model_judge_event or (recent_voice and assistant_gap > 2.5)):
            return {"action": "log", "reason": "recent_assistant_turn", **info}

        min_gap = self._min_gap * (0.35 if recent_voice else 1.0)
        if now - self._last_spoke_at < min_gap and not (watch or model_judge_event):
            return {"action": "log", "reason": "global_cooldown", **info}

        key = info["key"]
        last_key_at = self._last_spoke_by_key.get(key, 0.0)
        repeat_gap = self._repeat_gap * (0.35 if recent_voice else (0.45 if watch else 1.0))
        if now - last_key_at < repeat_gap and not model_judge_event:
            return {"action": "silent", "reason": "repeated_event", **info}

        self._last_spoke_at = now
        self._last_spoke_by_key[key] = now
        return {"action": "speak", "reason": "send_to_gemini", "watch_mode": watch, "voice_context": recent_voice, **info}


class VoiceInputGovernor:
    """Gates always-on microphone input using VAD plus visual attention cues."""

    def __init__(self):
        self._session_until = 0.0
        self._tail_until = 0.0
        self._last_reason = "idle"

    @property
    def last_reason(self) -> str:
        return self._last_reason

    def decide(
        self,
        *,
        speaking: bool,
        probability: float,
        visual_available: bool,
        visual_attention: float,
        face_recent: bool,
        recognized_recent: bool,
        voice_match_recent: bool,
        mouth_active: bool,
        mouth_score: float,
        quiet_active: bool,
        assistant_recent: bool,
        manual_watch: bool,
    ) -> dict:
        now = time.time()
        strong_addressing = voice_match_recent or mouth_active or recognized_recent
        if quiet_active and not strong_addressing:
            self._last_reason = "quiet_mode"
            return {"forward": False, "reason": self._last_reason}

        if not speaking:
            forward_tail = now < self._tail_until
            self._last_reason = "tail" if forward_tail else "silence"
            return {"forward": forward_tail, "reason": self._last_reason}

        if assistant_recent:
            self._last_reason = "assistant_recent"
            return {"forward": False, "reason": self._last_reason}

        if voice_match_recent:
            self._session_until = now + 7.0
            self._tail_until = now + 1.4
            self._last_reason = "voiceprint"
            return {"forward": True, "reason": self._last_reason}

        if mouth_active and probability >= 0.14:
            self._session_until = now + 6.0
            self._tail_until = now + 1.3
            self._last_reason = "mouth_sync"
            return {"forward": True, "reason": self._last_reason, "mouth_score": mouth_score}

        if not visual_available:
            self._session_until = now + 6.0
            self._tail_until = now + 1.2
            self._last_reason = "no_vision_fallback"
            return {"forward": True, "reason": self._last_reason}

        threshold = 0.28 if manual_watch else 0.38
        visually_addressed = recognized_recent or (face_recent and visual_attention >= threshold)
        if visually_addressed:
            self._session_until = now + 6.0
            self._tail_until = now + 1.2
            self._last_reason = "visual_attention"
            return {"forward": True, "reason": self._last_reason}

        if now < self._session_until and probability >= 0.18:
            self._tail_until = now + 1.2
            self._last_reason = "active_session"
            return {"forward": True, "reason": self._last_reason}

        self._last_reason = "not_addressing"
        return {"forward": False, "reason": self._last_reason}


def _compact_for_ui(name: str, result: dict) -> dict:
    if name == "delegate_tool_task" and isinstance(result, dict):
        progress = result.get("progress_snapshot") if isinstance(result.get("progress_snapshot"), dict) else {}
        summary = _redact_secrets_for_ui(progress.get("summary") or "工具代理已完成，正在交給 Raphael 整合最終回覆。")
        if result.get("stopped_for_budget"):
            summary = "工具代理接近輪數上限，已停止繼續探索並整理目前進度。"
        if "error" in result:
            summary = _redact_secrets_for_ui(result.get("error") or summary)
            hint = _redact_secrets_for_ui(result.get("recovery_hint", ""))
            if hint:
                summary = f"{summary}；建議：{hint}"
        return {
            "summary": summary,
            "ok": result.get("ok", "error" not in result),
            "model": result.get("model"),
            "turns": result.get("turns"),
            "tool_count": result.get("tool_count"),
            "stuck_detected": result.get("stuck_detected"),
            "recovery_hint": _redact_secrets_for_ui(result.get("recovery_hint", "")),
            "strategy_count": len(result.get("strategy_events") or []),
            "stopped_for_budget": bool(result.get("stopped_for_budget")),
            "progress_snapshot": _redact_secrets_for_ui(progress),
            "duration_ms": result.get("duration_ms"),
        }
    if name in {"search_memories", "filtered_search_memories", "get_all_memories"} and isinstance(result, dict):
        return _redact_secrets_for_ui(result)
    return result


_AUTHORITATIVE_SUCCESS_TOOLS = {
    "copy_file", "move_file", "write_file", "download_file", "download_image",
    "make_directory", "zip_create", "zip_extract", "replace_in_file",
    "gmail_send", "calendar_create_event", "browser_login", "browser_open",
    "browser_follow_link", "browser_click", "browser_fill", "browser_press_key",
    "computer_control", "computer_focus_window",
}


def _delegate_completed_actions(result: dict) -> list[str]:
    actions: list[str] = []
    for event in result.get("tool_calls") or []:
        if not isinstance(event, dict) or not event.get("success"):
            continue
        tool = str(event.get("tool") or "")
        preview = str(event.get("result_preview") or "").strip()
        if tool in _AUTHORITATIVE_SUCCESS_TOOLS and preview:
            actions.append(f"{tool}: {preview}")
    deduped: list[str] = []
    seen: set[str] = set()
    for action in actions:
        key = action.lower()
        if key not in seen:
            seen.add(key)
            deduped.append(action)
    return deduped[-8:]


def _delegate_response_contract(result: dict) -> dict:
    progress = result.get("progress_snapshot") if isinstance(result.get("progress_snapshot"), dict) else {}
    summary = str(progress.get("summary") or result.get("answer") or result.get("error") or "").strip()
    completed_actions = _delegate_completed_actions(result)
    return {
        "instruction": (
            "回答使用者時必須根據工具實際結果與原始要求推進任務。"
            "completed_actions 內列出的成功副作用是權威事實，不可否認、弱化或改口說沒有完成。"
            "不可把工具已成功完成或已推進的部分改口說成無法做到，"
            "不可用泛用拒絕、道德勸告、時間管理建議或與任務無關的替代目標取代回覆。"
            "若任務尚未完成，請說明已完成的具體進度、下一個最小可執行步驟，"
            "以及只在必要時指出需要使用者本人確認、驗證或選擇的地方。"
        ),
        "current_progress": _redact_secrets_for_ui(summary),
        "completed_actions": _redact_secrets_for_ui(completed_actions),
        "stopped_for_budget": bool(result.get("stopped_for_budget")),
        "stuck_detected": bool(result.get("stuck_detected")),
    }


def _background_progress_decision(payload: dict, state: dict) -> tuple[bool, str]:
    """Decide whether a quiet background-browser result should still be shown."""
    if not isinstance(payload, dict):
        return False, ""
    has_error = bool(payload.get("error"))
    has_files = bool(payload.get("files"))
    result_preview = str(payload.get("result_preview", "") or "")
    needs_user = "需要使用者" in result_preview
    if has_error or has_files or needs_user:
        return True, result_preview

    progress = payload.get("progress_snapshot")
    if not isinstance(progress, dict):
        return False, ""
    try:
        tool_count = int(progress.get("tool_count") or 0)
    except Exception:
        tool_count = 0
    phase = str(progress.get("current_phase") or "")
    summary = str(progress.get("summary") or result_preview)
    if not summary:
        return False, ""

    last_phase = str(state.get("last_phase") or "")
    last_count = int(state.get("last_tool_count") or 0)
    should_publish = False
    if phase and phase != last_phase:
        should_publish = True
    elif tool_count and tool_count % 3 == 0 and tool_count != last_count:
        should_publish = True

    if not should_publish:
        return False, ""
    state["last_phase"] = phase
    state["last_tool_count"] = tool_count
    return True, summary


_FILE_PROGRESS_TOOLS = {
    "copy_file", "move_file", "write_file", "download_file", "download_image",
    "make_directory", "zip_create", "zip_extract", "replace_in_file",
}


def _short_voice_text(text: str, limit: int = 96) -> str:
    text = re.sub(r"\s+", " ", str(text or "").strip())
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "..."


def _task_voice_line(event: str, payload: dict) -> str:
    """Convert low-level tool events into short spoken progress updates."""
    if not isinstance(payload, dict):
        return ""
    tool = str(payload.get("tool") or "")
    preview = str(payload.get("result_preview") or payload.get("error") or "").strip()

    if event == "tool_start":
        if tool.startswith("browser_"):
            return "我正在背景瀏覽器處理。"
        if tool.startswith("computer_"):
            return "我正在確認目標視窗並操作。"
        if tool in {"web_search", "website_find", "site_memory_search", "dns_lookup"}:
            return "我正在尋找並驗證入口。"
        if tool in _FILE_PROGRESS_TOOLS:
            return "我正在處理檔案。"
        if tool.startswith("gmail_"):
            return "我正在處理郵件。"
        if tool.startswith("calendar_"):
            return "我正在處理行程。"
        return "我正在執行下一個工具步驟。"

    if event != "tool_done":
        return ""
    if payload.get("error"):
        return _short_voice_text(f"這一步遇到問題：{preview}", 86) if preview else "這一步遇到問題，我正在換策略。"
    if tool.startswith("browser_"):
        return _short_voice_text(f"背景瀏覽器進度：{preview}", 96) if preview else "背景瀏覽器已有進展。"
    if tool.startswith("computer_"):
        return _short_voice_text(f"桌面操作進度：{preview}", 96) if preview else "桌面操作已有進展。"
    if tool in _FILE_PROGRESS_TOOLS:
        return _short_voice_text(preview, 96) if preview else "檔案處理完成。"
    if tool.startswith("gmail_"):
        return _short_voice_text(f"郵件進度：{preview}", 96) if preview else "郵件處理完成。"
    return _short_voice_text(f"工具進度：{preview}", 96) if preview else ""


def _ssl_context() -> ssl.SSLContext:
    """Use certifi when available so Windows Python does not depend on OS CA quirks."""
    verify = os.environ.get("RAPHAEL_SSL_VERIFY", "1").strip().lower()
    if verify in {"0", "false", "no"}:
        log.warning("RAPHAEL_SSL_VERIFY=0，Gemini WebSocket 將略過 TLS 憑證驗證")
        return ssl._create_unverified_context()
    try:
        import certifi
    except Exception:
        return ssl.create_default_context()
    return ssl.create_default_context(cafile=certifi.where())


# ══════════════════════════════════════════════════════════════════════════════
# RaphaelCore
# ══════════════════════════════════════════════════════════════════════════════

class RaphaelCore:
    """
    Raphael 的核心引擎。

    用法：
        core = RaphaelCore(bridge, voice="Puck", thinking="medium")
        await core.start()
        # ... 感知模組和 UI 透過 bridge 與 core 互動 ...
        await core.stop()
    """

    def __init__(
        self,
        bridge: Bridge,
        *,
        voice: str = Config.VOICE,
        thinking: str | None = Config.THINKING,
        user_id: str = Config.USER_ID,
    ):
        self._bridge = bridge
        self._voice = voice
        self._thinking = self._normalize_thinking(thinking)

        self._ws: websockets.WebSocketClientProtocol | None = None
        self._closed = False
        self._resume_token: str | None = None
        # setup 相容性降級層級：0=全功能；數字越大代表去掉越多新欄位，
        # 用來在某些 API 版本不支援 mediaResolution/contextWindowCompression/sessionResumption 時仍能連上。
        self._setup_level = 0
        self._out_transcript = ""
        self._in_transcript = ""
        self._last_user_text = ""
        self._memory_written_this_turn = False
        self._visual_identity_written_this_turn = False
        self._voice_identity_written_this_turn = False
        self._visual_frame_sent_this_turn = False
        self._proactive_turn_pending = False
        self._proactive_pending_since = 0.0
        self._proactive_pending_is_fire = False
        self._proactive_out_buffer = ""
        self._proactive_audio_buffer: list[dict] = []

        self._latest_video_jpeg: bytes = b""
        self._last_identity_scan = 0.0
        self._visual_scan_busy = False
        self._identity_seen_at: dict[str, float] = {}
        self._last_user_activity = 0.0
        self._last_assistant_activity = 0.0
        # 半雙工：AI 出聲（含播放尾音）期間暫停轉發麥克風，避免喇叭→麥克風回授讓 AI 自我打斷/重講。
        self._assistant_speaking_until = 0.0
        self._proactive_governor = ProactiveGovernor()
        self._voice_governor = VoiceInputGovernor()
        self._latest_vad_speaking = False
        self._latest_vad_probability = 0.0
        self._prev_vad_speaking = False  # 用於偵測語音打斷
        self._voice_interrupt_triggered_this_turn = False  # 避免同一次說話多次打斷
        self._vision_available = False
        self._vision_face_count = 0
        self._vision_attention_score = 0.0
        self._last_face_seen = 0.0
        self._last_recognized_seen = 0.0
        self._recognized_people: list[str] = []
        self._last_audio_forward_at = 0.0
        self._last_audio_gate_publish = 0.0
        self._last_audio_gate_state = ""
        self._voice_pcm_ring = collections.deque(maxlen=420)
        self._last_voice_scan = 0.0
        self._voice_match_label = ""
        self._voice_match_score = 0.0
        self._last_voice_match_at = 0.0
        self._prev_mouth_crop = None
        self._mouth_motion_score = 0.0
        self._last_mouth_motion_at = 0.0
        self._deferred_proactive_task: asyncio.Task | None = None
        self._tasks: list[asyncio.Task] = []
        self._background_tasks: set[asyncio.Task] = set()
        self._connect_lock = asyncio.Lock()
        self._tool_rounds = 0
        self._user_speaking = False      # 使用者正在說話時不主動開口
        self._sources = {
            "audio": True,
            "vision": True,
            "tool": True,
            "memory": True,
            "proactive": True,
        }
        self._features = dict(_DEFAULT_FEATURES)

        self._memory = MemoryManager(
            bridge,
            user_id=user_id,
            qdrant_host=Config.QDRANT_HOST,
            qdrant_port=Config.QDRANT_PORT,
        )

    @staticmethod
    def _normalize_thinking(value: str | None) -> str | None:
        if value is None:
            return None
        return _THINKING_MAP.get(str(value).strip().lower(), str(value).strip().lower())

    def set_sources(self, sources: dict) -> None:
        """套用 UI 來源開關。未知鍵忽略，避免前後端版本不同步時炸掉。"""
        for key in self._sources:
            if key in sources:
                self._sources[key] = bool(sources[key])
        if isinstance(sources.get("features"), dict):
            self.set_features(sources["features"])

    def set_features(self, features: dict) -> dict:
        """套用 Demo 用細項功能開關。未知鍵忽略，方便前後端版本漸進更新。"""
        for key in self._features:
            if key in features:
                self._features[key] = bool(features[key])
        if not self._features.get("voice_identity", True):
            self._voice_match_label = ""
            self._voice_match_score = 0.0
            self._last_voice_match_at = 0.0
        if not self._features.get("mouth_sync", True):
            self._mouth_motion_score = 0.0
            self._last_mouth_motion_at = 0.0
            self._prev_mouth_crop = None
        return dict(self._features)

    def _feature_enabled(self, key: str) -> bool:
        return bool(self._features.get(key, True))

    @property
    def memory(self) -> MemoryManager:
        return self._memory

    def set_memory_user(self, user_id: str) -> dict:
        return self._memory.select_account(user_id)

    def create_memory_user(self, user_id: str) -> dict:
        return self._memory.create_account(user_id)

    def delete_memory_user(self, user_id: str) -> dict:
        return self._memory.delete_account(user_id)

    # ══════════════════════════════════════════════════════════════════════
    # 啟動 / 停止
    # ══════════════════════════════════════════════════════════════════════

    async def start(self) -> None:
        self._closed = False
        if not Config.API_KEY:
            await self._bridge.publish(Channel.ERROR, "缺少 GEMINI_API_KEY，Gemini Live 未連線")
            await self._bridge.publish(Channel.STATUS, "disconnected")
            return

        try:
            await self._connect()
        except Exception as e:
            err = f"Gemini 連線失敗: {e}"
            log.error(err)
            await self._bridge.publish(Channel.ERROR, err)
            await self._bridge.publish(Channel.STATUS, "disconnected")
            return

        self._tasks = [
            asyncio.create_task(self._gemini_recv_loop()),
            asyncio.create_task(self._audio_forward_loop()),
            asyncio.create_task(self._vad_state_loop()),
            asyncio.create_task(self._video_forward_loop()),
            asyncio.create_task(self._text_forward_loop()),
            asyncio.create_task(self._proactive_loop()),
            asyncio.create_task(self._proactive_heartbeat_loop()),
            asyncio.create_task(self._user_interrupt_loop()),
        ]
        self._interrupt_sub = self._bridge.subscribe({Channel.USER_INTERRUPT})
        await self._bridge.publish(Channel.STATUS, "connected")
        log.info("RaphaelCore 啟動完成")

    async def stop(self) -> None:
        self._closed = True
        try:
            for t in self._tasks:
                t.cancel()
            for t in list(self._background_tasks):
                t.cancel()
            if self._tasks:
                await asyncio.gather(*self._tasks, return_exceptions=True)
                self._tasks = []
            if self._background_tasks:
                await asyncio.gather(*self._background_tasks, return_exceptions=True)
                self._background_tasks.clear()
            if self._ws:
                with contextlib.suppress(Exception):
                    await self._ws.close()
                self._ws = None
        finally:
            self._memory.close()
            await self._bridge.publish(Channel.STATUS, "disconnected")
            log.info("RaphaelCore 已停止")

    def _track_background_task(self, task: asyncio.Task) -> None:
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)

    # ══════════════════════════════════════════════════════════════════════
    # Gemini WebSocket 連線
    # ══════════════════════════════════════════════════════════════════════

    async def _connect(self) -> None:
        log.info("連接 Gemini Live %s (setup level=%d)...", Config.MODEL, self._setup_level)
        self._ws = await websockets.connect(
            Config.WS_URL,
            ping_interval=Config.PING_INTERVAL,
            ssl=_ssl_context(),
        )
        try:
            await self._send_setup()
        except websockets.exceptions.ConnectionClosed as e:
            # setup 在完成前就被關閉，通常代表某個新欄位這個 API 版本不支援。
            # 逐級拿掉新欄位重試，確保至少能連上（不讓連線卡死在重連迴圈）。
            with contextlib.suppress(Exception):
                await self._ws.close()
            self._ws = None
            if self._setup_level < 4:
                self._setup_level += 1
                log.warning(
                    "setup 在完成前被關閉 (code=%s reason=%r)，降級到 level %d 後重試",
                    e.code, e.reason, self._setup_level,
                )
                await self._connect()
                return
            raise
        log.info("Gemini session 建立完成 (setup level=%d)", self._setup_level)

    def _ws_open(self) -> bool:
        if self._ws is None:
            return False
        if getattr(self._ws, "closed", False):
            return False
        if getattr(self._ws, "close_code", None) is not None:
            return False
        return True

    async def _ensure_live_session(self) -> bool:
        # 接收/重連由 _gemini_recv_loop 監督迴圈統一負責；這裡只負責「需要時立刻確保連上」。
        if self._closed:
            return False
        if self._ws_open():
            return True
        return await self._reconnect()

    async def _send_setup(self) -> None:
        system_prompt = self._memory.build_context("")

        speech_cfg: dict = {
            "voiceConfig": {
                "prebuiltVoiceConfig": {"voiceName": self._voice}
            }
        }
        # 釘住語言修正誤判（韓文/英文）。最可能不被某些模型支援，故放在降級階梯第一個被拿掉。
        if self._setup_level < 1 and Config.SPEECH_LANGUAGE.strip():
            speech_cfg["languageCode"] = Config.SPEECH_LANGUAGE.strip()

        gen_cfg: dict = {
            "responseModalities": ["AUDIO"],
            "speechConfig": speech_cfg,
        }
        if self._thinking:
            gen_cfg["thinkingConfig"] = {"thinkingLevel": self._thinking}
        # 讓 Gemini 用最高 token 預算解析影像（只在必要時送單張，成本可控）。level>=3 時拿掉以求相容。
        if self._setup_level < 3:
            gen_cfg["mediaResolution"] = "MEDIA_RESOLUTION_HIGH"

        tools = [
            MEMORY_TOOL_DECLS,
            _SIMPLE_TOOL_DECLS,
            _TOOL_AGENT_DECLS,
            _LOOK_TOOL_DECLS,
        ]

        setup_body: dict = {
            "model": f"models/{Config.MODEL}",
            "generationConfig": gen_cfg,
            "systemInstruction": {"parts": [{"text": system_prompt}]},
            "outputAudioTranscription": {},
            "inputAudioTranscription": {},
            "tools": tools,
        }
        # session 續接：新連線傳 {} 以開始接收 resume token；重連帶 handle 還原上下文。level>=4 時拿掉。
        if self._setup_level < 4:
            setup_body["sessionResumption"] = ({"handle": self._resume_token} if self._resume_token else {})
        # 上下文滑動視窗壓縮：讓 session 長時間運行不被上下文上限關閉。level>=2 時拿掉。
        if self._setup_level < 2:
            setup_body["contextWindowCompression"] = {"slidingWindow": {}}

        await self._ws.send(json.dumps({"setup": setup_body}))

        async for raw in self._ws:
            msg = json.loads(raw)
            if "setupComplete" in msg:
                break
            # 若 Gemini 在 setupComplete 前回了別的東西（通常是 setup 欄位錯誤），記下來方便診斷。
            log.warning("setup 期間收到非 setupComplete 訊息: %s", str(msg)[:400])

    # ══════════════════════════════════════════════════════════════════════
    # Gemini 接收主迴圈 → 發佈到 bridge
    # ══════════════════════════════════════════════════════════════════════

    async def _gemini_recv_loop(self) -> None:
        """唯一的接收者 + 重連監督迴圈。

        連線因時間/上下文上限或網路被關閉時，自動帶 resume token 重連，
        而不是直接死掉。這是展覽長時間運行不中斷的關鍵。
        """
        backoff = 1.0
        while not self._closed:
            if not self._ws_open():
                if not await self._reconnect():
                    await asyncio.sleep(backoff)
                    backoff = min(backoff * 2, 15.0)
                    continue
                backoff = 1.0
            ws = self._ws
            try:
                async for raw in ws:
                    if self._closed:
                        break
                    await self._dispatch(json.loads(raw))
            except websockets.exceptions.ConnectionClosed as e:
                log.warning("Gemini 斷線: %s %s", e.code, e.reason)
            except Exception as e:
                log.warning("Gemini 接收例外: %s", e)
            if self._closed:
                break
            # 連線中斷 → 標記重連（下一圈會帶 resume token 重新連上）
            log.info("Gemini 連線中斷，準備自動重連…")
            await self._bridge.publish(Channel.STATUS, "reconnecting")
            self._ws = None
        await self._bridge.publish(Channel.STATUS, "disconnected")

    async def _reconnect(self) -> bool:
        """建立（或重建）Gemini 連線。鎖保護，確保只有一條重連路徑。"""
        async with self._connect_lock:
            if self._ws_open():
                return True
            if self._ws is not None:
                with contextlib.suppress(Exception):
                    await self._ws.close()
                self._ws = None
            try:
                await self._connect()
            except Exception as e:
                self._ws = None
                log.warning("Gemini 重連失敗: %s", e)
                return False
            # 重連後舊的主動回合已不存在（turnComplete 不會再來），清旗標避免殘留 pending 擋住新事件。
            self._proactive_turn_pending = False
            self._proactive_pending_is_fire = False
            self._proactive_out_buffer = ""
            self._proactive_audio_buffer = []
            self._assistant_speaking_until = 0.0
            await self._bridge.publish(Channel.STATUS, "connected")
            return True

    def _note_assistant_audio(self, payload: dict) -> None:
        """累加 AI 輸出音訊的播放時長，延長半雙工抑制視窗到實際播放結束。"""
        data = payload.get("data") or ""
        approx_bytes = len(data) * 3 // 4          # base64 → 原始位元組
        duration = approx_bytes / 2 / Config.SPK_RATE  # PCM16 mono
        base = max(time.time(), self._assistant_speaking_until)
        self._assistant_speaking_until = base + duration

    async def _emit_audio_out(self, payload: dict) -> None:
        self._note_assistant_audio(payload)
        await self._bridge.publish(Channel.AUDIO_OUT, payload)

    async def _dispatch(self, msg: dict) -> None:
        b = self._bridge

        # ── serverContent ────────────────────────────────────────────
        if sc := msg.get("serverContent"):

            # interrupted
            if sc.get("interrupted"):
                self._out_transcript = ""
                self._in_transcript = ""
                self._assistant_speaking_until = 0.0  # 被打斷 → 立刻解除半雙工，讓麥克風恢復
                # 關鍵：被打斷時不會收到 turnComplete，必須在這裡清掉主動回合旗標，
                # 否則 _proactive_turn_pending 會卡到 30s 逾時，期間所有 fire(綠框)都被擋住而延遲。
                self._proactive_turn_pending = False
                self._proactive_pending_is_fire = False
                self._proactive_out_buffer = ""
                self._proactive_audio_buffer = []
                await b.publish(Channel.INTERRUPTED, {})

            # 音訊輸出 → UI 播放
            for part in sc.get("modelTurn", {}).get("parts", []):
                if inline := part.get("inlineData"):
                    if "audio" in inline.get("mimeType", ""):
                        audio_payload = {
                            "data": inline["data"],
                            "mimeType": inline["mimeType"],
                        }
                        if self._proactive_turn_pending:
                            self._proactive_audio_buffer.append(audio_payload)
                        else:
                            await self._emit_audio_out(audio_payload)

            # 輸出轉錄
            if ot := sc.get("outputTranscription"):
                if chunk := ot.get("text", ""):
                    self._last_assistant_activity = time.time()
                    self._out_transcript += chunk
                    if self._proactive_turn_pending:
                        self._proactive_out_buffer += chunk
                    else:
                        await b.publish(Channel.TRANSCRIPT_OUT, {"text": chunk, "done": False})

            # 輸入轉錄
            if it := sc.get("inputTranscription"):
                if chunk := it.get("text", ""):
                    self._last_user_activity = time.time()
                    self._user_speaking = True
                    self._in_transcript += chunk
                    await b.publish(Channel.TRANSCRIPT_IN, {"text": chunk, "done": False})
                    # 語音視覺意圖自動附幀：偵測到「你看到什麼/這是什麼」就在模型回答前先送當下幀，
                    # 省掉 look_now 那一次工具來回（降低語音讀圖延遲）。每輪只附一次。
                    if (
                        not self._visual_frame_sent_this_turn
                        and self._sources["vision"]
                        and self._latest_video_jpeg
                        and _VISUAL_INTENT_RE.search(self._in_transcript)
                    ):
                        self._visual_frame_sent_this_turn = True
                        await self._send_frame_to_gemini(self._latest_video_jpeg)

            # turnComplete
            if sc.get("turnComplete"):
                proactive_turn = self._proactive_turn_pending
                out_tr = self._proactive_out_buffer if proactive_turn else self._out_transcript
                in_tr = self._in_transcript or self._last_user_text
                if proactive_turn:
                    cleaned = out_tr.strip()
                    if not cleaned or _is_silent_response(cleaned):
                        log.debug("主動事件經 Gemini 判斷為 SILENT")
                    elif _is_observe_response(cleaned):
                        # 值得記錄但不值得打斷：不播語音，只把觀察當文字記錄顯示（Raphael 在看、但不出聲）
                        note = _strip_observe(cleaned)
                        if note:
                            await b.publish(Channel.TRANSCRIPT_OUT, {"text": f"（觀察）{note}", "done": False})
                            await b.publish(Channel.TRANSCRIPT_OUT, {"text": "", "done": True})
                            log.info("主動觀察(無語音): %s", note[:60])
                    else:
                        for audio_payload in self._proactive_audio_buffer:
                            await self._emit_audio_out(audio_payload)
                        await b.publish(Channel.TRANSCRIPT_OUT, {"text": cleaned, "done": False})
                        await b.publish(Channel.TRANSCRIPT_OUT, {"text": "", "done": True})
                else:
                    await b.publish(Channel.TRANSCRIPT_OUT, {"text": "", "done": True})
                    await b.publish(Channel.TRANSCRIPT_IN, {"text": "", "done": True})
                self._out_transcript = ""
                self._in_transcript = ""
                self._proactive_turn_pending = False
                self._proactive_pending_is_fire = False
                self._proactive_out_buffer = ""
                self._proactive_audio_buffer = []
                self._last_user_text = ""
                self._tool_rounds = 0
                self._user_speaking = False
                if out_tr.strip():
                    self._last_assistant_activity = time.time()
                # 預設由 Gemini 透過記憶工具負責寫入；安全網需明確開啟。
                if Config.AUTO_MEMORY_SAFETY_NET and in_tr.strip() and not self._memory_written_this_turn:
                    self._track_background_task(asyncio.create_task(self._auto_memory(in_tr, out_tr)))
                if in_tr.strip() and not self._visual_identity_written_this_turn:
                    self._track_background_task(asyncio.create_task(self._maybe_auto_enroll_visual_identity_after_turn(in_tr)))
                if in_tr.strip() and not self._voice_identity_written_this_turn:
                    self._track_background_task(asyncio.create_task(self._maybe_auto_enroll_voice_identity_after_turn(in_tr)))
                self._memory_written_this_turn = False
                self._visual_identity_written_this_turn = False
                self._voice_identity_written_this_turn = False
                self._visual_frame_sent_this_turn = False

        # ── toolCall ──────────────────────────────────────────────────
        if tc := msg.get("toolCall"):
            self._tool_rounds += 1
            if self._tool_rounds > Config.MAX_TOOL_ROUNDS:
                log.warning("工具呼叫超過 %d 輪，中止", Config.MAX_TOOL_ROUNDS)
                for fc in tc.get("functionCalls", []):
                    await self._ws.send(json.dumps({
                        "toolResponse": {
                            "functionResponses": [{
                                "id": fc.get("id", ""),
                                "name": fc.get("name", ""),
                                "response": {"error": f"超過工具呼叫上限 {Config.MAX_TOOL_ROUNDS} 輪"},
                            }]
                        }
                    }))
            else:
                for fc in tc.get("functionCalls", []):
                    call_id = fc.get("id", "")
                    name = fc.get("name", "")
                    args = fc.get("args", {})

                    await b.publish(Channel.TOOL_CALL, {"name": name, "args": _redact_secrets_for_ui(args)})

                    t0 = time.time()
                    result = await self._handle_tool_call(name, args)
                    duration_ms = round((time.time() - t0) * 1000, 1)

                    ok = "error" not in result
                    await b.publish(Channel.TOOL_RESULT, {
                        "name": name,
                        "result": _compact_for_ui(name, result),
                        "_meta": {"ok": ok, "tool": name, "duration_ms": duration_ms},
                    })

                    await self._ws.send(json.dumps({
                        "toolResponse": {
                            "functionResponses": [{
                                "id": call_id,
                                "name": name,
                                "response": result,
                            }]
                        }
                    }))

        # ── toolCallCancellation ──────────────────────────────────────
        if tcc := msg.get("toolCallCancellation"):
            log.info("工具呼叫取消: %s", tcc.get("ids", []))

        # ── goAway ────────────────────────────────────────────────────
        if ga := msg.get("goAway"):
            secs = ga.get("timeLeft", {}).get("seconds", "?")
            log.warning("Gemini 即將關閉: %ss", secs)
            await b.publish(Channel.ERROR, f"伺服器將在 {secs}s 後關閉，需重新連線")

        # ── sessionResumptionUpdate ───────────────────────────────────
        if sru := msg.get("sessionResumptionUpdate"):
            if token := sru.get("newHandle"):
                self._resume_token = token

        # ── usageMetadata ─────────────────────────────────────────────
        if msg.get("usageMetadata"):
            await b.publish(Channel.USAGE, msg["usageMetadata"])

    # ══════════════════════════════════════════════════════════════════════
    # 工具路由
    # ══════════════════════════════════════════════════════════════════════

    async def _handle_tool_call(self, name: str, args: dict) -> dict:
        if not self._sources["tool"]:
            return {"error": "工具來源目前已關閉"}
        if _is_smalltalk(self._last_user_text):
            return {"error": "本輪只是寒暄，不需要使用工具"}
        if self._memory.is_memory_tool(name):
            if not self._sources["memory"]:
                return {"error": "記憶來源目前已關閉"}
            if name == "store_visual_identity":
                if not self._feature_enabled("visual_identity"):
                    return {"error": "圖像身份功能目前已由 WebUI 關閉"}
                result = await self._store_visual_identity_tool(args)
            elif name == "store_voice_identity":
                if not self._feature_enabled("voice_identity"):
                    return {"error": "聲紋身份功能目前已由 WebUI 關閉"}
                result = await self._store_voice_identity_tool(args)
            else:
                result = await self._memory.execute(name, args)
            if name == "store_memory" and "error" not in result:
                self._memory_written_this_turn = True
            return result

        if name in _LOOK_TOOL_NAMES:
            return await self._handle_look_now()

        if name in _SIMPLE_TOOL_NAMES:
            return _execute_simple_tool(name, args)

        if name in _TOOL_AGENT_NAMES:
            if not self._feature_enabled("computer_tools"):
                task_text = str(args.get("task", "") if isinstance(args, dict) else "")
                if re.search(r"\bcomputer[_ ]|截圖|點擊|打字|滑鼠|鍵盤|操作電腦|桌面操作", task_text, re.IGNORECASE):
                    return {"error": "電腦操作工具目前已由 WebUI 關閉"}
            return await self._delegate_tool_task(args)

        return {"error": f"未知工具: {name}"}

    async def _handle_look_now(self) -> dict:
        """擷取當下畫面送進 Gemini，讓模型在下一輪能依畫面回答。"""
        if not self._sources["vision"]:
            return {"status": "視覺來源目前已關閉，看不到畫面", "frame_available": False}
        if self._visual_frame_sent_this_turn:
            return {
                "status": "你目前已經擁有本輪最即時的鏡頭畫面，不需要重複調用此工具。請直接根據已有的畫面回答使用者。",
                "frame_available": True,
            }
        jpeg = self._latest_video_jpeg
        if not jpeg:
            return {"status": "目前沒有可用的鏡頭畫面（攝影機未啟動或尚未取得幀）", "frame_available": False}
        sent = await self._send_frame_to_gemini(jpeg)
        if not sent:
            return {"status": "擷取畫面失敗，請稍後再試", "frame_available": False}
        self._visual_frame_sent_this_turn = True
        return {
            "status": "已擷取當下畫面並同步給你，請依這張畫面回答使用者。",
            "frame_available": True,
        }

    async def _delegate_tool_task(self, args: dict) -> dict:
        task = str(args.get("task", "")).strip()
        if not task:
            return {"error": "delegate_tool_task 缺少 task"}

        supplied_memory = str(args.get("memory_context", "") or "").strip()
        site_context = _site_memory_context_for_delegate(self._last_user_text, task)
        if supplied_memory:
            memory_context = supplied_memory
        elif self._sources["memory"] and _should_attach_memory_to_delegate(self._last_user_text, task):
            candidates = self._memory.retrieve(task, limit=8, min_score=0.18)
            if _LOGIN_OR_SITE_TASK_RE.search(f"{self._last_user_text}\n{task}"):
                seen_ids = {r.get("id") for r in candidates}
                for query in _delegate_memory_queries(self._last_user_text, task, site_context):
                    for row in self._memory.retrieve_filtered(query, category="credential", limit=8, min_score=0.0):
                        if row.get("id") not in seen_ids:
                            candidates.append(row)
                            seen_ids.add(row.get("id"))
                    for row in self._memory.retrieve_filtered(query, category="technical", limit=4, min_score=0.0):
                        if row.get("id") not in seen_ids:
                            candidates.append(row)
                            seen_ids.add(row.get("id"))
                    for row in self._memory.retrieve_filtered(query, category="project", limit=4, min_score=0.0):
                        if row.get("id") not in seen_ids:
                            candidates.append(row)
                            seen_ids.add(row.get("id"))
            relevant = _filter_delegate_memories(candidates, self._last_user_text, task, site_context)
            memory_context = "\n".join(f"- {r['memory']}" for r in relevant[:14])
        else:
            memory_context = ""

        if site_context:
            memory_context = (memory_context + "\n\n" + site_context).strip()

        def memory_search(query: str, limit: int = 5) -> dict:
            if not self._sources["memory"]:
                return {"error": "記憶來源目前已關閉"}
            raw_results = self._memory.retrieve(query, limit=max(int(limit or 5) * 3, 8), min_score=0.0)
            results = _filter_delegate_memories(raw_results, self._last_user_text, task, site_context, limit=int(limit or 5))
            return {
                "results": results,
                "count": len(results),
                "user_id": self._memory.user_id,
                "backend": self._memory.backend,
            }

        loop = asyncio.get_running_loop()
        background_progress_state = {"last_phase": "", "last_tool_count": 0}
        task_voice_state = {"last_text": "", "last_at": 0.0}

        def publish_task_voice(text: str, *, force: bool = False) -> None:
            text = _short_voice_text(_redact_secrets_for_ui(text), 110)
            if not text:
                return
            now = time.time()
            if not force and text == task_voice_state["last_text"] and now - task_voice_state["last_at"] < 4.0:
                return
            task_voice_state["last_text"] = text
            task_voice_state["last_at"] = now

            async def publish() -> None:
                await self._bridge.publish(Channel.TASK_VOICE, {"text": text})

            asyncio.run_coroutine_threadsafe(publish(), loop)

        publish_task_voice("收到任務，我開始處理。", force=True)

        def on_tool_event(event: str, payload: dict) -> None:
            tool_name = str(payload.get("tool") or "")
            quiet_background_browser = tool_name.startswith("browser_")
            summary_override = ""
            voice_text = _task_voice_line(event, payload)
            if quiet_background_browser and event == "tool_start":
                publish_task_voice(voice_text)
                return
            if quiet_background_browser and event != "tool_start":
                should_publish, summary_override = _background_progress_decision(payload, background_progress_state)
                if not should_publish:
                    return
                if summary_override:
                    voice_payload = dict(payload)
                    voice_payload["result_preview"] = summary_override
                    voice_text = _task_voice_line(event, voice_payload)

            publish_task_voice(voice_text)

            async def publish() -> None:
                channel = Channel.TOOL_CALL if event == "tool_start" else Channel.TOOL_RESULT
                if event == "tool_start":
                    await self._bridge.publish(channel, {
                        "name": f"minimax::{payload.get('tool')}",
                        "args": _redact_secrets_for_ui(payload.get("args", {})),
                        "_meta": {
                            "agent": "minimax",
                            "state": "running",
                            "tool_call_id": payload.get("tool_call_id"),
                        },
                    })
                else:
                    await self._bridge.publish(channel, {
                        "name": f"minimax::{payload.get('tool')}",
                        "result": {
                            "summary": _redact_secrets_for_ui(summary_override or payload.get("result_preview", "")),
                            "error": _redact_secrets_for_ui(payload.get("error", "")),
                            "files": _redact_secrets_for_ui(payload.get("files", [])),
                            "progress_snapshot": _redact_secrets_for_ui(payload.get("progress_snapshot", {})),
                        },
                        "_meta": {
                            "ok": payload.get("success", False),
                            "tool": payload.get("tool"),
                            "agent": "minimax",
                            "state": "done",
                            "tool_call_id": payload.get("tool_call_id"),
                            "duration_ms": payload.get("duration_ms"),
                        },
                    })
            asyncio.run_coroutine_threadsafe(publish(), loop)

        agent = MinimaxToolAgent(
            memory_search=memory_search,
            on_tool_event=on_tool_event,
            disabled_tool_prefixes=[] if self._feature_enabled("computer_tools") else ["computer_"],
        )

        try:
            result = await asyncio.to_thread(
                agent.run,
                task=task,
                memory_context=memory_context,
                user_id=self._memory.user_id,
                original_user_message=self._last_user_text,
            )
        except Exception as e:
            publish_task_voice(f"工具任務遇到問題：{e}", force=True)
            return {
                "error": f"Minimax 工具代理執行失敗: {e}",
                "tool_count": len(TOOL_AGENT_TOOLS),
            }

        if self._sources["memory"]:
            await self._store_delegate_learning_events(result)

        if isinstance(result, dict):
            result = dict(result)
            result["assistant_response_contract"] = _delegate_response_contract(result)
            progress = result.get("progress_snapshot") if isinstance(result.get("progress_snapshot"), dict) else {}
            final_summary = str(progress.get("summary") or result.get("answer") or result.get("error") or "").strip()
            if result.get("stopped_for_budget"):
                publish_task_voice("工具任務接近輪數上限，已整理目前進度。", force=True)
            elif result.get("error"):
                publish_task_voice(f"工具任務遇到問題：{final_summary}", force=True)
            else:
                publish_task_voice(f"工具任務完成：{final_summary or '已完成工具處理。'}", force=True)

        return result

    async def _store_delegate_learning_events(self, result: dict) -> None:
        if not isinstance(result, dict):
            return
        events = result.get("learning_events") or []
        if not isinstance(events, list):
            return
        for event in events[:8]:
            if not isinstance(event, dict):
                continue
            memory = str(event.get("memory") or "").strip()
            if not memory:
                continue
            category = str(event.get("category") or "technical").strip()
            if category not in {"technical", "project", "event", "other"}:
                category = "technical"
            try:
                importance = int(event.get("importance", 4))
            except Exception:
                importance = 4
            importance = max(1, min(5, importance))
            try:
                await self._memory.execute("store_memory", {
                    "memory": memory,
                    "category": category,
                    "importance": importance,
                })
                self._memory_written_this_turn = True
            except Exception as e:
                log.debug("委派學習事項寫入失敗: %s", e)

    async def _store_visual_identity_tool(self, args: dict) -> dict:
        if not self._latest_video_jpeg:
            return {"ok": False, "error": "目前沒有可用的最新鏡頭畫面，無法建立圖像身份記憶"}
        if self._visual_identity_written_this_turn:
            return {"ok": True, "stored": False, "message": "本輪已建立過圖像身份記憶，避免重複寫入"}

        label = str(args.get("label") or self._memory.user_id or "我").strip()
        if label in {"我", "自己", "目前使用者", "使用者"}:
            label = self._memory.user_id
        source_text = str(args.get("source_text") or self._last_user_text or "").strip()

        result = await asyncio.to_thread(
            self._memory.enroll_visual_identity_from_jpeg,
            self._latest_video_jpeg,
            label,
            source_text,
        )
        if result.get("ok"):
            self._visual_identity_written_this_turn = True
            await self._bridge.publish(Channel.MEMORY_WRITE, {
                "memory": f"圖像身份記憶：{result.get('label', label)}",
                "category": "personal",
                "importance": 5,
            })
            await self._bridge.publish(Channel.VISION_EVENT, {
                "reason": "identity_saved",
                "detail": f"已記住圖像身份：{result.get('label', label)}",
            })
        return result

    async def _maybe_auto_enroll_visual_identity(self, text: str) -> str:
        if not (self._feature_enabled("visual_identity") and self._feature_enabled("visual_identity_auto_enroll")):
            return ""
        label = _visual_identity_label_from_text(text, self._memory.user_id)
        if not label or self._visual_identity_written_this_turn:
            return ""
        result = await self._store_visual_identity_tool({
            "label": label,
            "source_text": text,
        })
        if result.get("ok"):
            return (
                "[Raphael 內部狀態：已根據目前鏡頭畫面建立圖像身份記憶，"
                f"名稱={result.get('label', label)}，帳號={self._memory.user_id}。]\n\n"
            )
        return (
            "[Raphael 內部狀態：使用者似乎要求建立圖像身份記憶，"
            f"但未完成：{result.get('error', '未知原因')}。]\n\n"
        )

    async def _maybe_auto_enroll_visual_identity_after_turn(self, text: str) -> None:
        try:
            await self._maybe_auto_enroll_visual_identity(text)
        finally:
            self._visual_identity_written_this_turn = False

    def _collect_voice_pcm(self, seconds: float = 4.0) -> bytes:
        cutoff = time.time() - max(0.5, seconds)
        chunks = [
            bytes(pcm)
            for ts, pcm in list(self._voice_pcm_ring)
            if ts >= cutoff and pcm
        ]
        return b"".join(chunks)

    async def _store_voice_identity_tool(self, args: dict) -> dict:
        if self._voice_identity_written_this_turn:
            return {"ok": True, "stored": False, "message": "本輪已建立過聲紋記憶，避免重複寫入"}

        pcm = self._collect_voice_pcm(seconds=5.0)
        if len(pcm) < int(Config.MIC_RATE * 2 * 0.9):
            return {"ok": False, "error": "目前沒有足夠的近期語音樣本，請用語音說一小段話後再註冊聲紋"}

        label = str(args.get("label") or self._memory.user_id or "我").strip()
        if label in {"我", "自己", "目前使用者", "使用者"}:
            label = self._memory.user_id
        source_text = str(args.get("source_text") or self._last_user_text or "").strip()

        result = await asyncio.to_thread(
            self._memory.enroll_voice_identity_from_pcm,
            pcm,
            label,
            source_text,
        )
        if result.get("ok"):
            self._voice_identity_written_this_turn = True
            await self._bridge.publish(Channel.MEMORY_WRITE, {
                "memory": f"聲紋身份記憶：{result.get('label', label)}",
                "category": "personal",
                "importance": 5,
            })
            await self._bridge.publish(Channel.VISION_EVENT, {
                "reason": "voice_identity_saved",
                "detail": f"已記住聲音身份：{result.get('label', label)}",
            })
        return result

    async def _maybe_auto_enroll_voice_identity(self, text: str) -> str:
        if not (self._feature_enabled("voice_identity") and self._feature_enabled("voice_identity_auto_enroll")):
            return ""
        label = _voice_identity_label_from_text(text, self._memory.user_id)
        if not label or self._voice_identity_written_this_turn:
            return ""
        result = await self._store_voice_identity_tool({
            "label": label,
            "source_text": text,
        })
        if result.get("ok"):
            return (
                "[Raphael 內部狀態：已根據近期麥克風音訊建立聲紋身份記憶，"
                f"名稱={result.get('label', label)}，帳號={self._memory.user_id}。"
                "聲紋只作為語音閘門輔助判斷，不是安全驗證。]\n\n"
            )
        return (
            "[Raphael 內部狀態：使用者似乎要求建立聲紋身份記憶，"
            f"但未完成：{result.get('error', '未知原因')}。]\n\n"
        )

    async def _maybe_auto_enroll_voice_identity_after_turn(self, text: str) -> None:
        try:
            await self._maybe_auto_enroll_voice_identity(text)
        finally:
            self._voice_identity_written_this_turn = False

    @staticmethod
    def _visual_attention_from_candidates(candidates: list[dict]) -> float:
        best = 0.0
        for item in candidates or []:
            box = item.get("box") or {}
            try:
                x1 = float(box.get("x1", 0.0))
                y1 = float(box.get("y1", 0.0))
                x2 = float(box.get("x2", 0.0))
                y2 = float(box.get("y2", 0.0))
            except Exception:
                continue
            w = max(0.0, x2 - x1)
            h = max(0.0, y2 - y1)
            area = w * h
            cx = x1 + w / 2
            cy = y1 + h / 2
            dist = ((cx - 0.5) ** 2 + (cy - 0.45) ** 2) ** 0.5
            centered = max(0.0, 1.0 - dist / 0.72)
            matched_bonus = 0.25 if item.get("matched") else 0.0
            score = min(1.0, area * 5.2 + centered * 0.48 + matched_bonus)
            best = max(best, score)
        return best

    def _update_mouth_motion_from_jpeg(self, jpeg: bytes, candidates: list[dict]) -> None:
        if not self._feature_enabled("mouth_sync"):
            self._mouth_motion_score = 0.0
            self._last_mouth_motion_at = 0.0
            self._prev_mouth_crop = None
            return
        if not jpeg or not candidates:
            self._mouth_motion_score = max(0.0, self._mouth_motion_score * 0.75)
            return

        best = None
        best_area = 0.0
        for item in candidates:
            box = item.get("box") or {}
            try:
                x1 = float(box.get("x1", 0.0))
                y1 = float(box.get("y1", 0.0))
                x2 = float(box.get("x2", 0.0))
                y2 = float(box.get("y2", 0.0))
            except Exception:
                continue
            area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
            if area > best_area:
                best_area = area
                best = (x1, y1, x2, y2)
        if not best or best_area < 0.01:
            self._mouth_motion_score = max(0.0, self._mouth_motion_score * 0.75)
            return

        try:
            import cv2
            import numpy as np

            frame = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                return
            h, w = frame.shape[:2]
            x1, y1, x2, y2 = best
            fx1 = int(max(0.0, min(1.0, x1)) * w)
            fy1 = int(max(0.0, min(1.0, y1)) * h)
            fx2 = int(max(0.0, min(1.0, x2)) * w)
            fy2 = int(max(0.0, min(1.0, y2)) * h)
            fw = max(1, fx2 - fx1)
            fh = max(1, fy2 - fy1)

            mx1 = fx1 + int(fw * 0.20)
            mx2 = fx1 + int(fw * 0.80)
            my1 = fy1 + int(fh * 0.55)
            my2 = fy1 + int(fh * 0.90)
            if mx2 <= mx1 or my2 <= my1:
                return
            crop = frame[my1:my2, mx1:mx2]
            if crop.size == 0:
                return
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (48, 24), interpolation=cv2.INTER_AREA)
            gray = cv2.equalizeHist(gray).astype("float32") / 255.0

            if self._prev_mouth_crop is None or self._prev_mouth_crop.shape != gray.shape:
                self._prev_mouth_crop = gray
                self._mouth_motion_score = max(0.0, self._mouth_motion_score * 0.75)
                return

            diff = float(np.mean(np.abs(gray - self._prev_mouth_crop)))
            self._prev_mouth_crop = gray
            score = max(0.0, min(1.0, diff * 2.6))
            self._mouth_motion_score = (self._mouth_motion_score * 0.45) + (score * 0.55)
            if self._mouth_motion_score >= Config.MOUTH_SYNC_THRESHOLD:
                self._last_mouth_motion_at = time.time()
        except Exception as e:
            log.debug("嘴部動作估計略過: %s", e)

    async def _scan_visual_identities(self, jpeg: bytes) -> None:
        visual_identity_enabled = self._feature_enabled("visual_identity")
        mouth_sync_enabled = self._feature_enabled("mouth_sync")
        if not jpeg or not self._sources["memory"] or not (visual_identity_enabled or mouth_sync_enabled):
            return
        if self._visual_scan_busy:
            return
        self._visual_scan_busy = True
        try:
            detector = self._memory.identify_visual_identities if visual_identity_enabled else self._memory.detect_visual_candidates
            result = await asyncio.to_thread(detector, jpeg, 5)
        finally:
            self._visual_scan_busy = False
        if not result.get("ok"):
            return

        matches = (result.get("matches", []) or []) if visual_identity_enabled else []
        candidates = result.get("candidates", []) or []
        now = time.time()
        self._vision_available = True
        self._vision_face_count = int(result.get("faces", len(candidates)) or 0)
        self._vision_attention_score = self._visual_attention_from_candidates(candidates)
        if self._vision_face_count > 0:
            self._last_face_seen = now
        if matches and visual_identity_enabled:
            self._last_recognized_seen = now
        self._recognized_people = [str(m.get("label") or "") for m in matches if str(m.get("label") or "").strip()]
        self._update_mouth_motion_from_jpeg(jpeg, candidates)

        boxes = []
        for match in matches if visual_identity_enabled else []:
            box = match.get("box") or {}
            label = str(match.get("label") or "未知")
            boxes.append({
                **box,
                "kind": "identity",
                "label": label,
                "score": match.get("score"),
                "active": True,
            })

        await self._bridge.publish(Channel.SENSOR_VIEW, {
            "identity_boxes": boxes,
            "recognized_people": [b["label"] for b in boxes],
            "vision_attention": self._vision_attention_score,
            "face_count": self._vision_face_count,
            "mouth_motion": round(self._mouth_motion_score, 3),
        })

        for match in matches if visual_identity_enabled else []:
            label = str(match.get("label") or "").strip()
            if not label:
                continue
            last_seen = self._identity_seen_at.get(label, 0.0)
            if now - last_seen < Config.VISUAL_IDENTITY_SEEN_COOLDOWN:
                continue
            self._identity_seen_at[label] = now
            score = match.get("score", 0)
            await self._bridge.publish(Channel.VISION_EVENT, {
                "reason": "identity_seen",
                "detail": f"辨識到已記住的人：{label}（相似度 {score}）",
            })
            await self._bridge.publish(Channel.PROACTIVE, {
                "type": "vision_identity",
                "label": label,
                "score": score,
            })

    async def _scan_voice_identity_if_due(self, *, speaking: bool, probability: float) -> None:
        now = time.time()
        if not self._feature_enabled("voice_identity"):
            self._voice_match_label = ""
            self._voice_match_score = 0.0
            self._last_voice_match_at = 0.0
            return
        if not speaking or probability < 0.12 or not self._sources["memory"]:
            return
        if now - self._last_voice_scan < Config.VOICE_IDENTITY_SCAN_GAP:
            return
        self._last_voice_scan = now
        pcm = self._collect_voice_pcm(seconds=3.5)
        if len(pcm) < int(Config.MIC_RATE * 2 * 0.9):
            return
        result = await asyncio.to_thread(self._memory.identify_voice_identity, pcm)
        if not result.get("ok"):
            return
        matches = result.get("matches", []) or []
        if matches:
            best = matches[0]
            self._voice_match_label = str(best.get("label") or "").strip()
            self._voice_match_score = float(best.get("score", 0.0) or 0.0)
            self._last_voice_match_at = now
        elif now - self._last_voice_match_at > 8.0:
            self._voice_match_label = ""
            self._voice_match_score = 0.0

    # ══════════════════════════════════════════════════════════════════════
    # Bridge → Gemini 轉發（感知資料）
    # ══════════════════════════════════════════════════════════════════════

    async def _audio_forward_loop(self) -> None:
        """AUDIO_IN（VAD 產生的 PCM16）→ Gemini"""
        sub = self._bridge.listen(Channel.AUDIO_IN)
        try:
            async for _, payload in sub:
                if self._closed:
                    break
                if not self._sources["audio"]:
                    continue
                pcm = payload.get("pcm", b"")
                speaking = bool(payload.get("speaking", self._latest_vad_speaking))
                probability = float(payload.get("probability", self._latest_vad_probability) or 0.0)
                if pcm and self._feature_enabled("voice_identity") and (speaking or probability >= 0.08):
                    self._voice_pcm_ring.append((time.time(), bytes(pcm)))
                self._latest_vad_speaking = speaking
                self._latest_vad_probability = probability
                await self._scan_voice_identity_if_due(speaking=speaking, probability=probability)
                now = time.time()
                voice_match_recent = self._feature_enabled("voice_identity") and (now - self._last_voice_match_at) < 5.0
                mouth_active = self._feature_enabled("mouth_sync") and (now - self._last_mouth_motion_at) < 0.85
                if self._feature_enabled("advanced_voice_gate"):
                    decision = self._voice_governor.decide(
                        speaking=speaking,
                        probability=probability,
                        visual_available=self._vision_available,
                        visual_attention=self._vision_attention_score,
                        face_recent=(now - self._last_face_seen) < 3.5,
                        recognized_recent=self._feature_enabled("visual_identity") and (now - self._last_recognized_seen) < 5.0,
                        voice_match_recent=voice_match_recent,
                        mouth_active=mouth_active,
                        mouth_score=self._mouth_motion_score,
                        quiet_active=self._proactive_governor.quiet_active,
                        assistant_recent=(now - self._last_assistant_activity) < 1.8 if self._last_assistant_activity else False,
                        manual_watch=self._proactive_governor.watch_active,
                    )
                else:
                    # 純串流模式（展覽預設）：把麥克風音訊「原封不動」持續送給 Gemini，
                    # 交由 Gemini 原生 VAD 判斷說話起訖與打斷。本地 VAD 只用來點亮 UI 音量條與
                    # 提供主動性「使用者正在說話」的脈絡——絕不拿來當轉發閘門，否則會截掉字詞開頭造成延遲。
                    decision = {"forward": True, "reason": "stream"}

                # 半雙工：AI 出聲（含 0.8s 播放尾音）期間暫停轉發麥克風，否則喇叭聲被收回去，
                # Gemini 原生 VAD 會把 AI 自己的聲音當成使用者插話 → 自我打斷、重複講同一句。
                # 接耳機/指向麥克風時可開 full_duplex 停用此抑制，恢復隨時插話。
                assistant_speaking = (
                    not self._feature_enabled("full_duplex")
                    and now < (self._assistant_speaking_until + 0.8)
                )
                if assistant_speaking:
                    decision = {"forward": False, "reason": "assistant_speaking"}

                await self._publish_audio_gate_state(speaking, probability, decision)
                if decision.get("forward") and pcm and self._ws_open():
                    try:
                        await self._ws.send(json.dumps({
                            "realtimeInput": {
                                "audio": {
                                    "data": base64.b64encode(pcm).decode(),
                                    "mimeType": f"audio/pcm;rate={Config.MIC_RATE}",
                                }
                            }
                        }))
                    except Exception:
                        # 連線剛好斷掉：交給 recv 監督迴圈自動重連，這裡丟棄這格音訊即可。
                        pass
                # _user_speaking / 語音脈絡時間戳以「實際 VAD 人聲」為準，與是否轉發解耦，
                # 確保連續串流時主動性不會因 user_speaking 永遠為真而被凍結。
                # AI 說話期間的收音多半是回授，不可當成使用者在說話（否則主動性又被壓死）。
                if speaking and not assistant_speaking:
                    self._last_audio_forward_at = now
                    self._user_speaking = True
                elif self._last_audio_forward_at and now - self._last_audio_forward_at > 1.5 and not self._in_transcript:
                    self._user_speaking = False
        finally:
            self._bridge.unsubscribe(sub)

    async def _vad_state_loop(self) -> None:
        """Track VAD state for server-side audio gating."""
        sub = self._bridge.listen(Channel.VAD_EVENT)
        try:
            async for _, payload in sub:
                if self._closed:
                    break
                prev_speaking = self._latest_vad_speaking
                self._latest_vad_speaking = bool(payload.get("speaking", False))
                self._latest_vad_probability = float(payload.get("probability", 0.0) or 0.0)

                # 語音打斷：偵測到使用者開始說話（從無聲到有聲），且功能開啟
                if (
                    self._feature_enabled("voice_interrupt")
                    and self._latest_vad_speaking
                    and not prev_speaking
                    and not self._voice_interrupt_triggered_this_turn
                ):
                    self._voice_interrupt_triggered_this_turn = True
                    log.info("語音打斷觸發：偵測到使用者開始說話")
                    asyncio.create_task(self._send_interrupt_to_gemini())

                # 當使用者停止說話後，重置標記，這樣下次說話可以再次打斷
                if not self._latest_vad_speaking:
                    self._voice_interrupt_triggered_this_turn = False
        finally:
            self._bridge.unsubscribe(sub)

    async def _publish_audio_gate_state(self, speaking: bool, probability: float, decision: dict) -> None:
        now = time.time()
        speaker_label = self._voice_match_label if (now - self._last_voice_match_at) < 5.0 else ""
        state = (
            f"{speaking}:{decision.get('forward')}:{decision.get('reason')}:"
            f"{speaker_label}:{round(self._voice_match_score, 2)}:{round(self._mouth_motion_score, 2)}"
        )
        if state == self._last_audio_gate_state and now - self._last_audio_gate_publish < 0.7:
            return
        self._last_audio_gate_state = state
        self._last_audio_gate_publish = now
        await self._bridge.publish(Channel.VAD_EVENT, {
            "speaking": speaking,
            "probability": probability,
            "listening": bool(decision.get("forward")),
            "gate_reason": decision.get("reason", ""),
            "speaker_label": speaker_label,
            "voice_score": round(self._voice_match_score, 3) if speaker_label else 0.0,
            "mouth_motion": round(self._mouth_motion_score, 3),
        })

    async def _video_forward_loop(self) -> None:
        """VIDEO_IN（Vision 產生的 JPEG）→ 只更新 _latest_video_jpeg 緩衝 + 身份掃描。

        重要：不再持續把未經處理的幀串流給 Gemini。畫面只在三種時機進入模型：
          1. gate 開火（_send_proactive_decision 送 gate 挑出的最佳幀）
          2. 使用者文字出現視覺意圖時自動附幀（_text_forward_loop）
          3. 模型主動呼叫 look_now 工具擷取當下幀（_handle_look_now）
        """
        sub = self._bridge.listen(Channel.VIDEO_IN)
        try:
            async for _, payload in sub:
                if self._closed:
                    break  # 重連期間 _ws 可能短暫為 None，仍要持續更新緩衝，不可中斷
                jpeg = payload.get("jpeg", b"")
                if jpeg:
                    self._latest_video_jpeg = jpeg
                if not self._sources["vision"]:
                    continue
                now = time.time()
                identity_interval = 1.0 / max(0.1, Config.VISUAL_IDENTITY_SCAN_FPS)
                visual_scan_enabled = self._feature_enabled("visual_identity") or self._feature_enabled("mouth_sync")
                if jpeg and self._sources["memory"] and visual_scan_enabled and now - self._last_identity_scan >= identity_interval:
                    self._last_identity_scan = now
                    self._track_background_task(asyncio.create_task(self._scan_visual_identities(jpeg)))
        finally:
            self._bridge.unsubscribe(sub)

    async def _send_frame_to_gemini(self, jpeg: bytes) -> bool:
        """把單張 JPEG 幀經 realtimeInput 送進 Gemini。回傳是否成功送出。"""
        if not jpeg or self._closed or not self._ws_open():
            return False
        try:
            await self._ws.send(json.dumps({
                "realtimeInput": {
                    "video": {
                        "data": base64.b64encode(jpeg).decode(),
                        "mimeType": "image/jpeg",
                    }
                }
            }))
            return True
        except Exception as e:
            log.warning("送畫面給 Gemini 失敗: %s", e)
            return False

    async def _text_forward_loop(self) -> None:
        """TEXT_IN（UI 文字輸入）→ 注入記憶 context → Gemini"""
        sub = self._bridge.listen(Channel.TEXT_IN)
        try:
            async for _, payload in sub:
                if self._closed:
                    break
                text = payload if isinstance(payload, str) else payload.get("text", "")
                if not text:
                    continue

                self._proactive_turn_pending = False
                self._proactive_out_buffer = ""
                self._proactive_audio_buffer = []
                self._user_speaking = True
                self._last_user_text = text
                self._last_user_activity = time.time()
                governor_note = self._proactive_governor.note_user_text(text)
                identity_context = await self._maybe_auto_enroll_visual_identity(text)
                voice_identity_context = await self._maybe_auto_enroll_voice_identity(text)

                # 送出前先搜尋相關記憶，注入為上下文
                mem_context = ""
                smalltalk = _is_smalltalk(text)
                if self._sources["memory"] and not smalltalk and self._memory.should_retrieve_for(text):
                    relevant = self._memory.retrieve(text, limit=4, min_score=0.2)
                    if relevant:
                        mem_context = (
                            "[Raphael 內部可見的相關記憶]\n"
                            + "\n".join(f"- {r['memory']}" for r in relevant)
                            + "\n\n"
                        )

                preflight_context = ""
                if not smalltalk and self._sources["tool"]:
                    preflight_context = await self._maybe_preflight_tool_delegate(text, mem_context)

                if smalltalk:
                    full_text = (
                        "[系統提示：本輪使用者只是寒暄。"
                        "請由你自然生成一句簡短回覆；不要搜尋記憶、不要呼叫任何工具、不要延續上一個任務。]\n"
                        f"使用者說：{text}"
                    )
                else:
                    governor_context = f"[Raphael 內部狀態：{governor_note}。]\n\n" if governor_note else ""
                    full_text = governor_context + identity_context + voice_identity_context + mem_context + preflight_context + text

                if not await self._ensure_live_session():
                    await self._bridge.publish(Channel.ERROR, "Gemini session 目前不可用，請稍後重試或重新整理連線")
                    self._user_speaking = False
                    continue

                # 視覺意圖：自動把當下畫面附給模型（取代過去持續串流）。
                # 模型仍可在其他情況自行呼叫 look_now。
                if (
                    not smalltalk
                    and self._sources["vision"]
                    and self._latest_video_jpeg
                    and _VISUAL_INTENT_RE.search(text)
                ):
                    self._visual_frame_sent_this_turn = True
                    text += "\n[系統提示：已自動擷取當下鏡頭畫面並同步給你，請直接依此畫面回答，不需再呼叫 look_now。]"
                    await self._send_frame_to_gemini(self._latest_video_jpeg)

                message = json.dumps({
                    "clientContent": {
                        "turns": [{"role": "user", "parts": [{"text": full_text}]}],
                        "turnComplete": True,
                    }
                })
                try:
                    await self._ws.send(message)
                except Exception as e:
                    log.warning("送往 Gemini 失敗，嘗試重連: %s", e)
                    if await self._ensure_live_session():
                        await self._ws.send(message)
                    else:
                        await self._bridge.publish(Channel.ERROR, f"送往 Gemini 失敗: {e}")
                        self._user_speaking = False
        finally:
            self._bridge.unsubscribe(sub)

    async def _maybe_preflight_tool_delegate(self, text: str, memory_context: str) -> str:
        try:
            decision = await asyncio.to_thread(
                route_user_request_for_tools,
                text,
                memory_context=memory_context,
                available_tools=TOOL_AGENT_TOOLS,
            )
        except Exception as e:
            log.warning("工具需求路由器失敗，交回 Gemini 自行決定工具: %s", e)
            return (
                "[Raphael 內部狀態：工具需求路由器暫時不可用。"
                "如果本輪回答需要目前狀態、外部資料或實際操作，你必須呼叫 delegate_tool_task，"
                "不可憑空描述、不可猜測已完成。]\n\n"
            )

        if not tool_route_requires_delegate(decision):
            return ""

        task = str(decision.get("task") or text).strip()
        await self._bridge.publish(Channel.TASK_VOICE, {"text": "我判斷這需要先用工具確認。"})
        await self._bridge.publish(Channel.TOOL_CALL, {
            "name": "delegate_tool_task",
            "args": _redact_secrets_for_ui({
                "task": task,
                "route_reason": decision.get("reason", ""),
                "confidence": decision.get("confidence", 0),
            }),
            "_meta": {"agent": "router", "state": "running"},
        })
        started = time.perf_counter()
        result = await self._delegate_tool_task({
            "task": task,
            "memory_context": memory_context,
        })
        duration_ms = round((time.perf_counter() - started) * 1000, 2)
        compact = _compact_for_ui("delegate_tool_task", result)
        await self._bridge.publish(Channel.TOOL_RESULT, {
            "name": "delegate_tool_task",
            "result": compact,
            "_meta": {
                "ok": compact.get("ok", "error" not in result) if isinstance(compact, dict) else "error" not in result,
                "tool": "delegate_tool_task",
                "agent": "router",
                "state": "done",
                "duration_ms": duration_ms,
            },
        })

        evidence = _redact_secrets_for_ui(result)
        evidence_text = json.dumps(evidence, ensure_ascii=False, default=str)
        if len(evidence_text) > 6000:
            evidence_text = evidence_text[:6000] + "... [truncated]"
        return (
            "[Raphael 內部工具預檢結果]\n"
            f"路由判斷：本輪需要先使用工具；原因：{decision.get('reason', '')}。\n"
            f"已委派工具任務：{task}\n"
            f"工具結果：{evidence_text}\n"
            "回答使用者時必須依照上述工具結果；不可說沒有調用工具，"
            "不可憑空補畫面或外部狀態，若工具只完成部分進度，請說明已完成與下一步。\n\n"
        )

    # ══════════════════════════════════════════════════════════════════════
    # 半主動邏輯
    # ══════════════════════════════════════════════════════════════════════

    async def _proactive_loop(self) -> None:
        """
        收到 PROACTIVE 事件 → ProactiveGovernor 判斷 → 必要時觸發 Gemini 主動開口。
        """
        sub = self._bridge.listen(Channel.PROACTIVE)
        try:
            async for _, payload in sub:
                if self._closed:
                    break  # 重連期間不可中斷，連線狀態交給送出時 _ws_open 判斷
                if not self._ws_open():
                    continue
                if not self._sources["proactive"] or not self._sources["vision"]:
                    continue
                if not self._feature_enabled("vision_proactive"):
                    continue
                if isinstance(payload, dict) and payload.get("type") == "vision_identity" and not self._feature_enabled("visual_identity"):
                    continue

                # 動態候選（藍紅框）：type 為 vision motion 但「沒有 frame_jpeg」＝低階動態訊號，不送 Gemini。
                # 否則每幾秒一個 turn 會灌爆 context、佔住 _proactive_turn_pending，把真正帶最佳幀的綠框 FIRE 擠掉，
                # 造成「綠框觸發卻沒輸出觀察」。藍紅框仍在 UI 顯示；只有 FIRE（帶最佳幀）與心跳會觸發 Gemini。
                is_fire = False
                if isinstance(payload, dict):
                    ptype = str(payload.get("type") or "")
                    if (
                        ptype in {"vision:object_motion", "vision:fast_burst", "vision:motion", "vision:semantic", "vision:object"}
                        and not payload.get("frame_jpeg")
                    ):
                        log.debug("略過無 frame_jpeg 的 motion 候選: ptype=%s", ptype)
                        continue
                    is_fire = bool(payload.get("frame_jpeg"))

                blocked = self._proactive_blocked(is_fire=is_fire)
                if blocked:
                    log.debug(
                        "主動回合 blocked: is_fire=%s proactive_pending=%s proactive_is_fire=%s",
                        is_fire,
                        getattr(self, "_proactive_turn_pending", False),
                        getattr(self, "_proactive_pending_is_fire", False),
                    )
                decision = self._proactive_governor.decide(
                    payload,
                    # user_busy 需「真的在說話」才算；純串流下瀏覽器 VAD 門檻低(peak>0.025)，
                    # 現場雜音會讓 _user_speaking 長期為真而永遠壓掉主動反應，故再加機率門檻。
                    user_busy=(self._user_speaking and self._latest_vad_probability >= 0.45),
                    last_user_activity=self._last_user_activity,
                    last_assistant_activity=self._last_assistant_activity,
                    voice_active=self._voice_context_active(),
                    recent_voice=self._recent_voice_context(),
                    visual_attention=self._vision_attention_score,
                    proactive_blocked=blocked,
                )
                action = decision.get("action")
                log.debug(
                    "PROACTIVE 事件: type=%s is_fire=%s action=%s reason=%s blocked=%s",
                    payload.get("type") if isinstance(payload, dict) else payload,
                    is_fire,
                    action,
                    decision.get("reason"),
                    blocked,
                )
                if action == "defer":
                    self._schedule_deferred_proactive(payload, decision)
                    continue
                if action != "speak":
                    log.debug(
                        "主動開口略過: action=%s reason=%s event=%s",
                        action,
                        decision.get("reason"),
                        decision.get("type"),
                    )
                    continue

                frame_jpeg = payload.get("frame_jpeg") if isinstance(payload, dict) else None
                await self._send_proactive_decision(decision, frame_jpeg=frame_jpeg)
        finally:
            self._bridge.unsubscribe(sub)

    def _voice_context_active(self) -> bool:
        now = time.time()
        return bool(self._latest_vad_speaking and self._last_audio_forward_at and now - self._last_audio_forward_at < 2.0)

    def _recent_voice_context(self) -> bool:
        now = time.time()
        return bool(
            self._last_audio_forward_at
            and now - self._last_audio_forward_at < Config.PROACTIVE_AUDIO_CONTEXT_WINDOW
        )

    def _schedule_deferred_proactive(self, payload, decision: dict) -> None:
        task = self._deferred_proactive_task
        if task is not None and not task.done():
            return
        delay = max(0.2, float(decision.get("defer_ms", 900) or 900) / 1000.0)
        self._deferred_proactive_task = asyncio.create_task(self._deferred_proactive(payload, delay))
        self._track_background_task(self._deferred_proactive_task)

    async def _deferred_proactive(self, payload, delay: float) -> None:
        await asyncio.sleep(delay)
        deadline = time.time() + 4.0
        while self._user_speaking and time.time() < deadline:
            await asyncio.sleep(0.25)
        if self._closed or not self._ws_open():
            return
        frame_jpeg = payload.get("frame_jpeg") if isinstance(payload, dict) else None
        decision = self._proactive_governor.decide(
            payload,
            user_busy=False,
            last_user_activity=self._last_user_activity,
            last_assistant_activity=self._last_assistant_activity,
            voice_active=False,
            recent_voice=True,
            visual_attention=self._vision_attention_score,
            proactive_blocked=self._proactive_blocked(is_fire=frame_jpeg is not None),
        )
        if decision.get("action") == "speak":
            frame_jpeg = payload.get("frame_jpeg") if isinstance(payload, dict) else None
            await self._send_proactive_decision(decision, frame_jpeg=frame_jpeg)

    def _proactive_blocked(self, is_fire: bool = False) -> bool:
        """是否該擋下這個主動/插入 turn，避免打斷自己、重複講。

        兩種獨立的擋法：
        - AI「正在出聲」(`_assistant_speaking_until`)：任何事件都不可打斷，避免切斷發話中途。
        - 主動回合執行中（`_proactive_turn_pending`）：
          當前一個主動回合還在等待模型回答或執行工具時，我們需要保護它不被打斷。
          然而，如果是高價值的「綠框事件 (is_fire=True)」：
          1. 若當前 pending 的只是「背景心跳 (is_fire=False)」，允許無條件立刻打斷，保證 100% 即時性！
          2. 若當前 pending 的也是綠框，給予較短的等待窗 (5.0s)。
          若新事件只是一般心跳，則維持 25.0 秒安全保護超時。
        """
        now = time.time()
        if now < self._assistant_speaking_until + 0.8:
            return True
            
        # 絕對優先權：綠框事件無條件打斷背景心跳
        if is_fire and self._proactive_turn_pending and not getattr(self, "_proactive_pending_is_fire", False):
            return False

        timeout = 5.0 if is_fire else 25.0
        if self._proactive_turn_pending and (now - self._proactive_pending_since) < timeout:
            return True
        return False

    async def _send_proactive_decision(self, decision: dict, frame_jpeg: bytes | None = None) -> None:
        # fire(綠框)帶最佳畫面、屬重要事件，優先權高於動態候選；思考窗等更短，較不會被低優先事件卡住。
        if self._proactive_blocked(is_fire=frame_jpeg is not None):
            log.debug("主動回合進行中或 AI 正在說話，略過此事件")
            return
        now = time.time()

        description = decision.get("description", "環境發生變化")
        event_type = decision.get("type", "vision:event")
        detail = str(decision.get("detail") or "").strip()
        metrics = decision.get("metrics") if isinstance(decision.get("metrics"), dict) else {}
        boxes = decision.get("boxes") if isinstance(decision.get("boxes"), list) else []
        compact_boxes = boxes[:3]
        score = decision.get("score")
        extra_lines = []
        if detail:
            extra_lines.append(f"細節：{detail}")
        if metrics:
            metric_text = ", ".join(
                f"{k}={v}" for k, v in metrics.items()
                if k in {"drift", "roi_drift", "jitter", "motion_area", "flow_pixels", "objects", "fast_motion_hit"}
            )
            if metric_text:
                extra_lines.append(f"偵測數據：{metric_text}")
        if score is not None:
            extra_lines.append(f"score={score}")
        if compact_boxes:
            extra_lines.append("變化區域：" + json.dumps(compact_boxes, ensure_ascii=False))
        extra_context = ("；".join(extra_lines) + "。") if extra_lines else ""
        watch_line = "使用者目前希望你協助觀察畫面。" if decision.get("watch_mode") else ""
        voice_line = "這個事件發生在使用者正在說話或剛說完附近；請優先理解它可能和使用者當下語音有關。" if decision.get("voice_context") else ""

        # 完全 Raphael 主動性：不再限制成「一句話或 SILENT」。
        # 模型自己判斷是否值得介入；若值得，可以說話、也可以呼叫工具去查證或執行對使用者有幫助的動作。
        prompt = (
            "[系統提示：這是你（Raphael）主動察覺到的環境事件，不是使用者直接提問。"
            f"事件：{description}。事件類型：{event_type}。{extra_context}{watch_line}{voice_line}"
            "我已把此刻 gate 為你挑出的最佳畫面同步給你。"
            "請像史萊姆裡的智慧之王拉斐爾一樣，先真正看懂這張畫面發生了什麼，再判斷它對使用者當下的目標或處境是否重要、現在介入是否有幫助。"
            "依需要二選一回覆："
            "（A）需要開口回答或提醒使用者→直接用一兩句自然、不打擾的繁體中文說出來。注意：此處為背景事件處理，請【絕對不要】呼叫任何工具（如 delegate_tool_task、search_memories、look_now 等），直接給出純文字回答。"
            "（B）不需要開口（包含值得記下、小晃動、無關緊要的光線變化等）→一律在開頭加上 OBSERVE 標記，寫下一句你觀察到的重點（例如：OBSERVE 只是光線變化，或 OBSERVE 使用者正在看書）。系統會把它當成無語音的觀察記錄，不會發出聲音。注意：請【絕對不要】呼叫任何工具。"
            "不要硬找話講；如果不值得開口打擾，請一律使用 OBSERVE 記錄。]"
        )

        if not self._ws_open():
            return
        using_frame = "fire_frame" if frame_jpeg else "latest_video (fallback!)"
        log.info("送 Gemini 判斷主動事件: %s (%s) [frame=%s]", event_type, decision.get("reason"), using_frame)
        self._proactive_turn_pending = True
        self._proactive_pending_since = now
        self._proactive_pending_is_fire = True
        self._proactive_out_buffer = ""
        self._proactive_audio_buffer = []
        # 把畫面內嵌進同一個 clientContent turn（inlineData），與提示詞一起原子送出。
        # 原本走 realtimeInput 串流通道送幀、再用 clientContent 觸發回合，但兩條通道處理時序不同：
        # 幀常常還沒併進當前回合，turnComplete 就已叫模型開始生成，於是模型看到的是上一次觸發已吸收的舊幀（慢一拍）。
        # 改成內嵌後，圖和提示詞在同一回合，必定對應到本次觸發的畫面。
        jpeg = frame_jpeg or self._latest_video_jpeg
        parts: list[dict] = [{"text": prompt}]
        if jpeg:
            parts.insert(0, {"inlineData": {"mimeType": "image/jpeg", "data": base64.b64encode(jpeg).decode()}})
        try:
            await self._ws.send(json.dumps({
                "clientContent": {
                    "turns": [{"role": "user", "parts": parts}],
                    "turnComplete": True,
                }
            }))
        except Exception as e:
            log.warning("送出主動事件失敗（交給重連）：%s", e)
            self._proactive_turn_pending = False

    async def _proactive_heartbeat_loop(self) -> None:
        """認知心跳：閒置時定期戳模型自我檢視「現在有沒有值得主動說/做的事」。

        這補上主動性最大的缺口——原本主動性只由視覺事件觸發，場景一靜止就永遠沒機會開口。
        心跳只送純文字（不送畫面，避免吃爆 context），靠 Live session 既有的對話記憶判斷時機。
        """
        interval = max(8.0, Config.PROACTIVE_HEARTBEAT_SEC)
        while not self._closed:
            await asyncio.sleep(interval)
            if self._closed or not self._ws_open():
                continue
            if not self._sources["proactive"]:
                continue
            if not self._feature_enabled("vision_proactive"):
                continue
            if self._proactive_governor.quiet_active:
                continue
            if self._user_speaking or self._proactive_blocked():
                continue
            now = time.time()
            idle_user = (now - self._last_user_activity) if self._last_user_activity else 9999.0
            idle_asst = (now - self._last_assistant_activity) if self._last_assistant_activity else 9999.0
            # 只在真的閒置一段時間後才檢視，避免剛講完又馬上自言自語
            if min(idle_user, idle_asst) < interval:
                continue
            await self._send_proactive_review()

    async def _send_proactive_review(self) -> None:
        """閒置時的一次內部自我檢視（純文字、不送畫面）。回覆走和環境事件相同的 速/觀察/SILENT 邏輯。"""
        if self._proactive_blocked() or not self._ws_open():
            return
        prompt = (
            "[系統提示：這是一個安靜時刻的內部自我檢視，不是使用者提問，使用者也看不到這段提示。"
            "請像史萊姆裡的智慧之王拉斐爾一樣，回顧你和使用者到目前為止的對話、他提過的目標、待辦或在意的事，"
            "以及你已知道的脈絡，判斷現在是否有「真正值得主動說或做」的事——"
            "例如他可能忘了的下一步、一個此刻有用的提醒、或你能先替他完成並回報的事。"
            "依需要三選一："
            "（A）需要開口回答或提醒使用者→用一兩句自然、不打擾的繁體中文說。注意：此處為純背景檢視，請【絕對不要】呼叫任何工具（如 delegate_tool_task、search_memories、look_now 等），直接給出純文字回覆。"
            "（B）只是值得默默記下但不必出聲打擾的事→以 OBSERVE 開頭寫一句你的觀察重點。注意：請【絕對不要】呼叫任何工具。"
            "（C）目前沒有特別的事、無事需處理、或不值得出聲→請【必須】直接回覆 `SILENT` 四個字母，不要寫任何多餘的標點或文字。]"
        )
        now = time.time()
        log.debug("送 Gemini 主動自我檢視（心跳）")
        self._proactive_turn_pending = True
        self._proactive_pending_since = now
        self._proactive_pending_is_fire = False
        self._proactive_out_buffer = ""
        self._proactive_audio_buffer = []
        try:
            await self._ws.send(json.dumps({
                "clientContent": {
                    "turns": [{"role": "user", "parts": [{"text": prompt}]}],
                    "turnComplete": True,
                }
            }))
        except Exception as e:
            log.warning("送出主動自我檢視失敗：%s", e)
            self._proactive_turn_pending = False

    # ══════════════════════════════════════════════════════════════════════
    # 使用者打斷
    # ══════════════════════════════════════════════════════════════════════

    async def _user_interrupt_loop(self) -> None:
        """監聽使用者點擊打斷按鈕，直接送中斷信號給 Gemini。"""
        sub = self._bridge.listen(Channel.USER_INTERRUPT)
        try:
            async for _, payload in sub:
                if self._closed:
                    break
                await self._send_interrupt_to_gemini()
        finally:
            self._bridge.unsubscribe(sub)

    async def _send_interrupt_to_gemini(self) -> None:
        """送中斷信號給 Gemini：空白 clientContent + turnComplete=False"""
        log.info("收到使用者打斷請求")
        self._out_transcript = ""
        self._in_transcript = ""
        self._assistant_speaking_until = 0.0
        self._proactive_turn_pending = False
        self._proactive_pending_is_fire = False
        self._proactive_out_buffer = ""
        self._proactive_audio_buffer = []
        self._user_speaking = False
        self._voice_interrupt_triggered_this_turn = False  # 重置語音打斷標記
        self._tool_rounds = 0
        if not self._ws_open():
            return
        try:
            await self._ws.send(json.dumps({
                "clientContent": {
                    "turns": [],
                    "turnComplete": False,
                }
            }))
            await self._bridge.publish(Channel.INTERRUPTED, {})
            log.info("已送出 Gemini 中斷信號")
        except Exception as e:
            log.warning("送出中斷信號失敗: %s", e)

    # ══════════════════════════════════════════════════════════════════════
    # 自動記憶儲存（對話結束後）
    # ══════════════════════════════════════════════════════════════════════

    async def _auto_memory(self, user_text: str, ai_text: str) -> None:
        """Safety net: store obvious stable facts if the model did not call store_memory."""
        if not self._sources["memory"] or not user_text.strip():
            return
        await self._memory.maybe_auto_store(user_text, ai_text)
