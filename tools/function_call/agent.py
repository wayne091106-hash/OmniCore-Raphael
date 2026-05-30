"""
tools/function_call/agent.py — Minimax tool executor
═════════════════════════════════════════════════════

Gemini Live remains the dialogue brain and memory owner.  When Gemini decides a
task needs non-memory tools, core.py delegates a rewritten task prompt here.
Minimax then performs OpenAI-compatible tool calling over implementations.py and
returns a structured execution report to Gemini.
"""

from __future__ import annotations

import json
import hashlib
import os
import re
import ssl
import time
import urllib.parse
from collections.abc import Callable
from copy import deepcopy
from pathlib import Path
from typing import Any

import certifi
import httpx
from dotenv import load_dotenv
from openai import OpenAI

from .definitions import TOOLS
from .implementations import registry

try:
    import truststore
except ImportError:  # pragma: no cover - optional Windows SSL helper
    truststore = None

load_dotenv()

BASE_URL = os.environ.get("NIM_BASE_URL", "https://integrate.api.nvidia.com/v1")
MODEL = os.environ.get("NIM_MODEL", "minimaxai/minimax-m2.7")
MAX_TOOL_ROUNDS = int(os.environ.get("MAX_TOOL_ROUNDS", "16"))
MODEL_TIMEOUT = float(os.environ.get("NIM_REQUEST_TIMEOUT", "180"))
TEMPERATURE = float(os.environ.get("NIM_TEMPERATURE", "0.2"))

_MINIMAX_SETTINGS = {
    "base_url": BASE_URL,
    "model": MODEL,
    "max_tool_rounds": MAX_TOOL_ROUNDS,
    "request_timeout": MODEL_TIMEOUT,
    "temperature": TEMPERATURE,
}

MEMORY_TOOL_NAMES = {"store_memory", "recall_memory"}
EXPLICIT_EMAIL_RE = re.compile(
    r"(寄信|發信|寄送.*信|發送.*信|電子郵件|郵件|gmail|email|e-mail|寄到|寄給|回信)",
    re.IGNORECASE,
)
EMAIL_ADDRESS_RE = re.compile(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}")
REPEAT_GUARDED_TOOLS = {
    "browser_open",
    "browser_login",
    "web_search",
    "website_find",
    "dns_lookup",
    "open_url",
    "site_memory_search",
}
GENERIC_SITE_SEARCH_TERMS = {
    "moodle", "learning", "login", "portal", "course", "courses",
    "網站", "平台", "登入", "入口", "網址", "教學平台", "數位學習",
    "學習平台", "數位", "教學",
}
SITE_SERVICE_TERMS = {
    "moodle", "lms", "classroom", "portal", "login", "sso",
    "教學平台", "數位學習", "學習平台", "後台", "課程系統",
}
CONTEXT_NOISE_TERMS = {
    "原始使用者訊息", "Gemini", "Raphael", "委派任務", "使用者", "工具",
    "登入", "帳號", "密碼", "網站", "平台", "課程", "作業", "請", "幫我",
}

MEMORY_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "search_memories",
        "description": "依照目前 Raphael 記憶帳號搜尋長期記憶。只能讀取，不可寫入。",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "要搜尋的記憶查詢",
                },
                "limit": {
                    "type": "integer",
                    "description": "最多回傳幾筆，預設 5",
                },
            },
            "required": ["query"],
        },
    },
}

SECRET_FIELD_RE = re.compile(
    r"(?i)(密碼|password|passwd|pwd|passcode|token|secret|client_secret)\s*(?:[:=：]|為|为|是|is)?\s*([^\s,，;；。]+)"
)


def redact_secrets(value: Any):
    if isinstance(value, str):
        return SECRET_FIELD_RE.sub(lambda m: f"{m.group(1)}=********", value)
    if isinstance(value, list):
        return [redact_secrets(v) for v in value]
    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            if re.search(r"(?i)(密碼|password|passwd|pwd|passcode|token|secret|client_secret)", str(key)):
                out[key] = "********"
            else:
                out[key] = redact_secrets(item)
        return out
    return value

TOOL_AGENT_TOOLS = [
    schema
    for schema in TOOLS
    if schema.get("function", {}).get("name") not in MEMORY_TOOL_NAMES
] + [MEMORY_SEARCH_TOOL]


def _clamp_number(value: Any, *, default: float, minimum: float, maximum: float, integer: bool = False) -> int | float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = float(default)
    parsed = max(minimum, min(maximum, parsed))
    return int(round(parsed)) if integer else parsed


def get_minimax_settings() -> dict:
    """Return the current runtime Minimax settings used by delegated tool work."""
    return deepcopy(_MINIMAX_SETTINGS)


def update_minimax_settings(settings: dict | None) -> dict:
    """Validate and apply runtime Minimax settings from the WebUI."""
    if not isinstance(settings, dict):
        return get_minimax_settings()

    current = get_minimax_settings()
    if "base_url" in settings:
        base_url = str(settings.get("base_url") or "").strip()
        if base_url and re.match(r"^https?://", base_url, re.IGNORECASE):
            current["base_url"] = base_url.rstrip("/")
    if "model" in settings:
        model = str(settings.get("model") or "").strip()
        if model:
            current["model"] = model
    if "max_tool_rounds" in settings:
        current["max_tool_rounds"] = _clamp_number(
            settings.get("max_tool_rounds"),
            default=current["max_tool_rounds"],
            minimum=4,
            maximum=64,
            integer=True,
        )
    if "request_timeout" in settings:
        current["request_timeout"] = _clamp_number(
            settings.get("request_timeout"),
            default=current["request_timeout"],
            minimum=20,
            maximum=600,
            integer=True,
        )
    if "temperature" in settings:
        current["temperature"] = round(float(_clamp_number(
            settings.get("temperature"),
            default=current["temperature"],
            minimum=0,
            maximum=1.5,
        )), 2)

    _MINIMAX_SETTINGS.update(current)
    return get_minimax_settings()

SYSTEM_PROMPT = """你是 Raphael 的 Minimax 工具執行代理。

分工：
- Gemini 是主對話模型，也是長期記憶的寫入與最終回答負責人。
- 你只負責完成 Gemini 委派的工具任務，並把成功或失敗的真實結果回傳。
- 你可以使用 implementations.py 內註冊的工具，以及只讀的 search_memories。
- 你不可自行寫入、修改、刪除長期記憶；若發現值得記錄的事項，在最終報告的 memory_suggestions 中列出，交給 Gemini 決定。
- computer_* 工具會真實操作使用者電腦（截圖、點擊、打字、快捷鍵）。只有在任務明確需要桌面操作、或 Gemini 委派提示要求時才使用；每次操作前必須先用 computer_active_window 或 computer_list_windows 確認要操作哪個視窗。
- 若 Gemini 委派你登入網站，且原始使用者訊息或 memory_context 已提供該網站的帳號密碼，視為使用者授權你在本輪代填登入；必須優先使用 browser_* 背景瀏覽器工具完成登入，避免干擾使用者目前操作的瀏覽器與滑鼠鍵盤。只有 browser_* 回報不可用、或頁面必須由使用者可見操作時，才改用 open_url/computer_*。

行為規則：
- 需要工具就主動呼叫，不要猜測或捏造。
- 對原始使用者已授權且不涉及危險、詐欺、繞過驗證或資料竊取的任務，以完成為預設；不要用泛用拒絕、道德勸告或時間管理建議取代工具工作。
- 最高優先順序是「原始使用者訊息」與「Gemini 委派任務」。相關記憶只能提供登入、入口、檔案位置、使用者偏好或過去經驗；不可把記憶中的舊任務、舊收件人、舊查詢詞改成這一輪的目標。
- 每輪開始先用一句內部判斷確認目標服務、目標動作與禁止事項；若後續工具呼叫與目標無關，必須停下改回原任務。
- 若工具失敗，保留錯誤、原因、可修復步驟，仍然回報已完成哪些部分。
- 同一個工具用同一組參數失敗後，不要原地重試；第二次失敗必須改策略，第三次會被系統視為卡住並停止。看到 repeated_call 或 recovery_hint 時，立刻換工具、換入口、或回報卡住原因。
- 若任務涉及使用者個人脈絡，優先使用委派提示中提供的 memory_context；不足時才呼叫 search_memories。
- 登入任務只能使用與目標網站/服務相符的帳密；不可把某網站帳密拿去嘗試其他網站。不要在最終回報中完整顯示密碼，只說明是否已登入。
- 若登入需要驗證碼、二階段驗證、手機確認或使用者本人操作，停下來回報需要使用者處理，不要猜測或繞過。
- 網站/登入任務開始時，先用 site_memory_search 查已學過的入口與已知失敗網址；若沒有明確入口，再用 website_find 搭配目標機構/服務名稱解析並驗證網址。不要用「登入 教學平台」這種泛用查詢直接挑第一個結果。
- 搜尋網站時必須比對目標機構與服務是否相符；例如使用者指定某學校或公司，就不可登入其他機構的同類平台。若搜尋結果與目標不符，換查詢或回報找不到，不要硬試。
- 每次確認可用的網站入口，用 site_memory_remember 記住服務與 URL；DNS 失敗、404、明顯錯站時，用 site_memory_mark_failure 記住，避免下次重踩。
- 桌面操作必須具備視窗意識：若任務提到「目前視窗、已登入視窗、我的瀏覽器、某個程式」，先讀取前景視窗；若不確定，列出可見視窗並依標題/程序選目標。找不到明確目標時停下回報，不要用全螢幕截圖猜座標。
- 需要截圖桌面時優先使用 computer_screenshot_window，只截取目標視窗；只有使用者明確要求整個螢幕或跨視窗比對時，才使用 computer_screenshot。
- 要連續點擊/輸入時，可用 computer_control，但步驟開頭應包含 active_window、list_windows 或 focus_window，讓結果中保留實際操作的視窗依據。
- 使用 browser_get_page 讀取頁面文字與 controls 後，優先透過 selector/文字操作；需要瀏覽頁面入口時用 browser_links 取得連結，再用 browser_follow_link 依 index 前進；等待動態頁面用 browser_wait，走錯頁用 browser_back。不要把截圖當文字檔讀取。只有需要讓使用者檢查畫面或遇到卡關時才回傳 browser_screenshot。
- 若 browser_open/browser_login 因網址或 DNS 失敗，先搜尋官方目前網址或回報網址錯誤；不得因此直接改操作使用者前景視窗，除非已用視窗工具確認那就是本任務目標。
- 網站網址需要搜尋時，優先選官方、目前使用中的主站或登入頁；避免選擇帶舊學年度、年份、封存或教學範例字樣的結果，除非使用者明確要求舊平台。
- 處理使用者上傳檔案時，優先使用委派提示中的本機路徑，依檔案類型選擇 read_file、pdf_extract_text、docx_extract_text、xlsx_read_sheet、image_info 等工具。
- Gmail 工具若回傳 found=true、id 或 snippet，就代表有找到郵件；必須根據 snippet 判斷內容，不可說「沒有收到」。
- 若 Gmail snippet 內含「不要、不用、不參加、拒絕、可以、好、同意」等明確回覆，必須在結果中直接說明對方回了什麼。
- gmail_send 是外部副作用工具。只有原始使用者訊息明確要求寄信/發郵件/email，且指定收件人或信箱時才可使用。使用者說「傳給我、給我、截圖給我」代表在 WebUI 對話中提供檔案或連結，不代表寄 Gmail。
- 使用者要求處理任何已授權網站、平台、課程、文件、資料或任務時，工作是依照本輪目標持續操作、讀取、整理、建立草稿或提出可執行步驟；不要改成搜尋無關專案、查 Gmail、寄信或泛用建議。若需要使用者本人判斷、驗證或最終確認，先完成可執行部分，再清楚標示需要確認的地方。
- 網路圖片任務應使用 web_image_search、download_image、computer_screenshot 等合適工具；不要改寫成寄信任務。
- 使用者要求「找圖片、截圖、傳給我、給我」時，正確完成方式是把圖片下載到 data/outputs 並在結果中回傳檔案路徑/file_url，讓 WebUI 顯示或下載。
- 不可把與本輪任務無關的記憶、舊任務、舊收件人帶進工具任務。
- 嚴格貼合 Gemini 委派任務，不要改寫成無關的泛用報告。
- 不要輸出 markdown 表格、長篇清單、工具原始 JSON 或多餘標題。
- 最終輸出必須是繁體中文純文字，最多 8 句，格式固定為：
  結果：...
  狀態：成功/部分成功/失敗
  需要使用者處理：...
  可記憶事項：...
"""


TOOL_ROUTER_SYSTEM = """你是 Raphael 的工具需求路由器，只負責判斷使用者本輪要求是否必須先取得外部證據或執行工具。

判斷原則：
- 只有純聊天、一般知識解釋、創作、改寫、翻譯、推理題，且不依賴目前狀態或外部資料時，才選 direct。
- 如果使用者要求「告訴我你看到什麼」、「看鏡頭」、「看眼前畫面」，系統會在底層自動把鏡頭畫面附給主模型，這不需要委派工具，請回傳 direct。
- 任何需要目前狀態、螢幕視窗截圖（注意：螢幕截圖與鏡頭畫面不同）、網站/檔案/郵件/日曆/雲端/系統/API/網路/最新資訊/登入/下載/查證/實際操作的要求，都選 delegate。
- 如果使用者要求「幫我確認、去做、幫我找、幫我登入、幫我下載、看目前狀態」，這些都需要可驗證證據，不能直接回答。
- 你的輸出只能是 JSON，不可回答使用者問題本身。

JSON 格式：
{"mode":"direct" 或 "delegate","confidence":0到1,"reason":"一句話原因","task":"若 mode=delegate，寫給工具代理的完整任務；否則空字串"}
"""


def _client(settings: dict | None = None) -> OpenAI:
    settings = settings or get_minimax_settings()
    api_key = os.environ.get("NIM_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("缺少 NIM_API_KEY，Minimax 工具代理無法啟動。")

    if truststore:
        verify = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    else:
        verify = certifi.where()

    timeout = float(settings["request_timeout"])
    http_client = httpx.Client(verify=verify, timeout=timeout)
    return OpenAI(
        api_key=api_key,
        base_url=str(settings["base_url"]),
        timeout=timeout,
        max_retries=1,
        http_client=http_client,
    )


def _safe_json_loads(raw: str | None) -> dict:
    if not raw:
        return {}
    try:
        obj = json.loads(raw)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def _tool_catalog_for_router(tools: list[dict] | None = None, limit: int = 80) -> str:
    lines: list[str] = []
    for schema in (tools or TOOL_AGENT_TOOLS)[:limit]:
        fn = schema.get("function", {}) if isinstance(schema, dict) else {}
        name = str(fn.get("name") or "").strip()
        desc = re.sub(r"\s+", " ", str(fn.get("description") or "").strip())
        if not name:
            continue
        lines.append(f"- {name}: {desc[:120]}")
    return "\n".join(lines)


def normalize_tool_route_decision(raw: dict, user_text: str = "") -> dict:
    mode = str(raw.get("mode") or "").strip().lower()
    if mode not in {"direct", "delegate"}:
        mode = "direct"
    try:
        confidence = float(raw.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))
    task = str(raw.get("task") or "").strip()
    if mode == "delegate" and not task:
        task = (
            "請依照原始使用者要求完成需要工具或外部證據的部分；"
            f"原始使用者要求：{user_text}"
        )
    if mode == "direct":
        task = ""
    return {
        "mode": mode,
        "confidence": confidence,
        "reason": str(raw.get("reason") or "").strip()[:240],
        "task": task[:2000],
    }


def route_user_request_for_tools(
    user_text: str,
    *,
    memory_context: str = "",
    available_tools: list[dict] | None = None,
) -> dict:
    """Ask a small router pass whether the user request needs tools before Gemini answers."""
    text = str(user_text or "").strip()
    if not text:
        return {"mode": "direct", "confidence": 1.0, "reason": "空訊息", "task": ""}

    # ── 本地快速判定通道（避免 10 秒 NVIDIA API 延遲，全面提升對話反應速度） ──
    lower_text = text.lower()
    # 1. 簡單符號、問號、或極短字元（例如：?, ??, hi, 123）
    if len(text) <= 5 or not re.search(r"[\u4e00-\u9fff\w]", text):
        return {"mode": "direct", "confidence": 1.0, "reason": "簡單字元/符號，本地直接對話", "task": ""}
    
    # 2. 視覺意圖與主觀提問（由 core 附當下幀給 Gemini 即可，絕不需 Minimax 電腦/瀏覽器操作）
    if re.search(
        r"(你?看到|看得到|看一下|看看|畫面|鏡頭|攝影機|眼前|現在.*前面|這是什麼|這個是什麼|我手上|我拿的|我穿|這張|這邊有什麼|前面有什麼|描述.*畫面|看得出|認得.*嗎|what do you see|look at|on screen|in front of)",
        lower_text
    ):
        return {"mode": "direct", "confidence": 1.0, "reason": "鏡頭與視覺提問，本地直接對話", "task": ""}
    
    # 3. 常見寒暄與打招呼
    if re.search(
        r"^(你好|您好|嗨|哈囉|哈啰|hello|hi|hey|早安|午安|晚安|在嗎|在不在|謝謝|謝謝你|感恩|ok|okay|好的|收到|掰掰|再見)",
        lower_text
    ):
        return {"mode": "direct", "confidence": 1.0, "reason": "常規寒暄/答覆，本地直接對話", "task": ""}

    settings = get_minimax_settings()
    client = _client(settings)
    response = client.chat.completions.create(
        model=str(settings["model"]),
        messages=[
            {"role": "system", "content": TOOL_ROUTER_SYSTEM},
            {
                "role": "user",
                "content": (
                    "【使用者本輪要求】\n"
                    f"{text}\n\n"
                    "【已注入的相關記憶摘要】\n"
                    f"{memory_context or '(無)'}\n\n"
                    "【可委派工具目錄】\n"
                    f"{_tool_catalog_for_router(available_tools)}"
                ),
            },
        ],
        temperature=0,
    )
    raw_text = response.choices[0].message.content or ""
    return normalize_tool_route_decision(_safe_json_loads(raw_text), text)


def tool_route_requires_delegate(decision: dict, threshold: float = 0.55) -> bool:
    return (
        isinstance(decision, dict)
        and decision.get("mode") == "delegate"
        and float(decision.get("confidence") or 0.0) >= threshold
    )


def _compact_result(result: Any, limit: int = 1800) -> str:
    result = redact_secrets(result)
    text = json.dumps(result, ensure_ascii=False, default=str)
    if len(text) <= limit:
        return text
    return text[:limit] + f"... [truncated {len(text) - limit} chars]"


def _preview(value: Any, limit: int = 240) -> str:
    value = redact_secrets(value)
    text = json.dumps(value, ensure_ascii=False, default=str)
    if len(text) <= limit:
        return text
    return text[:limit] + "..."


def _site_query_terms(text: str) -> set[str]:
    text = (text or "").lower()
    terms = {t for t in re.findall(r"[a-z0-9][a-z0-9._-]{1,}", text)}
    institution_terms: set[str] = set()
    for m in re.finditer(
        r"(?:我是|我就讀|使用者是|就讀|來自|在|到)?([\u4e00-\u9fff]{1,8}?(?:高級中學|高中|中學|大學|國中|國小|學院|學校|公司))",
        text,
    ):
        institution_terms.add(m.group(1))
    for m in re.finditer(r"(?:我是|使用者是)?([\u4e00-\u9fff]{2,8}?)(?:學生|生|校友)", text):
        institution_terms.add(m.group(1))
    for term in institution_terms:
        terms.add(term)
        for suffix, short in (("中學", "中"), ("高中", "高"), ("大學", "大"), ("國中", "中"), ("國小", "小")):
            if term.endswith(suffix) and len(term) > len(suffix):
                terms.add(term[0] + short)
    return {t for t in terms if t}


def _specific_site_terms(text: str) -> list[str]:
    terms = []
    generic_fragments = {
        frag
        for item in GENERIC_SITE_SEARCH_TERMS | SITE_SERVICE_TERMS | CONTEXT_NOISE_TERMS
        for frag in ({item} | ({item[i:i + 2] for i in range(max(0, len(item) - 1))} if re.search(r"[\u4e00-\u9fff]", item) else set()))
    }
    for term in _site_query_terms(text):
        if term in generic_fragments:
            continue
        if re.search(r"[\u4e00-\u9fff]", term) and re.search(r"(作業|登入|使用|查詢|幫|帮|請|请|看|要|去|我是|你是|他是|她是|用者)", term):
            continue
        if re.fullmatch(r"(https?|www|com|edu|tw|org|net|site)", term):
            continue
        terms.append(term)
    noise = {"學生", "中學", "高中", "大學", "國中", "國小", "使用", "用者", "登入", "網站", "平台", "課程", "作業"}
    terms = [term for term in terms if term not in noise]
    def rank(item: str) -> tuple[int, int, str]:
        if re.search(r"[a-z0-9-]+(?:\.[a-z0-9-]+)+", item):
            return (0, -len(item), item)
        if re.search(r"[\u4e00-\u9fff]", item):
            return (1, -len(item), item)
        return (2, -len(item), item)
    return sorted(terms, key=rank)


def generic_site_search_guard(tool_name: str, args: dict, task_context: str = "") -> dict | None:
    if tool_name != "web_search":
        return None
    query = str((args or {}).get("query") or "").strip()
    if not query:
        return None
    if not (_site_service_label(query) or re.search(r"(login|portal|登入|教學平台|數位學習|學習平台|網站|平台)", query, re.I)):
        return None
    if _specific_site_terms(query):
        return None

    context_terms = _specific_site_terms(task_context)
    target = (context_terms[0] if context_terms else "").strip()
    service = _site_service_label(query) or _site_service_label(task_context) or "目標服務"
    specific_query = f"{target} {service}".strip() if target else f"目標機構 {service}"
    return {
        "error": f"阻止泛用網站搜尋：{query}",
        "blocked": True,
        "recovery_hint": (
            f"不要用泛用網站/登入查詢；先用 site_memory_search 查已學入口，"
            f"或改用 website_find('{specific_query}') 這種包含機構名稱的查詢。"
        ),
        "suggested_query": specific_query,
    }


def _site_service_label(text: str) -> str:
    lower = (text or "").lower()
    service_priority = ["lms", "moodle", "classroom", "portal", "sso", "login", "課程系統", "學習平台", "數位學習", "教學平台", "後台"]
    for term in service_priority:
        if term.lower() in lower:
            return term
    if re.search(r"登入|login|sign.?in", lower, re.I):
        return "登入入口"
    if re.search(r"網站|平台|portal", lower, re.I):
        return "網站入口"
    return ""


def summarize_tool_result(tool_name: str, result: dict) -> str:
    result = redact_secrets(result)
    if not isinstance(result, dict):
        return _preview(result)
    if "error" in result:
        base = str(result.get("error", ""))[:220]
        hint = str(result.get("recovery_hint", "") or "")
        return base + (f"；建議：{hint[:160]}" if hint else "")
    if tool_name == "list_processes":
        top = result.get("top_processes") or []
        top_text = "、".join(
            f"{p.get('name')} x{p.get('count')}"
            for p in top[:5]
            if isinstance(p, dict) and p.get("name")
        )
        return f"列出 {result.get('process_count', 0)} 個程序" + (f"；常見：{top_text}" if top_text else "")
    if tool_name == "site_memory_search":
        sites = result.get("sites") or []
        failures = result.get("failures") or []
        first = sites[0] if sites and isinstance(sites[0], dict) else {}
        return f"找到 {len(sites)} 個已學入口、{len(failures)} 個失敗紀錄" + (f"；優先：{first.get('service', '')} {first.get('url', '')}" if first else "")
    if tool_name == "site_memory_remember" and result.get("success"):
        site = result.get("site") or {}
        return f"已記住網站入口：{site.get('service', '')} {site.get('url', '')}"
    if tool_name == "site_memory_mark_failure" and result.get("success"):
        failure = result.get("failure") or {}
        return f"已記住失敗網址：{failure.get('url', '')}"
    if tool_name == "website_find":
        best = result.get("best") or {}
        if best:
            return f"已找到並驗證入口：{best.get('title', '')} {best.get('final_url') or best.get('url', '')}"
        return f"沒有找到可驗證入口；搜尋錯誤：{result.get('search_error', '')}"
    if tool_name == "browser_links":
        links = result.get("links") or []
        first = links[0] if links and isinstance(links[0], dict) else {}
        return f"列出 {result.get('count', len(links))} 個背景頁面連結" + (f"；第一個：{first.get('text') or first.get('href', '')}" if first else "")
    if tool_name in {"browser_follow_link", "browser_back", "browser_wait"} and result.get("success"):
        return f"背景瀏覽器：{result.get('title') or result.get('url') or '完成'}"
    if tool_name == "computer_active_window":
        win = result.get("active_window") or {}
        return f"目前前景視窗：{win.get('process', '')} - {win.get('title', '')}".strip(" -")
    if tool_name == "computer_list_windows":
        active = result.get("active_window") or {}
        wins = result.get("windows") or []
        candidates = "；".join(
            f"{w.get('process', '')}:{w.get('title', '')}"[:80]
            for w in wins[:3]
            if isinstance(w, dict)
        )
        return f"找到 {result.get('count', len(wins))} 個可見視窗；目前：{active.get('title', '')}" + (f"；候選：{candidates}" if candidates else "")
    if tool_name == "computer_focus_window":
        win = result.get("active_window") or result.get("selected_window") or {}
        return f"已切換到視窗：{win.get('process', '')} - {win.get('title', '')}".strip(" -")
    if tool_name == "computer_screenshot_window" and result.get("success"):
        win = result.get("window") or {}
        return f"已截取視窗：{win.get('title', '')}；檔案：{result.get('filename') or result.get('path', '')}"
    if tool_name == "computer_screenshot" and result.get("success"):
        win = result.get("active_window") or {}
        scope = "指定區域" if result.get("region") else "全螢幕"
        return f"已截圖（{scope}）：{result.get('filename') or result.get('path', '')}；前景：{win.get('title', '')}"
    if tool_name == "computer_control" and result.get("success") is not None:
        steps = result.get("steps") or []
        shot = result.get("screenshot") or {}
        win = (shot.get("window") or shot.get("active_window") or {}) if isinstance(shot, dict) else {}
        return f"執行 {len(steps)} 個電腦操作步驟；最後視窗：{win.get('title', '') or '未取得'}"
    if tool_name == "calculator":
        return f"計算結果：{result.get('result')}"
    if tool_name == "web_search" and isinstance(result.get("results"), list):
        titles = [str(r.get("title", "")) for r in result["results"][:3] if isinstance(r, dict)]
        return f"搜尋到 {len(result['results'])} 筆：" + "；".join(titles)
    if tool_name == "web_image_search" and isinstance(result.get("results"), list):
        titles = [str(r.get("title", "")) for r in result["results"][:3] if isinstance(r, dict)]
        return f"搜尋到 {len(result['results'])} 張圖片：" + "；".join(titles)
    if tool_name == "download_image" and result.get("success"):
        return f"圖片已下載：{result.get('filename') or result.get('path')}"
    if tool_name == "copy_file" and result.get("success"):
        return f"已複製檔案到：{result.get('path') or result.get('dst') or result.get('filename')}"
    if tool_name == "move_file" and result.get("success"):
        return f"已移動檔案到：{result.get('path') or result.get('dst') or result.get('filename')}"
    if tool_name == "write_file" and result.get("success"):
        return f"已寫入檔案：{result.get('path')}"
    if tool_name == "download_file" and result.get("success"):
        return f"已下載檔案到：{result.get('path')}"
    if tool_name == "make_directory" and result.get("success"):
        return f"已建立資料夾：{result.get('path')}"
    if tool_name == "zip_create" and result.get("success"):
        return f"已建立壓縮檔：{result.get('zip_path')}，加入 {result.get('files_added', 0)} 個檔案"
    if tool_name == "zip_extract" and result.get("success"):
        return f"已解壓縮到：{result.get('dest_dir')}，共 {result.get('count', 0)} 個項目"
    if tool_name == "replace_in_file" and result.get("success"):
        return f"已更新檔案：{result.get('path')}，取代 {result.get('replaced', 0)} 處"
    if tool_name == "weather_get" and isinstance(result.get("main"), dict):
        weather = result.get("weather") or [{}]
        desc = weather[0].get("description", "") if isinstance(weather[0], dict) else ""
        main = result.get("main", {})
        wind = result.get("wind", {})
        place = result.get("query_used") or result.get("name") or ""
        return (
            f"{place}：{desc}，氣溫 {main.get('temp')}°C，"
            f"體感 {main.get('feels_like')}°C，濕度 {main.get('humidity')}%，"
            f"風速 {wind.get('speed')} m/s"
        )
    if tool_name == "gmail_send" and result.get("success"):
        return f"郵件已送出，id={result.get('id', '')}"
    if tool_name == "gmail_read":
        if result.get("found") is False:
            return f"沒有找到郵件：{result.get('query', '')}"
        snippet = result.get("snippet", "")
        subject = result.get("subject", "(無主旨)")
        sender = result.get("from", "")
        return f"找到郵件：{subject}；寄件者：{sender}；內容摘要：{snippet}"
    if tool_name == "gmail_list" and isinstance(result.get("messages"), list):
        if not result["messages"]:
            return "收件匣沒有列出郵件"
        first = result["messages"][0]
        return f"列出 {len(result['messages'])} 封郵件；最新：{first.get('subject', '(無主旨)')}，來自 {first.get('from', '')}"
    if tool_name.startswith("gmail_") and ("messages" in result or "snippet" in result):
        return _preview({k: result.get(k) for k in ("messages", "snippet", "id", "subject", "from")})
    if tool_name.startswith("calendar_") and ("events" in result or result.get("success")):
        return _preview({k: result.get(k) for k in ("success", "events", "htmlLink", "id")})
    if tool_name.startswith("drive_") and ("files" in result or "content" in result):
        return _preview({k: result.get(k) for k in ("files", "name", "content")})
    for key in ("result", "summary", "message", "value", "status"):
        if key in result:
            return str(result[key])[:300]
    return _preview(result)


def progress_snapshot_from_events(tool_events: list[dict], strategy_events: list[str] | None = None) -> dict:
    events = [event for event in (tool_events or []) if isinstance(event, dict)]
    successes = [event for event in events if event.get("success")]
    failures = [event for event in events if not event.get("success")]
    last = events[-1] if events else {}
    last_tool = str(last.get("tool") or "")
    last_preview = str(last.get("result_preview") or last.get("error") or "")

    phase = "尚未開始"
    if last_tool:
        if last_tool in {"site_memory_search", "website_find", "web_search", "dns_lookup"}:
            phase = "尋找或驗證入口"
        elif last_tool.startswith("browser_"):
            phase = "背景瀏覽器操作"
        elif last_tool.startswith("computer_"):
            phase = "桌面視窗操作"
        elif last_tool.startswith("gmail_"):
            phase = "郵件處理"
        elif last_tool.startswith("drive_"):
            phase = "雲端檔案處理"
        elif last_tool.startswith("calendar_"):
            phase = "行事曆處理"
        else:
            phase = "工具執行"

    if failures:
        next_focus = "先處理最近失敗或改換策略"
    elif events:
        next_focus = "根據目前頁面或結果繼續下一個最小步驟"
    else:
        next_focus = "開始第一個可驗證步驟"

    summary = (
        f"進度：已執行 {len(events)} 次工具，成功 {len(successes)}、失敗 {len(failures)}；"
        f"目前階段：{phase}"
    )
    if last_preview:
        summary += f"；最近：{last_preview[:120]}"

    return {
        "tool_count": len(events),
        "success_count": len(successes),
        "failure_count": len(failures),
        "current_phase": phase,
        "last_tool": last_tool,
        "last_result": last_preview[:240],
        "next_focus": next_focus,
        "strategy_count": len(strategy_events or []),
        "summary": summary,
    }


def browser_state_signature(tool_name: str, result: dict) -> str:
    if not str(tool_name or "").startswith("browser_") or not isinstance(result, dict):
        return ""
    url = str(result.get("url") or "")
    title = str(result.get("title") or "")
    if not url and not title:
        return ""
    text = re.sub(r"\s+", " ", str(result.get("text") or ""))[:1200]
    controls = result.get("controls") or []
    controls_count = len(controls) if isinstance(controls, list) else 0
    parsed = urllib.parse.urlsplit(url)
    normalized_url = urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, parsed.path, parsed.query, ""))
    digest = hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()[:12]
    return f"{normalized_url}|{title}|{digest}|controls={controls_count}|needs_user={bool(result.get('needs_user_action'))}"


def browser_stagnation_guardrail(tool_events: list[dict], threshold: int = 3) -> str:
    browser_events = [
        event for event in (tool_events or [])
        if isinstance(event, dict)
        and str(event.get("tool") or "").startswith("browser_")
        and event.get("browser_state_signature")
    ]
    if len(browser_events) < threshold:
        return ""
    recent = browser_events[-threshold:]
    signatures = {event.get("browser_state_signature") for event in recent}
    if len(signatures) != 1:
        return ""
    tools = " → ".join(str(event.get("tool") or "") for event in recent)
    last = recent[-1]
    return (
        "【背景瀏覽器停滯偵測】\n"
        f"- 背景瀏覽器頁面狀態連續 {threshold} 次沒有變化。\n"
        f"- 最近工具路徑：{tools}。\n"
        f"- 目前頁面：{last.get('title', '')} {last.get('args', {}).get('url', '')}。\n"
        "- 下一步不要繼續在同一頁盲目點擊或按鍵。\n"
        "- 請改用 browser_links/browser_get_page 重新判斷可操作目標；必要時回傳 browser_screenshot 請使用者確認，或明確回報需要使用者處理。"
    )


def next_step_guardrail(tool_name: str, args: dict, result: dict, repeat_count: int = 1, failure_repeat_count: int = 0) -> str:
    """Build a plain-language correction message after risky or failed tool calls."""
    if not isinstance(result, dict):
        return ""
    has_error = "error" in result
    hint = str(result.get("recovery_hint", "") or "")
    repeated = bool(result.get("repeated_call")) or repeat_count >= 2 or failure_repeat_count >= 2
    if not (has_error or hint or repeated):
        return ""

    args_preview = json.dumps(summarize_args(args), ensure_ascii=False, default=str)
    lines = [
        "【下一步策略約束】",
        f"- 剛才工具：{tool_name}",
        f"- 已用參數：{args_preview}",
        f"- 結果摘要：{summarize_tool_result(tool_name, result)}",
        "- 下一步不可用同一工具與同一組參數原地重試。",
    ]
    if hint:
        lines.append(f"- 修復方向：{hint}")
    if repeat_count >= 2:
        lines.append(f"- 這組參數已嘗試 {repeat_count} 次；必須換入口、換查詢、換工具，或回報明確卡住原因。")
    if failure_repeat_count >= 2:
        lines.append(f"- 同一失敗狀態已出現 {failure_repeat_count} 次；必須停止這條路徑。")
    if tool_name.startswith("browser_"):
        lines.append("- 優先留在背景瀏覽器工作；不要因背景瀏覽器失敗就改操作使用者前景視窗。")
    return "\n".join(lines)


def round_budget_guardrail(turn: int, max_rounds: int, tool_events: list[dict]) -> str:
    remaining = max(0, int(max_rounds) - int(turn))
    failed = [event for event in tool_events if isinstance(event, dict) and not event.get("success", False)]
    recent = tool_events[-5:]
    recent_text = "；".join(
        f"{event.get('tool', '')}:{event.get('result_preview', '')}"[:120]
        for event in recent
        if isinstance(event, dict)
    )
    return (
        "【工具輪數收斂】\n"
        f"- 已用工具輪數：{turn}/{max_rounds}，剩餘：{remaining}。\n"
        f"- 已執行工具：{len(tool_events)} 次；失敗：{len(failed)} 次。\n"
        f"- 最近進度：{recent_text or '尚無工具結果'}。\n"
        "- 下一步不要再開新探索分支，不要重複查同一件事。\n"
        "- 請立刻整理目前成果：已完成哪些、卡在哪裡、下一個最小可行步驟是什麼、是否需要使用者處理。\n"
        "- 若尚未完成任務，狀態請回報「部分成功」或「失敗」，不要假裝成功。"
    )


def _domain(url: str) -> str:
    try:
        return urllib.parse.urlsplit(str(url or "")).netloc
    except Exception:
        return ""


def _canonical_args(args: dict) -> str:
    safe = redact_secrets(args or {})
    try:
        return json.dumps(safe, ensure_ascii=False, sort_keys=True, default=str)
    except Exception:
        return str(safe)


def _tool_signature(tool_name: str, args: dict) -> str:
    return f"{tool_name}:{_canonical_args(args)}"


def _failure_signature(tool_name: str, args: dict, result: dict) -> str:
    if not isinstance(result, dict) or "error" not in result:
        return ""
    domain = _domain(args.get("url", "") if isinstance(args, dict) else "")
    target = domain or str((args or {}).get("query") or (args or {}).get("host") or (args or {}).get("target") or "")
    error = re.sub(r"\s+", " ", str(result.get("error", "")))[:160]
    return f"{tool_name}:{target}:{error}"


def repeated_tool_guard(tool_name: str, args: dict, attempt_counts: dict[str, int]) -> tuple[int, dict | None]:
    """Track repeated tool calls and stop deterministic retry loops."""
    signature = _tool_signature(tool_name, args)
    attempt_counts[signature] = attempt_counts.get(signature, 0) + 1
    repeat_count = attempt_counts[signature]
    if tool_name in REPEAT_GUARDED_TOOLS and repeat_count >= 3:
        result = {
            "error": f"偵測到 {tool_name} 用同一組參數重複 {repeat_count} 次，已停止原地重試。",
            "repeated_call": True,
        }
        result["recovery_hint"] = recovery_hint_for_tool(tool_name, args, result, repeat_count)
        return repeat_count, result
    return repeat_count, None


def recovery_hint_for_tool(tool_name: str, args: dict, result: dict, repeat_count: int = 1) -> str:
    if not isinstance(result, dict):
        return ""
    error = str(result.get("error", "") or "")
    message = str(result.get("message", "") or "")
    url = str((args or {}).get("url") or result.get("url") or "")
    query = str((args or {}).get("query") or "")
    text = f"{error}\n{message}"

    if result.get("repeated_call"):
        return "已偵測到同一工具同一參數重複呼叫；停止原地重試，改查已學入口、換工具，或回報具體卡住原因。"
    if tool_name in {"browser_open", "browser_login"} and re.search(r"ERR_NAME_NOT_RESOLVED|getaddrinfo|DNS", text, re.I):
        return f"不要再重試 {url}；先用 site_memory_mark_failure 記錄失敗，再用 site_memory_search/website_find 找同服務的已驗證入口。"
    if tool_name == "web_search" and query and repeat_count >= 2:
        return "同一查詢已重複；改用更具體的機構名稱、網域限制或 site_memory_search，不要再次用泛用查詢。"
    if tool_name == "website_find" and "best" not in result and repeat_count >= 2:
        return "網站解析沒有找到入口；改用已知機構網域、縮小查詢詞，或回報需要使用者提供入口。"
    if "未知工具" in error:
        return "工具不存在；改用可用工具清單中的相近工具，不能用不存在的工具名繼續嘗試。"
    if "找不到密碼欄位" in message or "找不到密碼欄位" in error:
        return "登入頁可能需要先點 Google/SSO/登入連結；先用 browser_get_page 檢查 controls，再點正確入口，不要改操作前景視窗。"
    return ""


def learning_events_from_tool(tool_name: str, args: dict, result: dict) -> list[dict]:
    if not isinstance(result, dict):
        return []
    args = redact_secrets(args or {})
    result = redact_secrets(result)
    events: list[dict] = []

    if tool_name == "website_find":
        best = result.get("best") or {}
        if best:
            url = best.get("final_url") or best.get("url") or ""
            title = best.get("title") or ""
            query = args.get("query") or ""
            events.append({
                "category": "technical",
                "importance": 5,
                "memory": f"網站入口經驗：查詢「{query}」時，已驗證可用入口為 {url}（標題：{title}）。下次同類任務應優先使用此入口。",
            })
        for item in (result.get("failed_candidates") or [])[:3]:
            url = item.get("url") or ""
            error = item.get("error") or ""
            if url:
                events.append({
                    "category": "technical",
                    "importance": 4,
                    "memory": f"網站入口失敗經驗：{url} 在查詢「{args.get('query') or ''}」時驗證失敗（{error}），下次除非使用者指定否則避免重試。",
                })

    if tool_name == "site_memory_remember" and result.get("success"):
        site = result.get("site") or {}
        if site.get("url"):
            events.append({
                "category": "technical",
                "importance": 5,
                "memory": f"網站入口經驗：{site.get('service', '網站')} 的可用入口是 {site.get('url')}（標題：{site.get('title', '')}）。",
            })

    if tool_name == "site_memory_mark_failure" and result.get("success"):
        failure = result.get("failure") or {}
        if failure.get("url"):
            events.append({
                "category": "technical",
                "importance": 4,
                "memory": f"網站入口失敗經驗：{failure.get('service', '網站')} 的 {failure.get('url')} 失敗（{failure.get('error', '') or failure.get('note', '')}），下次應避開。",
            })

    if tool_name == "browser_login":
        url = result.get("url") or args.get("url") or ""
        title = result.get("title") or _domain(url)
        if result.get("logged_in"):
            events.append({
                "category": "project",
                "importance": 5,
                "memory": f"登入操作經驗：已能透過背景瀏覽器登入 {title}（{url}）。下次同服務登入應優先使用背景瀏覽器與已驗證入口。",
            })
        elif url:
            reason = result.get("error") or result.get("message") or "未確認登入成功"
            events.append({
                "category": "technical",
                "importance": 4,
                "memory": f"登入操作未完成經驗：背景瀏覽器登入 {title}（{url}）未成功（{reason}），下次需先確認入口、SSO 流程或是否需要使用者驗證。",
            })

    if tool_name == "browser_open":
        url = result.get("url") or args.get("url") or ""
        if "error" in result and url:
            events.append({
                "category": "technical",
                "importance": 4,
                "memory": f"網站入口失敗經驗：背景瀏覽器開啟 {url} 失敗（{result.get('error', '')}），下次應先查已學入口或重新驗證網址。",
            })
        elif result.get("success") and url and urllib.parse.urlsplit(str(url)).scheme in {"http", "https"} and urllib.parse.urlsplit(str(url)).netloc:
            events.append({
                "category": "technical",
                "importance": 3,
                "memory": f"網站入口經驗：背景瀏覽器可開啟 {result.get('title', _domain(url))}（{url}）。",
            })

    return events


def collect_file_refs(value: Any, out: list[dict] | None = None) -> list[dict]:
    if out is None:
        out = []
    if value is None or len(out) >= 12:
        return out
    if isinstance(value, list):
        for item in value:
            collect_file_refs(item, out)
        return out
    if isinstance(value, dict):
        url = value.get("file_url") or value.get("url")
        path = value.get("path") or value.get("dest") or value.get("dest_path")
        name = value.get("filename") or value.get("name") or (Path(str(path)).name if path else "檔案")
        if isinstance(url, str) and (url.startswith("/files/") or url.startswith("http")):
            out.append({
                "file_url": url,
                "filename": str(name or "檔案"),
                "mime": value.get("mime") or "",
                "path": str(path or ""),
            })
        for key, item in value.items():
            if key not in {"file_url", "url", "path", "dest", "dest_path", "filename", "name", "mime"}:
                collect_file_refs(item, out)
    return out


def summarize_args(args: dict, limit: int = 120) -> dict:
    compact = {}
    for key, value in (args or {}).items():
        if re.search(r"(?i)(密碼|password|passwd|pwd|passcode|token|secret|client_secret)", str(key)):
            text = "********"
        else:
            text = str(redact_secrets(value))
        compact[key] = text if len(text) <= limit else text[:limit] + "..."
    return compact


def _gmail_send_allowed(original_user_message: str, args: dict) -> tuple[bool, str]:
    text = original_user_message or ""
    if not EXPLICIT_EMAIL_RE.search(text):
        return False, "原始使用者沒有明確要求寄信或發電子郵件；「傳給我/給我」只能在 WebUI 提供結果。"
    to = str((args or {}).get("to", "") or "")
    if not (EMAIL_ADDRESS_RE.search(to) or EMAIL_ADDRESS_RE.search(text)):
        return False, "原始使用者沒有明確指定收件信箱，禁止從記憶或舊任務推斷收件人。"
    return True, ""


class MinimaxToolAgent:
    def __init__(
        self,
        *,
        memory_search: Callable[[str, int], dict] | None = None,
        on_tool_event: Callable[[str, dict], None] | None = None,
        disabled_tool_prefixes: list[str] | None = None,
    ):
        self.memory_search = memory_search
        self.on_tool_event = on_tool_event
        self._original_user_message = ""
        self._task_context = ""
        self.disabled_tool_prefixes = tuple(disabled_tool_prefixes or [])

    @property
    def tool_count(self) -> int:
        return len(self._available_tools())

    def _available_tools(self) -> list[dict]:
        if not self.disabled_tool_prefixes:
            return TOOL_AGENT_TOOLS
        out = []
        for schema in TOOL_AGENT_TOOLS:
            name = schema.get("function", {}).get("name", "")
            if any(name.startswith(prefix) for prefix in self.disabled_tool_prefixes):
                continue
            out.append(schema)
        return out

    def _emit(self, event: str, payload: dict) -> None:
        if self.on_tool_event:
            self.on_tool_event(event, payload)

    def _execute_tool(self, tool_name: str, args: dict) -> dict:
        guard = generic_site_search_guard(tool_name, args, self._task_context)
        if guard:
            return {
                **guard,
                "_meta": {"ok": False, "tool": tool_name, "duration_ms": 0},
            }
        if tool_name == "gmail_send":
            ok, reason = _gmail_send_allowed(self._original_user_message, args)
            if not ok:
                return {
                    "error": f"安全閘阻止寄信：{reason}",
                    "blocked": True,
                    "_meta": {"ok": False, "tool": tool_name, "duration_ms": 0},
                }
        if any(tool_name.startswith(prefix) for prefix in self.disabled_tool_prefixes):
            return {
                "error": f"工具 {tool_name} 目前已由 WebUI 關閉",
                "blocked": True,
                "_meta": {"ok": False, "tool": tool_name, "duration_ms": 0},
            }
        if tool_name == "search_memories":
            if not self.memory_search:
                return {"error": "Minimax 未取得記憶搜尋通道"}
            return self.memory_search(args.get("query", ""), int(args.get("limit", 5)))
        return registry.dispatch_with_meta(tool_name, args)

    def _finalize_without_tools(self, client: OpenAI, messages: list[dict], *, settings: dict, turn: int, tool_events: list[dict], learning_events: list[dict], strategy_events: list[str], started_at: float, stopped_for_budget: bool = False) -> dict:
        response = client.chat.completions.create(
            model=str(settings["model"]),
            messages=messages,
            temperature=float(settings["temperature"]),
        )
        msg = response.choices[0].message
        progress_snapshot = progress_snapshot_from_events(tool_events, strategy_events)
        return {
            "ok": True,
            "model": settings["model"],
            "tool_count": len(self._available_tools()),
            "turns": turn,
            "answer": (msg.content or "").strip(),
            "tool_calls": tool_events,
            "learning_events": learning_events,
            "strategy_events": strategy_events,
            "progress_snapshot": progress_snapshot,
            "stopped_for_budget": stopped_for_budget,
            "duration_ms": round((time.perf_counter() - started_at) * 1000, 2),
        }

    def run(
        self,
        *,
        task: str,
        memory_context: str = "",
        user_id: str = "default",
        original_user_message: str = "",
    ) -> dict:
        started_at = time.perf_counter()
        self._original_user_message = original_user_message or ""
        self._task_context = "\n".join(
            part for part in (original_user_message or "", task or "", memory_context or "") if part
        )
        messages: list[dict] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"【目前記憶帳號】{user_id}\n"
                    f"【原始使用者訊息】\n{original_user_message or '(未提供)'}\n\n"
                    f"【Gemini 委派任務】\n{task}\n\n"
                    f"【Gemini 提供的相關記憶】\n{memory_context or '(無)'}\n\n"
                    "請完成任務。最後只回報任務結果，不要輸出表格、markdown 報告或工具原始 JSON。"
                ),
            },
        ]
        tool_events: list[dict] = []
        learning_events: list[dict] = []
        strategy_events: list[str] = []
        learning_seen: set[str] = set()
        attempt_counts: dict[str, int] = {}
        failure_counts: dict[str, int] = {}
        settings = get_minimax_settings()
        max_tool_rounds = int(settings["max_tool_rounds"])
        client = _client(settings)
        available_tools = self._available_tools()

        for turn in range(1, max_tool_rounds + 1):
            response = client.chat.completions.create(
                model=str(settings["model"]),
                messages=messages,
                tools=available_tools,
                tool_choice="auto",
                temperature=float(settings["temperature"]),
            )
            msg = response.choices[0].message
            messages.append(msg.model_dump(exclude_unset=True))

            if not msg.tool_calls:
                progress_snapshot = progress_snapshot_from_events(tool_events, strategy_events)
                return {
                    "ok": True,
                    "model": settings["model"],
                    "tool_count": len(available_tools),
                    "turns": turn,
                    "answer": (msg.content or "").strip(),
                    "tool_calls": tool_events,
                    "learning_events": learning_events,
                    "strategy_events": strategy_events,
                    "progress_snapshot": progress_snapshot,
                    "duration_ms": round((time.perf_counter() - started_at) * 1000, 2),
                }

            turn_guardrails: list[str] = []
            for tc in msg.tool_calls:
                tool_name = tc.function.name
                args = _safe_json_loads(tc.function.arguments)
                event_base = {
                    "tool_call_id": tc.id,
                    "tool": tool_name,
                    "args": summarize_args(args),
                    "agent": "minimax",
                }
                self._emit("tool_start", event_base)

                tool_started = time.perf_counter()
                stuck_reason = ""
                repeat_count, guarded_result = repeated_tool_guard(tool_name, args, attempt_counts)
                if guarded_result is not None:
                    result = guarded_result
                    stuck_reason = result["error"]
                else:
                    result = self._execute_tool(tool_name, args)
                    hint = recovery_hint_for_tool(tool_name, args, result, repeat_count)
                    if hint and isinstance(result, dict) and "recovery_hint" not in result:
                        result = {**result, "recovery_hint": hint}

                failure_sig = _failure_signature(tool_name, args, result)
                failure_repeat_count = 0
                if failure_sig:
                    failure_counts[failure_sig] = failure_counts.get(failure_sig, 0) + 1
                    failure_repeat_count = failure_counts[failure_sig]
                    if failure_counts[failure_sig] >= 3:
                        stuck_reason = (
                            f"同一失敗狀態已出現 {failure_counts[failure_sig]} 次，"
                            "工具代理判定目前策略卡住。"
                        )
                for learning in learning_events_from_tool(tool_name, args, result):
                    memory = str(learning.get("memory") or "")
                    if memory and memory not in learning_seen:
                        learning_seen.add(memory)
                        learning_events.append(learning)
                duration_ms = result.get("_meta", {}).get(
                    "duration_ms",
                    round((time.perf_counter() - tool_started) * 1000, 2),
                )
                success = "error" not in result
                event = {
                    **event_base,
                    "success": success,
                    "duration_ms": duration_ms,
                    "result_preview": summarize_tool_result(tool_name, result),
                    "files": collect_file_refs(result),
                    "error": result.get("error", "") if isinstance(result, dict) else "",
                }
                state_signature = browser_state_signature(tool_name, result)
                if state_signature:
                    event["browser_state_signature"] = state_signature
                tool_events.append(event)
                event["progress_snapshot"] = progress_snapshot_from_events(tool_events, strategy_events)
                self._emit("tool_done", event)

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": _compact_result(result),
                })
                guardrail = next_step_guardrail(tool_name, args, result, repeat_count, failure_repeat_count)
                if guardrail:
                    strategy_events.append(guardrail)
                    turn_guardrails.append(guardrail)
                stagnation_guardrail = browser_stagnation_guardrail(tool_events)
                if stagnation_guardrail and (not strategy_events or strategy_events[-1] != stagnation_guardrail):
                    strategy_events.append(stagnation_guardrail)
                    turn_guardrails.append(stagnation_guardrail)

                if stuck_reason:
                    progress_snapshot = progress_snapshot_from_events(tool_events, strategy_events)
                    return {
                        "ok": False,
                        "model": settings["model"],
                        "tool_count": len(available_tools),
                        "turns": turn,
                        "error": stuck_reason,
                        "stuck_detected": True,
                        "recovery_hint": result.get("recovery_hint", "") if isinstance(result, dict) else "",
                        "tool_calls": tool_events,
                        "learning_events": learning_events,
                        "strategy_events": strategy_events,
                        "progress_snapshot": progress_snapshot,
                        "duration_ms": round((time.perf_counter() - started_at) * 1000, 2),
                    }

            if turn_guardrails:
                messages.append({
                    "role": "user",
                    "content": "\n\n".join(turn_guardrails[-3:]),
                })
            remaining_after_turn = max_tool_rounds - turn
            if remaining_after_turn <= 1:
                guardrail = round_budget_guardrail(turn, max_tool_rounds, tool_events)
                strategy_events.append(guardrail)
                messages.append({
                    "role": "user",
                    "content": guardrail,
                })
                return self._finalize_without_tools(
                    client,
                    messages,
                    settings=settings,
                    turn=turn,
                    tool_events=tool_events,
                    learning_events=learning_events,
                    strategy_events=strategy_events,
                    started_at=started_at,
                    stopped_for_budget=True,
                )

        guardrail = round_budget_guardrail(max_tool_rounds, max_tool_rounds, tool_events)
        strategy_events.append(guardrail)
        progress_snapshot = progress_snapshot_from_events(tool_events, strategy_events)
        return {
            "ok": False,
            "model": settings["model"],
            "tool_count": len(available_tools),
            "error": f"工具代理已用完 {max_tool_rounds} 輪，已整理可用進度而不是繼續重試。",
            "recovery_hint": "請縮小任務、使用已驗證入口，或讓代理從最近一次成功頁面繼續。",
            "tool_calls": tool_events,
            "learning_events": learning_events,
            "strategy_events": strategy_events,
            "progress_snapshot": progress_snapshot,
            "stopped_for_budget": True,
            "duration_ms": round((time.perf_counter() - started_at) * 1000, 2),
        }
