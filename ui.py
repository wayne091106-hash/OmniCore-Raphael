"""
ui.py — FastAPI Web 介面 + 系統入口
════════════════════════════════════
從 gemini_live_ui.py 拆出 FastAPI，讀 UI/index.html。

責任：
  1. 提供 HTTP 靜態檔（UI/index.html + UI/app.js）
  2. WebSocket /ws 端點：bridge ↔ browser 雙向橋接
  3. 系統入口：啟動 Bridge → Core → (可選) Perception

WebSocket 協議（與 app.js 一致）：
  後端 → 前端：{"channel": "transcript_out", "payload": {"text": "...", "done": false}}
  前端 → 後端：{"channel": "text_in", "payload": {"text": "..."}}

啟動：
  python ui.py                        # 純 Web 模式（瀏覽器提供音訊/鏡頭）
  python ui.py --perception           # 啟動本機感知模組（VAD + Vision）
"""

import argparse
import asyncio
import base64
import datetime
import json
import logging
import math
import mimetypes
import os
import re
import sys
import uuid
from contextlib import asynccontextmanager, suppress
from pathlib import Path


def _handle_dependency_cli() -> None:
    if "--install-deps" not in sys.argv and "--check-deps" not in sys.argv:
        return
    from dependency_bootstrap import install, print_status

    install_mode = "--install-deps" in sys.argv
    flag = "--install-deps" if install_mode else "--check-deps"
    idx = sys.argv.index(flag)
    profile = sys.argv[idx + 1] if idx + 1 < len(sys.argv) and not sys.argv[idx + 1].startswith("--") else "core"
    include_torch = "--with-torch" in sys.argv

    if install_mode:
        raise SystemExit(install(profile, include_torch=include_torch))
    print_status(profile, include_torch=include_torch)
    raise SystemExit(0)


_handle_dependency_cli()

from dotenv import load_dotenv

load_dotenv()

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from bridge import Bridge, Channel
from core import RaphaelCore
from tools.function_call.agent import get_minimax_settings, update_minimax_settings

log = logging.getLogger("ui")

# ── 路徑 ─────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
UI_DIR = BASE_DIR / "UI"
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
OUTPUT_DIR = DATA_DIR / "outputs"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── 不轉發給瀏覽器的內部頻道 ─────────────────────────────────────────────────
_INTERNAL_CHANNELS = {Channel.AUDIO_IN, Channel.VIDEO_IN, Channel.TEXT_IN}

# ── 全域狀態（lifespan 管理）─────────────────────────────────────────────────
_bridge: Bridge | None = None
_core: RaphaelCore | None = None
_core_lock: asyncio.Lock | None = None
_vad_mod = None
_vision_mod = None
_browser_vision_mod = None
_use_perception = False
_core_config = {
    "voice": os.environ.get("RAPHAEL_VOICE", "Puck"),
    "thinking": os.environ.get("RAPHAEL_THINKING", "") or None,
}


def _json_safe(value, depth: int = 0):
    """Convert bridge payloads to values the browser can always JSON.parse."""
    if depth > 20:
        return str(value)
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (bytes, bytearray, memoryview)):
        return base64.b64encode(bytes(value)).decode("ascii")
    if isinstance(value, (datetime.datetime, datetime.date, datetime.time)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, uuid.UUID):
        return str(value)
    if isinstance(value, dict):
        return {str(_json_safe(k, depth + 1)): _json_safe(v, depth + 1) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v, depth + 1) for v in value]

    item = getattr(value, "item", None)
    if callable(item):
        try:
            return _json_safe(item(), depth + 1)
        except Exception:
            pass

    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        try:
            return _json_safe(tolist(), depth + 1)
        except Exception:
            pass

    return str(value)


def _normalize_thinking(value):
    if value is None:
        return None
    text = str(value).strip().lower()
    return {
        "": None,
        "off": None,
        "none": None,
        "low": "low",
        "med": "medium",
        "medium": "medium",
        "high": "high",
        "minimal": "minimal",
    }.get(text, text)


async def _restart_core_if_needed(config: dict) -> None:
    """連線前設定改變時，重建 Gemini session 讓 voice/thinking 真的套用。"""
    global _core, _core_config
    if _bridge is None:
        return

    next_config = {
        "voice": config.get("voice") or _core_config["voice"],
        "thinking": _normalize_thinking(config.get("thinking")),
    }
    if next_config == _core_config and _core is not None:
        return

    if _core_lock is None:
        return

    async with _core_lock:
        if next_config == _core_config and _core is not None:
            return
        if _core is not None:
            await _core.stop()
            _core = None
        _core_config = next_config
        _core = RaphaelCore(
            _bridge,
            voice=_core_config["voice"],
            thinking=_core_config["thinking"],
        )
        await _core.start()


async def _publish_memory_accounts() -> None:
    if _bridge is not None and _core is not None:
        await _bridge.publish(Channel.MEMORY_ACCOUNTS, _core.memory.list_accounts())


def _apply_perception_config(config: dict) -> None:
    vision_cfg = {}
    if isinstance(config.get("vision_settings"), dict):
        vision_cfg.update(config["vision_settings"])
    features = config.get("features")
    if isinstance(features, dict):
        if "vision_gate" in features:
            vision_cfg["enable_gate"] = bool(features["vision_gate"])
        if "vision_overlay" in features:
            vision_cfg["render_feedback"] = bool(features["vision_overlay"])
        if "vision_proactive" in features:
            vision_cfg["emit_proactive"] = bool(features["vision_proactive"])
    for key in ("drift_threshold", "semantic_fps", "target_fps", "capture_fps"):
        if key in config:
            vision_cfg[key] = config[key]
    if _vision_mod is not None and vision_cfg and hasattr(_vision_mod, "update_config"):
        _vision_mod.update_config(vision_cfg)
    if _browser_vision_mod is not None and vision_cfg and hasattr(_browser_vision_mod, "update_config"):
        _browser_vision_mod.update_config(vision_cfg)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """應用程式生命週期：啟動 bridge + core + (可選) perception"""
    global _bridge, _core, _core_lock, _vad_mod, _vision_mod, _browser_vision_mod, _use_perception

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    _bridge = Bridge()
    _core_lock = asyncio.Lock()

    # 解析啟動參數
    use_perception = "--perception" in sys.argv
    _use_perception = use_perception

    # ── 啟動 Core ─────────────────────────────────────────────────────
    _core = RaphaelCore(
        _bridge,
        voice=_core_config["voice"],
        thinking=_core_config["thinking"],
    )
    await _core.start()
    await _publish_memory_accounts()

    # ── 可選：啟動本機感知模組 ────────────────────────────────────────
    _vad_mod = None
    _vision_mod = None
    _browser_vision_mod = None

    if use_perception:
        try:
            from perception.vad import VadModule
            _vad_mod = VadModule(_bridge)
            await _vad_mod.start()
            log.info("本機 VAD 已啟動")
        except Exception as e:
            log.warning("VAD 啟動失敗（跳過）: %s", e)
            _vad_mod = None

        try:
            from perception.vision import VisionModule, GateConfig
            _vision_mod = VisionModule(_bridge, GateConfig())
            await _vision_mod.start()
            log.info("本機 Vision 已啟動")
        except Exception as e:
            log.warning("Vision 啟動失敗（跳過）: %s", e)
            _vision_mod = None

    try:
        from perception.vision import BrowserVisionAnalyzer, GateConfig
        browser_vision_cfg = GateConfig()
        _browser_vision_mod = BrowserVisionAnalyzer(_bridge, browser_vision_cfg)
        await _browser_vision_mod.start()
    except Exception as e:
        log.warning("Browser Vision Gate 啟動失敗（Web 鏡頭將不顯示語意/光流框）: %s", e)
        _browser_vision_mod = None

    log.info("系統就緒 — http://localhost:8765")

    try:
        yield
    finally:
        # ── 關閉 ──────────────────────────────────────────────────────
        if _vad_mod:
            with suppress(Exception):
                _vad_mod.stop()
        if _vision_mod:
            with suppress(Exception):
                _vision_mod.stop()
        if _browser_vision_mod:
            with suppress(Exception):
                _browser_vision_mod.stop()
        if _core:
            await _core.stop()
        log.info("系統已關閉")


app = FastAPI(lifespan=lifespan)
app.mount("/files", StaticFiles(directory=DATA_DIR), name="files")


def _safe_upload_name(name: str) -> str:
    raw = Path(name or "upload.bin").name
    safe = re.sub(r"[^\w._()\-一-龥]+", "_", raw, flags=re.UNICODE)
    return safe[:120] or "upload.bin"


def _save_uploaded_file(payload: dict) -> dict:
    name = _safe_upload_name(str(payload.get("name") or "upload.bin"))
    b64 = str(payload.get("data_b64") or "")
    if "," in b64 and b64.startswith("data:"):
        b64 = b64.split(",", 1)[1]
    raw = base64.b64decode(b64)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    file_id = f"{stamp}_{uuid.uuid4().hex[:8]}_{name}"
    path = UPLOAD_DIR / file_id
    path.write_bytes(raw)
    mime = payload.get("mime") or mimetypes.guess_type(name)[0] or "application/octet-stream"
    rel = path.relative_to(DATA_DIR)
    return {
        "name": name,
        "path": str(path.resolve()),
        "file_url": "/files/" + "/".join(rel.parts),
        "mime": mime,
        "size": len(raw),
    }


# ══════════════════════════════════════════════════════════════════════════════
# HTTP：提供前端靜態檔
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/")
async def index():
    return FileResponse(UI_DIR / "index.html", headers={"Cache-Control": "no-store"})


@app.get("/app.js")
async def app_js():
    return FileResponse(UI_DIR / "app.js", headers={"Cache-Control": "no-store"})


@app.get("/runtime")
async def runtime_info():
    return {
        "perception": bool(_use_perception),
        "local_vision": _vision_mod is not None,
        "local_vad": _vad_mod is not None,
        "browser_vision": _browser_vision_mod is not None,
        "minimax_settings": get_minimax_settings(),
    }


# ══════════════════════════════════════════════════════════════════════════════
# WebSocket：bridge ↔ browser 雙向橋接
# ══════════════════════════════════════════════════════════════════════════════

@app.websocket("/ws")
async def ws_endpoint(client: WebSocket):
    await client.accept()

    if _bridge is None:
        await client.close(code=1011, reason="系統未就緒")
        return

    # 訂閱 bridge（排除內部頻道）
    ui_channels = {ch for ch in Channel if ch not in _INTERNAL_CHANNELS}
    sub = _bridge.subscribe(ui_channels)

    async def bridge_to_browser():
        """bridge 訊息 → 瀏覽器"""
        try:
            async for channel, payload in sub:
                try:
                    msg = _json_safe({"channel": channel.value, "payload": payload})
                    await client.send_text(json.dumps(msg, ensure_ascii=False, allow_nan=False))
                except WebSocketDisconnect:
                    break
                except Exception as e:
                    log.warning("WebSocket 丟棄無法轉發的 %s 訊息: %s", channel.value, e)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            log.warning("WebSocket bridge_to_browser 中斷: %s", e)

    async def browser_to_bridge():
        """瀏覽器訊息 → bridge"""
        try:
            while True:
                data = await client.receive_json()
                ch_str = data.get("channel", "")
                payload = data.get("payload")

                if ch_str == "text_in":
                    text = payload if isinstance(payload, str) else payload.get("text", "")
                    if text:
                        await _bridge.publish(Channel.TEXT_IN, {"text": text})

                elif ch_str == "file_upload":
                    file_info = _save_uploaded_file(payload if isinstance(payload, dict) else {})
                    note = (
                        f"使用者上傳檔案：{file_info['name']}\n"
                        f"本機路徑：{file_info['path']}\n"
                        f"下載/預覽：{file_info['file_url']}\n"
                        f"MIME：{file_info['mime']}，大小：{file_info['size']} bytes\n"
                        "如果需要處理這個檔案，請委派工具代理使用對應檔案工具讀取或分析。"
                    )
                    await _bridge.publish(Channel.TEXT_IN, {"text": note})

                elif ch_str == "ping":
                    await client.send_json({"channel": "pong", "payload": payload or {}})

                elif ch_str == "user_interrupt":
                    await _bridge.publish(Channel.USER_INTERRUPT, payload or {})

                elif ch_str == "session_config":
                    cfg = payload if isinstance(payload, dict) else {}
                    minimax_settings = cfg.get("minimax_settings")
                    if isinstance(minimax_settings, dict):
                        update_minimax_settings(minimax_settings)
                    await _restart_core_if_needed(cfg)
                    user_id = cfg.get("memory_user") or cfg.get("user_id")
                    if _core and user_id:
                        _core.set_memory_user(user_id)
                    sources = cfg.get("sources", {})
                    if _core and isinstance(sources, dict):
                        _core.set_sources(sources)
                    features = cfg.get("features", {})
                    if _core and isinstance(features, dict):
                        _core.set_features(features)
                    _apply_perception_config(cfg)
                    await _publish_memory_accounts()

                elif ch_str == "memory_account":
                    req = payload if isinstance(payload, dict) else {}
                    action = req.get("action", "list")
                    user_id = req.get("user_id", "")
                    result = None
                    if _core:
                        if action == "create":
                            result = _core.create_memory_user(user_id)
                        elif action == "delete":
                            result = _core.delete_memory_user(user_id)
                        elif action == "select":
                            result = _core.set_memory_user(user_id)
                        else:
                            result = _core.memory.list_accounts()
                    if result:
                        await _bridge.publish(Channel.MEMORY_ACCOUNTS, result)

                elif ch_str == "source_control":
                    sources = payload if isinstance(payload, dict) else {}
                    if _core:
                        _core.set_sources(sources)
                    if isinstance(sources.get("features"), dict):
                        _apply_perception_config({"features": sources["features"]})

                elif ch_str == "feature_control":
                    features = payload if isinstance(payload, dict) else {}
                    if _core:
                        _core.set_features(features)
                    _apply_perception_config({"features": features})

                elif ch_str == "audio_in":
                    audio_payload = payload if isinstance(payload, dict) else {}
                    pcm_b64 = audio_payload.get("pcm_b64", "") if isinstance(audio_payload, dict) else payload
                    if pcm_b64:
                        pcm_bytes = base64.b64decode(pcm_b64)
                        await _bridge.publish(Channel.AUDIO_IN, {
                            "pcm": pcm_bytes,
                            "speaking": bool(audio_payload.get("speaking", False)),
                            "probability": float(audio_payload.get("probability", 0.0) or 0.0),
                            "source": "browser",
                        })

                elif ch_str == "video_in":
                    jpeg_b64 = payload.get("jpeg_b64", "") if isinstance(payload, dict) else payload
                    if jpeg_b64:
                        jpeg_bytes = base64.b64decode(jpeg_b64)
                        if _browser_vision_mod is not None:
                            _browser_vision_mod.submit_jpeg(jpeg_bytes)
                        await _bridge.publish(Channel.VIDEO_IN, {"jpeg": jpeg_bytes})

        except WebSocketDisconnect:
            log.info("WebSocket client disconnected")
        except Exception as e:
            log.exception("WebSocket browser_to_bridge 錯誤: %s", e)
            try:
                await _bridge.publish(Channel.ERROR, f"WebSocket 收訊錯誤: {e}")
            except Exception:
                pass

    forward_task = asyncio.create_task(bridge_to_browser())

    try:
        await browser_to_bridge()
    finally:
        forward_task.cancel()
        _bridge.unsubscribe(sub)


# ══════════════════════════════════════════════════════════════════════════════
# 入口
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════╗
║   RAPHAEL · Ambient Perception           ║
║   http://localhost:8765                   ║
╚══════════════════════════════════════════╝

啟動選項：
  python ui.py                  純 Web 模式
  python ui.py --perception     啟動本機 VAD + Vision
  python ui.py --check-deps all  檢查依賴
  python ui.py --install-deps core
  python ui.py --install-deps perception --with-torch
  python ui.py --install-deps identity
  python ui.py --install-deps identity-strong
""")
    uvicorn.run(
        "ui:app",
        host="0.0.0.0",
        port=8765,
        log_level="warning",
    )
