"""
╔══════════════════════════════════════════════════════════════╗
║      Gemini Live UI — Web 介面版                              ║
║      模型: gemini-3.1-flash-live-preview                     ║
║                                                              ║
║  • FastAPI + WebSocket 橋接 Gemini Live API                  ║
║  • 支援文字 / 音訊 / 影像模式                                  ║
║  • 修正：打斷清空佇列、轉錄串流顯示、完整事件處理               ║
║                                                              ║
║  安裝: pip install fastapi uvicorn websockets pyaudio opencv-python
║  執行: python gemini_live_ui.py                              ║
║  瀏覽: http://localhost:8765                                  ║
╚══════════════════════════════════════════════════════════════╝
"""

import asyncio
import base64
import json
import os
import sys
import threading
import time
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ─── 依賴 ─────────────────────────────────────────────────────────────────────

def _need(pkg, pip_name=None):
    import importlib
    try:
        return importlib.import_module(pkg)
    except ImportError:
        print(f"[錯誤] 缺少 {pkg}，請執行: pip install {pip_name or pkg}")
        sys.exit(1)

_need("fastapi", "fastapi uvicorn")
_need("uvicorn")
_need("websockets")

import websockets as _ws
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

try:
    import pyaudio
    HAS_AUDIO = True
except ImportError:
    HAS_AUDIO = False

try:
    import cv2
    HAS_VIDEO = True
except ImportError:
    HAS_VIDEO = False

# ─── 設定 ─────────────────────────────────────────────────────────────────────

API_KEY   = os.environ.get("GEMINI_API_KEY", "")
MODEL     = "gemini-3.1-flash-live-preview"
WS_URL    = (
    "wss://generativelanguage.googleapis.com/ws/"
    "google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent"
    f"?key={API_KEY}"
)
MIC_RATE  = 16000
SPK_RATE  = 24000
CHANNELS  = 1
CHUNK     = 1024
VOICE     = "Puck"
SYSTEM_PROMPT = "你是一個友善的繁體中文AI助理，請用繁體中文回答，簡潔清晰。"

# ─── Gemini 橋接核心 ──────────────────────────────────────────────────────────

class GeminiBridge:
    """
    管理一條 Gemini Live WebSocket 連線。
    前端 UI 透過 FastAPI WebSocket 與此橋接溝通。

    修正項目：
    - interrupted → 立刻清空播放佇列 + 通知前端
    - outputTranscription → 串流累積，turnComplete 才換行
    - generationComplete / goAway / sessionResumptionUpdate / usageMetadata 全部處理
    """

    def __init__(self, ui_ws: WebSocket, mode: str, voice: str, thinking: str | None):
        self.ui_ws    = ui_ws
        self.mode     = mode
        self.voice    = voice
        self.thinking = thinking
        self.gemini   = None          # Gemini WS
        self._closed  = False

        # 音訊播放（server-side，僅 audio/video 模式）
        self._player  = None
        self._play_q  = asyncio.Queue()

        # 轉錄累積
        self._out_transcript = ""
        self._in_transcript  = ""

        # session 續接 token
        self._resume_token: str | None = None

    # ── 發送給 UI ─────────────────────────────────────────────────────────────

    async def _ui(self, **kwargs):
        if not self._closed:
            try:
                await self.ui_ws.send_json(kwargs)
            except Exception:
                pass

    # ── 連線 ─────────────────────────────────────────────────────────────────

    async def connect(self):
        self.gemini = await _ws.connect(WS_URL, ping_interval=20)
        await self._send_setup()
        await self._ui(type="status", text="connected", model=MODEL, voice=self.voice)

    async def _send_setup(self):
        # generationConfig 只放生成參數，轉錄設定在 setup 頂層
        gen_cfg: dict = {
            "responseModalities": ["AUDIO"],
            "speechConfig": {
                "voiceConfig": {
                    "prebuiltVoiceConfig": {"voiceName": self.voice}
                }
            },
        }
        if self.thinking:
            gen_cfg["thinkingConfig"] = {"thinkingLevel": self.thinking}

        # outputAudioTranscription / inputAudioTranscription 是 setup 頂層欄位
        setup_body: dict = {
            "model": f"models/{MODEL}",
            "generationConfig": gen_cfg,
            "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
            "outputAudioTranscription": {},   # 所有模式都開啟，取得 AI 回應文字
        }
        if self.mode in ("audio", "video"):
            setup_body["inputAudioTranscription"] = {}  # 語音/影像模式才需要使用者語音轉錄

        # session 續接
        if self._resume_token:
            setup_body["sessionResumption"] = {"handle": self._resume_token}

        await self.gemini.send(json.dumps({"setup": setup_body}))

        # 等 setupComplete
        async for raw in self.gemini:
            msg = json.loads(raw)
            if "setupComplete" in msg:
                break

    # ── 發送方法 ─────────────────────────────────────────────────────────────

    async def send_text(self, text: str):
        await self.gemini.send(json.dumps({
            "clientContent": {
                "turns": [{"role": "user", "parts": [{"text": text}]}],
                "turnComplete": True
            }
        }))

    async def send_audio(self, pcm_b64: str):
        await self.gemini.send(json.dumps({
            "realtimeInput": {
                "audio": {"data": pcm_b64, "mimeType": f"audio/pcm;rate={MIC_RATE}"}
            }
        }))

    async def send_video(self, jpeg_b64: str):
        await self.gemini.send(json.dumps({
            "realtimeInput": {
                "video": {"data": jpeg_b64, "mimeType": "image/jpeg"}
            }
        }))

    async def send_audio_stream_end(self):
        await self.gemini.send(json.dumps({
            "realtimeInput": {"audioStreamEnd": True}
        }))

    # ── 接收主迴圈 ───────────────────────────────────────────────────────────

    async def recv_loop(self):
        try:
            async for raw in self.gemini:
                msg = json.loads(raw)
                await self._handle(msg)
        except _ws.exceptions.ConnectionClosed as e:
            await self._ui(type="error", text=f"Gemini 斷線: {e.code} {e.reason}")
        finally:
            self._closed = True
            await self._ui(type="status", text="disconnected")

    async def _handle(self, msg: dict):

        # ── serverContent ──────────────────────────────────────────────────
        if sc := msg.get("serverContent"):

            # ★ interrupted：AI 被打斷，立刻清空播放佇列
            if sc.get("interrupted"):
                await self._flush_audio()
                await self._ui(type="interrupted")
                self._out_transcript = ""

            # 音訊輸出 → 直接回傳給前端播放（base64）
            for part in sc.get("modelTurn", {}).get("parts", []):
                if inline := part.get("inlineData"):
                    if "audio" in inline.get("mimeType", ""):
                        await self._ui(
                            type="audio",
                            data=inline["data"],
                            mimeType=inline["mimeType"]
                        )

            # 輸出轉錄（串流累積，不逐行換行）
            if ot := sc.get("outputTranscription"):
                chunk = ot.get("text", "")
                if chunk:
                    self._out_transcript += chunk
                    await self._ui(type="transcript_out", text=chunk, done=False)

            # 輸入轉錄
            if it := sc.get("inputTranscription"):
                chunk = it.get("text", "")
                if chunk:
                    self._in_transcript += chunk
                    await self._ui(type="transcript_in", text=chunk, done=False)

            # generationComplete（模型生成完畢，還沒等播放結束）
            if sc.get("generationComplete"):
                await self._ui(type="generation_complete")

            # turnComplete（含等待播放完畢）
            if sc.get("turnComplete"):
                await self._ui(
                    type="turn_complete",
                    transcript=self._out_transcript,
                    user_transcript=self._in_transcript,
                )
                self._out_transcript = ""
                self._in_transcript  = ""

            # groundingMetadata（Google Search 引用）
            if gm := sc.get("groundingMetadata"):
                sources = []
                for chunk in gm.get("groundingChunks", []):
                    if web := chunk.get("web"):
                        sources.append({"title": web.get("title",""), "uri": web.get("uri","")})
                if sources:
                    await self._ui(type="sources", sources=sources)

        # ── toolCall ──────────────────────────────────────────────────────
        if tc := msg.get("toolCall"):
            for fc in tc.get("functionCalls", []):
                call_id = fc.get("id", "")
                name    = fc.get("name", "")
                args    = fc.get("args", {})
                await self._ui(type="tool_call", name=name, args=args)
                result  = self._execute_tool(name, args)
                await self._ui(type="tool_result", name=name, result=result)
                await self.gemini.send(json.dumps({
                    "toolResponse": {
                        "functionResponses": [{"id": call_id, "name": name, "response": result}]
                    }
                }))

        # ── toolCallCancellation ───────────────────────────────────────────
        if tcc := msg.get("toolCallCancellation"):
            ids = tcc.get("ids", [])
            await self._ui(type="tool_cancelled", ids=ids)

        # ── goAway（伺服器即將關閉）────────────────────────────────────────
        if ga := msg.get("goAway"):
            secs = ga.get("timeLeft", {}).get("seconds", "?")
            await self._ui(type="go_away", seconds=secs)

        # ── sessionResumptionUpdate ────────────────────────────────────────
        if sru := msg.get("sessionResumptionUpdate"):
            if token := sru.get("newHandle"):
                self._resume_token = token
                await self._ui(type="session_token", token=token)

        # ── usageMetadata ─────────────────────────────────────────────────
        if um := msg.get("usageMetadata"):
            await self._ui(type="usage", data=um)

    async def _flush_audio(self):
        """清空播放佇列（interrupted 時呼叫）"""
        while not self._play_q.empty():
            try:
                self._play_q.get_nowait()
            except asyncio.QueueEmpty:
                break

    def _execute_tool(self, name: str, args: dict) -> dict:
        if name == "get_current_time":
            return {"time": time.strftime("%Y-%m-%d %H:%M:%S")}
        elif name == "get_weather":
            city = args.get("city", "未知")
            return {"city": city, "temperature": "26°C", "condition": "多雲", "note": "模擬資料"}
        return {"error": f"未知工具: {name}"}

    async def close(self):
        self._closed = True
        if self.gemini:
            await self.gemini.close()


# ─── FastAPI App ──────────────────────────────────────────────────────────────

app = FastAPI()

HTML = r"""<!DOCTYPE html>
<html lang="zh-TW">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Gemini Live</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;600&family=Noto+Sans+TC:wght@300;400;500&display=swap" rel="stylesheet">
<style>
:root {
  --bg:       #0a0a0f;
  --surface:  #111118;
  --border:   #1e1e2e;
  --border2:  #2a2a3e;
  --text:     #e0e0f0;
  --muted:    #5a5a7a;
  --accent:   #6c63ff;
  --accent2:  #a78bfa;
  --ai:       #34d399;
  --user:     #60a5fa;
  --warn:     #fbbf24;
  --danger:   #f87171;
  --tool:     #f472b6;
  --radius:   12px;
  --font-mono: 'JetBrains Mono', monospace;
  --font-ui:   'Noto Sans TC', sans-serif;
}

* { box-sizing: border-box; margin: 0; padding: 0; }

body {
  background: var(--bg);
  color: var(--text);
  font-family: var(--font-ui);
  height: 100dvh;
  display: flex;
  flex-direction: column;
  overflow: hidden;
}

/* ── Header ── */
header {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 12px 20px;
  border-bottom: 1px solid var(--border);
  background: var(--surface);
  flex-shrink: 0;
}

.logo {
  font-family: var(--font-mono);
  font-size: 13px;
  font-weight: 600;
  color: var(--accent2);
  letter-spacing: 0.05em;
}

.logo span { color: var(--muted); font-weight: 300; }

.status-dot {
  width: 7px; height: 7px;
  border-radius: 50%;
  background: var(--muted);
  transition: background 0.3s;
  flex-shrink: 0;
}
.status-dot.connected  { background: var(--ai); box-shadow: 0 0 8px var(--ai); }
.status-dot.error      { background: var(--danger); }
.status-dot.thinking   { background: var(--warn); animation: pulse 1s infinite; }

#status-text {
  font-family: var(--font-mono);
  font-size: 11px;
  color: var(--muted);
}

.header-right {
  margin-left: auto;
  display: flex;
  align-items: center;
  gap: 8px;
}

.usage-badge {
  font-family: var(--font-mono);
  font-size: 10px;
  color: var(--muted);
  background: var(--border);
  padding: 3px 8px;
  border-radius: 20px;
  display: none;
}
.usage-badge.visible { display: block; }

/* ── Controls bar ── */
#controls {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 10px 20px;
  border-bottom: 1px solid var(--border);
  background: var(--surface);
  flex-shrink: 0;
  flex-wrap: wrap;
}

select, .btn {
  font-family: var(--font-mono);
  font-size: 12px;
  background: var(--border);
  color: var(--text);
  border: 1px solid var(--border2);
  border-radius: 8px;
  padding: 6px 12px;
  cursor: pointer;
  transition: all 0.15s;
  outline: none;
}
select:hover, .btn:hover { border-color: var(--accent); }
select:focus { border-color: var(--accent2); }

.btn-primary {
  background: var(--accent);
  border-color: var(--accent);
  color: #fff;
  font-weight: 600;
}
.btn-primary:hover { background: var(--accent2); border-color: var(--accent2); }
.btn-primary:disabled { opacity: 0.4; cursor: not-allowed; }

.btn-danger { border-color: var(--danger); color: var(--danger); }
.btn-danger:hover { background: var(--danger); color: #000; }
.btn-danger:disabled { opacity: 0.3; cursor: not-allowed; }

.btn-warn { border-color: var(--warn); color: var(--warn); }
.btn-warn:hover { background: var(--warn); color: #000; }
.btn-warn:disabled { opacity: 0.3; cursor: not-allowed; }

.sep { width: 1px; height: 24px; background: var(--border2); }

/* ── Main layout ── */
main {
  display: flex;
  flex: 1;
  overflow: hidden;
}

/* ── Chat column ── */
#chat-col {
  flex: 1;
  display: flex;
  flex-direction: column;
  overflow: hidden;
}

#messages {
  flex: 1;
  overflow-y: auto;
  padding: 16px 20px;
  display: flex;
  flex-direction: column;
  gap: 12px;
  scroll-behavior: smooth;
}

#messages::-webkit-scrollbar { width: 4px; }
#messages::-webkit-scrollbar-track { background: transparent; }
#messages::-webkit-scrollbar-thumb { background: var(--border2); border-radius: 2px; }

.msg {
  display: flex;
  flex-direction: column;
  gap: 4px;
  animation: fadeUp 0.2s ease;
  max-width: 100%;
}

@keyframes fadeUp {
  from { opacity: 0; transform: translateY(6px); }
  to   { opacity: 1; transform: translateY(0); }
}

.msg-label {
  font-family: var(--font-mono);
  font-size: 10px;
  color: var(--muted);
  text-transform: uppercase;
  letter-spacing: 0.08em;
}

.msg-bubble {
  padding: 10px 14px;
  border-radius: var(--radius);
  font-size: 14px;
  line-height: 1.6;
  word-break: break-word;
  white-space: pre-wrap;
  max-width: 680px;
}

.msg.user .msg-bubble {
  background: #1a2035;
  border: 1px solid #2a3550;
  color: var(--user);
  align-self: flex-end;
  border-bottom-right-radius: 4px;
}
.msg.user .msg-label { text-align: right; color: var(--user); opacity: 0.6; }

.msg.ai .msg-bubble {
  background: #0f1f1a;
  border: 1px solid #1a3028;
  color: var(--ai);
  border-bottom-left-radius: 4px;
}
.msg.ai .msg-label { color: var(--ai); opacity: 0.6; }

.msg.system .msg-bubble {
  background: var(--border);
  border: 1px solid var(--border2);
  color: var(--muted);
  font-family: var(--font-mono);
  font-size: 12px;
  border-radius: 8px;
}

.msg.tool .msg-bubble {
  background: #1f1020;
  border: 1px solid #3a1540;
  color: var(--tool);
  font-family: var(--font-mono);
  font-size: 12px;
}
.msg.tool .msg-label { color: var(--tool); opacity: 0.6; }

.msg.warn .msg-bubble {
  background: #1f1a00;
  border: 1px solid #3a3000;
  color: var(--warn);
  font-family: var(--font-mono);
  font-size: 12px;
}

/* 串流中的 AI 訊息（typing indicator） */
.msg.ai.streaming .msg-bubble::after {
  content: "▋";
  animation: blink 0.7s steps(1) infinite;
  color: var(--ai);
  opacity: 0.7;
}
@keyframes blink { 50% { opacity: 0; } }

/* ── Input bar ── */
#input-bar {
  display: flex;
  gap: 8px;
  padding: 12px 20px;
  border-top: 1px solid var(--border);
  background: var(--surface);
  flex-shrink: 0;
}

#text-input {
  flex: 1;
  background: var(--border);
  border: 1px solid var(--border2);
  border-radius: 10px;
  color: var(--text);
  font-family: var(--font-ui);
  font-size: 14px;
  padding: 10px 14px;
  outline: none;
  resize: none;
  min-height: 42px;
  max-height: 120px;
  transition: border-color 0.15s;
}
#text-input:focus { border-color: var(--accent); }
#text-input:disabled { opacity: 0.4; }

/* ── Video panel ── */
#video-panel {
  width: 240px;
  border-left: 1px solid var(--border);
  background: var(--surface);
  display: none;
  flex-direction: column;
  align-items: center;
  padding: 12px;
  gap: 10px;
  flex-shrink: 0;
}
#video-panel.visible { display: flex; }

#video-el {
  width: 100%;
  border-radius: 8px;
  background: #000;
  aspect-ratio: 4/3;
  object-fit: cover;
}

#canvas-el { display: none; }

.panel-label {
  font-family: var(--font-mono);
  font-size: 10px;
  color: var(--muted);
  text-transform: uppercase;
  letter-spacing: 0.08em;
}

/* ── Audio viz ── */
#audio-viz {
  width: 100%;
  height: 36px;
  display: none;
}
#audio-viz.visible { display: block; }

/* ── Pulse anim ── */
@keyframes pulse {
  0%, 100% { opacity: 1; }
  50%       { opacity: 0.3; }
}

/* ── Interrupted banner ── */
#interrupted-banner {
  position: fixed;
  top: 60px;
  left: 50%;
  transform: translateX(-50%);
  background: var(--warn);
  color: #000;
  font-family: var(--font-mono);
  font-size: 12px;
  font-weight: 600;
  padding: 6px 16px;
  border-radius: 20px;
  opacity: 0;
  pointer-events: none;
  transition: opacity 0.2s;
  z-index: 100;
}
#interrupted-banner.show { opacity: 1; }

/* ── Sources ── */
.sources {
  margin-top: 6px;
  display: flex;
  flex-direction: column;
  gap: 4px;
}
.source-link {
  font-family: var(--font-mono);
  font-size: 11px;
  color: var(--accent2);
  text-decoration: none;
  opacity: 0.8;
}
.source-link:hover { opacity: 1; text-decoration: underline; }
</style>
</head>
<body>

<header>
  <div class="logo">GEMINI <span>LIVE /</span> RAPHAEL</div>
  <div class="status-dot" id="status-dot"></div>
  <div id="status-text">未連線</div>
  <div class="header-right">
    <div class="usage-badge" id="usage-badge">— tokens</div>
  </div>
</header>

<div id="controls">
  <select id="sel-mode">
    <option value="text">📝 文字</option>
    <option value="audio">🎤 語音</option>
    <option value="video">📷 影像+語音</option>
  </select>
  <select id="sel-voice">
    <option value="Puck">Puck</option>
    <option value="Charon">Charon</option>
    <option value="Kore">Kore</option>
    <option value="Fenrir">Fenrir</option>
    <option value="Aoede">Aoede</option>
  </select>
  <select id="sel-thinking">
    <option value="">Thinking OFF</option>
    <option value="minimal">Thinking: minimal</option>
    <option value="low">Thinking: low</option>
    <option value="medium">Thinking: medium</option>
    <option value="high">Thinking: high</option>
  </select>
  <div class="sep"></div>
  <button class="btn btn-primary" id="btn-connect">連線</button>
  <button class="btn btn-danger"  id="btn-disconnect" disabled>斷線</button>
  <button class="btn btn-warn"    id="btn-interrupt"  disabled title="打斷 AI 說話">⏹ 打斷</button>
</div>

<main>
  <div id="chat-col">
    <div id="messages"></div>
    <div id="input-bar">
      <textarea id="text-input" placeholder="輸入訊息… (Enter 發送，Shift+Enter 換行)" rows="1" disabled></textarea>
      <button class="btn btn-primary" id="btn-send" disabled>發送</button>
    </div>
  </div>

  <div id="video-panel">
    <div class="panel-label">鏡頭預覽</div>
    <video id="video-el" autoplay muted playsinline></video>
    <canvas id="canvas-el"></canvas>
    <canvas id="audio-viz" class="visible"></canvas>
    <div class="panel-label" id="cam-fps-label">1 fps → Gemini</div>
  </div>
</main>

<div id="interrupted-banner">⚡ 打斷成功</div>

<script>
// ── 狀態 ──────────────────────────────────────────────────────────────────────

let ws = null;
let audioCtx = null;
let mediaStream = null;
let micProcessor = null;
let videoInterval = null;
let playQueue = [];
let isPlaying = false;
let isAISpeaking = false;
let currentAiMsg = null;
let ignoreTranscript = false;  // true 時忽略打斷後的殘餘 transcript

const MODE    = () => document.getElementById('sel-mode').value;
const VOICE   = () => document.getElementById('sel-voice').value;
const THINK   = () => document.getElementById('sel-thinking').value;

// ── DOM refs ──────────────────────────────────────────────────────────────────

const $dot        = document.getElementById('status-dot');
const $statusText = document.getElementById('status-text');
const $messages   = document.getElementById('messages');
const $input      = document.getElementById('text-input');
const $btnConnect = document.getElementById('btn-connect');
const $btnDisconn = document.getElementById('btn-disconnect');
const $btnInterr  = document.getElementById('btn-interrupt');
const $btnSend    = document.getElementById('btn-send');
const $videoPanel = document.getElementById('video-panel');
const $videoEl    = document.getElementById('video-el');
const $canvasEl   = document.getElementById('canvas-el');
const $vizCanvas  = document.getElementById('audio-viz');
const $usageBadge = document.getElementById('usage-badge');
const $intBanner  = document.getElementById('interrupted-banner');

// ── UI helpers ────────────────────────────────────────────────────────────────

function setStatus(state, text) {
  $dot.className = 'status-dot ' + state;
  $statusText.textContent = text;
}

function addMsg(role, text, extra = {}) {
  const wrap = document.createElement('div');
  wrap.className = `msg ${role}`;
  const label = document.createElement('div');
  label.className = 'msg-label';
  label.textContent = {
    ai: 'Gemini', user: '你', system: 'System', tool: 'Tool', warn: 'Notice'
  }[role] || role;
  const bubble = document.createElement('div');
  bubble.className = 'msg-bubble';
  bubble.textContent = text;
  wrap.appendChild(label);
  wrap.appendChild(bubble);
  if (extra.sources) {
    const src = document.createElement('div');
    src.className = 'sources';
    extra.sources.forEach(s => {
      const a = document.createElement('a');
      a.className = 'source-link';
      a.href = s.uri; a.target = '_blank';
      a.textContent = '↗ ' + (s.title || s.uri);
      src.appendChild(a);
    });
    wrap.appendChild(src);
  }
  $messages.appendChild(wrap);
  $messages.scrollTop = $messages.scrollHeight;
  return wrap;
}

function showInterruptedBanner() {
  $intBanner.classList.add('show');
  setTimeout(() => $intBanner.classList.remove('show'), 1500);
}

// ── WebSocket 連線 ────────────────────────────────────────────────────────────

async function connect() {
  const mode = MODE();
  setStatus('thinking', '連線中…');
  $btnConnect.disabled = true;

  // 請求媒體（需 https 或 localhost，否則瀏覽器拒絕）
  if (mode === 'audio' || mode === 'video') {
    const isSecure = location.protocol === 'https:' || location.hostname === 'localhost' || location.hostname === '127.0.0.1';
    if (!isSecure) {
      addMsg('system', '⚠ 麥克風/鏡頭需要安全環境（https 或 localhost）。\n請用 http://localhost:8765 而非 IP 開啟頁面。');
      setStatus('error', '需要 localhost');
      $btnConnect.disabled = false;
      return;
    }
    try {
      const constraints = { audio: true, video: mode === 'video' };
      mediaStream = await navigator.mediaDevices.getUserMedia(constraints);
    } catch (e) {
      const hint = e.name === 'NotAllowedError'
        ? '請在瀏覽器允許麥克風權限後重試'
        : e.message;
      addMsg('system', '無法取得麥克風/鏡頭：' + hint);
      setStatus('error', '媒體錯誤');
      $btnConnect.disabled = false;
      return;
    }
  }

  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  const url = `${proto}://${location.host}/ws?mode=${mode}&voice=${VOICE()}&thinking=${THINK()}`;
  ws = new WebSocket(url);
  ws.binaryType = 'arraybuffer';

  ws.onopen = () => {
    addMsg('system', `已連線 | 模式: ${mode} | 音色: ${VOICE()}`);
    $btnDisconn.disabled = false;
    $btnInterr.disabled = false;
    if (mode === 'text') {
      $input.disabled = false;
      $btnSend.disabled = false;
      $input.focus();
    }
    if (mode === 'video') {
      $videoPanel.classList.add('visible');
      $videoEl.srcObject = mediaStream;
      startVideoCapture();
    }
    if (mode === 'audio' || mode === 'video') {
      startMic();
    }
    setupAudio();
  };

  ws.onmessage = (ev) => {
    const msg = JSON.parse(ev.data);
    handleServerMsg(msg);
  };

  ws.onclose = (ev) => {
    cleanup();
    setStatus('', `斷線 ${ev.code}`);
    addMsg('system', `連線關閉 (${ev.code})`);
    $btnConnect.disabled = false;
  };

  ws.onerror = () => {
    setStatus('error', '連線錯誤');
  };
}

function disconnect() {
  if (ws) ws.close();
}

// ── 訊息處理 ──────────────────────────────────────────────────────────────────

function handleServerMsg(msg) {
  switch (msg.type) {

    case 'status':
      if (msg.text === 'connected') setStatus('connected', `已連線 · ${msg.model}`);
      if (msg.text === 'disconnected') setStatus('', '已斷線');
      break;

    case 'audio':
      // 把 base64 PCM 加入播放佇列
      isAISpeaking = true;
      $btnInterr.classList.add('btn-warn');
      enqueueAudio(msg.data, msg.mimeType);
      break;

    case 'interrupted':
      // 清空瀏覽器端播放佇列，並忽略後續舊回合 transcript
      playQueue = [];
      isPlaying = false;
      isAISpeaking = false;
      ignoreTranscript = true;  // 直到 turn_complete 才重置
      if (currentAiMsg) {
        currentAiMsg.classList.remove('streaming');
        currentAiMsg = null;
      }
      showInterruptedBanner();
      break;

    case 'transcript_out': {
      // 打斷後忽略舊回合的殘餘 transcript，直到 turn_complete 重置
      if (ignoreTranscript) break;
      if (!currentAiMsg) {
        currentAiMsg = addMsg('ai', '');
        currentAiMsg.classList.add('streaming');
      }
      const bubble = currentAiMsg.querySelector('.msg-bubble');
      bubble.textContent += msg.text;
      $messages.scrollTop = $messages.scrollHeight;
      setStatus('thinking', 'AI 說話中…');
      break;
    }

    case 'transcript_in':
      // 使用者語音轉錄
      break; // 已在 send 時顯示；若要顯示可在此加

    case 'generation_complete':
      setStatus('connected', '生成完畢');
      break;

    case 'turn_complete':
      isAISpeaking = false;
      ignoreTranscript = false;  // 重置，下一回合正常顯示
      if (currentAiMsg) {
        currentAiMsg.classList.remove('streaming');
        currentAiMsg = null;
      }
      setStatus('connected', '等待輸入');
      break;

    case 'tool_call':
      addMsg('tool', `⚙ 呼叫 ${msg.name}(${JSON.stringify(msg.args)})`);
      break;

    case 'tool_result':
      addMsg('tool', `✓ ${msg.name} → ${JSON.stringify(msg.result)}`);
      break;

    case 'tool_cancelled':
      addMsg('tool', `✕ 取消: ${msg.ids.join(', ')}`);
      break;

    case 'sources':
      if (currentAiMsg) {
        const src = document.createElement('div');
        src.className = 'sources';
        msg.sources.forEach(s => {
          const a = document.createElement('a');
          a.className = 'source-link';
          a.href = s.uri; a.target = '_blank';
          a.textContent = '↗ ' + (s.title || s.uri);
          src.appendChild(a);
        });
        currentAiMsg.appendChild(src);
      }
      break;

    case 'go_away':
      addMsg('warn', `⚠ 伺服器將在 ${msg.seconds}s 後關閉連線，請重新連線`);
      break;

    case 'session_token':
      // 可儲存 token 供續接使用
      break;

    case 'usage':
      const d = msg.data;
      const total = d.totalTokenCount || '–';
      $usageBadge.textContent = `${total} tokens`;
      $usageBadge.classList.add('visible');
      break;

    case 'error':
      addMsg('system', '❌ ' + msg.text);
      setStatus('error', '錯誤');
      break;
  }
}

// ── 打斷 ──────────────────────────────────────────────────────────────────────

function interrupt() {
  // 清空本地播放佇列
  playQueue = [];
  isPlaying = false;
  isAISpeaking = false;
  // 通知 server（server 會通知 Gemini 透過 VAD 打斷）
  // Live API 的 barge-in 透過持續送麥克風聲音自動觸發，這裡只清本地佇列
  if (currentAiMsg) currentAiMsg.classList.remove('streaming');
  currentAiMsg = null;
  showInterruptedBanner();
}

// ── 音訊播放（Web Audio API，支援 24kHz PCM16）───────────────────────────────

function setupAudio() {
  audioCtx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 24000 });
}

function enqueueAudio(b64, mimeType) {
  const raw = atob(b64);
  const buf = new Int16Array(raw.length / 2);
  for (let i = 0; i < buf.length; i++) {
    buf[i] = (raw.charCodeAt(i * 2)) | (raw.charCodeAt(i * 2 + 1) << 8);
  }
  // 轉 Float32
  const float = new Float32Array(buf.length);
  for (let i = 0; i < buf.length; i++) float[i] = buf[i] / 32768;

  playQueue.push(float);
  if (!isPlaying) drainQueue();
}

function drainQueue() {
  if (playQueue.length === 0) { isPlaying = false; return; }
  isPlaying = true;
  const float = playQueue.shift();
  const audioBuf = audioCtx.createBuffer(1, float.length, 24000);
  audioBuf.getChannelData(0).set(float);
  const src = audioCtx.createBufferSource();
  src.buffer = audioBuf;
  src.connect(audioCtx.destination);
  src.onended = drainQueue;
  src.start();
}

// ── 麥克風串流 ────────────────────────────────────────────────────────────────

function startMic() {
  const micCtx = new AudioContext({ sampleRate: 16000 });
  const src = micCtx.createMediaStreamSource(mediaStream);
  micProcessor = micCtx.createScriptProcessor(4096, 1, 1);
  micProcessor.onaudioprocess = (e) => {
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    const float = e.inputBuffer.getChannelData(0);
    const pcm = new Int16Array(float.length);
    for (let i = 0; i < float.length; i++)
      pcm[i] = Math.max(-32768, Math.min(32767, float[i] * 32768));
    const b64 = btoa(String.fromCharCode(...new Uint8Array(pcm.buffer)));
    ws.send(JSON.stringify({ type: 'audio', data: b64 }));
  };
  src.connect(micProcessor);
  micProcessor.connect(micCtx.destination);
}

// ── 鏡頭擷取 ─────────────────────────────────────────────────────────────────

function startVideoCapture() {
  const ctx = $canvasEl.getContext('2d');
  videoInterval = setInterval(() => {
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    $canvasEl.width  = 320;
    $canvasEl.height = 240;
    ctx.drawImage($videoEl, 0, 0, 320, 240);
    $canvasEl.toBlob(blob => {
      const reader = new FileReader();
      reader.onload = () => {
        const b64 = reader.result.split(',')[1];
        ws.send(JSON.stringify({ type: 'video', data: b64 }));
      };
      reader.readAsDataURL(blob);
    }, 'image/jpeg', 0.8);
  }, 1000);
}

// ── 文字發送 ──────────────────────────────────────────────────────────────────

function sendText() {
  const text = $input.value.trim();
  if (!text || !ws || ws.readyState !== WebSocket.OPEN) return;
  ws.send(JSON.stringify({ type: 'text', data: text }));
  addMsg('user', text);
  $input.value = '';
  $input.style.height = 'auto';
  setStatus('thinking', '等待回應…');
}

$input.addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendText(); }
});
$input.addEventListener('input', () => {
  $input.style.height = 'auto';
  $input.style.height = Math.min($input.scrollHeight, 120) + 'px';
});

$btnSend.addEventListener('click', sendText);
$btnConnect.addEventListener('click', connect);
$btnDisconn.addEventListener('click', disconnect);
$btnInterr.addEventListener('click', interrupt);

// 已連線時切換 voice / mode / thinking → 自動重連套用新設定
async function reconnectIfNeeded() {
  if (!ws || ws.readyState !== WebSocket.OPEN) return;
  addMsg('system', '設定變更，重新連線中…');
  ws.close();
  // 等舊連線清理完畢再重連
  await new Promise(r => setTimeout(r, 400));
  connect();
}

document.getElementById('sel-voice').addEventListener('change', reconnectIfNeeded);
document.getElementById('sel-mode').addEventListener('change', reconnectIfNeeded);
document.getElementById('sel-thinking').addEventListener('change', reconnectIfNeeded);

// ── cleanup ───────────────────────────────────────────────────────────────────

function cleanup() {
  playQueue = []; isPlaying = false; isAISpeaking = false; currentAiMsg = null; ignoreTranscript = false;
  if (mediaStream) { mediaStream.getTracks().forEach(t => t.stop()); mediaStream = null; }
  if (videoInterval) { clearInterval(videoInterval); videoInterval = null; }
  if (micProcessor) { micProcessor.disconnect(); micProcessor = null; }
  $videoPanel.classList.remove('visible');
  $btnDisconn.disabled = true;
  $btnInterr.disabled = true;
  $input.disabled = true;
  $btnSend.disabled = true;
}

setStatus('', '未連線');
</script>
</body>
</html>"""


@app.get("/")
async def index():
    return HTMLResponse(HTML)


@app.websocket("/ws")
async def ws_endpoint(
    client: WebSocket,
    mode:     str = "text",
    voice:    str = "Puck",
    thinking: str = "",
):
    await client.accept()
    bridge = GeminiBridge(
        ui_ws=client,
        mode=mode,
        voice=voice,
        thinking=thinking or None,
    )

    try:
        await bridge.connect()

        # 並行：接收 Gemini 回應 + 接收前端訊息
        recv_task = asyncio.create_task(bridge.recv_loop())

        async def frontend_loop():
            try:
                while True:
                    data = await client.receive_json()
                    msg_type = data.get("type")
                    if msg_type == "text":
                        await bridge.send_text(data["data"])
                    elif msg_type == "audio":
                        await bridge.send_audio(data["data"])
                    elif msg_type == "video":
                        await bridge.send_video(data["data"])
            except WebSocketDisconnect:
                pass

        await asyncio.gather(recv_task, frontend_loop())

    except Exception as e:
        try:
            await client.send_json({"type": "error", "text": str(e)})
        except Exception:
            pass
    finally:
        recv_task.cancel() if 'recv_task' in dir() else None
        await bridge.close()


# ─── 入口 ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if not API_KEY:
        print("[錯誤] 請設定 GEMINI_API_KEY 環境變數（或寫入 .env）")
        sys.exit(1)

    print(f"""
╔══════════════════════════════════════════╗
║   Gemini Live UI                         ║
║   http://localhost:8765                  ║
╚══════════════════════════════════════════╝
安裝依賴: pip install fastapi uvicorn websockets
執行後開瀏覽器到 http://localhost:8765
""")
    uvicorn.run(app, host="0.0.0.0", port=8765, log_level="warning")