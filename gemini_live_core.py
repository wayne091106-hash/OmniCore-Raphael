"""
gemini_live_core.py
═══════════════════
純後端 Gemini Live API 客戶端
模型: gemini-3.1-flash-live-preview

架構分層（由底到上）：
  1. Config          — 所有常數集中於此，改這裡就好
  2. AudioPlayer     — 本機喇叭播放 24kHz PCM16
  3. Microphone      — 本機麥克風擷取 16kHz PCM16
  4. Camera          — 本機鏡頭擷取 JPEG（1fps）
  5. ToolRegistry    — Function Calling 工具定義 + 執行器
  6. GeminiSession   — Gemini WebSocket 連線、setup、收發、事件
  7. App             — 三種執行模式的主迴圈（text / audio / video）

已實作的 Gemini Live API 功能：
  ✓ 文字輸入 / 音訊輸入 / 影像幀輸入
  ✓ 音訊輸出（24kHz PCM16）+ outputAudioTranscription
  ✓ inputAudioTranscription（語音/影像模式）
  ✓ interrupted（打斷 → 清空播放佇列）
  ✓ generationComplete / turnComplete
  ✓ goAway（伺服器即將關閉通知）
  ✓ sessionResumptionUpdate（session token 保存）
  ✓ usageMetadata（token 用量）
  ✓ groundingMetadata（Google Search 引用來源）
  ✓ Function Calling（同步，含 toolCallCancellation）
  ✓ Google Search grounding
  ✓ Thinking（thinkingLevel）
  ✓ audioStreamEnd（靜音訊號）
  ✓ VAD 輔助方法（activityStart / activityEnd）

安裝依賴:
  pip install websockets pyaudio opencv-python

使用方式:
  export GEMINI_API_KEY=你的金鑰

  python gemini_live_core.py                    # 文字對話
  python gemini_live_core.py --audio            # 語音對話
  python gemini_live_core.py --video            # 影像 + 語音
  python gemini_live_core.py --thinking high    # 啟用 Thinking
  python gemini_live_core.py --tools            # 啟用 Function Calling + Google Search
  python gemini_live_core.py --voice Aoede      # 指定音色
"""

import asyncio
import base64
import json
import os
import sys
import threading
import time

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


# ══════════════════════════════════════════════════════════════════════════════
# 1. CONFIG  ── 改這裡就好，不用動其他地方
# ══════════════════════════════════════════════════════════════════════════════

class Config:
    # ── Gemini API ────────────────────────────────────────────────────────────
    API_KEY   : str = os.environ.get("GEMINI_API_KEY", "")
    MODEL     : str = "gemini-3.1-flash-live-preview"

    # ── 語音設定 ──────────────────────────────────────────────────────────────
    # 可選音色: Puck / Charon / Kore / Fenrir / Aoede
    VOICE     : str = "Puck"

    # ── System Prompt ─────────────────────────────────────────────────────────
    SYSTEM_PROMPT: str = (
        "你是一個友善的繁體中文AI助理，請用繁體中文回答，簡潔清晰。"
    )

    # ── 音訊規格（Live API 規定，勿更動）────────────────────────────────────
    MIC_RATE  : int = 16_000   # 麥克風輸入取樣率（16kHz PCM16）
    SPK_RATE  : int = 24_000   # 喇叭輸出取樣率（24kHz PCM16）
    CHANNELS  : int = 1
    CHUNK     : int = 1024     # 每次讀取的 frame 數

    # ── 影像規格 ──────────────────────────────────────────────────────────────
    VIDEO_FPS     : int = 1    # Live API 建議 ≤ 1fps
    VIDEO_WIDTH   : int = 320
    VIDEO_HEIGHT  : int = 240
    JPEG_QUALITY  : int = 85

    # ── 網路 ─────────────────────────────────────────────────────────────────
    WS_URL: str = (
        "wss://generativelanguage.googleapis.com/ws/"
        "google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent"
        f"?key={API_KEY}"
    )
    PING_INTERVAL: int = 20


# ══════════════════════════════════════════════════════════════════════════════
# 依賴檢查
# ══════════════════════════════════════════════════════════════════════════════

def _check(pkg: str) -> bool:
    import importlib
    try:
        importlib.import_module(pkg)
        return True
    except ImportError:
        return False

if not _check("websockets"):
    print("[錯誤] 請執行: pip install websockets")
    sys.exit(1)

import websockets
import websockets.exceptions

HAS_PYAUDIO = _check("pyaudio")
HAS_CV2     = _check("cv2")

if HAS_PYAUDIO:
    import pyaudio
if HAS_CV2:
    import cv2


# ══════════════════════════════════════════════════════════════════════════════
# 終端機顏色輸出
# ══════════════════════════════════════════════════════════════════════════════

class C:
    R  = "\033[91m"   # 紅
    G  = "\033[92m"   # 綠
    Y  = "\033[93m"   # 黃
    B  = "\033[94m"   # 藍
    M  = "\033[95m"   # 紫
    CY = "\033[96m"   # 青
    W  = "\033[0m"    # 重置

def log(tag: str, msg: str, color: str = C.W) -> None:
    ts = time.strftime("%H:%M:%S")
    print(f"{color}[{ts}][{tag}]{C.W} {msg}", flush=True)


# ══════════════════════════════════════════════════════════════════════════════
# 2. AudioPlayer  ── 本機喇叭播放 24kHz PCM16
# ══════════════════════════════════════════════════════════════════════════════

class AudioPlayer:
    """
    非阻塞式 PCM16 音訊播放器（24kHz）。

    用法:
        player = AudioPlayer()
        await player.start()
        await player.push(pcm_bytes)   # 加入佇列
        await player.flush()           # 打斷時清空佇列
        await player.stop()            # 結束
    """

    def __init__(self):
        if not HAS_PYAUDIO:
            raise RuntimeError("需要 pyaudio：pip install pyaudio")
        cfg = Config()
        self._pa = pyaudio.PyAudio()
        self._stream = self._pa.open(
            format=pyaudio.paInt16,
            channels=cfg.CHANNELS,
            rate=cfg.SPK_RATE,
            output=True,
            frames_per_buffer=cfg.CHUNK,
        )
        self._q    : asyncio.Queue[bytes | None] = asyncio.Queue()
        self._task : asyncio.Task | None = None

    async def start(self) -> None:
        self._task = asyncio.get_event_loop().create_task(self._play_loop())

    async def _play_loop(self) -> None:
        loop = asyncio.get_event_loop()
        while True:
            data = await self._q.get()
            if data is None:
                break
            await loop.run_in_executor(None, self._stream.write, data)

    async def push(self, pcm: bytes) -> None:
        await self._q.put(pcm)

    async def flush(self) -> None:
        """清空播放佇列（打斷時呼叫）"""
        while not self._q.empty():
            try:
                self._q.get_nowait()
            except asyncio.QueueEmpty:
                break

    async def stop(self) -> None:
        await self._q.put(None)
        if self._task:
            await self._task
        self._stream.stop_stream()
        self._stream.close()
        self._pa.terminate()


# ══════════════════════════════════════════════════════════════════════════════
# 3. Microphone  ── 本機麥克風擷取 16kHz PCM16
# ══════════════════════════════════════════════════════════════════════════════

class Microphone:
    """
    背景執行緒連續擷取麥克風 PCM16（16kHz）。

    用法:
        mic = Microphone()
        mic.start(asyncio.get_event_loop())
        chunk = await mic.read()
        mic.stop()
    """

    def __init__(self):
        if not HAS_PYAUDIO:
            raise RuntimeError("需要 pyaudio：pip install pyaudio")
        cfg = Config()
        self._pa = pyaudio.PyAudio()
        self._stream = self._pa.open(
            format=pyaudio.paInt16,
            channels=cfg.CHANNELS,
            rate=cfg.MIC_RATE,
            input=True,
            frames_per_buffer=cfg.CHUNK,
        )
        self._q      : asyncio.Queue[bytes] = asyncio.Queue()
        self._active : bool = False

    def start(self, loop: asyncio.AbstractEventLoop) -> None:
        self._active = True
        threading.Thread(
            target=self._capture, args=(loop,), daemon=True
        ).start()

    def _capture(self, loop: asyncio.AbstractEventLoop) -> None:
        while self._active:
            try:
                data = self._stream.read(Config.CHUNK, exception_on_overflow=False)
                asyncio.run_coroutine_threadsafe(self._q.put(data), loop)
            except Exception:
                break

    async def read(self) -> bytes:
        return await self._q.get()

    def stop(self) -> None:
        self._active = False
        self._stream.stop_stream()
        self._stream.close()
        self._pa.terminate()


# ══════════════════════════════════════════════════════════════════════════════
# 4. Camera  ── 本機鏡頭擷取 JPEG（1fps）
# ══════════════════════════════════════════════════════════════════════════════

class Camera:
    """
    背景執行緒以 VIDEO_FPS 速率擷取鏡頭 JPEG 幀。

    用法:
        cam = Camera()
        cam.start(asyncio.get_event_loop())
        jpeg_bytes = await cam.read()
        cam.stop()
    """

    def __init__(self, device: int = 0):
        if not HAS_CV2:
            raise RuntimeError("需要 opencv-python：pip install opencv-python")
        cfg = Config()
        self._cap     = cv2.VideoCapture(device)
        self._q       : asyncio.Queue[bytes] = asyncio.Queue(maxsize=2)
        self._active  : bool = False
        self._interval: float = 1.0 / cfg.VIDEO_FPS
        self._w = cfg.VIDEO_WIDTH
        self._h = cfg.VIDEO_HEIGHT
        self._q_val = cfg.JPEG_QUALITY

    def start(self, loop: asyncio.AbstractEventLoop) -> None:
        self._active = True
        threading.Thread(
            target=self._capture, args=(loop,), daemon=True
        ).start()
        log("CAM", f"鏡頭啟動（{Config.VIDEO_FPS} fps, {self._w}×{self._h}）", C.G)

    def _capture(self, loop: asyncio.AbstractEventLoop) -> None:
        while self._active:
            t0 = time.time()
            ret, frame = self._cap.read()
            if ret:
                frame = cv2.resize(frame, (self._w, self._h))
                ok, buf = cv2.imencode(
                    ".jpg", frame,
                    [cv2.IMWRITE_JPEG_QUALITY, self._q_val]
                )
                if ok:
                    asyncio.run_coroutine_threadsafe(
                        self._safe_put(buf.tobytes()), loop
                    )
            elapsed = time.time() - t0
            time.sleep(max(0.0, self._interval - elapsed))

    async def _safe_put(self, data: bytes) -> None:
        """佇列滿時丟棄最舊的幀（保持低延遲）"""
        if self._q.full():
            try:
                self._q.get_nowait()
            except asyncio.QueueEmpty:
                pass
        await self._q.put(data)

    async def read(self) -> bytes:
        return await self._q.get()

    def stop(self) -> None:
        self._active = False
        self._cap.release()


# ══════════════════════════════════════════════════════════════════════════════
# 5. ToolRegistry  ── Function Calling 工具定義 + 執行器
# ══════════════════════════════════════════════════════════════════════════════

class ToolRegistry:
    """
    在這裡新增自訂工具：
      1. 在 DECLARATIONS 加入 JSON schema
      2. 在 execute() 加入對應的 elif 分支
    """

    # ── 工具宣告（傳給 Gemini setup）────────────────────────────────────────
    DECLARATIONS: list[dict] = [
        {
            "functionDeclarations": [
                {
                    "name": "get_current_time",
                    "description": "取得目前的本地時間",
                    "parameters": {
                        "type": "OBJECT",
                        "properties": {},
                        "required": []
                    }
                },
                {
                    "name": "get_weather",
                    "description": "查詢某城市的天氣（模擬）",
                    "parameters": {
                        "type": "OBJECT",
                        "properties": {
                            "city": {
                                "type": "STRING",
                                "description": "城市名稱，例如 台北"
                            }
                        },
                        "required": ["city"]
                    }
                },
                # ── 在此新增更多工具宣告 ─────────────────────────────────
            ]
        }
    ]

    @staticmethod
    def execute(name: str, args: dict) -> dict:
        """
        執行工具並回傳結果。
        在這裡新增 elif 分支來處理新工具。
        """
        if name == "get_current_time":
            return {"time": time.strftime("%Y-%m-%d %H:%M:%S")}

        elif name == "get_weather":
            city = args.get("city", "未知")
            # TODO: 替換為真實天氣 API（e.g. OpenWeatherMap）
            return {
                "city": city,
                "temperature": "26°C",
                "condition": "多雲",
                "note": "模擬資料"
            }

        # ── 在此新增更多工具邏輯 ─────────────────────────────────────────
        else:
            return {"error": f"未知工具: {name}"}


# ══════════════════════════════════════════════════════════════════════════════
# 6. GeminiSession  ── Gemini WebSocket 連線核心
# ══════════════════════════════════════════════════════════════════════════════

class GeminiSession:
    """
    管理單一 Gemini Live WebSocket session。

    事件回呼（可覆寫或外掛 handler）：
      on_audio(pcm_bytes)              — 收到音訊輸出
      on_transcript_out(text, done)   — 收到 AI 轉錄片段
      on_transcript_in(text, done)    — 收到使用者語音轉錄片段
      on_interrupted()                — AI 被打斷
      on_generation_complete()        — 模型生成完畢
      on_turn_complete(out_tr, in_tr) — 回合結束（含完整轉錄）
      on_sources(sources)             — Google Search 引用來源
      on_tool_call(name, args) → dict — Function Call（需回傳結果）
      on_tool_cancelled(ids)          — 工具呼叫取消
      on_go_away(seconds)             — 伺服器即將關閉
      on_session_token(token)         — session 續接 token
      on_usage(data)                  — token 用量
      on_error(code, reason)          — 連線錯誤
    """

    def __init__(
        self,
        mode      : str        = "text",   # text | audio | video
        voice     : str        = Config.VOICE,
        thinking  : str | None = None,     # None | minimal | low | medium | high
        use_tools : bool       = False,
        player    : AudioPlayer | None = None,
    ):
        self.mode       = mode
        self.voice      = voice
        self.thinking   = thinking
        self.use_tools  = use_tools
        self.player     = player           # 若傳入 player，on_audio 預設自動播放

        self._ws             : websockets.WebSocketClientProtocol | None = None
        self._closed         : bool = False
        self._resume_token   : str | None = None
        self._out_transcript : str = ""
        self._in_transcript  : str = ""

    # ── 連線 ─────────────────────────────────────────────────────────────────

    async def connect(self) -> None:
        log("WS", f"連接 {Config.MODEL} ...", C.B)
        self._ws = await websockets.connect(
            Config.WS_URL,
            ping_interval=Config.PING_INTERVAL
        )
        log("WS", "WebSocket 連線成功", C.G)
        await self._send_setup()

    async def _send_setup(self) -> None:
        """
        建立 session 設定。
        欄位結構（camelCase，raw WebSocket JSON）：
          setup.generationConfig       ← 生成參數
          setup.systemInstruction
          setup.outputAudioTranscription  ← setup 頂層，非 generationConfig 子項
          setup.inputAudioTranscription   ← setup 頂層
          setup.tools
          setup.sessionResumption
        """
        cfg = Config()

        # generationConfig
        gen_cfg: dict = {
            "responseModalities": ["AUDIO"],   # TEXT modality 在此模型有 bug，永遠用 AUDIO
            "speechConfig": {
                "voiceConfig": {
                    "prebuiltVoiceConfig": {"voiceName": self.voice}
                }
            },
        }
        if self.thinking:
            gen_cfg["thinkingConfig"] = {"thinkingLevel": self.thinking}

        # setup 頂層
        setup_body: dict = {
            "model": f"models/{Config.MODEL}",
            "generationConfig": gen_cfg,
            "systemInstruction": {"parts": [{"text": Config.SYSTEM_PROMPT}]},
            "outputAudioTranscription": {},   # 所有模式啟用，讓 text 模式也能讀到回應
        }

        # 語音/影像模式才需要輸入轉錄
        if self.mode in ("audio", "video"):
            setup_body["inputAudioTranscription"] = {}

        # 工具
        if self.use_tools:
            setup_body["tools"] = (
                ToolRegistry.DECLARATIONS + [{"googleSearch": {}}]
            )

        # session 續接（goAway 後重連用）
        if self._resume_token:
            setup_body["sessionResumption"] = {"handle": self._resume_token}

        await self._ws.send(json.dumps({"setup": setup_body}))

        # 等 setupComplete
        async for raw in self._ws:
            msg = json.loads(raw)
            if "setupComplete" in msg:
                log("SETUP", "Session 初始化完成 ✓", C.G)
                self._log_config()
                break

    def _log_config(self) -> None:
        parts = [f"模式={self.mode}", f"音色={self.voice}"]
        if self.thinking:
            parts.append(f"thinking={self.thinking}")
        if self.use_tools:
            parts.append("tools=ON")
        log("CFG", " | ".join(parts), C.M)

    # ── 發送方法 ─────────────────────────────────────────────────────────────

    async def send_text(self, text: str) -> None:
        """文字輸入（使用 clientContent，適合 text 模式）"""
        await self._ws.send(json.dumps({
            "clientContent": {
                "turns": [{"role": "user", "parts": [{"text": text}]}],
                "turnComplete": True
            }
        }))

    async def send_text_realtime(self, text: str) -> None:
        """文字輸入（使用 realtimeInput，適合對話中插入文字）"""
        await self._ws.send(json.dumps({
            "realtimeInput": {"text": text}
        }))

    async def send_audio(self, pcm: bytes) -> None:
        """發送麥克風 PCM16（16kHz）"""
        await self._ws.send(json.dumps({
            "realtimeInput": {
                "audio": {
                    "data": base64.b64encode(pcm).decode(),
                    "mimeType": f"audio/pcm;rate={Config.MIC_RATE}"
                }
            }
        }))

    async def send_video(self, jpeg: bytes) -> None:
        """發送鏡頭 JPEG 幀（≤ 1fps）"""
        await self._ws.send(json.dumps({
            "realtimeInput": {
                "video": {
                    "data": base64.b64encode(jpeg).decode(),
                    "mimeType": "image/jpeg"
                }
            }
        }))

    async def send_audio_stream_end(self) -> None:
        """通知 Gemini 麥克風已靜音（用於手動 VAD）"""
        await self._ws.send(json.dumps({
            "realtimeInput": {"audioStreamEnd": True}
        }))

    async def send_vad_start(self) -> None:
        """手動 VAD：語音開始（需在 setup 中 disableAutomaticVad: true）"""
        await self._ws.send(json.dumps({
            "realtimeInput": {"activityStart": {}}
        }))

    async def send_vad_end(self) -> None:
        """手動 VAD：語音結束"""
        await self._ws.send(json.dumps({
            "realtimeInput": {"activityEnd": {}}
        }))

    async def _send_tool_response(
        self, call_id: str, name: str, result: dict
    ) -> None:
        await self._ws.send(json.dumps({
            "toolResponse": {
                "functionResponses": [{
                    "id":       call_id,
                    "name":     name,
                    "response": result
                }]
            }
        }))

    # ── 接收主迴圈 ───────────────────────────────────────────────────────────

    async def recv_loop(self) -> None:
        """
        持續接收 Gemini 訊息並分派到對應的事件 handler。
        """
        try:
            async for raw in self._ws:
                await self._dispatch(json.loads(raw))
        except websockets.exceptions.ConnectionClosed as e:
            self.on_error(e.code, e.reason)
        finally:
            self._closed = True

    async def _dispatch(self, msg: dict) -> None:

        # ── serverContent ──────────────────────────────────────────────────
        if sc := msg.get("serverContent"):

            # ★ interrupted：打斷，清空播放佇列
            if sc.get("interrupted"):
                if self.player:
                    await self.player.flush()
                self._out_transcript = ""
                self._in_transcript  = ""
                self.on_interrupted()

            # 音訊輸出
            for part in sc.get("modelTurn", {}).get("parts", []):
                if inline := part.get("inlineData"):
                    if "audio" in inline.get("mimeType", ""):
                        pcm = base64.b64decode(inline["data"])
                        if self.player:
                            await self.player.push(pcm)
                        self.on_audio(pcm)

            # 輸出轉錄（串流累積）
            if ot := sc.get("outputTranscription"):
                if chunk := ot.get("text", ""):
                    self._out_transcript += chunk
                    self.on_transcript_out(chunk, done=False)

            # 輸入轉錄
            if it := sc.get("inputTranscription"):
                if chunk := it.get("text", ""):
                    self._in_transcript += chunk
                    self.on_transcript_in(chunk, done=False)

            # generationComplete
            if sc.get("generationComplete"):
                self.on_generation_complete()

            # turnComplete
            if sc.get("turnComplete"):
                self.on_transcript_out("", done=True)
                self.on_transcript_in("", done=True)
                self.on_turn_complete(self._out_transcript, self._in_transcript)
                self._out_transcript = ""
                self._in_transcript  = ""

            # groundingMetadata（Google Search 引用）
            if gm := sc.get("groundingMetadata"):
                sources = [
                    {"title": w.get("title", ""), "uri": w.get("uri", "")}
                    for chunk in gm.get("groundingChunks", [])
                    if (w := chunk.get("web"))
                ]
                if sources:
                    self.on_sources(sources)

        # ── toolCall ──────────────────────────────────────────────────────
        if tc := msg.get("toolCall"):
            for fc in tc.get("functionCalls", []):
                call_id = fc.get("id", "")
                name    = fc.get("name", "")
                args    = fc.get("args", {})
                result  = self.on_tool_call(name, args)
                await self._send_tool_response(call_id, name, result)

        # ── toolCallCancellation ───────────────────────────────────────────
        if tcc := msg.get("toolCallCancellation"):
            self.on_tool_cancelled(tcc.get("ids", []))

        # ── goAway ────────────────────────────────────────────────────────
        if ga := msg.get("goAway"):
            secs = ga.get("timeLeft", {}).get("seconds", "?")
            self.on_go_away(int(secs) if isinstance(secs, (int, float)) else secs)

        # ── sessionResumptionUpdate ────────────────────────────────────────
        if sru := msg.get("sessionResumptionUpdate"):
            if token := sru.get("newHandle"):
                self._resume_token = token
                self.on_session_token(token)

        # ── usageMetadata ─────────────────────────────────────────────────
        if um := msg.get("usageMetadata"):
            self.on_usage(um)

    # ── 事件 handler（覆寫這些來自訂行為）───────────────────────────────────

    def on_audio(self, pcm: bytes) -> None:
        """收到 AI 音訊輸出（已自動推入 player，這裡可額外處理如存檔）"""
        pass

    def on_transcript_out(self, text: str, done: bool) -> None:
        """AI 語音轉錄片段；done=True 表示回合結束"""
        pass

    def on_transcript_in(self, text: str, done: bool) -> None:
        """使用者語音轉錄片段；done=True 表示回合結束"""
        pass

    def on_interrupted(self) -> None:
        """AI 被打斷"""
        pass

    def on_generation_complete(self) -> None:
        """模型生成完畢（音訊可能仍在播放）"""
        pass

    def on_turn_complete(self, out_transcript: str, in_transcript: str) -> None:
        """回合結束，含本回合完整轉錄"""
        pass

    def on_sources(self, sources: list[dict]) -> None:
        """Google Search grounding 引用來源"""
        pass

    def on_tool_call(self, name: str, args: dict) -> dict:
        """Function Call，回傳結果 dict"""
        return ToolRegistry.execute(name, args)

    def on_tool_cancelled(self, ids: list[str]) -> None:
        """工具呼叫取消"""
        pass

    def on_go_away(self, seconds) -> None:
        """伺服器即將關閉，seconds 為剩餘秒數"""
        pass

    def on_session_token(self, token: str) -> None:
        """session 續接 token，可儲存後用於重連"""
        pass

    def on_usage(self, data: dict) -> None:
        """token 用量資料"""
        pass

    def on_error(self, code: int, reason: str) -> None:
        """WebSocket 錯誤"""
        pass

    # ── 關閉 ─────────────────────────────────────────────────────────────────

    async def close(self) -> None:
        self._closed = True
        if self._ws:
            await self._ws.close()
            log("WS", "連線已關閉", C.Y)


# ══════════════════════════════════════════════════════════════════════════════
# 7. App  ── 三種執行模式的主迴圈（覆寫 GeminiSession handler 加入印出邏輯）
# ══════════════════════════════════════════════════════════════════════════════

class ConsoleSession(GeminiSession):
    """
    GeminiSession 的終端機版實作。
    覆寫所有 on_* handler 加入 log 輸出。
    """

    # ── 事件 handler ─────────────────────────────────────────────────────────

    def on_transcript_out(self, text: str, done: bool) -> None:
        if text:
            print(text, end="", flush=True)
        elif done and self._out_transcript:
            print()  # 回合結束換行

    def on_transcript_in(self, text: str, done: bool) -> None:
        if text:
            print(f"\r{C.Y}[你說] {text}{C.W}", flush=True)

    def on_interrupted(self) -> None:
        log("⚡", "打斷", C.Y)

    def on_generation_complete(self) -> None:
        log("GEN", "生成完畢", C.CY)

    def on_turn_complete(self, out_tr: str, in_tr: str) -> None:
        log("TURN", "回合結束 ✓", C.G)

    def on_sources(self, sources: list[dict]) -> None:
        log("SEARCH", "引用來源：", C.B)
        for s in sources:
            print(f"  {C.B}↗{C.W} {s['title']}  {C.M}{s['uri']}{C.W}")

    def on_tool_call(self, name: str, args: dict) -> dict:
        log("TOOL", f"呼叫 {name}({args})", C.M)
        result = ToolRegistry.execute(name, args)
        log("TOOL", f"結果 → {result}", C.M)
        return result

    def on_tool_cancelled(self, ids: list[str]) -> None:
        log("TOOL", f"取消 {ids}", C.Y)

    def on_go_away(self, seconds) -> None:
        log("WARN", f"伺服器將在 {seconds}s 後關閉，請重新執行", C.R)

    def on_session_token(self, token: str) -> None:
        log("SESSION", f"Resume token 已更新（{token[:12]}…）", C.CY)

    def on_usage(self, data: dict) -> None:
        total = data.get("totalTokenCount", "?")
        log("USAGE", f"總 token 用量: {total}", C.M)

    def on_error(self, code: int, reason: str) -> None:
        log("ERR", f"WebSocket 斷線: {code} {reason}", C.R)


# ── 文字對話模式 ──────────────────────────────────────────────────────────────

async def run_text(session: ConsoleSession) -> None:
    log("MODE", "📝 文字對話（quit 離開）", C.M)
    loop = asyncio.get_event_loop()

    # text 模式仍需播放器接收音訊（否則 WS 緩衝區卡住）
    player = None
    if HAS_PYAUDIO:
        player = AudioPlayer()
        session.player = player
        await player.start()
    else:
        log("WARN", "pyaudio 未安裝，音訊將被靜默丟棄", C.Y)

    recv_task = asyncio.create_task(session.recv_loop())

    try:
        while True:
            try:
                user = await loop.run_in_executor(
                    None, lambda: input(f"{C.CY}你: {C.W}")
                )
            except (EOFError, KeyboardInterrupt):
                break

            if user.strip().lower() in ("quit", "exit", "q", "bye"):
                log("INFO", "結束對話", C.Y)
                break
            if not user.strip():
                continue

            await session.send_text(user)
            print(f"{C.G}Gemini: {C.W}", end="", flush=True)

    finally:
        recv_task.cancel()
        if player:
            await player.stop()


# ── 語音對話模式 ──────────────────────────────────────────────────────────────

async def run_audio(session: ConsoleSession) -> None:
    if not HAS_PYAUDIO:
        log("ERR", "語音模式需要 pyaudio：pip install pyaudio", C.R)
        return

    log("MODE", "🎤 語音對話（Ctrl+C 結束）", C.M)
    loop   = asyncio.get_event_loop()
    player = AudioPlayer()
    mic    = Microphone()

    session.player = player
    await player.start()
    mic.start(loop)
    log("MIC", "麥克風已啟動，開始監聽...", C.G)

    async def mic_loop() -> None:
        while True:
            chunk = await mic.read()
            await session.send_audio(chunk)

    recv_task = asyncio.create_task(session.recv_loop())
    mic_task  = asyncio.create_task(mic_loop())

    try:
        await asyncio.gather(recv_task, mic_task)
    except asyncio.CancelledError:
        pass
    finally:
        recv_task.cancel()
        mic_task.cancel()
        mic.stop()
        await player.stop()


# ── 影像 + 語音對話模式 ───────────────────────────────────────────────────────

async def run_video(session: ConsoleSession) -> None:
    if not HAS_PYAUDIO:
        log("ERR", "影像模式需要 pyaudio：pip install pyaudio", C.R)
        return
    if not HAS_CV2:
        log("ERR", "影像模式需要 opencv-python：pip install opencv-python", C.R)
        return

    log("MODE", "📷 影像 + 語音（Ctrl+C 結束）", C.M)
    loop   = asyncio.get_event_loop()
    player = AudioPlayer()
    mic    = Microphone()
    cam    = Camera()

    session.player = player
    await player.start()
    mic.start(loop)
    cam.start(loop)

    async def mic_loop() -> None:
        while True:
            chunk = await mic.read()
            await session.send_audio(chunk)

    async def cam_loop() -> None:
        while True:
            frame = await cam.read()
            await session.send_video(frame)

    recv_task = asyncio.create_task(session.recv_loop())
    mic_task  = asyncio.create_task(mic_loop())
    cam_task  = asyncio.create_task(cam_loop())

    try:
        await asyncio.gather(recv_task, mic_task, cam_task)
    except asyncio.CancelledError:
        pass
    finally:
        for t in (recv_task, mic_task, cam_task):
            t.cancel()
        mic.stop()
        cam.stop()
        await player.stop()


# ══════════════════════════════════════════════════════════════════════════════
# 入口
# ══════════════════════════════════════════════════════════════════════════════

def parse_args() -> dict:
    args = sys.argv[1:]
    mode      = "text"
    voice     = Config.VOICE
    thinking  = None
    use_tools = False

    if "--audio"   in args: mode = "audio"
    if "--video"   in args: mode = "video"
    if "--tools"   in args: use_tools = True
    if "--thinking" in args:
        idx = args.index("--thinking")
        thinking = args[idx + 1] if idx + 1 < len(args) else "minimal"
    if "--voice" in args:
        idx = args.index("--voice")
        voice = args[idx + 1] if idx + 1 < len(args) else Config.VOICE

    return dict(mode=mode, voice=voice, thinking=thinking, use_tools=use_tools)


async def main() -> None:
    opts = parse_args()

    if not Config.API_KEY:
        log("ERR", "請設定 GEMINI_API_KEY 環境變數（或寫入 .env）", C.R)
        sys.exit(1)

    print(f"""
{C.B}╔══════════════════════════════════════════════════════╗
║      Gemini Live Core — 純後端客戶端                    ║
║      模型: {Config.MODEL:<38}║
╚══════════════════════════════════════════════════════╝{C.W}
  模式     : {opts['mode']}
  音色     : {opts['voice']}
  Thinking : {opts['thinking'] or 'OFF'}
  Tools    : {'ON' if opts['use_tools'] else 'OFF'}
""")

    session = ConsoleSession(**opts)

    try:
        await session.connect()
        if opts["mode"] == "video":
            await run_video(session)
        elif opts["mode"] == "audio":
            await run_audio(session)
        else:
            await run_text(session)
    except KeyboardInterrupt:
        log("INFO", "使用者中斷 (Ctrl+C)", C.Y)
    except websockets.exceptions.ConnectionClosedError as e:
        log("ERR", f"連線異常斷開: {e.code} {e.reason}", C.R)
    except Exception as e:
        log("ERR", f"{type(e).__name__}: {e}", C.R)
        raise
    finally:
        await session.close()


if __name__ == "__main__":
    asyncio.run(main())