"""
bridge.py — 統一收發匯流排
═══════════════════════════
所有模組透過 Bridge 溝通，不直接互相引用。

Channel enum 與 UI/app.js 的 applyMessage(channel, payload) 一一對應。

用法：
    bridge = Bridge()

    # 發送（任何模組皆可呼叫）
    await bridge.publish(Channel.TRANSCRIPT_OUT, {"text": "你好", "done": False})

    # 訂閱所有 channel（core / ui 用）
    sub = bridge.subscribe()
    async for channel, payload in sub:
        ...

    # 只聽特定 channel（感知模組用）
    async for payload in bridge.listen(Channel.VAD_EVENT):
        ...

    # 結束時取消訂閱
    bridge.unsubscribe(sub)
"""

import asyncio
from enum import Enum
from typing import AsyncIterator


class Channel(str, Enum):
    """
    訊息頻道。值 = app.js applyMessage 第一個參數。
    新增頻道時，同步更新 app.js 的 switch-case。
    """
    # ── 感知 → UI ─────────────────────────────────────────
    SENSOR_VIEW   = "sensor_view"     # 視覺幀 + VAD + gate 狀態
    VAD_EVENT     = "vad_event"       # VAD 即時音量 / speaking
    VISION_EVENT  = "vision_event"    # 視覺事件日誌（reason, detail）

    # ── 對話轉錄 ──────────────────────────────────────────
    TRANSCRIPT_IN  = "transcript_in"  # 使用者語音 ASR
    TRANSCRIPT_OUT = "transcript_out" # AI 回應字幕

    # ── 工具 ──────────────────────────────────────────────
    TOOL_CALL   = "tool_call"         # 工具呼叫（name, args）
    TOOL_RESULT = "tool_result"       # 工具結果（name, result）
    TASK_VOICE  = "task_voice"        # 工具任務狀態語音（收到、進度、完成、卡住）

    # ── 記憶 ──────────────────────────────────────────────
    MEMORY_WRITE = "memory_write"     # 記憶寫入（memory, category, importance）

    # ── 主動性 ────────────────────────────────────────────
    PROACTIVE = "proactive"           # 主動開口觸發原因

    # ── 音訊 ──────────────────────────────────────────────
    AUDIO_OUT = "audio_out"           # AI 音訊 PCM（base64）→ 前端播放

    # ── 控制 / 狀態 ──────────────────────────────────────
    INTERRUPTED = "interrupted"       # AI 被打斷
    STATUS      = "status"            # 連線狀態
    USAGE       = "usage"             # token / session 用量
    MEMORY_ACCOUNTS = "memory_accounts" # 記憶帳號清單 / 目前帳號
    ERROR       = "error"             # 錯誤訊息

    # ── 前端 → 後端指令 ──────────────────────────────────
    TEXT_IN       = "text_in"         # 使用者文字輸入
    AUDIO_IN      = "audio_in"       # 使用者麥克風 PCM
    VIDEO_IN      = "video_in"       # 使用者鏡頭 JPEG
    USER_INTERRUPT = "user_interrupt" # 使用者點擊打斷


class _Subscription:
    """一個訂閱者的獨立佇列，支援 async for 迭代。"""

    def __init__(self, channels: set[Channel] | None = None, maxsize: int = 256):
        self._queue: asyncio.Queue[tuple[Channel, dict]] = asyncio.Queue(maxsize=maxsize)
        self._channels = channels  # None = 全部

    def wants(self, channel: Channel) -> bool:
        return self._channels is None or channel in self._channels

    async def put(self, channel: Channel, payload) -> None:
        try:
            self._queue.put_nowait((channel, payload))
        except asyncio.QueueFull:
            # 背壓：丟棄最舊的訊息，防無限積壓
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
            self._queue.put_nowait((channel, payload))

    def __aiter__(self):
        return self

    async def __anext__(self) -> tuple[Channel, dict]:
        return await self._queue.get()


class Bridge:
    """
    中央匯流排。

    所有感知模組、核心、UI 都持有同一個 Bridge 實例。
    publish → fan-out 到所有符合的 subscriber。
    """

    def __init__(self):
        self._subs: list[_Subscription] = []
        self._lock = asyncio.Lock()

    async def publish(self, channel: Channel, payload) -> None:
        """發送訊息到指定頻道，fan-out 給所有訂閱者。"""
        async with self._lock:
            subs = list(self._subs)
        for sub in subs:
            if sub.wants(channel):
                await sub.put(channel, payload)

    def subscribe(self, channels: set[Channel] | None = None) -> _Subscription:
        """
        訂閱訊息。

        channels=None → 收全部頻道
        channels={Channel.VAD_EVENT, Channel.VISION_EVENT} → 只收指定頻道

        回傳 _Subscription，可 async for channel, payload in sub 迭代。
        """
        sub = _Subscription(channels)
        self._subs.append(sub)
        return sub

    def listen(self, channel: Channel) -> _Subscription:
        """subscribe 的便利版：只聽單一頻道。"""
        return self.subscribe({channel})

    def unsubscribe(self, sub: _Subscription) -> None:
        """取消訂閱，釋放資源。"""
        try:
            self._subs.remove(sub)
        except ValueError:
            pass
