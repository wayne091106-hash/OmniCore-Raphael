"""
perception/vad.py — Silero VAD 語音活動偵測
════════════════════════════════════════════
從 test-vad.py 改接 bridge。

功能：
  1. sounddevice 擷取麥克風音訊（16kHz mono float32）
  2. 軟體增益 + Silero VAD 即時推論（背景執行緒，不阻塞 event loop）
  3. 發佈 VAD 狀態 + 原始 PCM16 到 bridge

輸出頻道：
  Channel.VAD_EVENT  → {speaking: bool, probability: float}
  Channel.AUDIO_IN   → {pcm: bytes}   (PCM16 16kHz mono, 所有音訊皆送，不做 gate)

依賴：torch, numpy, sounddevice
"""

import asyncio
import queue
import struct
import threading
import logging

import numpy as np
import sounddevice as sd
import torch

from bridge import Bridge, Channel

log = logging.getLogger("vad")

# ── 預設參數（可由外部覆寫）──────────────────────────────────
SAMPLE_RATE = 16_000
CHUNK_SIZE = 512       # 32ms per chunk
VAD_THRESHOLD = 0.55
GAIN_FACTOR = 2.5


def _find_input_device() -> int | None:
    """優先綁定 NVIDIA Broadcast；找不到就回傳 None（用系統預設）。"""
    devices = sd.query_devices()
    for i, dev in enumerate(devices):
        if "NVIDIA Broadcast" in dev["name"] and dev["max_input_channels"] > 0:
            log.info("綁定 NVIDIA Broadcast：[%d] %s", i, dev["name"])
            return i
    log.info("未偵測 NVIDIA Broadcast，使用系統預設麥克風")
    return None


def _load_silero():
    """載入 Silero VAD 模型（首次會從網路下載）。"""
    log.info("載入 Silero VAD 模型...")
    model, _ = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        force_reload=False,
        trust_repo=True,
    )
    log.info("Silero VAD 載入完成")
    return model


def _float32_to_pcm16(arr: np.ndarray) -> bytes:
    """float32 [-1,1] → PCM16 little-endian bytes。"""
    pcm = np.clip(arr, -1.0, 1.0)
    pcm = (pcm * 32767).astype(np.int16)
    return pcm.tobytes()


class VadModule:
    """
    VAD 感知模組。

    用法：
        vad = VadModule(bridge)
        await vad.start()      # 開始擷取 + 推論
        ...
        vad.stop()             # 結束
    """

    def __init__(
        self,
        bridge: Bridge,
        *,
        threshold: float = VAD_THRESHOLD,
        gain: float = GAIN_FACTOR,
        device: int | None = None,
    ):
        self._bridge = bridge
        self._threshold = threshold
        self._gain = gain
        self._device = device       # None = auto-detect

        self._raw_q: queue.Queue[np.ndarray] = queue.Queue()
        self._active = False
        self._loop: asyncio.AbstractEventLoop | None = None
        self._model = None

    # ── 啟動 / 停止 ──────────────────────────────────────────

    async def start(self) -> None:
        self._loop = asyncio.get_running_loop()

        if self._device is None:
            self._device = _find_input_device()

        self._model = await self._loop.run_in_executor(None, _load_silero)

        self._active = True
        self._stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            blocksize=CHUNK_SIZE,
            device=self._device,
            dtype="float32",
            channels=1,
            callback=self._audio_callback,
        )
        self._stream.start()

        threading.Thread(
            target=self._process_loop, daemon=True, name="vad-process"
        ).start()

        log.info(
            "VAD 啟動 (sr=%d, chunk=%d, th=%.2f, gain=%.1f)",
            SAMPLE_RATE, CHUNK_SIZE, self._threshold, self._gain,
        )

    def stop(self) -> None:
        self._active = False
        try:
            self._stream.stop()
            self._stream.close()
        except Exception:
            pass
        log.info("VAD 已停止")

    # ── sounddevice callback（音訊驅動執行緒）────────────────

    def _audio_callback(self, indata, frames, time_info, status):
        if status:
            log.warning("音訊驅動回報: %s", status)
        self._raw_q.put(indata.copy())

    # ── 處理執行緒（VAD 推論 + 發佈到 bridge）───────────────

    def _process_loop(self) -> None:
        while self._active:
            try:
                raw = self._raw_q.get(timeout=0.5)
            except queue.Empty:
                continue

            processed = np.clip(raw * self._gain, -1.0, 1.0)
            tensor = torch.from_numpy(processed).view(1, -1)
            prob = self._model(tensor, SAMPLE_RATE).item()
            speaking = prob >= self._threshold

            pcm_bytes = _float32_to_pcm16(processed.flatten())

            loop = self._loop
            if loop is None or loop.is_closed():
                break

            asyncio.run_coroutine_threadsafe(
                self._publish(speaking, prob, pcm_bytes), loop
            )

    async def _publish(self, speaking: bool, prob: float, pcm: bytes) -> None:
        await self._bridge.publish(Channel.VAD_EVENT, {
            "speaking": speaking,
            "probability": prob,
        })
        await self._bridge.publish(Channel.AUDIO_IN, {
            "pcm": pcm,
            "speaking": speaking,
            "probability": prob,
        })
