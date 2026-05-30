"""
perception/vision.py — CLIP 語義漂移 + 光流混合視覺閘門
═══════════════════════════════════════════════
從 test-yolo-complete.py 的語意/差幀與光流核心改接 bridge。

核心邏輯（RaphaelVisionGate）完整保留：
  軌道一：CLIP 語義漂移 EMA（大物 / 場景變化）
  軌道二：光流四重濾網瞬發（高速小物體）

輸出頻道：
  Channel.SENSOR_VIEW   → {jpeg, gate_state, drift, fps_capture, fps_semantic, feedback_boxes}
  Channel.VISION_EVENT  → {jpeg, reason, detail}       （gate 觸發時）
  Channel.VIDEO_IN      → {jpeg: bytes}                 （每幀 JPEG，core 節流後送 Gemini）
  Channel.PROACTIVE     → {type, detail, score, boxes}  （畫面變化→交給 Gemini 判斷是否主動開口）

依賴：torch, opencv-python, clip, numpy
      選用：mediapipe（Pose，預設關閉）
"""

import asyncio
import base64
import collections
import logging
import queue
import sys
import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import cv2
import numpy as np
import torch
import clip
from PIL import Image

from bridge import Bridge, Channel

log = logging.getLogger("vision")


# ══════════════════════════════════════════════════════════════════════════════
# 設定（保留 test-yolo-complete.py 的語意/光流核心）
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class GateConfig:
    # ── 擷取 ──
    camera_source: int | str = 0
    target_fps: int = 10
    semantic_fps: int = 4
    capture_fps: int = 24
    capture_width: int = 1280
    capture_height: int = 720

    # ── 決策軸：CLIP 語義漂移 ──
    clip_model: str = "ViT-B/32"
    clip_device: str = "cuda" if torch.cuda.is_available() else "cpu"
    drift_threshold: float = 0.15
    drift_ema_alpha: float = 0.4
    new_object_th_factor: float = 0.6
    anchor_update_rate: float = 0.005

    # ── 高速輔助通道：光流 ──
    enable_optical_flow: bool = True
    optical_flow_mag_threshold: float = 2.0
    fast_motion_max_area: float = 0.02
    inner_zone_margin: float = 0.1

    # ── 時機閘：穩定度 / 銳利度 ──
    stable_jitter: float = 0.04
    sharp_buffer_size: int = 8
    max_armed_wait: float = 2.0

    # ── L1 frame diff ──
    diff_pixel_threshold: int = 15

    # ── 情境：Pose（僅顯示，不參與決策）──
    enable_pose: bool = False
    pose_model_complexity: int = 0
    pose_full_delta: float = 0.10
    pose_key_indices: tuple = (11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28)

    # ── 冷卻 ──
    cooldown_sec: float = 1
    proactive_min_gap: float = 2.0
    proactive_repeat_gap: float = 4.0

    # ── JPEG 輸出品質 ──
    # 高質檔位：送 Gemini 的單張畫面（gate 最佳幀 / 串流幀緩衝）拉高，提升 VLM 理解力。
    # analysis_width 是 CLIP 的輸入寬度，維持 640 以免拖慢 gate 推論。
    jpeg_quality: int = 90
    preview_width: int = 1024
    preview_height: int = 576
    analysis_width: int = 640
    video_publish_fps: int = 5
    enable_gate: bool = True
    render_feedback: bool = True
    emit_events: bool = True
    emit_proactive: bool = True


class GateState(Enum):
    WATCHING = "WATCHING"
    ARMED = "ARMED"
    COOLDOWN = "COOLDOWN"


def _normalize_box(box, shape: tuple[int, int]) -> dict | None:
    if box is None:
        return None
    h, w = shape
    if h <= 0 or w <= 0:
        return None
    x1, y1, x2, y2 = [float(v) for v in box]
    if x2 <= x1 or y2 <= y1:
        return None
    return {
        "x1": max(0.0, min(1.0, x1 / w)),
        "y1": max(0.0, min(1.0, y1 / h)),
        "x2": max(0.0, min(1.0, x2 / w)),
        "y2": max(0.0, min(1.0, y2 / h)),
    }


def _feedback_boxes(result: dict, shape: tuple[int, int]) -> list[dict]:
    boxes = []
    motion = _normalize_box(result.get("motion_box"), shape)
    if motion:
        boxes.append({
            **motion,
            "kind": "semantic",
            "label": "GLOBAL",
            "active": bool(result.get("drift_ema", 0.0) > result.get("eff_th", 1.0)),
        })
    fast = _normalize_box(result.get("fast_motion_box"), shape)
    if fast:
        hit = bool(result.get("fast_motion_hit"))
        boxes.append({
            **fast,
            "kind": "fast" if hit else "fast_rejected",
            "label": "FAST",
            "active": hit,
        })
    return boxes


def _proactive_type(kind: str) -> str:
    if kind == "object_motion":
        return "vision:object_motion"
    return f"vision:{kind or 'semantic'}"


def _proactive_payload(kind: str, detail: str, result: dict, frame_jpeg: bytes | None = None) -> dict:
    boxes = result.get("feedback_boxes") or []
    metrics = {
        "drift": round(float(result.get("drift_ema", result.get("drift", 0.0)) or 0.0), 4),
        "roi_drift": round(float(result.get("roi_drift", 0.0) or 0.0), 4),
        "jitter": round(float(result.get("jitter", 0.0) or 0.0), 4),
        "motion_area": round(float(result.get("motion_area", 0.0) or 0.0), 4),
        "flow_pixels": int(result.get("flow_pixels", 0) or 0),
        "objects": int(result.get("change_box_count", len(boxes)) or 0),
        "fast_motion_hit": bool(result.get("fast_motion_hit", False)),
    }
    score = metrics["roi_drift"] or metrics["drift"] or metrics["jitter"]
    payload = {
        "type": _proactive_type(kind),
        "detail": detail,
        "score": score,
        "boxes": boxes[:3],
        "metrics": metrics,
    }
    # gate 開火時挑出的最佳幀（最銳利、最穩定）；core 會把這張送進 Gemini，
    # 而不是隨手抓的串流幀，確保模型看到的就是 gate 認為值得看的畫面。
    if frame_jpeg:
        payload["frame_jpeg"] = frame_jpeg
        log.debug("_proactive_payload: frame_jpeg included (size=%d bytes)", len(frame_jpeg))
    else:
        log.warning("_proactive_payload: frame_jpeg is EMPTY! kind=%s detail=%s", kind, detail)
    return payload


def _flow_mask_preview(result: dict, frame_shape: tuple[int, int], cfg: GateConfig) -> str:
    mask = result.get("motion_mask")
    if mask is None:
        return ""
    try:
        mask_disp = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        mh, mw = mask_disp.shape[:2]
        fh, fw = frame_shape
        if fh <= 0 or fw <= 0:
            return ""

        ix1 = int(mw * cfg.inner_zone_margin)
        iy1 = int(mh * cfg.inner_zone_margin)
        ix2 = int(mw * (1.0 - cfg.inner_zone_margin))
        iy2 = int(mh * (1.0 - cfg.inner_zone_margin))
        cv2.rectangle(mask_disp, (ix1, iy1), (ix2, iy2), (100, 100, 100), 1)
        cv2.putText(mask_disp, "INNER ZONE", (ix1, max(10, iy1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (100, 100, 100), 1, cv2.LINE_AA)

        def draw_scaled(box, color, label):
            if box is None:
                return
            x1, y1, x2, y2 = box
            sx1 = int(x1 * mw / fw)
            sy1 = int(y1 * mh / fh)
            sx2 = int(x2 * mw / fw)
            sy2 = int(y2 * mh / fh)
            cv2.rectangle(mask_disp, (sx1, sy1), (sx2, sy2), color, 2)
            cv2.putText(mask_disp, label, (sx1, max(14, sy1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)

        draw_scaled(result.get("motion_box"), (235, 210, 60), "GLOBAL")
        fast_color = (0, 0, 255) if result.get("fast_motion_hit") else (100, 100, 100)
        draw_scaled(result.get("fast_motion_box"), fast_color, "FAST")

        ok, buf = cv2.imencode(".jpg", mask_disp, [cv2.IMWRITE_JPEG_QUALITY, 74])
        if not ok:
            return ""
        return base64.b64encode(buf.tobytes()).decode()
    except Exception:
        return ""


# ══════════════════════════════════════════════════════════════════════════════
# RaphaelVisionGate — 核心決策引擎（語意漂移 + 差幀 ROI + 光流）
# ══════════════════════════════════════════════════════════════════════════════

class RaphaelVisionGate:

    def __init__(self, config: GateConfig):
        self.cfg = config
        self.device = config.clip_device
        log.info("初始化（device=%s）", self.device)

        self.mp_pose = None
        if config.enable_pose:
            import mediapipe as mp_lib
            log.info("載入 MediaPipe Pose...")
            self.mp_pose = mp_lib.solutions.pose.Pose(
                model_complexity=config.pose_model_complexity,
                min_detection_confidence=0.5, min_tracking_confidence=0.5,
            )

        log.info("載入 CLIP %s...", config.clip_model)
        self.clip_model, self.clip_preprocess = clip.load(config.clip_model, device=self.device)
        self.clip_model.eval()

        # ── 狀態 ──
        self.state = GateState.WATCHING
        self._anchor_emb: Optional[torch.Tensor] = None
        self._anchor_frame: Optional[np.ndarray] = None
        self._prev_full_emb: Optional[torch.Tensor] = None
        self._drift_ema: float = 0.0
        self._prev_gray: Optional[np.ndarray] = None
        self._prev_gray_small: Optional[np.ndarray] = None

        self._prev_keypoints: Optional[list] = None

        self._ring = collections.deque(maxlen=config.sharp_buffer_size)
        self._best_armed_frame: Optional[np.ndarray] = None
        self._best_armed_sharpness: float = -1.0
        self._fired_best_frame: Optional[np.ndarray] = None  # 專門保存 FIRE 事件使用的最佳幀

        self._armed_since: float = 0.0
        self._cooldown_start: float = 0.0
        self._last_trigger_time: float = 0.0
        self._trigger_count: int = 0
        self._last_fire_kind: str = ""

        log.info("初始化完成")

    def evaluate(self, frame: np.ndarray) -> dict:
        now = time.time()
        out = {
            "drift": 0.0, "drift_ema": 0.0, "eff_th": self.cfg.drift_threshold,
            "jitter": 0.0, "stable": False, "sharpness": 0.0,
            "new_objects": [], "pose_energy": 0.0,
            "state": self.state.value, "fired": False, "fire_kind": "",
            "cooldown_left": 0.0, "armed_wait": 0.0, "reason": "",
            "motion_box": None, "motion_mask": None,
            "flow_pixels": 0, "fast_motion_hit": False, "fast_motion_box": None,
            "roi_drift": 0.0, "motion_area": 0.0, "motion_candidate": False,
            "feedback_boxes": [],
        }

        # ── 1. Jitter + Motion Bounding Box ──────────────────────────────
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_b = cv2.GaussianBlur(gray, (5, 5), 0)

        jitter = 0.0
        motion_box = None
        valid_boxes = []
        if self._prev_gray is not None:
            d = cv2.absdiff(self._prev_gray, gray_b)
            thresh = cv2.threshold(d, self.cfg.diff_pixel_threshold, 255, cv2.THRESH_BINARY)[1]
            non_zero = np.count_nonzero(thresh)
            jitter = float(non_zero) / d.size

            if non_zero > 30:
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for c in contours:
                    if cv2.contourArea(c) > 60:
                        valid_boxes.append(cv2.boundingRect(c))

                if valid_boxes:
                    x_min = min(b[0] for b in valid_boxes)
                    y_min = min(b[1] for b in valid_boxes)
                    x_max = max(b[0] + b[2] for b in valid_boxes)
                    y_max = max(b[1] + b[3] for b in valid_boxes)

                    pad = 40
                    x_min_pad = max(0, x_min - pad)
                    y_min_pad = max(0, y_min - pad)
                    x_max_pad = min(frame.shape[1], x_max + pad)
                    y_max_pad = min(frame.shape[0], y_max + pad)

                    area_ratio = (x_max_pad - x_min_pad) * (y_max_pad - y_min_pad) / (frame.shape[1] * frame.shape[0])
                    if 0.001 < area_ratio < 0.7:
                        motion_box = (x_min_pad, y_min_pad, x_max_pad, y_max_pad)
                        out["motion_area"] = float(area_ratio)

        self._prev_gray = gray_b
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        out["jitter"] = jitter
        out["sharpness"] = sharpness
        out["motion_box"] = motion_box
        out["motion_candidate"] = motion_box is not None

        stable = jitter < self.cfg.stable_jitter
        out["stable"] = stable
        self._ring.append((frame.copy(), sharpness))

        # ── 1.5 光流輔助判斷 ─────────────────────────────────────────────
        fast_motion_hit = False
        flow_pixels = 0
        motion_mask = None
        fast_motion_box = None

        if self.cfg.enable_optical_flow:
            gray_small = cv2.resize(gray_b, (320, 240))
            if self._prev_gray_small is not None:
                flow = cv2.calcOpticalFlowFarneback(
                    self._prev_gray_small, gray_small, None, 0.5, 3, 15, 3, 5, 1.2, 0,
                )
                mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                motion_mask = cv2.threshold(
                    mag, self.cfg.optical_flow_mag_threshold, 255, cv2.THRESH_BINARY,
                )[1].astype(np.uint8)

                if valid_boxes:
                    H, W = frame.shape[:2]

                    # 分群合併
                    max_dist = 60
                    merged_boxes = []
                    for b in valid_boxes:
                        merged_boxes.append([b[0], b[1], b[0] + b[2], b[1] + b[3]])

                    changed = True
                    while changed:
                        changed = False
                        new_merged = []
                        for b in merged_boxes:
                            b_x1, b_y1, b_x2, b_y2 = b
                            merged_with_existing = False
                            for i, m in enumerate(new_merged):
                                mx1, my1, mx2, my2 = m
                                dist_x = max(0, max(b_x1, mx1) - min(b_x2, mx2))
                                dist_y = max(0, max(b_y1, my1) - min(b_y2, my2))
                                if dist_x < max_dist and dist_y < max_dist:
                                    new_merged[i] = [
                                        min(b_x1, mx1), min(b_y1, my1),
                                        max(b_x2, mx2), max(b_y2, my2),
                                    ]
                                    merged_with_existing = True
                                    changed = True
                                    break
                            if not merged_with_existing:
                                new_merged.append(b)
                        merged_boxes = new_merged

                    ix1 = int(W * self.cfg.inner_zone_margin)
                    iy1 = int(H * self.cfg.inner_zone_margin)
                    ix2 = int(W * (1.0 - self.cfg.inner_zone_margin))
                    iy2 = int(H * (1.0 - self.cfg.inner_zone_margin))

                    for mx1, my1, mx2, my2 in merged_boxes:
                        intersect_inner = not (mx2 < ix1 or mx1 > ix2 or my2 < iy1 or my1 > iy2)
                        box_area_ratio = ((mx2 - mx1) * (my2 - my1)) / (W * H)

                        if intersect_inner and box_area_ratio < self.cfg.fast_motion_max_area:
                            sx1 = int(mx1 * 320 / W)
                            sy1 = int(my1 * 240 / H)
                            sx2 = int(mx2 * 320 / W)
                            sy2 = int(my2 * 240 / H)

                            roi_mask = motion_mask[sy1:sy2, sx1:sx2]
                            local_flow_pixels = np.count_nonzero(roi_mask)

                            if local_flow_pixels >= 15:
                                fast_motion_hit = True
                                flow_pixels = max(flow_pixels, local_flow_pixels)
                                p = 40
                                fast_motion_box = (
                                    max(0, mx1 - p), max(0, my1 - p),
                                    min(W, mx2 + p), min(H, my2 + p),
                                )
                                break

            self._prev_gray_small = gray_small

        out["fast_motion_hit"] = fast_motion_hit
        out["flow_pixels"] = flow_pixels
        out["motion_mask"] = motion_mask
        out["fast_motion_box"] = fast_motion_box

        # ── 2. CLIP 語義 embedding ───────────────────────────────────────
        images_to_encode = [frame]
        target_crop_box = fast_motion_box if fast_motion_box is not None else motion_box

        if target_crop_box is not None and self._anchor_frame is not None:
            x1, y1, x2, y2 = target_crop_box
            images_to_encode.append(frame[y1:y2, x1:x2])
            images_to_encode.append(self._anchor_frame[y1:y2, x1:x2].astype(np.uint8))

        embs = self._clip_encode(images_to_encode)
        curr_full_emb = embs[0:1]

        if self._anchor_emb is None:
            self._anchor_emb = curr_full_emb
            self._anchor_frame = frame.copy().astype(np.float32)

        drift_full = 1.0 - float(torch.cosine_similarity(curr_full_emb, self._anchor_emb, dim=1).item())
        max_drift_val = max(0.0, drift_full)

        drift_roi = 0.0
        if target_crop_box is not None and len(embs) == 3:
            curr_roi_emb = embs[1:2]
            anchor_roi_emb = embs[2:3]
            drift_roi = 1.0 - float(torch.cosine_similarity(curr_roi_emb, anchor_roi_emb, dim=1).item())
            max_drift_val = max(max_drift_val, float(drift_roi))

        self._drift_ema = (
            self.cfg.drift_ema_alpha * max_drift_val
            + (1 - self.cfg.drift_ema_alpha) * self._drift_ema
        )
        out["drift"] = max_drift_val
        out["drift_ema"] = self._drift_ema
        out["roi_drift"] = drift_roi

        # ── 有效門檻 ──
        eff_th = self.cfg.drift_threshold

        # ── 3. 第四重濾網（語義濾網）─────────────────────────────────────
        if fast_motion_hit:
            if drift_roi < (eff_th * 0.5):
                fast_motion_hit = False
                out["reason"] = f"fast motion rejected (roi_drift {drift_roi:.3f} < {eff_th * 0.5:.3f})"

        out["fast_motion_hit"] = fast_motion_hit

        # ── Pose（情境，僅顯示）──────────────────────────────────────────
        out["pose_energy"] = 0.0
        if self.cfg.enable_pose and self.mp_pose is not None:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pose_res = self.mp_pose.process(rgb)
            pose_inst = 0.0
            if pose_res.pose_landmarks:
                kp = [(lm.x, lm.y, lm.visibility) for lm in pose_res.pose_landmarks.landmark]
                if self._prev_keypoints is not None:
                    ds = [
                        ((kp[i][0] - self._prev_keypoints[i][0]) ** 2
                         + (kp[i][1] - self._prev_keypoints[i][1]) ** 2) ** 0.5
                        for i in self.cfg.pose_key_indices if kp[i][2] > 0.5
                    ]
                    pose_inst = max(ds) if ds else 0.0
                self._prev_keypoints = kp
            out["pose_energy"] = min(pose_inst / self.cfg.pose_full_delta, 1.0)

        eff_th = self.cfg.drift_threshold
        out["eff_th"] = eff_th

        semantic_hit = (self._drift_ema > eff_th) or fast_motion_hit

        # ══ 狀態機 ════════════════════════════════════════════════════════
        if self.state == GateState.COOLDOWN:
            left = self.cfg.cooldown_sec - (now - self._cooldown_start)
            out["cooldown_left"] = max(0.0, left)
            if left <= 0:
                self._anchor_emb = curr_full_emb
                self._anchor_frame = frame.copy().astype(np.float32)
                self._drift_ema = 0.0
                self.state = GateState.WATCHING
                out["reason"] = "cooldown end -> WATCHING (anchor reset)"
            else:
                out["reason"] = f"cooldown {left:.1f}s"

        elif self.state == GateState.WATCHING:
            if stable and not semantic_hit and self._drift_ema < eff_th * 0.5:
                self._anchor_emb = (
                    (1 - self.cfg.anchor_update_rate) * self._anchor_emb
                    + self.cfg.anchor_update_rate * curr_full_emb
                )
                self._anchor_frame = cv2.addWeighted(
                    frame.astype(np.float32), self.cfg.anchor_update_rate,
                    self._anchor_frame, 1.0 - self.cfg.anchor_update_rate, 0.0,
                )

            if semantic_hit:
                self.state = GateState.ARMED
                self._armed_since = now
                if fast_motion_hit:
                    kind = "fast_burst"
                else:
                    kind = "semantic"
                self._last_fire_kind = kind
                out["reason"] = f"armed ({kind}, drift={self._drift_ema:.3f}>{eff_th:.3f})"

                if self._ring:
                    self._best_armed_frame, self._best_armed_sharpness = max(self._ring, key=lambda x: x[1])
                else:
                    self._best_armed_frame, self._best_armed_sharpness = frame.copy(), sharpness

        elif self.state == GateState.ARMED:
            waited = now - self._armed_since
            out["armed_wait"] = waited

            if sharpness > self._best_armed_sharpness:
                self._best_armed_frame = frame.copy()
                self._best_armed_sharpness = sharpness

            leaving = (self._drift_ema < eff_th * 0.7) and not fast_motion_hit

            if stable or waited >= self.cfg.max_armed_wait or leaving:
                self._anchor_emb = curr_full_emb
                self._anchor_frame = frame.copy().astype(np.float32)
                self._drift_ema = 0.0
                self._last_trigger_time = now
                self._cooldown_start = now
                self._trigger_count += 1
                self.state = GateState.COOLDOWN
                out["fired"] = True
                out["fire_kind"] = self._last_fire_kind

                reason_prefix = "stable" if stable else ("leaving" if leaving else "timeout")
                forced = "" if stable else f" [{reason_prefix}]"
                out["reason"] = (
                    f"FIRE #{self._trigger_count} [{self._last_fire_kind}]"
                    f"{forced} sharp={self._best_armed_sharpness:.0f}"
                )
                log.info(out["reason"])

                # 保存一份獨立的 copy，避免後續清除 _best_armed_frame 導致 result["best_frame"] 失效
                self._fired_best_frame = self._best_armed_frame.copy() if self._best_armed_frame is not None else None
                out["best_frame"] = self._fired_best_frame
                log.debug("_analyze: FIRE best_frame saved id=%s is None=%s", id(self._fired_best_frame) if self._fired_best_frame is not None else None, self._fired_best_frame is None)
                self._best_armed_frame = None
                self._best_armed_sharpness = -1.0
            else:
                out["reason"] = f"armed, waiting clean frame ({waited:.1f}s)"

        out["state"] = self.state.value
        out["feedback_boxes"] = _feedback_boxes(out, frame.shape[:2])
        out["change_box_count"] = len(out["feedback_boxes"])
        return out

    def _clip_encode(self, images: list[np.ndarray]) -> torch.Tensor:
        tensors = []
        for img_arr in images:
            img_pil = Image.fromarray(cv2.cvtColor(img_arr, cv2.COLOR_BGR2RGB))
            tensors.append(self.clip_preprocess(img_pil))
        batch = torch.stack(tensors).to(self.device)
        with torch.no_grad():
            return self.clip_model.encode_image(batch).float()

    @property
    def seconds_since_trigger(self) -> Optional[float]:
        if self._last_trigger_time == 0:
            return None
        return time.time() - self._last_trigger_time


# ══════════════════════════════════════════════════════════════════════════════
# VisionModule — 接 bridge 的包裝層
# ══════════════════════════════════════════════════════════════════════════════

class VisionModule:
    """
    視覺感知模組。

    用法：
        vision = VisionModule(bridge, GateConfig())
        await vision.start()
        ...
        vision.stop()
    """

    def __init__(self, bridge: Bridge, config: GateConfig | None = None):
        self._bridge = bridge
        self.cfg = config or GateConfig()
        self._active = False
        self._loop: asyncio.AbstractEventLoop | None = None
        self._frame_q: queue.Queue[np.ndarray] = queue.Queue(maxsize=4)
        self._state_lock = threading.Lock()
        self._latest_view = {
            "gate_state": GateState.WATCHING.value,
            "drift": 0.0,
            "fps_capture": 0.0,
            "fps_semantic": 0.0,
            "objects": 0,
            "feedback_boxes": [],
            "triggered": False,
            "flow_pixels": 0,
            "roi_drift": 0.0,
            "jitter": 0.0,
            "stable": False,
            "reason": "",
        }
        self._last_event_signature = ""
        self._trigger_flash_until = 0.0
        self._last_proactive_emit = 0.0
        self._last_proactive_kind = ""

        # fps 計數
        self._capture_count = 0
        self._eval_count = 0
        self._fps_capture = 0.0
        self._fps_semantic = 0.0
        self._fps_capture_reset_time = 0.0
        self._fps_semantic_reset_time = 0.0
        self._last_eval_time = 0.0
        self._last_video_publish = 0.0
        self._last_preview_publish = 0.0
        self._last_semantic_queue = 0.0

    async def start(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._active = True
        now = time.time()
        self._fps_capture_reset_time = now
        self._fps_semantic_reset_time = now

        threading.Thread(target=self._capture_loop, daemon=True, name="vision-capture").start()
        threading.Thread(target=self._eval_loop, daemon=True, name="vision-eval").start()

        log.info(
            "Vision 啟動 (camera=%s, capture=%dfps, semantic=%dfps)",
            self.cfg.camera_source,
            self.cfg.capture_fps,
            self.cfg.semantic_fps,
        )

    def stop(self) -> None:
        self._active = False
        log.info("Vision 已停止")

    def update_config(self, values: dict) -> None:
        """套用 WebUI 傳來的感知設定；config 物件會被 Gate 即時讀取。"""
        if not isinstance(values, dict):
            return
        if "target_fps" in values:
            try:
                fps = max(1, min(30, int(values["target_fps"])))
                self.cfg.target_fps = fps
                self.cfg.semantic_fps = fps
            except (TypeError, ValueError):
                pass
        if "semantic_fps" in values:
            try:
                fps = max(1, min(30, int(values["semantic_fps"])))
                self.cfg.target_fps = fps
                self.cfg.semantic_fps = fps
            except (TypeError, ValueError):
                pass
        if "capture_fps" in values:
            try:
                self.cfg.capture_fps = max(15, min(60, int(values["capture_fps"])))
            except (TypeError, ValueError):
                pass
        if "video_publish_fps" in values:
            try:
                self.cfg.video_publish_fps = max(1, min(20, int(values["video_publish_fps"])))
            except (TypeError, ValueError):
                pass
        if "drift_threshold" in values:
            try:
                self.cfg.drift_threshold = max(0.01, min(1.0, float(values["drift_threshold"])))
            except (TypeError, ValueError):
                pass
        for key in ("enable_gate", "render_feedback", "emit_events", "emit_proactive"):
            if key in values:
                setattr(self.cfg, key, bool(values[key]))

    # ── 擷取執行緒 ──────────────────────────────────────────────────────

    def _capture_loop(self) -> None:
        backend = cv2.CAP_DSHOW if sys.platform.startswith("win") and isinstance(self.cfg.camera_source, int) else 0
        cap = cv2.VideoCapture(self.cfg.camera_source, backend) if backend else cv2.VideoCapture(self.cfg.camera_source)
        if not cap.isOpened():
            log.error("無法開啟攝影機：%s", self.cfg.camera_source)
            return
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FPS, self.cfg.capture_fps)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cfg.capture_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg.capture_height)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        while self._active:
            t0 = time.time()
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.05)
                continue

            self._capture_count += 1
            now = time.time()
            cap_dt = now - self._fps_capture_reset_time
            if cap_dt >= 1.0:
                self._fps_capture = self._capture_count / cap_dt
                self._capture_count = 0
                self._fps_capture_reset_time = now

            # 1. 降低語義分析佇列推幀頻率（配合 semantic_fps 節省 CPU 縮放計算）
            semantic_interval = 1.0 / max(1, int(self.cfg.semantic_fps))
            if now - self._last_semantic_queue >= semantic_interval:
                self._last_semantic_queue = now
                analysis_frame = self._resize_for_analysis(frame)
                if self._frame_q.full():
                    try:
                        self._frame_q.get_nowait()
                    except queue.Empty:
                        pass
                self._frame_q.put(analysis_frame)

            # 2. 降低預覽串流編碼與發送頻率（限制在 max 8 FPS，大幅降低 CPU 繪圖與 JPEG/Base64 編碼耗能）
            preview_interval = 1.0 / 8.0
            if now - self._last_preview_publish >= preview_interval:
                self._last_preview_publish = now
                display_frame = self._draw_gate_feedback(frame)
                jpeg_bytes, jpeg_b64 = self._encode_preview(display_frame)
                if jpeg_bytes:
                    self._publish_sensor_frame(jpeg_b64, jpeg_bytes, raw_frame=frame)

            interval = 1.0 / max(1, int(self.cfg.capture_fps))
            elapsed = time.time() - t0
            time.sleep(max(0.0, interval - elapsed))

        cap.release()

    # ── 評估執行緒（CLIP 語義漂移 + 光流，重量級推論在此）──────────────

    def _eval_loop(self) -> None:
        gate = RaphaelVisionGate(self.cfg)

        while self._active:
            semantic_interval = 1.0 / max(1, int(self.cfg.semantic_fps))
            now = time.time()
            wait = semantic_interval - (now - self._last_eval_time)
            if wait > 0:
                time.sleep(min(wait, semantic_interval))

            try:
                frame = self._frame_q.get(timeout=0.5)
            except queue.Empty:
                continue

            while True:
                try:
                    frame = self._frame_q.get_nowait()
                except queue.Empty:
                    break

            self._last_eval_time = time.time()
            if not self.cfg.enable_gate:
                with self._state_lock:
                    self._latest_view.update({
                        "gate_state": GateState.WATCHING.value,
                        "drift": 0.0,
                        "fps_semantic": 0.0,
                        "objects": 0,
                        "feedback_boxes": [],
                        "triggered": False,
                        "flow_pixels": 0,
                        "roi_drift": 0.0,
                        "jitter": 0.0,
                        "stable": False,
                        "fast_motion_hit": False,
                        "reason": "vision gate disabled",
                    })
                continue

            result = gate.evaluate(frame)
            self._eval_count += 1

            # 更新 fps 計數（每秒重算一次）
            now = time.time()
            dt = now - self._fps_semantic_reset_time
            if dt >= 1.0:
                self._fps_semantic = self._eval_count / dt
                self._eval_count = 0
                self._fps_semantic_reset_time = now

            loop = self._loop
            if loop is None or loop.is_closed():
                break

            with self._state_lock:
                if result.get("fired"):
                    self._trigger_flash_until = time.time() + 0.65
                self._latest_view.update({
                    "gate_state": result["state"],
                    "drift": result["drift_ema"],
                    "fps_capture": self._fps_capture,
                    "fps_semantic": self._fps_semantic,
                    "objects": result.get("change_box_count", 0),
                    "feedback_boxes": result.get("feedback_boxes", []),
                    "triggered": time.time() < self._trigger_flash_until,
                    "flow_pixels": result.get("flow_pixels", 0),
                    "roi_drift": result.get("roi_drift", 0.0),
                    "jitter": result.get("jitter", 0.0),
                    "stable": result.get("stable", False),
                    "fast_motion_hit": result.get("fast_motion_hit", False),
                    "reason": result.get("reason", ""),
                })

            # ── gate 觸發時：VISION_EVENT + PROACTIVE ────────────────────
            if result["fired"]:
                best = result.get("best_frame")
                log.debug("_analyze_frame: FIRE best_frame id=%s is None=%s", id(best) if best is not None else None, best is None)
                fire_bytes, fire_jpeg = self._encode_preview(best if best is not None else frame)

                kind = result["fire_kind"]
                objs = result.get("new_objects", [])
                detail = result["reason"]
                log.debug("_analyze_frame: detail=%s", detail[:80] if detail else None)

                asyncio.run_coroutine_threadsafe(
                    self._publish_fire(kind, detail, fire_jpeg, objs, result, fire_bytes),
                    loop,
                ).result(timeout=5.0)
            else:
                self._publish_state_event(result)
                self._publish_motion_proactive(result)

    def _encode_preview(self, frame: np.ndarray) -> tuple[bytes, str]:
        h, w = frame.shape[:2]
        max_w = max(1, int(self.cfg.preview_width))
        max_h = max(1, int(self.cfg.preview_height))
        scale = min(max_w / w, max_h / h, 1.0)
        out_w = max(1, int(w * scale))
        out_h = max(1, int(h * scale))
        preview = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA) if scale != 1.0 else frame
        ok, buf = cv2.imencode(".jpg", preview, [cv2.IMWRITE_JPEG_QUALITY, self.cfg.jpeg_quality])
        if not ok:
            return b"", ""
        jpeg_bytes = buf.tobytes()
        return jpeg_bytes, base64.b64encode(jpeg_bytes).decode()

    def _resize_for_analysis(self, frame: np.ndarray) -> np.ndarray:
        max_w = max(1, int(self.cfg.analysis_width))
        h, w = frame.shape[:2]
        if w <= max_w:
            return frame.copy()
        scale = max_w / w
        return cv2.resize(frame, (max_w, max(1, int(h * scale))), interpolation=cv2.INTER_AREA)

    def _draw_gate_feedback(self, frame: np.ndarray) -> np.ndarray:
        cfg = getattr(self, "cfg", None)
        if cfg is not None and not getattr(cfg, "render_feedback", True):
            return frame
        with self._state_lock:
            boxes = list(self._latest_view.get("feedback_boxes") or [])
            triggered = time.time() < self._trigger_flash_until
        if not boxes and not triggered:
            return frame
        out = frame.copy()
        h, w = out.shape[:2]
        if triggered:
            overlay = out.copy()
            overlay[:] = (90, 230, 120)
            cv2.addWeighted(overlay, 0.35, out, 0.65, 0, out)
            cv2.rectangle(out, (0, 0), (w - 1, h - 1), (90, 230, 120), max(3, round(w / 140)))
        thickness = max(2, round(w / 420))
        font_scale = max(0.45, min(0.75, w / 1100))
        for box in boxes:
            try:
                x1 = int(float(box.get("x1", 0)) * w)
                y1 = int(float(box.get("y1", 0)) * h)
                x2 = int(float(box.get("x2", 0)) * w)
                y2 = int(float(box.get("y2", 0)) * h)
                kind = str(box.get("kind", "semantic"))
                text = str(box.get("label", "GLOBAL"))
                color = (235, 210, 60) if kind == "semantic" else ((70, 70, 245) if kind == "fast" else (100, 100, 100))
                cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
                (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
                ty = max(th + 6, y1)
                cv2.rectangle(out, (x1, ty - th - 8), (x1 + tw + 8, ty + 4), (0, 0, 0), -1)
                cv2.putText(out, text, (x1 + 4, ty - 3), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 1, cv2.LINE_AA)
            except Exception:
                continue
        return out

    def _publish_sensor_frame(self, jpeg_b64: str, jpeg_bytes: bytes, raw_frame: np.ndarray | None = None) -> None:
        loop = self._loop
        if loop is None or loop.is_closed():
            return
        with self._state_lock:
            view = dict(self._latest_view)
            view.update({
                "jpeg": jpeg_b64,
                "fps_capture": self._fps_capture,
                "fps_semantic": self._fps_semantic,
                "triggered": time.time() < self._trigger_flash_until,
                "rendered_feedback": True,
            })
        asyncio.run_coroutine_threadsafe(self._bridge.publish(Channel.SENSOR_VIEW, view), loop)
        now = time.time()
        video_interval = 1.0 / max(1, int(self.cfg.video_publish_fps))
        if now - self._last_video_publish >= video_interval:
            self._last_video_publish = now
            if raw_frame is not None:
                raw_bytes, _ = self._encode_preview(raw_frame)
                if raw_bytes:
                    jpeg_bytes = raw_bytes
            asyncio.run_coroutine_threadsafe(self._bridge.publish(Channel.VIDEO_IN, {"jpeg": jpeg_bytes}), loop)

    def _publish_state_event(self, result: dict) -> None:
        if not self.cfg.emit_events:
            return
        reason = result.get("reason", "")
        state = result.get("state", "")
        if not reason:
            return
        if not (reason.startswith("armed (") or reason.startswith("cooldown end")):
            return
        signature = f"{state}:{reason}"
        if signature == self._last_event_signature:
            return
        self._last_event_signature = signature
        loop = self._loop
        if loop is None or loop.is_closed():
            return
        event_reason = "armed" if reason.startswith("armed (") else "watching"
        asyncio.run_coroutine_threadsafe(
            self._bridge.publish(Channel.VISION_EVENT, {
                "reason": event_reason,
                "detail": reason,
            }),
            loop,
        )


    # gate fire_kind → app.js EVT_MAP key
    _KIND_TO_REASON = {
        "semantic":   "semantic",
        "fast_burst": "fast_motion",
        "object_motion": "object_appeared",
    }

    def _should_emit_proactive(self, kind: str) -> bool:
        now = time.time()
        min_gap = max(0.2, float(getattr(self.cfg, "proactive_min_gap", 2.0)))
        repeat_gap = max(min_gap, float(getattr(self.cfg, "proactive_repeat_gap", 4.0)))
        elapsed = now - self._last_proactive_emit
        if elapsed < min_gap:
            return False
        if kind == self._last_proactive_kind and elapsed < repeat_gap:
            return False
        self._last_proactive_emit = now
        self._last_proactive_kind = kind
        return True

    def _publish_motion_proactive(self, result: dict) -> None:
        if not self.cfg.emit_proactive or not result.get("motion_candidate"):
            return
        kind = "fast_burst" if result.get("fast_motion_hit") else "object_motion"
        if not self._should_emit_proactive(kind):
            return
        loop = self._loop
        if loop is None or loop.is_closed():
            return
        detail = result.get("reason") or "motion candidate"
        asyncio.run_coroutine_threadsafe(
            self._bridge.publish(Channel.PROACTIVE, _proactive_payload(kind, detail, result)),
            loop,
        )

    async def _publish_fire(
        self, kind: str, detail: str, jpeg_b64: str, objects: list[str], result: dict,
        frame_jpeg: bytes | None = None,
    ) -> None:
        if not self.cfg.emit_events:
            return
        evt_reason = self._KIND_TO_REASON.get(kind, kind)

        await self._bridge.publish(Channel.VISION_EVENT, {
            "jpeg": jpeg_b64,
            "reason": evt_reason,
            "detail": detail,
        })

        if self.cfg.emit_proactive:
            # FIRE 綠框事件屬高價值事件，不應在感知層被 _should_emit_proactive(kind) 的冷卻時間（如 2-4 秒）丟棄。
            # 直接發送 PROACTIVE 給 core.py，由核心主對話模組的 ProactiveGovernor 進行全域決策。
            await self._bridge.publish(Channel.PROACTIVE, _proactive_payload(kind, detail, result, frame_jpeg))


class BrowserVisionAnalyzer:
    """Semantic drift + optical-flow path for frames supplied by the WebUI browser camera."""

    _KIND_TO_REASON = {
        "semantic": "semantic",
        "fast_burst": "fast_motion",
        "object_motion": "object_appeared",
    }

    def __init__(self, bridge: Bridge, config: GateConfig | None = None):
        self._bridge = bridge
        self.cfg = config or GateConfig()
        self._active = False
        self._loop: asyncio.AbstractEventLoop | None = None
        self._frame_q: queue.Queue[bytes] = queue.Queue(maxsize=2)
        self._gate: RaphaelVisionGate | None = None
        self._eval_count = 0
        self._fps_semantic = 0.0
        self._fps_reset_time = 0.0
        self._last_eval_time = 0.0
        self._last_error = ""
        self._trigger_flash_until = 0.0
        self._last_event_signature = ""
        self._last_proactive_emit = 0.0
        self._last_proactive_kind = ""

    async def start(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._active = True
        self._fps_reset_time = time.time()
        threading.Thread(target=self._loop_worker, daemon=True, name="browser-vision-gate").start()
        log.info("Browser Vision Gate 已啟動 (semantic=%dfps)", self.cfg.semantic_fps)

    def stop(self) -> None:
        self._active = False
        log.info("Browser Vision Gate 已停止")

    def update_config(self, values: dict) -> None:
        if not isinstance(values, dict):
            return
        for key in ("target_fps", "semantic_fps"):
            if key in values:
                try:
                    self.cfg.semantic_fps = max(1, min(30, int(values[key])))
                except (TypeError, ValueError):
                    pass
        if "analysis_width" in values:
            try:
                self.cfg.analysis_width = max(320, min(1280, int(values["analysis_width"])))
            except (TypeError, ValueError):
                pass
        for key in ("enable_gate", "render_feedback", "emit_events", "emit_proactive"):
            if key in values:
                setattr(self.cfg, key, bool(values[key]))

    def submit_jpeg(self, jpeg_bytes: bytes) -> None:
        if not self._active or not jpeg_bytes or not self.cfg.enable_gate:
            return
        if self._frame_q.full():
            try:
                self._frame_q.get_nowait()
            except queue.Empty:
                pass
        try:
            self._frame_q.put_nowait(jpeg_bytes)
        except queue.Full:
            pass

    def _ensure_gate(self) -> RaphaelVisionGate:
        if self._gate is None:
            log.info("載入 Browser Vision Gate...")
            self._gate = RaphaelVisionGate(self.cfg)
        return self._gate

    def _loop_worker(self) -> None:
        while self._active:
            interval = 1.0 / max(1, int(self.cfg.semantic_fps))
            wait = interval - (time.time() - self._last_eval_time)
            if wait > 0:
                time.sleep(min(wait, interval))

            try:
                jpeg_bytes = self._frame_q.get(timeout=0.5)
            except queue.Empty:
                continue

            while True:
                try:
                    jpeg_bytes = self._frame_q.get_nowait()
                except queue.Empty:
                    break

            self._last_eval_time = time.time()
            try:
                frame = cv2.imdecode(np.frombuffer(jpeg_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
                if frame is None:
                    continue
                analysis = self._resize_for_analysis(frame)
                gate = self._ensure_gate()
                result = gate.evaluate(analysis)
                result["_frame_shape"] = analysis.shape[:2]
                if result.get("fired"):
                    self._trigger_flash_until = time.time() + 0.65
                self._eval_count += 1
                now = time.time()
                dt = now - self._fps_reset_time
                if dt >= 1.0:
                    self._fps_semantic = self._eval_count / dt
                    self._eval_count = 0
                    self._fps_reset_time = now
                self._publish(result, jpeg_bytes)
            except Exception as e:
                msg = str(e)
                if msg != self._last_error:
                    self._last_error = msg
                    log.warning("Browser Vision Gate 偵測失敗: %s", e)

    def _resize_for_analysis(self, frame: np.ndarray) -> np.ndarray:
        max_w = max(1, int(self.cfg.analysis_width))
        h, w = frame.shape[:2]
        if w <= max_w:
            return frame
        scale = max_w / w
        return cv2.resize(frame, (max_w, max(1, int(h * scale))), interpolation=cv2.INTER_AREA)

    def _should_emit_proactive(self, kind: str) -> bool:
        now = time.time()
        min_gap = max(0.2, float(getattr(self.cfg, "proactive_min_gap", 2.0)))
        repeat_gap = max(min_gap, float(getattr(self.cfg, "proactive_repeat_gap", 4.0)))
        elapsed = now - self._last_proactive_emit
        if elapsed < min_gap:
            return False
        if kind == self._last_proactive_kind and elapsed < repeat_gap:
            return False
        self._last_proactive_emit = now
        self._last_proactive_kind = kind
        return True

    def _publish(self, result: dict, jpeg_bytes: bytes) -> None:
        loop = self._loop
        if loop is None or loop.is_closed():
            return
        boxes = result.get("feedback_boxes", []) if self.cfg.render_feedback else []
        asyncio.run_coroutine_threadsafe(
            self._bridge.publish(Channel.SENSOR_VIEW, {
                "gate_state": result.get("state", GateState.WATCHING.value),
                "drift": result.get("drift_ema", 0.0),
                "fps_semantic": self._fps_semantic,
                "objects": len(boxes),
                "feedback_boxes": boxes,
                "triggered": time.time() < self._trigger_flash_until,
                "flow_pixels": result.get("flow_pixels", 0),
                "roi_drift": result.get("roi_drift", 0.0),
                "jitter": result.get("jitter", 0.0),
                "stable": result.get("stable", False),
                "fast_motion_hit": result.get("fast_motion_hit", False),
                "reason": result.get("reason", ""),
                "source": "browser_vision_gate",
            }),
            loop,
        )
        if result.get("fired"):
            kind = result.get("fire_kind", "")
            if self.cfg.emit_events:
                reason = self._KIND_TO_REASON.get(kind, kind)
                asyncio.run_coroutine_threadsafe(
                    self._bridge.publish(Channel.VISION_EVENT, {
                        "jpeg": base64.b64encode(jpeg_bytes).decode(),
                        "reason": reason,
                        "detail": result.get("reason", ""),
                    }),
                    loop,
                )
            # FIRE 綠框事件屬高價值事件，不應在感知層被 _should_emit_proactive 的冷卻時間丟棄。
            # 直接發送 PROACTIVE 給 core.py，由核心主對話模組的 ProactiveGovernor 進行全域決策。
            if self.cfg.emit_proactive:
                asyncio.run_coroutine_threadsafe(
                    self._bridge.publish(Channel.PROACTIVE, _proactive_payload(kind, result.get("reason", ""), result, jpeg_bytes)),
                    loop,
                ).result(timeout=5.0)
            return

        if self.cfg.emit_proactive and result.get("motion_candidate"):
            kind = "fast_burst" if result.get("fast_motion_hit") else "object_motion"
            if self._should_emit_proactive(kind):
                asyncio.run_coroutine_threadsafe(
                    self._bridge.publish(
                        Channel.PROACTIVE,
                        _proactive_payload(kind, result.get("reason", "motion candidate"), result),
                    ),
                    loop,
                )

        if self.cfg.emit_events:
            reason = result.get("reason", "")
            state = result.get("state", "")
            if reason and (reason.startswith("armed (") or reason.startswith("cooldown end")):
                signature = f"{state}:{reason}"
                if signature != self._last_event_signature:
                    self._last_event_signature = signature
                    asyncio.run_coroutine_threadsafe(
                        self._bridge.publish(Channel.VISION_EVENT, {
                            "reason": "armed" if reason.startswith("armed (") else "watching",
                            "detail": reason,
                        }),
                        loop,
                    )
