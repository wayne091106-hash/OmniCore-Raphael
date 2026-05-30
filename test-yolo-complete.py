"""
Raphael Vision Gate  (v6 — 混合雙軌決策完整版)
==================================================
核心設計：混合（Hybrid）雙軌制，徹底解決大物誤判與小物漏抓的矛盾。

軌道一：慢速/大物體主軌 (Semantic Drift)
    - 使用穩定平滑的 CLIP 語義漂移 EMA (`_drift_ema > eff_th`)。
    - 負責捕捉一般人物走動、場景改變等大面積語義變化。

軌道二：高速小物體特判副軌 (Fast Motion Bypass)
    針對極速飛過、面積極小，且容易被 EMA 抹平的物體，採用「四重濾網」瞬間觸發：
    1. 空間濾網 (Inner Zone)：物體必須觸碰內圈 (預設距邊緣 15% 內)，防堵大物體從邊緣切入造成的殘缺碎塊被誤認。
    2. 大小濾網 (Max Area)：動態面積必須小於 3%，只針對真正的「小物體」特判。
    3. 物理濾網 (Optical Flow)：該區域的光流必須留下移動痕跡 (>3 像素)，過濾純相機雜訊。
    4. 語義濾網 (Raw Drift)：給予大範圍 Padding (80px) 提供上下文後，原始語義必須產生變化，過濾純光影閃爍。
    四重條件同時滿足 ➔ 繞過 EMA，1 幀瞬發進入 ARMED！

視覺反饋：
    - 主視窗：顯示彩色畫面與實時參數。
    - 光流遮罩：額外顯示黑白光流偵測圖，內含 INNER ZONE 內圈輔助線與 FLOW FOCUS 實時定位框。

環境: Python 3.12, CUDA 12.8, RTX 5070
"""

import time
import logging
import collections
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import cv2
import numpy as np
import torch
import clip
import mediapipe as mp
from ultralytics import YOLO
from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("RaphaelGate")


# ══════════════════════════════════════════════════════════════════════════════
# 設定
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class GateConfig:
    # ── 擷取 ──
    camera_source: int | str = 0
    target_fps: int = 10

    # ── 決策軸：CLIP 語義漂移 ──
    clip_model: str = "ViT-B/32"
    clip_device: str = "cuda" if torch.cuda.is_available() else "cpu"
    drift_threshold: float = 0.15      # drift = 1-cos；超過 = 語義改變夠多
    drift_ema_alpha: float = 0.4       # EMA 平滑係數（越大越靈敏越吵）
    new_object_th_factor: float = 0.6  # 有新物件時，門檻乘上此值（變積極）
    anchor_update_rate: float = 0.005  # Anchor 緩慢更新率，吸收一天中的光影漸變

    # ── 高速輔助通道：光流 (Optical Flow) ──
    enable_optical_flow: bool = True   # 開啟以捕捉小物體
    optical_flow_mag_threshold: float = 2.0  # 判定為移動的光流速度門檻
    fast_motion_max_area: float = 0.02       # 判定為小物體的面積上限 (佔全畫面比例)
    inner_zone_margin: float = 0.1          # 內圈邊界比例 (0.15 = 距四邊 15% 內為內圈)

    # ── 時機閘：穩定度 / 銳利度 ──
    stable_jitter: float = 0.04        # frame_diff 比例低於此 = 畫面夠穩，可拍
    sharp_buffer_size: int = 8         # 保留最近 N 幀供挑最銳利
    max_armed_wait: float = 2.0        # ARMED 等多久還不穩就強制拍（秒）

    # ── L1 frame diff 參數 ──
    diff_pixel_threshold: int = 15

    # ── 物件通道：YOLO ──
    yolo_model: str = "yolov8n.pt"
    yolo_conf: float = 0.45
    object_persist_frames: int = 2     # 新類別需連續出現幾幀才算數（去閃爍）

    # ── 情境：Pose（僅顯示，不參與決策）──
    enable_pose: bool = False          # 預設關閉以節省效能
    pose_model_complexity: int = 0
    pose_full_delta: float = 0.10
    pose_key_indices: tuple = (11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28)

    # ── 冷卻 ──
    cooldown_sec: float = 1

    # ── 顯示 ──
    show_debug_window: bool = True
    display_height: int = 560
    panel_width: int = 460


class GateState(Enum):
    WATCHING = "WATCHING"   # 監看語義漂移
    ARMED = "ARMED"         # 語義已變，等清晰幀
    COOLDOWN = "COOLDOWN"


# ══════════════════════════════════════════════════════════════════════════════
# 主體
# ══════════════════════════════════════════════════════════════════════════════

class RaphaelVisionGate:

    def __init__(self, config: GateConfig):
        self.cfg = config
        self.device = config.clip_device
        log.info(f"初始化（device={self.device}）")

        log.info("載入 YOLO...")
        self.yolo = YOLO(config.yolo_model)
        self.yolo.to(self.device)

        self.mp_pose = None
        if config.enable_pose:
            log.info("載入 MediaPipe Pose...")
            self.mp_pose = mp.solutions.pose.Pose(
                model_complexity=config.pose_model_complexity,
                min_detection_confidence=0.5, min_tracking_confidence=0.5,
            )

        log.info(f"載入 CLIP {config.clip_model}...")
        self.clip_model, self.clip_preprocess = clip.load(config.clip_model, device=self.device)
        self.clip_model.eval()

        # ── 狀態 ──
        self.state = GateState.WATCHING
        self._anchor_emb: Optional[torch.Tensor] = None
        self._anchor_frame: Optional[np.ndarray] = None
        self._drift_ema: float = 0.0
        self._prev_gray: Optional[np.ndarray] = None
        self._prev_gray_small: Optional[np.ndarray] = None # 供光流降解析度使用

        self._prev_keypoints: Optional[list] = None
        self._prev_yolo_classes: set = set()
        self._new_class_streak: dict = {}

        # ring buffer：(frame, sharpness)
        self._ring = collections.deque(maxlen=config.sharp_buffer_size)
        self._best_armed_frame: Optional[np.ndarray] = None
        self._best_armed_sharpness: float = -1.0

        self._armed_since: float = 0.0
        self._cooldown_start: float = 0.0
        self._last_trigger_time: float = 0.0
        self._trigger_count: int = 0
        self._last_fire_kind: str = ""

        log.info("初始化完成 ✓")

    def evaluate(self, frame: np.ndarray) -> dict:
        now = time.time()
        out = {
            "drift": 0.0, "drift_ema": 0.0, "eff_th": self.cfg.drift_threshold,
            "jitter": 0.0, "stable": False, "sharpness": 0.0,
            "new_objects": [], "pose_energy": 0.0,
            "state": self.state.value, "fired": False, "fire_kind": "",
            "cooldown_left": 0.0, "armed_wait": 0.0, "reason": "",
            "motion_box": None,
            "motion_mask": None,
            "flow_pixels": 0,
            "fast_motion_hit": False,
            "fast_motion_box": None,
            "roi_drift": 0.0
        }

        # ── 1. Jitter (動態遮罩) 與 Motion Bounding Box (動態聚焦) ───────────
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_b = cv2.GaussianBlur(gray, (5, 5), 0)
        
        jitter = 0.0
        motion_box = None
        raw_motion_box = None
        valid_boxes = []
        if self._prev_gray is not None:
            d = cv2.absdiff(self._prev_gray, gray_b)
            thresh = cv2.threshold(d, self.cfg.diff_pixel_threshold, 255, cv2.THRESH_BINARY)[1]
            non_zero = np.count_nonzero(thresh)
            jitter = float(non_zero) / d.size
            
            # 如果畫面有變動，抓出涵蓋動態範圍的 Bounding Box (ROI)
            if non_zero > 30: 
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for c in contours:
                    if cv2.contourArea(c) > 60:
                        valid_boxes.append(cv2.boundingRect(c))
                
                if valid_boxes:
                    x_min = min(b[0] for b in valid_boxes)
                    y_min = min(b[1] for b in valid_boxes)
                    x_max = max(b[0]+b[2] for b in valid_boxes)
                    y_max = max(b[1]+b[3] for b in valid_boxes)
                    
                    pad = 40  # 擴張一點範圍，包含周圍上下文
                    x_min_pad = max(0, x_min - pad)
                    y_min_pad = max(0, y_min - pad)
                    x_max_pad = min(frame.shape[1], x_max + pad)
                    y_max_pad = min(frame.shape[0], y_max + pad)
                    
                    # 動態區域介於 0.1% 到 70% 之間
                    area_ratio = (x_max_pad - x_min_pad) * (y_max_pad - y_min_pad) / (frame.shape[1] * frame.shape[0])
                    if 0.001 < area_ratio < 0.7:
                        motion_box = (x_min_pad, y_min_pad, x_max_pad, y_max_pad)
                        raw_motion_box = (x_min, y_min, x_max, y_max)
                        
        self._prev_gray = gray_b
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        out["jitter"] = jitter
        out["sharpness"] = sharpness
        out["motion_box"] = motion_box
        
        stable = jitter < self.cfg.stable_jitter
        out["stable"] = stable
        self._ring.append((frame.copy(), sharpness))

        # ── 1.5 光流輔助判斷 (如果有框，且框夠小，檢查該區域有無光流痕跡) ──
        fast_motion_hit = False
        flow_pixels = 0
        motion_mask = None
        fast_motion_box = None

        if self.cfg.enable_optical_flow:
            gray_small = cv2.resize(gray_b, (320, 240))
            if self._prev_gray_small is not None:
                flow = cv2.calcOpticalFlowFarneback(self._prev_gray_small, gray_small, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                
                # 二值化光流速度
                motion_mask = cv2.threshold(mag, self.cfg.optical_flow_mag_threshold, 255, cv2.THRESH_BINARY)[1].astype(np.uint8)
                
                if valid_boxes:
                    H, W = frame.shape[:2]
                    
                    # ── 分群：合併距離近的變動框，避免大物體碎裂成多個小框而誤判為小物體 ──
                    max_dist = 60  # 像素，視為同一物體的最大距離
                    merged_boxes = []
                    for b in valid_boxes:
                        bx1, by1, bx2, by2 = b[0], b[1], b[0]+b[2], b[1]+b[3]
                        merged_boxes.append([bx1, by1, bx2, by2])
                        
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
                                    new_merged[i] = [min(b_x1, mx1), min(b_y1, my1), max(b_x2, mx2), max(b_y2, my2)]
                                    merged_with_existing = True
                                    changed = True
                                    break
                            if not merged_with_existing:
                                new_merged.append(b)
                        merged_boxes = new_merged
                    
                    # 計算內圈邊界
                    ix1 = int(W * self.cfg.inner_zone_margin)
                    iy1 = int(H * self.cfg.inner_zone_margin)
                    ix2 = int(W * (1.0 - self.cfg.inner_zone_margin))
                    iy2 = int(H * (1.0 - self.cfg.inner_zone_margin))
                    
                    for mx1, my1, mx2, my2 in merged_boxes:
                        # 相交條件：不完全在外圍
                        intersect_inner = not (mx2 < ix1 or mx1 > ix2 or my2 < iy1 or my1 > iy2)
                        
                        # 這是分群後的完整物件面積
                        box_area_ratio = ((mx2 - mx1) * (my2 - my1)) / (W * H)
                        
                        # 框的大小足夠小，且有切到內圈
                        if intersect_inner and box_area_ratio < self.cfg.fast_motion_max_area:
                            # 映射到 320x240 小圖的座標
                            sx1 = int(mx1 * 320 / W)
                            sy1 = int(my1 * 240 / H)
                            sx2 = int(mx2 * 320 / W)
                            sy2 = int(my2 * 240 / H)
                            
                            # 取出框內的區域
                            roi_mask = motion_mask[sy1:sy2, sx1:sx2]
                            local_flow_pixels = np.count_nonzero(roi_mask)
                            
                            # 光流在該處留下白色小面積痕跡 (例如超過 3 個像素即代表有真實移動)
                            if local_flow_pixels >= 3:
                                fast_motion_hit = True
                                flow_pixels = max(flow_pixels, local_flow_pixels)
                                
                                pad = 40
                                fast_motion_box = (
                                    max(0, mx1 - pad),
                                    max(0, my1 - pad),
                                    min(W, mx2 + pad),
                                    min(H, my2 + pad)
                                )
                                break
                            
            self._prev_gray_small = gray_small
            
        out["fast_motion_hit"] = fast_motion_hit
        out["flow_pixels"] = flow_pixels
        out["motion_mask"] = motion_mask
        out["fast_motion_box"] = fast_motion_box

        # ── 2. CLIP 語義 embedding (動態導向語義聚焦 Dynamic Motion Crop) ────
        images_to_encode = [frame]
        
        # 決定要 encode 的焦點框 (如果有特判小框就用小框，否則用全域大框)
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

        # 全圖漂移
        drift_full = 1.0 - float(torch.cosine_similarity(curr_full_emb, self._anchor_emb, dim=1).item())
        max_drift_val = max(0.0, drift_full)
        
        # ROI 局部漂移 (如果存在)
        drift_roi = 0.0
        if target_crop_box is not None and len(embs) == 3:
            curr_roi_emb = embs[1:2]
            anchor_roi_emb = embs[2:3]
            drift_roi = 1.0 - float(torch.cosine_similarity(curr_roi_emb, anchor_roi_emb, dim=1).item())
            max_drift_val = max(max_drift_val, float(drift_roi))

        self._drift_ema = (self.cfg.drift_ema_alpha * max_drift_val
                           + (1 - self.cfg.drift_ema_alpha) * self._drift_ema)
        out["drift"] = max_drift_val
        out["drift_ema"] = self._drift_ema
        out["roi_drift"] = drift_roi

        # ── 有效門檻 ──
        eff_th = self.cfg.drift_threshold

        # ── 3. 第四重濾網 (語義濾網)：過濾純手腳揮動 ──
        # 如果是 fast_motion_hit，必須檢查其 ROI 的「真實語義變化」是否夠大
        # 因為手腳揮動時，加上 40px 的 Context，背景與人的特徵還是很相似 (drift_roi 低)
        # 但如果是新物體(球)飛入/飛出，drift_roi 會大幅飆高！
        if fast_motion_hit:
            # 只有局部語義變化 > 門檻的一半 (例如 > 0.075)，才承認這是一個「異物」，否則當作是身體動作或光影閃爍
            if drift_roi < (eff_th * 0.5):
                fast_motion_hit = False
                # 注意：這裡不把 fast_motion_box 設為 None，保留給 UI 顯示用，並用不同顏色標示被過濾
                out["reason"] = f"fast motion rejected (roi_drift {drift_roi:.3f} < {eff_th*0.5:.3f})"
        
        out["fast_motion_hit"] = fast_motion_hit

        # ── Pose（情境，僅顯示）────────────────────────────────────────────────
        out["pose_energy"] = 0.0
        if self.cfg.enable_pose and self.mp_pose is not None:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pose_res = self.mp_pose.process(rgb)
            pose_inst = 0.0
            if pose_res.pose_landmarks:
                kp = [(lm.x, lm.y, lm.visibility) for lm in pose_res.pose_landmarks.landmark]
                if self._prev_keypoints is not None:
                    ds = [(((kp[i][0]-self._prev_keypoints[i][0])**2 +
                            (kp[i][1]-self._prev_keypoints[i][1])**2)**0.5)
                          for i in self.cfg.pose_key_indices if kp[i][2] > 0.5]
                    pose_inst = max(ds) if ds else 0.0
                self._prev_keypoints = kp
            out["pose_energy"] = min(pose_inst / self.cfg.pose_full_delta, 1.0)

        # ── 物件通道：去閃爍的新類別偵測 ───────────────────────────────────────
        yolo_res = self.yolo(frame, verbose=False, conf=self.cfg.yolo_conf)[0]
        curr = {int(b.cls[0]) for b in yolo_res.boxes}
        raw_new = curr - self._prev_yolo_classes
        for c in raw_new:
            self._new_class_streak[c] = self._new_class_streak.get(c, 0) + 1
        for c in list(self._new_class_streak):
            if c not in curr:
                self._new_class_streak.pop(c, None)
        confirmed_new = [c for c, n in self._new_class_streak.items()
                         if n >= self.cfg.object_persist_frames]
        self._prev_yolo_classes = curr
        out["new_objects"] = [self.yolo.names[c] for c in confirmed_new]

        # ── 有效門檻（新物件壓低）──────────────────────────────────────────────
        eff_th = self.cfg.drift_threshold
        if confirmed_new:
            eff_th *= self.cfg.new_object_th_factor
        out["eff_th"] = eff_th

        # 核心觸發條件：語義超過門檻，或者光流偵測到小物體 (繞過 EMA)
        semantic_hit = (self._drift_ema > eff_th) or fast_motion_hit

        # ══ 狀態機 ════════════════════════════════════════════════════════════
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
                self._anchor_emb = (1 - self.cfg.anchor_update_rate) * self._anchor_emb + self.cfg.anchor_update_rate * curr_full_emb
                self._anchor_frame = cv2.addWeighted(frame.astype(np.float32), self.cfg.anchor_update_rate, self._anchor_frame, 1.0 - self.cfg.anchor_update_rate, 0.0)

            if semantic_hit:
                self.state = GateState.ARMED
                self._armed_since = now
                if fast_motion_hit:
                    kind = "fast_burst"
                elif confirmed_new:
                    kind = "object"
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

            # 取消條件：漂移掉回 70% 門檻下，且沒有光流強勢觸發
            leaving = (self._drift_ema < eff_th * 0.7) and not fast_motion_hit
            
            if stable or waited >= self.cfg.max_armed_wait or leaving:
                snap = self._best_armed_frame
                snap_sharp = self._best_armed_sharpness
                
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
                objs = (" +" + ",".join(out["new_objects"])) if out["new_objects"] else ""
                out["reason"] = (f"FIRE #{self._trigger_count} [{self._last_fire_kind}{objs}]"
                                 f"{forced} sharp={snap_sharp:.0f}")
                log.info(out["reason"])
                
                self._best_armed_frame = None
                self._best_armed_sharpness = -1.0
            else:
                out["reason"] = f"armed, waiting clean frame ({waited:.1f}s)"

        out["state"] = self.state.value
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


class DebugRenderer:
    C = {
        "bg": (22, 22, 30), "green": (90, 230, 120), "yellow": (60, 210, 235),
        "orange": (40, 150, 250), "red": (70, 70, 245), "cyan": (235, 210, 60),
        "white": (235, 235, 235), "gray": (130, 130, 140), "dim": (75, 75, 85),
    }

    def __init__(self, cfg: GateConfig):
        self.cfg = cfg
        self._flash = 0.0
        self._last_fire = ""

    def render(self, frame, r, since_trigger):
        if r.get("fired"):
            self._flash = time.time()
            self._last_fire = r.get("reason", "")

        H = self.cfg.display_height
        sc = H / frame.shape[0]
        disp = cv2.resize(frame, (int(frame.shape[1]*sc), H))

        fa = time.time() - self._flash
        if fa < 0.6:
            a = max(0.0, 0.55 - fa)
            ov = disp.copy(); ov[:] = (90, 230, 120)
            cv2.addWeighted(ov, a, disp, 1-a, 0, disp)
            cv2.rectangle(disp, (0, 0), (disp.shape[1]-1, H-1), self.C["green"], 6)

        motion_box = r.get("motion_box")
        if motion_box is not None:
            x1, y1, x2, y2 = motion_box
            cv2.rectangle(disp, (int(x1*sc), int(y1*sc)), (int(x2*sc), int(y2*sc)), self.C["cyan"], 2)
            cv2.putText(disp, "GLOBAL", (int(x1*sc), max(20, int(y1*sc)-8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.C["cyan"], 1, cv2.LINE_AA)

        fast_motion_box = r.get("fast_motion_box")
        if fast_motion_box is not None:
            fx1, fy1, fx2, fy2 = fast_motion_box
            # 如果被第四重語義濾網擋下，畫成灰色；如果有過，畫成紅色
            f_color = self.C["red"] if r.get("fast_motion_hit") else (100, 100, 100)
            cv2.rectangle(disp, (int(fx1*sc), int(fy1*sc)), (int(fx2*sc), int(fy2*sc)), f_color, 2)
            cv2.putText(disp, "FAST", (int(fx1*sc), max(20, int(fy1*sc)-8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, f_color, 1, cv2.LINE_AA)

        panel = np.full((H, self.cfg.panel_width, 3), self.C["bg"], np.uint8)
        self._panel(panel, r, since_trigger)
        return np.hstack([disp, panel])

    def _panel(self, p, r, since_trigger):
        W = self.cfg.panel_width

        def txt(s, y, x=14, c="white", sc=0.5, th=1):
            cv2.putText(p, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, sc, self.C[c], th, cv2.LINE_AA)

        def bar(val, y, c, mark=None, h=18):
            x0, x1 = 14, W-14
            cv2.rectangle(p, (x0, y), (x1, y+h), self.C["dim"], -1)
            fw = int(min(max(val, 0), 1.0)*(x1-x0))
            cv2.rectangle(p, (x0, y), (x0+fw, y+h), self.C[c], -1)
            if mark is not None:
                mx = x0 + int(min(mark, 1.0)*(x1-x0))
                cv2.line(p, (mx, y-3), (mx, y+h+3), self.C["white"], 2)
            txt(f"{val:.3f}", y+h-4, x=W-62, c="white", sc=0.4)
            return y+h+8

        y = 38
        txt("RAPHAEL VISION GATE", y, c="cyan", sc=0.6, th=2); y += 12
        cv2.line(p, (14, y), (W-14, y), self.C["gray"], 1); y += 32

        state = r.get("state", "WATCHING")
        scol = {"WATCHING": "gray", "ARMED": "orange", "COOLDOWN": "yellow"}.get(state, "white")
        txt(f"STATE: {state}", y, c=scol, sc=0.64, th=2); y += 6
        if state == "COOLDOWN":
            txt(f"  {r.get('cooldown_left',0):.1f}s", y+20, c="yellow", sc=0.48); y += 24
        elif state == "ARMED":
            txt(f"  waiting clean frame {r.get('armed_wait',0):.1f}s", y+20, c="orange", sc=0.44); y += 24
        y += 30

        # 決策軸：語義漂移
        txt("SEMANTIC DRIFT  (decision)", y, c="cyan", sc=0.52); y += 24
        dcol = "green" if r["drift_ema"] > r["eff_th"] else "yellow"
        y = bar(r["drift_ema"], y, dcol, mark=r["eff_th"], h=24)
        txt(f"  raw={r['drift']:.3f}  threshold={r['eff_th']:.3f}", y, c="dim", sc=0.4); y += 28

        cv2.line(p, (14, y), (W-14, y), self.C["dim"], 1); y += 26

        # 時機閘：穩定度
        txt("STABILITY  (when to shoot)", y, c="white", sc=0.46); y += 22
        stab = max(0.0, 1.0 - r["jitter"]/max(self.cfg.stable_jitter*3, 1e-6))
        scol2 = "green" if r["stable"] else "orange"
        y = bar(stab, y, scol2)
        txt(f"  jitter={r['jitter']:.3f}  {'STABLE' if r['stable'] else 'moving'}  sharp={r['sharpness']:.0f}",
            y, c="dim", sc=0.4); y += 26

        # 物件通道
        objs = r.get("new_objects", [])
        ocol = "green" if objs else "dim"
        txt(f"NEW OBJECTS: {', '.join(objs) if objs else '—'}", y, c=ocol, sc=0.46); y += 22

        # 輔助高速通道
        if self.cfg.enable_optical_flow:
            txt("FAST MOTION (flow inside ROI)", y, c="white", sc=0.46); y += 22
            f_hit = r.get("fast_motion_hit", False)
            f_pix = r.get("flow_pixels", 0)
            roi_drift = r.get("roi_drift", 0.0)
            fcol = "red" if f_hit else "dim"
            txt(f"  flow_pixels={f_pix}  trigger={f_hit}  roi_drift={roi_drift:.3f}", y, c=fcol, sc=0.4); y += 26

        # Pose
        txt("pose (context only)", y, c="dim", sc=0.42); y += 18
        y = bar(r["pose_energy"], y, "dim", h=12) + 8

        cv2.line(p, (14, y), (W-14, y), self.C["gray"], 1); y += 26

        if since_trigger is not None:
            txt(f"last trigger: {since_trigger:.1f}s ago", y, c="green", sc=0.48); y += 24
        if self._last_fire:
            txt("last fire:", y, c="gray", sc=0.4); y += 16
            for i in range(0, len(self._last_fire), 36):
                txt(self._last_fire[i:i+36], y, c="white", sc=0.4); y += 16
        y += 6
        rs = r.get("reason", "")
        if rs:
            txt("> " + rs[:38], y, c="dim", sc=0.4)


def run(cfg: GateConfig):
    gate = RaphaelVisionGate(cfg)
    renderer = DebugRenderer(cfg)

    cap = cv2.VideoCapture(cfg.camera_source)
    if not cap.isOpened():
        log.error(f"無法開啟攝影機：{cfg.camera_source}")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    interval = 1.0 / cfg.target_fps
    last_eval = 0.0
    last_result = {
        "drift": 0, "drift_ema": 0, "eff_th": cfg.drift_threshold, "jitter": 0,
        "stable": False, "sharpness": 0, "new_objects": [], "pose_energy": 0,
        "state": "WATCHING", "fired": False, "fire_kind": "",
        "cooldown_left": 0, "armed_wait": 0, "reason": "starting...",
        "motion_mask": None,
        "motion_box": None,
    }

    log.info(f"開始（{cfg.target_fps} fps）。Q=離開")
    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.05); continue
        now = time.time()
        if now - last_eval >= interval:
            last_eval = now
            last_result = gate.evaluate(frame)
        if cfg.show_debug_window:
            disp = renderer.render(frame, last_result, gate.seconds_since_trigger)
            cv2.imshow("Raphael Vision Gate", disp)
            
            # 附上黑白光流偵測圖 (B&W Optical Flow Mask) 視窗
            motion_mask = last_result.get("motion_mask")
            if motion_mask is not None:
                mask_disp = cv2.cvtColor(motion_mask, cv2.COLOR_GRAY2BGR)
                motion_box = last_result.get("motion_box")
                if motion_box is not None:
                    # 繪製 ROI 範圍
                    sc_x = 320.0 / frame.shape[1]
                    sc_y = 240.0 / frame.shape[0]
                    
                    # 繪製內圈 (Inner Zone) 輔助線 (暗灰色)
                    ix1 = int(320 * gate.cfg.inner_zone_margin)
                    iy1 = int(240 * gate.cfg.inner_zone_margin)
                    ix2 = int(320 * (1.0 - gate.cfg.inner_zone_margin))
                    iy2 = int(240 * (1.0 - gate.cfg.inner_zone_margin))
                    cv2.rectangle(mask_disp, (ix1, iy1), (ix2, iy2), (100, 100, 100), 1)
                    cv2.putText(mask_disp, "INNER ZONE", (ix1, iy1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (100, 100, 100), 1, cv2.LINE_AA)

                    x1 = int(motion_box[0] * sc_x)
                    y1 = int(motion_box[1] * sc_y)
                    x2 = int(motion_box[2] * sc_x)
                    y2 = int(motion_box[3] * sc_y)
                    cv2.rectangle(mask_disp, (x1, y1), (x2, y2), (235, 210, 60), 2)
                    cv2.putText(mask_disp, "GLOBAL", (x1, max(15, y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (235, 210, 60), 1, cv2.LINE_AA)
                
                fast_motion_box = last_result.get("fast_motion_box")
                if fast_motion_box is not None:
                    sc_x = 320.0 / frame.shape[1]
                    sc_y = 240.0 / frame.shape[0]
                    x1 = int(fast_motion_box[0] * sc_x)
                    y1 = int(fast_motion_box[1] * sc_y)
                    x2 = int(fast_motion_box[2] * sc_x)
                    y2 = int(fast_motion_box[3] * sc_y)
                    cv2.rectangle(mask_disp, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(mask_disp, "FAST", (x1, max(15, y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)
                
                cv2.imshow("Optical Flow Mask (B&W)", mask_disp)

            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break
    cap.release()
    cv2.destroyAllWindows()
    log.info(f"結束。總觸發 {gate._trigger_count} 次。")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Raphael Vision Gate v3")
    ap.add_argument("--camera", default=0)
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--cooldown", type=float, default=1.0, help="觸發後的冷卻時間(秒)")
    ap.add_argument("--drift-threshold", type=float, default=0.15, help="語義漂移觸發門檻")
    ap.add_argument("--stable-jitter", type=float, default=0.04, help="畫面穩定門檻")
    ap.add_argument("--fast-motion-max-area", type=float, default=0.03, help="小物體判定面積上限(預設0.03)")
    ap.add_argument("--inner-zone-margin", type=float, default=0.15, help="內圈邊緣比例(預設0.15)")
    ap.add_argument("--no-window", action="store_true")
    args = ap.parse_args()

    cam = args.camera
    try:
        cam = int(cam)
    except ValueError:
        pass

    cfg = GateConfig(
        camera_source=cam, target_fps=args.fps, cooldown_sec=args.cooldown,
        drift_threshold=args.drift_threshold, stable_jitter=args.stable_jitter,
        fast_motion_max_area=args.fast_motion_max_area,
        inner_zone_margin=args.inner_zone_margin,
        show_debug_window=not args.no_window,
    )
    run(cfg)
