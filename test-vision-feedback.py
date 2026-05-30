"""
Minimal checks for the WebUI Vision Gate feedback contract.

This intentionally verifies the semantic-drift / optical-flow box format,
not YOLO object detection.
"""

import base64
import collections
import threading
import time
import types

import numpy as np
import torch

from perception.vision import GateConfig, GateState, RaphaelVisionGate, VisionModule, _feedback_boxes, _flow_mask_preview


def test_feedback_boxes_contract():
    result = {
        "motion_box": (10, 20, 110, 120),
        "fast_motion_box": (30, 40, 70, 90),
        "fast_motion_hit": True,
        "drift_ema": 0.22,
        "eff_th": 0.15,
    }
    boxes = _feedback_boxes(result, (200, 300))
    assert len(boxes) == 2
    assert boxes[0]["kind"] == "semantic"
    assert boxes[0]["label"] == "GLOBAL"
    assert boxes[0]["active"] is True
    assert boxes[1]["kind"] == "fast"
    assert boxes[1]["label"] == "FAST"
    assert boxes[1]["active"] is True
    assert all(0.0 <= boxes[0][key] <= 1.0 for key in ("x1", "y1", "x2", "y2"))
    assert boxes[0]["x1"] < boxes[0]["x2"]
    assert boxes[0]["y1"] < boxes[0]["y2"]


def test_rejected_fast_motion_is_gray_contract():
    result = {
        "motion_box": None,
        "fast_motion_box": (5, 6, 20, 26),
        "fast_motion_hit": False,
        "drift_ema": 0.02,
        "eff_th": 0.15,
    }
    boxes = _feedback_boxes(result, (100, 100))
    assert len(boxes) == 1
    assert boxes[0]["kind"] == "fast_rejected"
    assert boxes[0]["label"] == "FAST"
    assert boxes[0]["active"] is False


def _fake_gate(config: GateConfig | None = None) -> RaphaelVisionGate:
    cfg = config or GateConfig(enable_optical_flow=False, clip_device="cpu")
    gate = RaphaelVisionGate.__new__(RaphaelVisionGate)
    gate.cfg = cfg
    gate.device = "cpu"
    gate.mp_pose = None
    gate.state = GateState.WATCHING
    gate._anchor_emb = None
    gate._anchor_frame = None
    gate._drift_ema = 0.0
    gate._prev_gray = None
    gate._prev_gray_small = None
    gate._prev_keypoints = None
    gate._ring = collections.deque(maxlen=cfg.sharp_buffer_size)
    gate._best_armed_frame = None
    gate._best_armed_sharpness = -1.0
    gate._armed_since = 0.0
    gate._cooldown_start = 0.0
    gate._last_trigger_time = 0.0
    gate._trigger_count = 0
    gate._last_fire_kind = ""

    def fake_clip_encode(self, images):
        return torch.tensor([[1.0, 0.0] for _ in images], dtype=torch.float32)

    gate._clip_encode = types.MethodType(fake_clip_encode, gate)
    return gate


def test_frame_diff_produces_semantic_feedback_box():
    gate = _fake_gate(GateConfig(enable_optical_flow=False, clip_device="cpu"))
    frame0 = np.zeros((240, 320, 3), dtype=np.uint8)
    frame1 = frame0.copy()
    frame1[85:130, 120:175] = 255

    gate.evaluate(frame0)
    result = gate.evaluate(frame1)

    boxes = result["feedback_boxes"]
    assert any(box["kind"] == "semantic" and box["label"] == "GLOBAL" for box in boxes)
    assert result["motion_box"] is not None
    assert result["change_box_count"] >= 1
    assert result["jitter"] > 0


def test_triggered_state_can_be_rendered_green():
    result = {
        "motion_box": (0, 0, 100, 100),
        "fast_motion_box": None,
        "fast_motion_hit": False,
        "drift_ema": 0.2,
        "eff_th": 0.15,
        "fired": True,
        "state": "COOLDOWN",
        "flow_pixels": 0,
        "roi_drift": 0.0,
    }
    boxes = _feedback_boxes(result, (100, 100))
    flash_until = time.time() + 0.65
    assert boxes[0]["kind"] == "semantic"
    assert time.time() < flash_until


def test_flow_mask_preview_contract():
    mask = np.zeros((240, 320), dtype=np.uint8)
    mask[80:110, 140:170] = 255
    result = {
        "motion_mask": mask,
        "motion_box": (100, 60, 220, 150),
        "fast_motion_box": (140, 80, 170, 110),
        "fast_motion_hit": True,
    }
    encoded = _flow_mask_preview(result, (240, 320), GateConfig())
    assert encoded
    assert base64.b64decode(encoded)[:2] == b"\xff\xd8"


def _fake_vision_module(feedback_boxes, triggered=False):
    vm = VisionModule.__new__(VisionModule)
    vm._state_lock = threading.Lock()
    vm._trigger_flash_until = time.time() + 1.0 if triggered else 0.0
    vm._latest_view = {"feedback_boxes": feedback_boxes}
    return vm


def test_local_preview_draws_blue_red_and_green_feedback():
    frame = np.zeros((100, 160, 3), dtype=np.uint8)
    boxes = [
        {"x1": 0.10, "y1": 0.10, "x2": 0.60, "y2": 0.70, "kind": "semantic", "label": "GLOBAL"},
        {"x1": 0.68, "y1": 0.20, "x2": 0.90, "y2": 0.50, "kind": "fast", "label": "FAST"},
    ]
    rendered = _fake_vision_module(boxes, triggered=True)._draw_gate_feedback(frame)
    assert rendered.shape == frame.shape
    # OpenCV uses BGR. Semantic box is cyan-ish (235,210,60), fast box is red-ish (70,70,245).
    assert np.any((rendered[:, :, 0] > 200) & (rendered[:, :, 1] > 170) & (rendered[:, :, 2] < 120))
    assert np.any((rendered[:, :, 2] > 200) & (rendered[:, :, 0] < 140) & (rendered[:, :, 1] < 140))
    assert np.mean(rendered[:, :, 1]) > 20


if __name__ == "__main__":
    test_feedback_boxes_contract()
    test_rejected_fast_motion_is_gray_contract()
    test_frame_diff_produces_semantic_feedback_box()
    test_triggered_state_can_be_rendered_green()
    test_flow_mask_preview_contract()
    test_local_preview_draws_blue_red_and_green_feedback()
    print("vision feedback contract ok")
