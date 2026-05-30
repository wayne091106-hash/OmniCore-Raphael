"""
dependency_bootstrap.py — built-in dependency installer for Raphael.

Usage:
    python ui.py --install-deps core
    python ui.py --install-deps perception
    python ui.py --install-deps memory
    python ui.py --install-deps all
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys


PROFILES = {
    "core": [
        ("dotenv", "python-dotenv>=1.0.0"),
        ("websockets", "websockets>=12.0"),
        ("fastapi", "fastapi>=0.110.0"),
        ("uvicorn", "uvicorn>=0.27.0"),
        ("certifi", "certifi>=2024.0.0"),
    ],
    "memory": [
        ("qdrant_client", "qdrant-client>=1.9.0"),
    ],
    "memory-semantic": [
        ("qdrant_client", "qdrant-client>=1.9.0"),
        ("sentence_transformers", "sentence-transformers>=3.0.0"),
        ("fastembed", "fastembed>=0.3.0"),
    ],
    "perception": [
        ("numpy", "numpy>=1.24.0"),
        ("sounddevice", "sounddevice>=0.4.0"),
        ("cv2", "opencv-python>=4.8.0"),
        ("PIL", "Pillow>=10.0.0"),
        ("ultralytics", "ultralytics>=8.0.0"),
        ("clip", "git+https://github.com/openai/CLIP.git"),
    ],
    "identity": [
        ("cv2", "opencv-python>=4.8.0"),
        ("qdrant_client", "qdrant-client>=1.9.0"),
    ],
    "identity-strong": [
        ("cv2", "opencv-python>=4.8.0"),
        ("qdrant_client", "qdrant-client>=1.9.0"),
        ("insightface", "insightface>=0.7.3"),
        ("onnxruntime", "onnxruntime>=1.17.0"),
    ],
    "tools": [
        ("openai", "openai>=1.30.0"),
        ("googleapiclient", "google-api-python-client>=2.0.0"),
        ("google_auth_oauthlib", "google-auth-oauthlib>=1.0.0"),
        ("ddgs", "ddgs>=9.0.0"),
        ("duckduckgo_search", "duckduckgo-search>=5.0.0"),
        ("httpx", "httpx>=0.27.0"),
        ("truststore", "truststore>=0.9.0"),
        ("pyautogui", "pyautogui>=0.9.54"),
        ("pyperclip", "pyperclip>=1.8.2"),
        ("playwright", "playwright>=1.44.0"),
        ("psutil", "psutil>=5.9.0"),
        ("pypdf", "pypdf>=4.0.0"),
        ("docx", "python-docx>=1.1.0"),
        ("openpyxl", "openpyxl>=3.1.0"),
    ],
}

TORCH_CPU = "torch>=2.0.0"


def _has_module(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def _profile_names(profile: str) -> list[str]:
    if profile == "all":
        return ["core", "perception", "memory", "identity", "tools"]
    if profile == "web":
        return ["core"]
    if profile not in PROFILES:
        raise SystemExit(f"未知依賴組：{profile}。可用：core, perception, memory, memory-semantic, identity, identity-strong, tools, all")
    return [profile]


def missing_packages(profile: str, include_torch: bool = False) -> list[str]:
    packages: list[str] = []
    for name in _profile_names(profile):
        for module_name, package_spec in PROFILES[name]:
            if not _has_module(module_name):
                packages.append(package_spec)
    if include_torch and not _has_module("torch"):
        packages.append(TORCH_CPU)
    return packages


def install(profile: str = "core", include_torch: bool = False) -> int:
    packages = missing_packages(profile, include_torch=include_torch)
    if not packages:
        print(f"[Raphael] {profile} 依賴已就緒。")
        return 0

    print("[Raphael] 將安裝缺少的依賴：")
    for package in packages:
        print(f"  - {package}")

    cmd = [sys.executable, "-m", "pip", "install", *packages]
    return subprocess.call(cmd)


def print_status(profile: str = "all", include_torch: bool = False) -> None:
    missing = missing_packages(profile, include_torch=include_torch)
    if not missing:
        print(f"[Raphael] {profile} 依賴已就緒。")
        return
    print(f"[Raphael] {profile} 缺少：")
    for package in missing:
        print(f"  - {package}")
