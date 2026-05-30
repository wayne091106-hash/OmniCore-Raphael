# OmniCore-Raphael

基於 Gemini Live API 的即時語音／影像多模態助理，具備感知（VAD、視覺辨識）、
記憶（Qdrant）、Function Calling 工具鏈與主動性對話邏輯。

---

## 快速開始

```bash
# 1. 安裝依賴
pip install -r requirements.txt

# 2. 設定環境變數：複製範本後填入自己的金鑰
cp .env.example .env      # Windows: copy .env.example .env
#   編輯 .env，至少要填 GEMINI_API_KEY

# 3. 啟動正式版 UI（FastAPI + 前端）
python ui.py
```

> 需要 Google 工具（Gmail/Calendar/Drive）時，另需放置 `credentials.json`，
> 首次授權後會自動產生 `token.json`。

---

## 架構分層

正式版由三個模組組成（由底到上）：

| 檔案 | 職責 |
|------|------|
| `perception/vad.py` | PCM 音訊 + VAD 狀態 → bridge |
| `perception/vision.py` | JPEG 幀 + gate 狀態 → bridge |
| `bridge.py` | 統一收發，asyncio Queue 路由 |
| `core.py` | GeminiSession + 主動性邏輯，從 bridge 收發 |
| `ui.py` | FastAPI + 前端，顯示 bridge 轉來的感知資料 |

**工具與記憶：**

| 路徑 | 職責 |
|------|------|
| `tools/memory/persona.py` | 角色設定、system prompt 模板 |
| `tools/memory/store.py` / `manager.py` | Qdrant 存取、組合 persona + 記憶 |
| `tools/function_call/` | 外部模型（NIM/Minimax）+ Function Calling 工具實作 |

**舊的獨立單檔版本**（已被 `core.py` / `ui.py` 拆分取代，僅供參考／單機測試）：
`gemini_live_core.py`、`gemini_live_ui.py`。

---

## 環境變數

所有金鑰與設定集中在 `.env`（已被 `.gitignore` 排除，不會上傳）。
完整欄位請見 `.env.example`。關鍵項目：

| 變數 | 說明 |
|------|------|
| `GEMINI_API_KEY` | **必填**，Gemini Live API 金鑰 |
| `NIM_API_KEY` | 額外模型 Function Calling（選填） |
| `OPENWEATHER_API_KEY` / `NEWS_API_KEY` | 天氣 / 新聞工具（選填） |
| `RAPHAEL_MEMORY_BACKEND` | `qdrant_local`（預設）或 `qdrant_remote` |
| `GOOGLE_CREDENTIALS_PATH` / `GOOGLE_TOKEN_PATH` | Google OAuth 憑證檔路徑 |

---

## 🔐 安全注意事項

- **絕不**把 `.env`、`credentials.json`、`token.json` 上傳到公開倉庫，這三者已列入 `.gitignore`。
- 程式碼中**不可**寫死任何金鑰；一律以 `os.environ.get("XXX", "")` 讀取，並由 `.env` 提供。
- 若金鑰曾經外洩（例如曾寫死在原始碼、或已 push 到遠端），請至各服務後台**撤銷並重新產生**。

---

## 建議的未來整理方向（目前未執行，避免破壞 import）

目前測試檔以「從專案根目錄執行」的方式 import（`from core import ...`），
若日後要進一步整理，建議：

```
src/                  # 應用程式碼（core / bridge / ui / perception / tools 歸位）
tests/                # test-*.py 集中，並補上 conftest.py 處理 sys.path
legacy/               # gemini_live_core.py、gemini_live_ui.py
data/                 # 執行期產出（已 gitignore）
```

搬動時需同步調整各檔案的相對 import 與啟動指令。
