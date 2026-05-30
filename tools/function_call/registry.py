import inspect
import json
import time
import traceback
from typing import Callable, Dict, Optional

class ToolRegistry:
    """
    動態工具註冊引擎。
    利用 Python 反射機制自動從函式簽章生成 OpenAI Function Schema。
    """
    def __init__(self):
        self.tools_schema = []
        self.handlers = {}

    def register(self, schema_override: Optional[Dict] = None):
        """
        工具裝飾器。
        :param schema_override: 允許覆寫或補充自動生成的 JSON Schema (例如 enum 限制)
        """
        def decorator(func: Callable):
            sig = inspect.signature(func)
            doc = inspect.getdoc(func) or "沒有提供描述。"

            properties = {}
            required = []

            for name, param in sig.parameters.items():
                if name == "self":
                    continue

                param_type = "string"
                if param.annotation == int: param_type = "integer"
                elif param.annotation == float: param_type = "number"
                elif param.annotation == bool: param_type = "boolean"
                elif param.annotation == dict: param_type = "object"
                elif param.annotation == list: param_type = "array"

                properties[name] = {
                    "type": param_type,
                    "description": f"參數 {name}"
                }

                if param.default == inspect.Parameter.empty:
                    required.append(name)

            base_schema = {
                "type": "function",
                "function": {
                    "name": func.__name__,
                    "description": doc,
                    "parameters": {
                        "type": "object",
                        "properties": properties,
                        "required": required
                    }
                }
            }

            if schema_override:
                if "description" in schema_override:
                    base_schema["function"]["description"] = schema_override["description"]
                if "properties" in schema_override:
                    for k, v in schema_override["properties"].items():
                        if k in base_schema["function"]["parameters"]["properties"]:
                            base_schema["function"]["parameters"]["properties"][k].update(v)
                        else:
                            base_schema["function"]["parameters"]["properties"][k] = v

            self.tools_schema.append(base_schema)
            self.handlers[func.__name__] = func
            return func

        return decorator

    def _build_error_payload(self, tool_name: str, tool_args: dict, exc: Exception, started_at: float) -> dict:
        duration_ms = round((time.perf_counter() - started_at) * 1000, 2)
        error_type = exc.__class__.__name__
        message = str(exc)
        suggestions = []
        cause = "工具執行期間發生未預期例外。"

        if isinstance(exc, TypeError):
            cause = "傳入參數與工具函式定義不相容，通常是缺少必填欄位、欄位名稱錯誤，或型別不符合預期。"
            suggestions = [
                "檢查工具呼叫時的參數名稱是否與 schema 一致",
                "確認必填參數都有提供",
                "若同名工具近期有改版，重新整理前後端定義",
            ]
        elif isinstance(exc, FileNotFoundError):
            cause = "目標檔案或憑證檔不存在。"
            suggestions = [
                "確認檔案路徑是否正確",
                "若是外部憑證，確認檔案已放在專案根目錄",
            ]
        elif isinstance(exc, PermissionError):
            cause = "程序沒有足夠權限操作指定資源。"
            suggestions = [
                "確認目前程序有對目標路徑或系統資源的存取權限",
                "避免對受保護目錄或系統層級資源直接操作",
            ]
        elif "timeout" in message.lower() or "timed out" in message.lower():
            cause = "工具執行逾時，通常代表外部服務回應過慢或本地任務阻塞。"
            suggestions = [
                "稍後重試一次，確認是否為暫時性延遲",
                "若是網路工具，確認目標服務可正常連線",
                "若是本地工具，檢查是否有長時間阻塞的命令或 I/O",
            ]
        elif "credential" in message.lower() or "token" in message.lower():
            cause = "憑證、權杖或授權狀態異常。"
            suggestions = [
                "確認憑證檔存在且內容有效",
                "必要時重新授權或重新產生 token",
            ]

        return {
            "error": f"執行時發生例外: {message}",
            "error_type": error_type,
            "analysis": {
                "cause": cause,
                "tool_name": tool_name,
                "tool_args": tool_args,
                "duration_ms": duration_ms,
            },
            "suggestions": suggestions,
            "traceback": traceback.format_exc(limit=8),
            "_meta": {
                "ok": False,
                "tool": tool_name,
                "duration_ms": duration_ms,
            }
        }

    def dispatch_with_meta(self, tool_name: str, tool_args: dict) -> dict:
        """執行工具並回傳結構化結果，供 Web UI 顯示更完整的診斷資訊。"""
        handler = self.handlers.get(tool_name)
        started_at = time.perf_counter()

        if not handler:
            return {
                "error": f"未知工具：{tool_name}",
                "error_type": "UnknownToolError",
                "analysis": {
                    "cause": "模型呼叫了未註冊的工具，通常是前後端工具清單不同步。",
                    "tool_name": tool_name,
                    "tool_args": tool_args,
                    "duration_ms": 0,
                },
                "suggestions": [
                    "確認工具已同步載入",
                    "若剛新增工具，請重啟服務讓 registry 重新註冊",
                ],
                "_meta": {"ok": False, "tool": tool_name, "duration_ms": 0},
            }

        try:
            result = handler(**tool_args)
            duration_ms = round((time.perf_counter() - started_at) * 1000, 2)
            if not isinstance(result, dict):
                result = {"result": result}

            result.setdefault("_meta", {})
            result["_meta"].update({
                "ok": "error" not in result,
                "tool": tool_name,
                "duration_ms": duration_ms,
            })
            return result
        except Exception as exc:
            return self._build_error_payload(tool_name, tool_args, exc, started_at)

    def dispatch(self, tool_name: str, tool_args: dict) -> str:
        """根據工具名稱執行對應函式，並攔截所有異常轉為 JSON 格式"""
        result = self.dispatch_with_meta(tool_name, tool_args)
        return json.dumps(result, ensure_ascii=False, default=str)

# 全域單例
registry = ToolRegistry()
tool = registry.register
