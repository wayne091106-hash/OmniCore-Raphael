"""
工具定義 — 透過動態註冊引擎自動生成。
import implementations 觸發所有 @tool 裝飾器註冊。
"""
from .implementations import registry

TOOLS = registry.tools_schema
