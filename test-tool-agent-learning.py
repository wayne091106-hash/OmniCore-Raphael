"""
Regression checks for Raphael's delegated tool learning loop.

These tests stay offline: they verify prompt context assembly, retry guards,
secret redaction, and readable progress summaries without opening websites.
"""

from contextlib import contextmanager
import json
from pathlib import Path
import tempfile

from core import (
    _background_progress_decision,
    _compact_for_ui,
    _delegate_response_contract,
    _delegate_memory_queries,
    _delegate_memory_terms,
    _filter_delegate_memories,
    _site_memory_context_for_delegate,
    _site_memory_queries_for_delegate,
    _task_voice_line,
    ProactiveGovernor,
)
from tools.function_call.agent import (
    TOOL_AGENT_TOOLS,
    _tool_signature,
    browser_stagnation_guardrail,
    browser_state_signature,
    generic_site_search_guard,
    get_minimax_settings,
    learning_events_from_tool,
    next_step_guardrail,
    normalize_tool_route_decision,
    progress_snapshot_from_events,
    recovery_hint_for_tool,
    repeated_tool_guard,
    round_budget_guardrail,
    summarize_tool_result,
    tool_route_requires_delegate,
    update_minimax_settings,
)
import tools.function_call.implementations as impl
from tools.function_call.implementations import computer_control


@contextmanager
def temporary_site_memory(data: dict):
    original = impl.SITE_MEMORY_FILE
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "site_memory.json"
        path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        impl.SITE_MEMORY_FILE = path
        try:
            yield path
        finally:
            impl.SITE_MEMORY_FILE = original


def generic_site_memory_fixture() -> dict:
    return {
        "sites": [
            {
                "service": "範例高中 LMS",
                "url": "https://lms.school.example.edu.tw/",
                "title": "範例高中學習平台",
                "note": "verified",
                "status": "success",
                "success_count": 3,
            },
            {
                "service": "範例高中 LMS 登入",
                "url": "https://lms.school.example.edu.tw/login/index.php",
                "title": "範例高中學習平台: 登入",
                "note": "verified",
                "status": "success",
                "success_count": 1,
            },
        ],
        "failures": [
            {
                "service": "範例高中 LMS",
                "url": "https://old-lms.school.example.edu.tw/",
                "count": 2,
                "error": "DNS failed in previous run",
                "note": "avoid old host",
            },
        ],
    }


def test_site_memory_context_prefers_learned_learning_platform():
    with temporary_site_memory(generic_site_memory_fixture()):
        context = _site_memory_context_for_delegate(
            "我是範例高中學生，去 LMS 上幫我看作業",
            "登入 LMS 並查看課程作業",
        )
    assert "https://lms.school.example.edu.tw/" in context
    assert "https://old-lms.school.example.edu.tw/" in context
    assert "已知失敗網址" in context


def test_site_memory_rejects_local_urls():
    assert impl._clean_url("file:///C:/tmp/start.html") == ""
    assert impl._clean_url("data:text/html,hello") == ""
    assert impl._clean_url("about:blank") == ""
    assert impl._clean_url("lms.school.example.edu.tw") == "https://lms.school.example.edu.tw/"


def test_website_find_uses_high_confidence_memory_before_web_search():
    calls = {"search": 0, "probe": 0, "urls": []}
    original_probe = impl._probe_website
    original_search = impl.web_search
    original_remember = impl._site_memory_remember

    def fake_probe(url, timeout=8):
        calls["probe"] += 1
        calls["urls"].append(url)
        assert url.startswith("https://lms.school.example.edu.tw")
        if url != "https://lms.school.example.edu.tw/":
            return {"ok": False, "url": url, "error": "not the main entry"}
        return {
            "ok": True,
            "url": url,
            "final_url": url,
            "status": 200,
            "title": "範例高中學習平台",
            "content_type": "text/html",
        }

    def fake_search(*args, **kwargs):
        calls["search"] += 1
        raise AssertionError("website_find should not search when high confidence memory verifies")

    def fake_remember(*args, **kwargs):
        return {"success": True}

    try:
        impl._probe_website = fake_probe
        impl.web_search = fake_search
        impl._site_memory_remember = fake_remember
        with temporary_site_memory(generic_site_memory_fixture()):
            result = impl.website_find("範例高中 LMS", max_results=3)
    finally:
        impl._probe_website = original_probe
        impl.web_search = original_search
        impl._site_memory_remember = original_remember

    assert result["success"] is True
    assert result["searched"] is False
    assert result["memory_first"] is True
    assert result["best"]["final_url"] == "https://lms.school.example.edu.tw/"
    assert calls["search"] == 0
    assert calls["probe"] >= 1
    assert calls["urls"][0] == "https://lms.school.example.edu.tw/"


def test_generic_learning_platform_web_search_is_blocked_with_specific_recovery():
    result = generic_site_search_guard(
        "web_search",
        {"query": "LMS 登入 教學平台"},
        "使用者是範例高中學生，要登入 LMS 看作業",
    )
    assert result is not None
    assert result["blocked"] is True
    assert "泛用網站搜尋" in result["error"]
    assert "site_memory_search" in result["recovery_hint"]
    assert "範例高中" in result["recovery_hint"]
    assert "lms" in result["recovery_hint"].lower()


def test_specific_learning_platform_web_search_is_not_blocked():
    assert generic_site_search_guard(
        "web_search",
        {"query": "範例高中 LMS 登入"},
        "使用者是範例高中學生",
    ) is None
    assert generic_site_search_guard(
        "web_search",
        {"query": "site:school.example.edu.tw lms"},
        "使用者是範例高中學生",
    ) is None


def test_site_queries_are_context_derived_not_fixed_school_constant():
    plain_queries = _site_memory_queries_for_delegate("去 LMS 登入", "")
    school_queries = _site_memory_queries_for_delegate("我是範例高中學生，去 LMS 登入", "")
    assert all("範例高中" not in query for query in plain_queries)
    assert any("範例高中" in query and "lms" in query.lower() for query in school_queries)
    assert all("範例高中學" not in query for query in school_queries)


def test_delegate_memory_queries_use_site_context_without_old_fixed_query():
    with temporary_site_memory(generic_site_memory_fixture()):
        context = _site_memory_context_for_delegate("我是範例高中學生，去 LMS 登入", "登入 LMS")
    queries = _delegate_memory_queries("我是範例高中學生，去 LMS 登入", "登入 LMS", context)
    assert "範例高中 固定 LMS 帳號 密碼 網址" not in queries
    assert any("lms.school.example.edu.tw" in query for query in queries)


def test_delegate_memory_filter_keeps_target_credentials_and_drops_stale_tasks():
    rows = [
        {
            "id": "target",
            "category": "credential",
            "memory": "使用者用於範例高中 Google 服務的帳號為 student@gl.school.example.edu.tw，密碼=********。",
        },
        {
            "id": "gmail",
            "category": "credential",
            "memory": "使用者用於個人郵件服務的私人帳號為 personal@gmail.com，密碼=********。",
        },
        {
            "id": "stale",
            "category": "project",
            "memory": "舊動畫專案最新進度需要寄送到 teammate@example.com。",
        },
    ]
    filtered = _filter_delegate_memories(
        rows,
        "我是範例高中學生，去 Moodle 幫我看作業",
        "登入範例高中 Moodle，查看未完成作業",
        "【Raphael 已學網站入口】\n- 優先入口：範例高中 Moodle → https://moodle.school.example.edu.tw/",
    )
    memories = "\n".join(row["memory"] for row in filtered)
    assert "student@gl.school.example.edu.tw" in memories
    assert "personal@gmail.com" not in memories
    assert "舊動畫專案" not in memories


def test_site_matching_derives_generic_school_abbreviation_and_entry_quality():
    terms = impl._site_terms("範例高級中學 Moodle 教學平台")
    assert "範中" in terms
    noisy_terms = impl._site_terms("使用者是範例高中學生，要登入 LMS 看作業")
    assert "範例高中" in noisy_terms
    assert "範例高中學" not in noisy_terms
    delegate_terms = _delegate_memory_terms("使用者是範例高中學生，要登入 LMS 看作業")
    assert "範例高中" in delegate_terms
    assert "範例高中學" not in delegate_terms
    bad = impl._site_entry_quality(
        {"url": "https://moodle.school.example.edu.tw/auth/sso/callback", "title": "錯誤"},
        "範例高中 Moodle",
    )
    good = impl._site_entry_quality(
        {"url": "https://moodle.school.example.edu.tw/", "title": "範例高中 Moodle"},
        "範例高中 Moodle",
    )
    assert bad < 0
    assert good > bad


def test_minimax_runtime_settings_are_validated():
    before = get_minimax_settings()
    try:
        updated = update_minimax_settings({
            "model": "test/model",
            "base_url": "https://example.test/v1/",
            "max_tool_rounds": 999,
            "request_timeout": 5,
            "temperature": 9,
        })
        assert updated["model"] == "test/model"
        assert updated["base_url"] == "https://example.test/v1"
        assert updated["max_tool_rounds"] == 64
        assert updated["request_timeout"] == 20
        assert updated["temperature"] == 1.5
    finally:
        update_minimax_settings(before)


def test_visual_proactive_events_are_sent_to_gemini_for_judgment():
    import time

    governor = ProactiveGovernor()
    now = time.time()
    governor._last_spoke_at = now
    governor._last_spoke_by_key["vision:object_motion"] = now
    decision = governor.decide(
        {"type": "vision:object_motion", "detail": "motion candidate"},
        user_busy=False,
        last_user_activity=now,
        last_assistant_activity=now,
    )
    assert decision["action"] == "speak"
    assert decision["reason"] == "send_to_gemini"


def test_delegate_response_contract_prevents_generic_refusal_after_progress():
    contract = _delegate_response_contract({
        "ok": True,
        "answer": "已進入目標網站並找到待處理項目",
        "tool_calls": [
            {
                "tool": "copy_file",
                "success": True,
                "result_preview": "已複製檔案到：C:\\Users\\Alex\\Desktop\\image.png",
            }
        ],
        "progress_snapshot": {
            "summary": "已進入目標網站並找到待處理項目",
            "current_phase": "背景瀏覽器操作",
        },
    })
    text = contract["instruction"]
    assert "不可把工具已成功完成或已推進的部分改口說成無法做到" in text
    assert "completed_actions 內列出的成功副作用是權威事實" in text
    assert "下一個最小可執行步驟" in text
    assert contract["current_progress"] == "已進入目標網站並找到待處理項目"
    assert contract["completed_actions"] == ["copy_file: 已複製檔案到：C:\\Users\\Alex\\Desktop\\image.png"]


def test_tool_signature_redacts_passwords():
    signature = _tool_signature(
        "browser_login",
        {"url": "https://example.test/", "username": "u", "password": "secret"},
    )
    assert "secret" not in signature
    assert "********" in signature


def test_repeated_guard_stops_third_identical_site_retry():
    counts = {}
    args = {"url": "https://old-lms.school.example.edu.tw/"}
    assert repeated_tool_guard("browser_open", args, counts)[1] is None
    assert repeated_tool_guard("browser_open", args, counts)[1] is None
    repeat_count, result = repeated_tool_guard("browser_open", args, counts)
    assert repeat_count == 3
    assert result["repeated_call"] is True
    assert "停止原地重試" in result["recovery_hint"]


def test_dns_failure_hint_pushes_site_memory_recovery():
    hint = recovery_hint_for_tool(
        "browser_open",
        {"url": "https://old-lms.school.example.edu.tw/"},
        {"error": "Page.goto: net::ERR_NAME_NOT_RESOLVED"},
        1,
    )
    assert "site_memory_mark_failure" in hint
    assert "website_find" in hint


def test_next_step_guardrail_turns_hint_into_hard_strategy_text():
    guardrail = next_step_guardrail(
        "browser_open",
        {"url": "https://old-lms.school.example.edu.tw/", "password": "secret"},
        {
            "error": "Page.goto: net::ERR_NAME_NOT_RESOLVED",
            "recovery_hint": "不要再重試這個網址；改查已學入口。",
        },
        repeat_count=2,
        failure_repeat_count=2,
    )
    assert "下一步不可用同一工具與同一組參數原地重試" in guardrail
    assert "必須換入口" in guardrail
    assert "背景瀏覽器" in guardrail
    assert "secret" not in guardrail


def test_round_budget_guardrail_forces_progress_summary():
    guardrail = round_budget_guardrail(
        15,
        16,
        [
            {"tool": "website_find", "success": True, "result_preview": "找到入口"},
            {"tool": "browser_open", "success": False, "result_preview": "DNS failed"},
        ],
    )
    assert "已用工具輪數：15/16" in guardrail
    assert "不要再開新探索分支" in guardrail
    assert "部分成功" in guardrail
    assert "失敗：1" in guardrail


def test_progress_snapshot_is_readable_and_redacted():
    snapshot = progress_snapshot_from_events(
        [
            {
                "tool": "website_find",
                "success": True,
                "result_preview": "已找到並驗證入口：https://lms.school.example.edu.tw/",
            },
            {
                "tool": "browser_login",
                "success": False,
                "result_preview": "找不到密碼欄位；password=********",
                "error": "找不到密碼欄位",
            },
        ],
        ["strategy"],
    )
    assert snapshot["tool_count"] == 2
    assert snapshot["success_count"] == 1
    assert snapshot["failure_count"] == 1
    assert snapshot["current_phase"] == "背景瀏覽器操作"
    assert snapshot["next_focus"] == "先處理最近失敗或改換策略"
    assert "secret" not in str(snapshot)
    assert "進度：已執行 2 次工具" in snapshot["summary"]


def test_browser_state_signature_is_stable_and_redacted():
    result = {
        "url": "https://lms.school.example.edu.tw/login/index.php#section",
        "title": "Login",
        "text": "password=secret same page",
        "controls": [{"selector": "#username"}],
    }
    sig = browser_state_signature("browser_get_page", result)
    assert "secret" not in sig
    assert "section" not in sig
    assert "lms.school.example.edu.tw" in sig
    assert browser_state_signature("web_search", result) == ""


def test_browser_stagnation_guardrail_detects_same_page_across_tools():
    result = {
        "url": "https://lms.school.example.edu.tw/login/index.php",
        "title": "Login",
        "text": "same login page",
        "controls": [{"selector": "#username"}],
    }
    sig = browser_state_signature("browser_get_page", result)
    events = [
        {"tool": "browser_get_page", "success": True, "args": {}, "browser_state_signature": sig},
        {"tool": "browser_click", "success": True, "args": {"target": "登入"}, "browser_state_signature": sig},
        {"tool": "browser_press_key", "success": True, "args": {"key": "Enter"}, "browser_state_signature": sig},
    ]
    guardrail = browser_stagnation_guardrail(events)
    assert "頁面狀態連續 3 次沒有變化" in guardrail
    assert "不要繼續在同一頁" in guardrail
    assert "browser_links" in guardrail


def test_delegate_ui_summary_reports_budget_stop():
    compact = _compact_for_ui(
        "delegate_tool_task",
        {
            "ok": True,
            "stopped_for_budget": True,
            "strategy_events": ["budget"],
            "duration_ms": 10,
        },
    )
    assert compact["stopped_for_budget"] is True
    assert compact["strategy_count"] == 1
    assert "輪數上限" in compact["summary"]


def test_background_progress_decision_throttles_quiet_browser_updates():
    state = {}
    first = {
        "tool": "browser_open",
        "result_preview": "背景瀏覽器完成",
        "progress_snapshot": {
            "tool_count": 1,
            "current_phase": "背景瀏覽器操作",
            "summary": "進度：已執行 1 次工具，成功 1、失敗 0；目前階段：背景瀏覽器操作",
        },
    }
    should_publish, summary = _background_progress_decision(first, state)
    assert should_publish is True
    assert "進度：已執行 1 次工具" in summary

    second = {
        "tool": "browser_get_page",
        "result_preview": "背景瀏覽器完成",
        "progress_snapshot": {
            "tool_count": 2,
            "current_phase": "背景瀏覽器操作",
            "summary": "進度：已執行 2 次工具，成功 2、失敗 0；目前階段：背景瀏覽器操作",
        },
    }
    assert _background_progress_decision(second, state) == (False, "")

    third = {
        "tool": "browser_click",
        "result_preview": "背景瀏覽器完成",
        "progress_snapshot": {
            "tool_count": 3,
            "current_phase": "背景瀏覽器操作",
            "summary": "進度：已執行 3 次工具，成功 3、失敗 0；目前階段：背景瀏覽器操作",
        },
    }
    should_publish, summary = _background_progress_decision(third, state)
    assert should_publish is True
    assert "進度：已執行 3 次工具" in summary

    error_payload = {"tool": "browser_click", "error": "找不到按鈕", "result_preview": "找不到按鈕"}
    assert _background_progress_decision(error_payload, state) == (True, "找不到按鈕")


def test_task_voice_lines_are_short_actionable_and_redacted_by_caller():
    assert _task_voice_line("tool_start", {"tool": "browser_open"}) == "我正在背景瀏覽器處理。"
    assert _task_voice_line("tool_start", {"tool": "copy_file"}) == "我正在處理檔案。"
    done = _task_voice_line(
        "tool_done",
        {"tool": "copy_file", "result_preview": "已複製檔案到：C:\\Users\\Alex\\Desktop\\image.png"},
    )
    assert done == "已複製檔案到：C:\\Users\\Alex\\Desktop\\image.png"
    error = _task_voice_line(
        "tool_done",
        {"tool": "browser_click", "error": "找不到按鈕", "result_preview": "找不到按鈕"},
    )
    assert "遇到問題" in error


def test_tool_route_decision_normalizes_delegate_tasks():
    decision = normalize_tool_route_decision(
        {
            "mode": "delegate",
            "confidence": 0.91,
            "reason": "需要目前狀態",
            "task": "",
        },
        "看一下我現在的畫面，告訴我你看到什麼。",
    )
    assert decision["mode"] == "delegate"
    assert decision["confidence"] == 0.91
    assert "原始使用者要求" in decision["task"]
    assert tool_route_requires_delegate(decision) is True

    direct = normalize_tool_route_decision(
        {"mode": "direct", "confidence": 0.8, "reason": "一般知識", "task": "不該保留"},
        "解釋什麼是 binary search",
    )
    assert direct["task"] == ""
    assert tool_route_requires_delegate(direct) is False


def test_learning_events_capture_site_success_and_login_failure():
    site_events = learning_events_from_tool(
        "website_find",
        {"query": "範例高中 LMS"},
        {"best": {"final_url": "https://lms.school.example.edu.tw/", "title": "範例高中學習平台"}},
    )
    assert any("lms.school.example.edu.tw" in event["memory"] for event in site_events)

    login_events = learning_events_from_tool(
        "browser_login",
        {"url": "https://lms.school.example.edu.tw/", "username": "user", "password": "secret"},
        {"url": "https://lms.school.example.edu.tw/login/index.php", "logged_in": False, "message": "需要使用者驗證"},
    )
    assert any("登入操作未完成" in event["memory"] for event in login_events)
    assert all("secret" not in event["memory"] for event in login_events)

    local_events = learning_events_from_tool(
        "browser_open",
        {"url": "file:///C:/tmp/start.html"},
        {"success": True, "url": "file:///C:/tmp/start.html", "title": "Start"},
    )
    assert local_events == []


def test_process_summary_is_readable():
    summary = summarize_tool_result(
        "list_processes",
        {
            "process_count": 12,
            "top_processes": [
                {"name": "chrome.exe", "count": 3},
                {"name": "python.exe", "count": 2},
            ],
        },
    )
    assert summary == "列出 12 個程序；常見：chrome.exe x3、python.exe x2"


def test_successful_file_actions_are_readable_and_authoritative():
    assert summarize_tool_result(
        "copy_file",
        {"success": True, "path": "C:\\Users\\Alex\\Desktop\\image.png"},
    ) == "已複製檔案到：C:\\Users\\Alex\\Desktop\\image.png"
    assert summarize_tool_result(
        "move_file",
        {"success": True, "path": "C:\\Users\\Alex\\Desktop\\renamed.png"},
    ) == "已移動檔案到：C:\\Users\\Alex\\Desktop\\renamed.png"


def test_background_browser_navigation_tools_are_registered_and_readable():
    names = {tool["function"]["name"] for tool in TOOL_AGENT_TOOLS}
    assert {"browser_links", "browser_follow_link", "browser_wait", "browser_back"} <= names

    summary = summarize_tool_result(
        "browser_links",
        {
            "count": 2,
            "links": [
                {"index": 0, "text": "我的課程", "href": "https://example.test/my/"},
                {"index": 1, "text": "作業", "href": "https://example.test/mod/assign/"},
            ],
        },
    )
    assert summary == "列出 2 個背景頁面連結；第一個：我的課程"


def test_computer_control_requires_window_awareness_before_input():
    result = computer_control([{"action": "click", "x": 10, "y": 10}], screenshot_after=False)
    assert "error" in result
    assert "第一步必須先確認或切換目標視窗" in result["error"]
    assert "focus_window" in result["recovery_hint"]


if __name__ == "__main__":
    test_site_memory_context_prefers_learned_learning_platform()
    test_site_memory_rejects_local_urls()
    test_website_find_uses_high_confidence_memory_before_web_search()
    test_generic_learning_platform_web_search_is_blocked_with_specific_recovery()
    test_specific_learning_platform_web_search_is_not_blocked()
    test_site_queries_are_context_derived_not_fixed_school_constant()
    test_delegate_memory_queries_use_site_context_without_old_fixed_query()
    test_delegate_memory_filter_keeps_target_credentials_and_drops_stale_tasks()
    test_site_matching_derives_generic_school_abbreviation_and_entry_quality()
    test_minimax_runtime_settings_are_validated()
    test_visual_proactive_events_are_sent_to_gemini_for_judgment()
    test_delegate_response_contract_prevents_generic_refusal_after_progress()
    test_tool_signature_redacts_passwords()
    test_repeated_guard_stops_third_identical_site_retry()
    test_dns_failure_hint_pushes_site_memory_recovery()
    test_next_step_guardrail_turns_hint_into_hard_strategy_text()
    test_round_budget_guardrail_forces_progress_summary()
    test_progress_snapshot_is_readable_and_redacted()
    test_browser_state_signature_is_stable_and_redacted()
    test_browser_stagnation_guardrail_detects_same_page_across_tools()
    test_delegate_ui_summary_reports_budget_stop()
    test_background_progress_decision_throttles_quiet_browser_updates()
    test_task_voice_lines_are_short_actionable_and_redacted_by_caller()
    test_tool_route_decision_normalizes_delegate_tasks()
    test_learning_events_capture_site_success_and_login_failure()
    test_process_summary_is_readable()
    test_successful_file_actions_are_readable_and_authoritative()
    test_background_browser_navigation_tools_are_registered_and_readable()
    test_computer_control_requires_window_awareness_before_input()
    print("tool agent learning contract ok")
