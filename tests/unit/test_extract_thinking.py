"""抽取类场景必须显式关掉「思考」（enable_thinking）。

为什么值得单独一个测试文件：
  2026-10-28 实测，简历解析慢的真正原因是模型默认开着思考 ——
  同一份简历 8634 个 completion token 里有 6300 个是思考过程，耗时 164 秒；
  关掉后 2301 token、41 秒，字段一个没少。邀约抽取更严重：
  含两轮面试的邮件开着思考会跑满 60 秒超时直接失败。

  所以「调用时带没带 enable_thinking=False」不是风格问题，而是这两条链路
  能不能在超时预算内跑完的开关。这里把它钉住，免得以后重构时被顺手删掉。
"""

import asyncio
import json
from unittest.mock import AsyncMock, patch

from app.services import invitation, llm, resume


def _run(coro):
    return asyncio.run(coro)


def test_thinking_body_is_omitted_when_not_specified():
    """不指定就必须真的不传 —— 传 None 有些网关会当成字符串。"""
    assert llm._thinking_body(None) is None
    assert llm._thinking_body(False) == {"enable_thinking": False}
    assert llm._thinking_body(True) == {"enable_thinking": True}


def test_resume_parse_turns_thinking_off():
    payload = json.dumps({"name": "张三", "skills": []})
    with patch("app.services.resume.llm_service.chat", new_callable=AsyncMock) as mock_chat:
        mock_chat.return_value = payload
        _run(resume.parse_resume("张三，Java 开发，3 年经验。"))

    assert mock_chat.await_count == 1
    assert mock_chat.await_args.kwargs["enable_thinking"] is False


def test_invitation_extract_turns_thinking_off():
    payload = json.dumps({"events": []})
    with patch("app.services.invitation.llm_service.chat", new_callable=AsyncMock) as mock_chat:
        mock_chat.return_value = payload
        _run(invitation.extract_invitation(text="你好，10 月 30 日下午 2 点来面试。"))

    assert mock_chat.await_count == 1
    assert mock_chat.await_args.kwargs["enable_thinking"] is False
