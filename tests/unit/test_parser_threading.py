"""同步重活不能卡住事件循环。

背景（2026-10-16 review 发现）：`parse()` 是同步函数，扫描件 PDF 会触发 OCR
（DashScope 每页 60s 超时 × 最多 30 页）、DOCX 也要读完整文档。
如果直接在 async 里调用，这一个请求会把整个服务冻住 —— 别人的对话、/health
全都卡死。所以调用点必须 `asyncio.to_thread` 让出线程池。

这里用「阻塞的 parse + 一个并发探测器协程」来验证：
若 parse 卡在事件循环里，探测器在阻塞期间一帧都跑不动；放进线程池后则照常 tick。
"""

import asyncio
import time
from types import SimpleNamespace

import pytest
from unittest.mock import AsyncMock

from app.api import internal

REQ = internal.ResumeParseRequest(
    userId="1",
    resumeId="2",
    fileUrl="https://example.com/mock-oss-url",
    fileType="PDF",
)


@pytest.fixture(autouse=True)
def _mock_download_and_llm(monkeypatch):
    """只测「解析是否让出线程池」，其它环节一律打桩。"""
    monkeypatch.setattr(internal, "download_bytes", AsyncMock(return_value=b"%PDF-fake"))
    monkeypatch.setattr(
        internal.resume,
        "parse_resume",
        AsyncMock(return_value={
            "summary": "三年 Java 后端",
            "structured": {},
            "skills": [],
            "truncated": False,
        }),
    )


@pytest.mark.anyio
async def test_resume_parse_lets_other_coroutines_run(monkeypatch):
    """解析阻塞 0.4 秒期间，事件循环仍能调度别的协程。"""

    def slow_parse(_content, _file_type):
        # 模拟 OCR：真同步阻塞
        time.sleep(0.4)
        return SimpleNamespace(text="解析出来的文本", page_count=3)

    monkeypatch.setattr(internal, "parse", slow_parse)

    ticks = 0
    stop = False

    async def probe() -> None:
        """每 20ms tick 一次；阻塞期间若能继续 tick 就说明循环没被占住。"""
        nonlocal ticks
        while not stop:
            await asyncio.sleep(0.02)
            ticks += 1

    probe_task = asyncio.create_task(probe())
    await internal.resume_parse(REQ)
    stop = True
    await probe_task

    # 若 parse 卡在事件循环：阻塞 0.4s 期间一帧都跑不了，ticks 会接近 0~1
    print(f'解析期间探针 tick 数：{ticks}')
    assert ticks >= 5, (
        "解析期间事件循环被阻塞（探针几乎没跑动）—— 同步 parse 必须放进线程池"
    )


@pytest.mark.anyio
async def test_parse_failure_still_propagates(monkeypatch):
    """让出线程池不能把异常吞掉 —— 解析失败要如实返回给 Java。"""

    def boom(_content, _file_type):
        raise ValueError("扫描件无文字层，OCR 也失败了")

    monkeypatch.setattr(internal, "parse", boom)

    with pytest.raises(ValueError, match="OCR 也失败了"):
        await internal.resume_parse(REQ)
