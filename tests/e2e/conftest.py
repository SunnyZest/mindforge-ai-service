"""e2e 套件的公共设施。

这一组打的是【真实链路】：真上传、真解析、真调大模型。所以它：
  · 需要四个服务都在跑（Java 8080 / Python 8000 / MySQL / Redis）
  · 慢（十几分钟）、要花钱（真调模型）
  · 因此默认不跑 —— pyproject.toml 里 addopts = "-m 'not e2e'"
  · 要跑：pytest -m e2e

它和 Java 侧的测试分工明确：
  Java 测试（src/test/java）用 mock 挡住 AI，覆盖接口契约、鉴权、状态机、落库；
  这里只测【mock 测不出来的部分】—— AI 真的输出得对不对。
"""

import os
from typing import Any

import httpx
import pytest

from tests.e2e.helpers import SAMPLE_DOC, SAMPLE_JD, SAMPLE_RESUME  # noqa: F401

BASE_URL = os.environ.get("MF_API_BASE", "http://localhost:8080/api")
APP_USER = os.environ.get("MF_E2E_USER", "zhangsan")
APP_PASSWORD = os.environ.get("MF_E2E_PASSWORD", "123456")
ADMIN_USER = os.environ.get("MF_E2E_ADMIN_USER", "admin")
ADMIN_PASSWORD = os.environ.get("MF_E2E_ADMIN_PASSWORD", "admin123")

# 保留测试数据（排查失败时有用）：MF_E2E_KEEP=1 pytest -m e2e
KEEP_DATA = os.environ.get("MF_E2E_KEEP") == "1"

# 单个 AI 调用最慢的实测约 3~4 分钟（简历优化 / 匹配），读超时给足
DEFAULT_TIMEOUT = httpx.Timeout(connect=10.0, read=900.0, write=60.0, pool=10.0)

# ⚠️ 这里不写 pytestmark —— conftest 里的 pytestmark 不生效，
#    e2e 标记写在各个 test_*.py 模块顶部。


@pytest.fixture(scope="session")
def api() -> httpx.Client:
    """打到 Java 后端的 HTTP 客户端。

    故意用同步 client（而不是 httpx.AsyncClient + pytest-asyncio）：
    e2e 的耗时全在等服务端，不在并发，多一个异步插件不值得。
    """
    with httpx.Client(base_url=BASE_URL, timeout=DEFAULT_TIMEOUT) as client:
        yield client


def _login(api: httpx.Client, path: str, username: str, password: str) -> str:
    try:
        response = api.post(path, json={"username": username, "password": password})
    except httpx.TransportError as exc:
        pytest.skip(f"连不上 Java 后端 {BASE_URL}：{exc}（先把四个服务起起来）")
    body = response.json()
    if body.get("code") != 0:
        pytest.skip(
            f"{path} 登录失败：HTTP {response.status_code} body={body}。"
            f"检查账号 {username} 是否存在（e2e 打的是开发库，不是测试库）"
        )
    return body["data"]["accessToken"]


@pytest.fixture(scope="session")
def token(api: httpx.Client) -> str:
    return _login(api, "/app/auth/login", APP_USER, APP_PASSWORD)


@pytest.fixture(scope="session")
def admin_token(api: httpx.Client) -> str:
    return _login(api, "/admin/auth/login", ADMIN_USER, ADMIN_PASSWORD)


@pytest.fixture(scope="session")
def auth(token: str) -> dict[str, str]:
    """带 token 的请求头。

    session 级：会话级的昂贵 fixture（已解析的简历/职位）要依赖它，
    函数级的话没法被 session fixture 引用。
    """
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture(scope="session", autouse=True)
def cleanup(api: httpx.Client, token: str):
    """跑完把这次造出来的数据删掉。

    ⚠️ 只走 API 删 —— 不能直接 DELETE knowledge_base / document，
       那样会绕过「删知识库 → 触发向量清理」那条链路，向量库里留下孤儿分块。
    """
    registry: list[tuple[str, str]] = []
    yield registry

    if KEEP_DATA:
        print(f"\n  MF_E2E_KEEP=1，保留 {len(registry)} 条测试数据：{registry}")
        return

    headers = {"Authorization": f"Bearer {token}"}
    # 倒序删：后建的先删（面试依赖简历/职位，先删面试）
    for path, label in reversed(registry):
        try:
            response = api.delete(path, headers=headers)
            if response.json().get("code") != 0:
                print(f"  清理 {label} 失败：HTTP {response.status_code} {response.text[:160]}")
        except httpx.HTTPError as exc:
            print(f"  清理 {label} 异常：{exc}")
