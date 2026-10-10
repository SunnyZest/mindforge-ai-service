"""e2e 套件共用的断言辅助和测试素材。

单独一个模块（而不是塞进 conftest.py）是为了让 test_*.py 能直接 import ——
conftest 是给 pytest 用的，从测试模块里 import 它容易踩重复导入的坑。
"""

from typing import Any

import httpx


def ok(response: httpx.Response) -> Any:
    """断言业务成功并返回 data。"""
    body = response.json()
    assert body.get("code") == 0, (
        f"期望 code=0，实际 HTTP {response.status_code} code={body.get('code')} "
        f"message={body.get('message')} body={str(body)[:400]}"
    )
    return body.get("data")


def find_snake_case(node: Any, path: str = "data") -> list[str]:
    """递归找出所有带下划线的键。

    ★ Java 侧也有一条同样的断言。这里再来一遍是因为：Python 服务的响应如果
      直接暴露（不走 Java），snake_case 不会有任何报错，但 Java 的 Jackson
      对不上会静默变 null —— 接口「通了」、日志正常、数据是空的。
    """
    offenders: list[str] = []
    if isinstance(node, dict):
        for key, value in node.items():
            if "_" in key:
                offenders.append(f"{path}.{key}")
            offenders.extend(find_snake_case(value, f"{path}.{key}"))
    elif isinstance(node, list):
        for index, value in enumerate(node):
            offenders.extend(find_snake_case(value, f"{path}[{index}]"))
    return offenders


def assert_camel_case(node: Any) -> None:
    offenders = find_snake_case(node)
    assert not offenders, f"响应里有 snake_case 的键：{offenders}"


# ---------- 测试素材 ----------
# 刻意是一份「有明显短板」的简历配一份「要求比它高」的 JD：
# 这样五类结论都能出现，能真正验证分类逻辑，而不是全判 MATCHED。

SAMPLE_RESUME = """张伟
男 | 28 岁 | 上海 | zhangwei@example.com
求职意向：Java 后端开发工程师

教育经历
2015.09 - 2019.06  某某大学  计算机科学与技术  本科

工作经历
2021.07 - 至今  某某科技  Java 后端开发工程师
- 负责订单中心开发，参与分库分表改造
- 主导下单链路重构，接口 P99 从 800ms 降到 120ms

技能
Java、Spring Boot、MySQL、Redis、Kafka、ShardingSphere
"""

SAMPLE_JD = """高级 Java 后端开发工程师

岗位职责
1. 负责核心交易系统的设计与开发
2. 参与系统性能优化与稳定性建设

任职要求
1. 5 年以上 Java 后端开发经验，本科及以上学历
2. 精通 JVM 内存模型与 GC 调优
3. 熟悉 MySQL，具备 SQL 优化和分库分表经验
4. 熟悉 Redis，理解缓存与数据库的一致性方案
5. 熟悉 Kafka 等消息中间件
6. 有带团队经验者优先
7. 能接受短期出差
"""

SAMPLE_DOC = """平台使用说明

第一步：创建知识库
在左侧菜单进入「知识库」，点右上角「新建知识库」，填写名称和描述。

第二步：上传文档
进入知识库详情页，点「上传文档」，支持 PDF、Word、TXT、Markdown 和 HTML。
上传后系统会自动解析、切分并向量化，通常需要十几秒到一分钟。

第三步：开始对话
在「对话」页面新建会话并选择知识库，就可以基于文档内容提问了。
回答会附带引用来源，点引用可以看到原文片段。

常见问题
问：文档一直显示「处理中」怎么办？
答：进入文档详情可以看处理状态。失败的话会显示失败原因，可以点「重试」。
"""
