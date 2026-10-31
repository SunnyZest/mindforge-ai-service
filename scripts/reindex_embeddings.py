#!/usr/bin/env python
"""重建向量索引 —— 换 embedding 模型后必须跑。

════════════════════════════════════════════════════════════════════════
为什么需要这个脚本
════════════════════════════════════════════════════════════════════════

Chroma 里的向量和新模型不在同一个语义空间 —— 哪怕维度相同也一样。
换模型后如果只写新文档、不重建旧文档，collection 里会混着两套坐标，
检索结果静默变差且不报任何错（详见 app/services/vector_store.py 的类注释）。

════════════════════════════════════════════════════════════════════════
用法
════════════════════════════════════════════════════════════════════════

    # 1) 先看会重算哪些文档，不真写
    python scripts/reindex_embeddings.py --dry-run

    # 2) 真跑（密码从环境变量读，不要写在命令行里 —— 会进 shell history）
    export MINDFORGE_DB_PASSWORD='...'
    python scripts/reindex_embeddings.py --yes

    # 3) 只重建某个知识库
    python scripts/reindex_embeddings.py --kb 3 --yes

依赖：pymysql（只有这个脚本用，所以没进 pyproject 的服务依赖）

    uv pip install pymysql

⚠️ 跑之前先把 data/chroma 整个目录备份或删掉 —— 这个脚本是「按当前分块重算并覆盖」，
   不会保留旧向量。删掉目录再跑是最干净的做法。

⚠️ 跑之前确认 Python 服务【停了】—— 它也在读写同一个 Chroma。
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

# 让 `import app.xxx` 能用（脚本在 scripts/ 下，app 在项目根）
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))


def _fail(msg: str) -> None:
    print(f"[错误] {msg}", file=sys.stderr)
    sys.exit(1)


def _load_pymysql():
    try:
        import pymysql  # noqa: PLC0415
        return pymysql
    except ImportError:
        _fail(
            "缺少 pymysql，先装一下：uv pip install pymysql\n"
            "（这个脚本是运维工具，依赖没进服务镜像，免得为它多打几十 MB）"
        )


def connect_db(args):
    pymysql = _load_pymysql()
    password = args.password or os.environ.get("MINDFORGE_DB_PASSWORD")
    if not password:
        _fail(
            "没有数据库密码。两种传法（推荐第一种，避免进 shell history）：\n"
            "  export MINDFORGE_DB_PASSWORD='...'\n"
            "  python scripts/reindex_embeddings.py --yes\n"
            "或者：--password '...'"
        )
    try:
        return pymysql.connect(
            host=args.host, port=args.port, user=args.user,
            password=password, database=args.database,
            charset="utf8mb4", cursorclass=pymysql.cursors.DictCursor,
        )
    except Exception as exc:  # noqa: BLE001
        _fail(f"连不上数据库：{exc}")


def load_documents(conn, kb_id: int | None) -> list[dict]:
    """所有「有分块且未删除」的文档，带上分块数。"""
    sql = """
        SELECT d.id            AS documentId,
               d.knowledge_base_id AS knowledgeBaseId,
               d.user_id       AS userId,
               d.file_name     AS fileName,
               COUNT(c.id)     AS chunkCount
        FROM document d
        JOIN document_chunk c ON c.document_id = d.id
        WHERE d.deleted = 0
    """
    params: list = []
    if kb_id is not None:
        sql += " AND d.knowledge_base_id = %s"
        params.append(kb_id)
    sql += " GROUP BY d.id, d.knowledge_base_id, d.user_id, d.file_name ORDER BY d.id"
    with conn.cursor() as cur:
        cur.execute(sql, params)
        return list(cur.fetchall())


def load_chunks(conn, document_id: int) -> list[str]:
    """按 chunk_index 升序取分块正文。

    ⚠️ 顺序必须严格是 chunk_index 的升序 —— vector_store 用
       `doc{documentId}-chunk{i}` 生成 id，i 是【写入顺序】。
       顺序错位的话，检索命中 A 块却会显示 B 块的正文。
    """
    with conn.cursor() as cur:
        cur.execute(
            "SELECT chunk_index, content FROM document_chunk"
            " WHERE document_id = %s ORDER BY chunk_index",
            (document_id,),
        )
        rows = list(cur.fetchall())

    # 校验 chunk_index 是否从 0 连续递增 —— 不连续说明有过删除，
    # 重建后 id 会和原来的错位，这种文档跳过并报出来。
    for pos, row in enumerate(rows):
        if row["chunk_index"] != pos:
            raise ValueError(
                f"chunk_index 不连续（第 {pos} 位是 {row['chunk_index']}）"
            )
    return [row["content"] for row in rows]


async def reindex(args) -> int:
    from app.services.embedding import embedding_service
    from app.services.vector_store import vector_store

    conn = connect_db(args)
    try:
        docs = load_documents(conn, args.kb)
        if not docs:
            print("没有可重建的文档（没有分块，或指定知识库里没有）。")
            return 0

        total_chunks = sum(d["chunkCount"] for d in docs)
        print(f"待重建：{len(docs)} 个文档 / {total_chunks} 个分块")
        if args.kb is not None:
            print(f"（只处理知识库 {args.kb}）")
        print()

        if args.dry_run:
            for d in docs:
                print(f"  [dry-run] 文档 {d['documentId']} "
                      f"({d['fileName']}) — {d['chunkCount']} 块, "
                      f"kb={d['knowledgeBaseId']}")
            print("\n--dry-run：没有写入任何东西。去掉这个参数才会真跑。")
            return 0

        ok, skipped, failed = 0, 0, 0
        for i, d in enumerate(docs, 1):
            doc_id = d["documentId"]
            try:
                chunks = load_chunks(conn, doc_id)
            except ValueError as exc:
                print(f"  [{i}/{len(docs)}] 跳过文档 {doc_id}：{exc}")
                skipped += 1
                continue

            try:
                vectors = await embedding_service.embed(chunks)
                # add 内部会先按 document_id 删旧向量，所以是幂等的
                await vector_store.add(
                    document_id=doc_id,
                    knowledge_base_id=d["knowledgeBaseId"],
                    user_id=d["userId"],
                    chunks=chunks,
                    embeddings=vectors,
                )
                print(f"  [{i}/{len(docs)}] 文档 {doc_id} — {len(chunks)} 块 OK")
                ok += 1
            except Exception as exc:  # noqa: BLE001
                print(f"  [{i}/{len(docs)}] 文档 {doc_id} 失败："
                      f"{type(exc).__name__}: {exc}")
                failed += 1

        print(f"\n完成：成功 {ok}，跳过 {skipped}，失败 {failed}")
        if failed:
            print("⚠️ 有失败的文档，它们的旧向量可能已被删掉 —— 重跑本脚本即可补齐"
                  "（add 是幂等的）。")
        return 1 if failed else 0
    finally:
        conn.close()


def main() -> None:
    p = argparse.ArgumentParser(
        description="按当前 embedding 配置重建所有文档的向量",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--dry-run", action="store_true", help="只列出要重算的文档，不写入")
    p.add_argument("--yes", action="store_true", help="跳过确认，直接执行")
    p.add_argument("--kb", type=int, default=None, help="只重建指定知识库 id")
    p.add_argument("--host", default=os.environ.get("MINDFORGE_DB_HOST", "127.0.0.1"))
    p.add_argument("--port", type=int, default=int(os.environ.get("MINDFORGE_DB_PORT", "3306")))
    p.add_argument("--user", default=os.environ.get("MINDFORGE_DB_USER", "mind"))
    p.add_argument("--password", default=None, help="不推荐：会进 shell history")
    p.add_argument("--database", default=os.environ.get("MINDFORGE_DB_NAME", "mindforge"))
    args = p.parse_args()

    if not args.dry_run and not args.yes:
        print("这个操作会覆盖 Chroma 里所有向量。确认无误后加 --yes 再跑；")
        print("想先看看影响范围就加 --dry-run。")
        sys.exit(2)

    sys.exit(asyncio.run(reindex(args)))


if __name__ == "__main__":
    main()
