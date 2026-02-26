#!/usr/bin/env python3
"""Clear incomplete skill executions (running, paused, failed) from Redis and MySQL.

Rerunnable. Leaves completed/cancelled runs and all other DB data untouched.
Usage: python scripts/clear_incomplete_skill_runs.py (from project root, with .venv active)
"""
from __future__ import annotations

import asyncio
import os
import sys
from urllib.parse import urlparse

# Load project config
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


async def main() -> None:
    from ella.config import get_settings

    settings = get_settings()
    redis_url = settings.redis_url
    database_url = (settings.database_url or "").strip()
    if not database_url:
        print("SKIP: database_url not set (EMOTION_ENABLED or DB not configured)")
        return

    def _mysql_kwargs(url: str) -> dict:
        u = url.replace("mysql+aiomysql://", "mysql://")
        p = urlparse(u)
        return {
            "host": p.hostname or "localhost",
            "port": p.port or 3306,
            "user": p.username or "root",
            "password": p.password or "",
            "db": (p.path or "/ella").lstrip("/").split("?")[0] or "ella",
        }

    import aiomysql
    import redis.asyncio as aioredis

    kwargs = _mysql_kwargs(database_url)

    # 1) Get incomplete run_ids from MySQL
    pool = await aiomysql.create_pool(**kwargs)
    run_ids: list[str] = []
    async with pool.acquire() as conn:
        async with conn.cursor(aiomysql.DictCursor) as cur:
            await cur.execute(
                "SELECT run_id FROM ella_skill_runs WHERE status IN ('running','paused','failed')"
            )
            rows = await cur.fetchall()
            run_ids = [r["run_id"] for r in rows]
    pool.close()
    await pool.wait_closed()

    if not run_ids:
        print("No incomplete skill runs found in MySQL.")
        return

    print(f"Found {len(run_ids)} incomplete run(s): {run_ids}")

    # 2) Delete Redis keys ella:skill:{run_id}
    red = aioredis.from_url(redis_url)
    deleted_redis = 0
    for run_id in run_ids:
        key = f"ella:skill:{run_id}"
        n = await red.delete(key)
        deleted_redis += n
    await red.aclose()
    print(f"Deleted {deleted_redis} Redis checkpoint key(s).")

    # 3) Delete from MySQL (CASCADE removes ella_skill_open_questions)
    pool2 = await aiomysql.create_pool(**kwargs)
    affected = 0
    async with pool2.acquire() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                "DELETE FROM ella_skill_runs WHERE status IN ('running','paused','failed')"
            )
            affected = cur.rowcount
        await conn.commit()
    pool2.close()
    await pool2.wait_closed()
    print(f"Deleted {affected} row(s) from ella_skill_runs (open_questions CASCADE).")
    print("Done.")


if __name__ == "__main__":
    asyncio.run(main())
