"""Tiny operational helper: list tables of the configured database.

Usage: python scripts/inspect_db.py
"""

from __future__ import annotations

import asyncio
import sys

from sqlalchemy import text

from app.core.config import get_settings
from app.core.database import close_db, get_engine


async def main() -> int:
    settings = get_settings()
    print(f"database: {settings.DATABASE_URL.split('@')[-1]}")
    async with get_engine().connect() as connection:
        rows = await connection.execute(
            text(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema = 'public' ORDER BY table_name"
            )
        )
        tables = [row[0] for row in rows]
    print(f"{len(tables)} tables:")
    for table in tables:
        print(f"  - {table}")
    await close_db()
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
