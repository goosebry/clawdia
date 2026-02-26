#!/usr/bin/env python3
"""One-time script to initialise Qdrant collections.

Run before starting Ella for the first time:
    python scripts/init_qdrant.py

The script is idempotent — safe to run multiple times.
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from ella.memory.knowledge import (
    COLLECTION_CONVERSATIONS,
    COLLECTION_TASK_PATTERNS,
    COLLECTION_USER_PREFS,
    VECTOR_DIM,
    ensure_collections,
)


async def main() -> None:
    print("Initialising Qdrant collections...")
    await ensure_collections()
    print(f"  ✓ {COLLECTION_CONVERSATIONS}  (dim={VECTOR_DIM})")
    print(f"  ✓ {COLLECTION_TASK_PATTERNS}  (dim={VECTOR_DIM})")
    print(f"  ✓ {COLLECTION_USER_PREFS}  (dim={VECTOR_DIM})")
    print("Done.")


if __name__ == "__main__":
    asyncio.run(main())
