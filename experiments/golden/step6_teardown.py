"""Step 6: drop the scratch database. Evidence lives in results/<run>/ (D3).

GOLDEN_KEEP_DB=1 skips the drop for a follow-up probe; drop it afterwards.
"""

from __future__ import annotations

import asyncio
import os

from lib import GOLDEN_DB, drop_database


async def main() -> None:
    db = os.environ.get("GOLDEN_REUSE_DB") or GOLDEN_DB
    if os.environ.get("GOLDEN_KEEP_DB") == "1":
        print(f"keeping {db} (GOLDEN_KEEP_DB=1)")
        return
    await drop_database(db)
    print(f"dropped {db}")


if __name__ == "__main__":
    asyncio.run(main())
