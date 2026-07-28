from __future__ import annotations

import asyncio


async def retry_async(fn, attempts: int = 2, backoff: float = 1.0):
    for i in range(attempts):
        try:
            return await fn()
        except Exception:
            if i == attempts - 1:
                raise
            await asyncio.sleep(backoff * (2 ** i))
