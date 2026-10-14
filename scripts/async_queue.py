"""Keep a bounded number of jobs in flight, replenishing each completed slot."""
import asyncio


async def consume(items, handle, workers):
    iterator = iter(items)

    async def worker():
        while True:
            item = next(iterator, None)
            if item is None:
                return
            await handle(item)

    async with asyncio.TaskGroup() as group:
        for _ in range(workers):
            group.create_task(worker())
