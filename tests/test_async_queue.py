import asyncio
import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from async_queue import consume


class ContinuousQueueTests(unittest.IsolatedAsyncioTestCase):
    async def test_replenishes_without_waiting_for_long_request(self):
        first_started = asyncio.Event()
        third_started = asyncio.Event()
        completed = []

        async def handle(item):
            if item == 0:
                first_started.set()
                # A fixed-batch implementation deadlocks here.
                await third_started.wait()
            elif item == 1:
                await first_started.wait()
            elif item == 2:
                self.assertNotIn(0, completed)
                third_started.set()
            completed.append(item)

        await asyncio.wait_for(consume(range(4), handle, workers=2), timeout=2)
        self.assertEqual(sorted(completed), list(range(4)))

    async def test_infrastructure_error_cancels_other_requests(self):
        running = asyncio.Event()
        cancelled = asyncio.Event()

        async def handle(item):
            if item == 0:
                await running.wait()
                raise RuntimeError('engine failure')
            try:
                running.set()
                await asyncio.Event().wait()
            finally:
                cancelled.set()

        with self.assertRaises(ExceptionGroup):
            await asyncio.wait_for(consume(range(2), handle, workers=2), timeout=2)
        self.assertTrue(cancelled.is_set())
