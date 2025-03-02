import asyncio
from fastapi import BackgroundTasks

class InferenceQueue:
    def __init__(self, max_concurrent=4):
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.queue = asyncio.Queue()
        self.processing = False
        
    async def add_task(self, task_data):
        await self.queue.put(task_data)
        if not self.processing:
            asyncio.create_task(self._process_queue())
            
    async def _process_queue(self):
        self.processing = True
        
        while not self.queue.empty():
            async with self.semaphore:
                task_data = await self.queue.get()
                try:
                    await self._process_task(task_data)
                finally:
                    self.queue.task_done()
                    
        self.processing = False 