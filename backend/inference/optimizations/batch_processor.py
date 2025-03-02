from collections import deque
import asyncio
import torch

class BatchProcessor:
    def __init__(self, batch_size=4, max_wait_time=0.1):
        self.batch_size = batch_size
        self.max_wait_time = max_wait_time
        self.queue = deque()
        self.processing = False
        self.processor = None  # Will be set by InferenceManager
        
    def set_processor(self, processor):
        """Set the processor function that will handle the batch"""
        self.processor = processor
        
    async def add_request(self, request_data):
        future = asyncio.Future()
        self.queue.append((request_data, future))
        
        if not self.processing:
            asyncio.create_task(self._process_batch())
            
        return await future
        
    async def _process_batch(self):
        if not self.processor:
            raise RuntimeError("Processor not set")
            
        self.processing = True
        
        while self.queue:
            batch = []
            futures = []
            
            # Collect batch
            while len(batch) < self.batch_size and self.queue:
                request_data, future = self.queue.popleft()
                batch.append(request_data)
                futures.append(future)
                
                if len(batch) < self.batch_size:
                    try:
                        await asyncio.sleep(self.max_wait_time)
                    except asyncio.TimeoutError:
                        break
                        
            # Process batch
            try:
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
                    results = await self.processor(batch)
                    
                # Set results
                for future, result in zip(futures, results):
                    future.set_result(result)
            except Exception as e:
                for future in futures:
                    future.set_exception(e)
                
        self.processing = False 