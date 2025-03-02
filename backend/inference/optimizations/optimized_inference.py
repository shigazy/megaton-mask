class OptimizedInferenceManager:
    def __init__(self):
        self.batch_processor = BatchProcessor()
        self.cache = MaskCache()
        self.queue = InferenceQueue()
        self.model = QuantizedInferenceManager()
        self.warmup = ModelWarmup(self)
        
    async def startup(self):
        """Called when the service starts"""
        await self.warmup.warmup()
        
    async def generate_mask(self, video_id, points, bbox):
        # Check cache first
        cached_result = self.cache.get(video_id, points, bbox)
        if cached_result is not None:
            return cached_result
            
        # Add to batch processing queue
        result = await self.batch_processor.add_request({
            'video_id': video_id,
            'points': points,
            'bbox': bbox
        })
        
        # Cache the result
        self.cache.set(video_id, points, bbox, result)
        
        return result 