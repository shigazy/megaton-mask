import torch
import numpy as np

class ModelWarmup:
    def __init__(self, inference_manager):
        self.inference_manager = inference_manager
        
    async def warmup(self):
        """Run inference with dummy data to warm up the model"""
        dummy_video_path = "path/to/dummy/video.mp4"
        dummy_points = {
            'positive': [[100, 100], [200, 200]],
            'negative': [[300, 300]]
        }
        dummy_bbox = [50, 50, 100, 100]
        
        # Run several warmup iterations
        for _ in range(3):
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
                await self.inference_manager.generate_mask(
                    dummy_video_path,
                    dummy_points,
                    dummy_bbox
                ) 