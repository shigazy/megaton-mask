from pathlib import Path
from .optimizations.batch_processor import BatchProcessor
from .optimizations.cache_manager import MaskCache
from .optimizations.queue_manager import InferenceQueue
from .optimizations.warmup import ModelWarmup
import torch
import sys
import os
import numpy as np
from functools import lru_cache
from typing import Dict, List
import cv2
import asyncio

sys.path.append("./sam2")
from sam2.build_sam import build_sam2_video_predictor

def determine_model_cfg(model_path):
    """Determine the model configuration based on the model path"""
    if "large" in model_path:
        return "configs/samurai/sam2.1_hiera_l.yaml"
    elif "base_plus" in model_path:
        return "configs/samurai/sam2.1_hiera_b+.yaml"
    elif "small" in model_path:
        return "configs/samurai/sam2.1_hiera_s.yaml"
    elif "tiny" in model_path:
        return "configs/samurai/sam2.1_hiera_t.yaml"
    else:
        raise ValueError("Unknown model size in path!")

@lru_cache(maxsize=1)
def get_predictor(model_path):
    """Cache the predictor to avoid reloading the model"""
    model_cfg = determine_model_cfg(model_path)
    return build_sam2_video_predictor(model_cfg, model_path, device="cuda:0")

class InferenceManager:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(InferenceManager, cls).__new__(cls)
            cls._instance.initialized = False
        return cls._instance

    def __init__(self):
        if self.initialized:
            return
            
        self.predictor = None
        self.preview_predictor = None  # Separate predictor for previews
        self.current_state = None
        self.cache = MaskCache()
        self.queue = InferenceQueue()
        self.warmup = None
        # Initialize batch processor with process_batch method
        self.batch_processor = BatchProcessor()
        self.batch_processor.set_processor(self.process_batch)
        self.initialized = True

    def setup_warmup(self):
        """Initialize warmup after instance is created"""
        if self.warmup is None:
            self.warmup = ModelWarmup(self)

    async def initialize(self, is_preview=False):
        """Initialize the appropriate predictor"""
        try:
            if is_preview:
                # Use tiny model for previews
                model_path = "/home/ec2-user/megaton-roto/backend/inference/sam2/checkpoints/sam2.1_hiera_tiny.pt"  # Adjust path as needed
                if self.preview_predictor is None:
                    model_cfg = determine_model_cfg(model_path)
                    self.preview_predictor = build_sam2_video_predictor(model_cfg, model_path, device="cuda:0")
                return self.preview_predictor
            else:
                # Use regular model for full masks
                model_path = "/home/ec2-user/megaton-roto/backend/inference/sam2/checkpoints/sam2.1_hiera_large.pt"  # Adjust path as needed
                if self.predictor is None:
                    model_cfg = determine_model_cfg(model_path)
                    self.predictor = build_sam2_video_predictor(model_cfg, model_path, device="cuda:0")
                return self.predictor
        except Exception as e:
            print(f"Error initializing predictor: {e}")
            raise

    async def generate_mask(self, video_path, points, bbox):
        """Generate preview mask using tiny model"""
        print(f"generate_mask called with video_path: {video_path}")
        
        # Use tiny model for previews
        predictor = await self.initialize(is_preview=True)
        
        # Check cache first
        cache_key = f"{video_path}:{str(points)}:{str(bbox)}"
        cached_result = self.cache.get(cache_key, None, None)
        if cached_result is not None:
            return cached_result

        try:
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
                if self.current_state is None:
                    print(f"Initializing state with video_path: {video_path}")
                    self.current_state = predictor.init_state(video_path, offload_video_to_cpu=True)

                # Prepare points
                all_points = []
                all_labels = []
                
                if points.get('positive'):
                    all_points.extend(points['positive'])
                    all_labels.extend([1] * len(points['positive']))
                    
                if points.get('negative'):
                    all_points.extend(points['negative'])
                    all_labels.extend([0] * len(points['negative']))

                points_array = np.array(all_points) if all_points else None
                labels_array = np.array(all_labels) if all_points else None

                # Generate mask
                frame_idx, obj_ids, masks = predictor.add_new_points_or_box(
                    self.current_state,
                    frame_idx=0,
                    obj_id=0,
                    points=points_array,
                    labels=labels_array,
                    box=bbox if bbox else None
                )

                if masks is None or len(masks) == 0:
                    raise ValueError("No masks generated")

                # Process first frame only for preview
                mask = masks[0][0].cpu().numpy()
                result = mask > 0.0

                # Cache the result
                self.cache.set(cache_key, None, None, result)
                
                return result

        except Exception as e:
            print(f"Error in generate_mask: {e}")
            raise

    def cleanup(self):
        """Cleanup resources"""
        if self.current_state is not None:
            del self.current_state
            self.current_state = None
            torch.cuda.empty_cache()

    async def process_batch(self, batch):
        """Process a batch of requests"""
        try:
            print(f"Processing batch: {batch}")
            results = []
            for request in batch:
                video_path = request.get('video_path')
                points = request.get('points')
                labels = request.get('labels')
                bbox = request.get('bbox')
                
                # Generate mask for each request in batch
                mask = await self.generate_mask(video_path, points, bbox)
                results.append(mask)
            
            return results
        except Exception as e:
            print(f"Error processing batch: {e}")
            raise

    async def generate_full_video_masks(
        self,
        video_path: str,
        points: Dict[str, List[List[float]]],
        bbox: List[float],
        super_mode: bool = False,
        method: str = "default"
    ):
        """Generate full video masks using regular model"""
        print(f"Starting full video mask generation: {video_path}")
        
        try:
            # Use regular model for full masks
            predictor = await self.initialize(is_preview=False)
            
            # Get video info
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            cap.release()

            print(f"Video info - frames: {total_frames}, size: {width}x{height}")

            # Run the heavy computation in a thread pool to not block the event loop
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, self._generate_masks,
                predictor, video_path, points, bbox, super_mode, method,
                total_frames, height, width)
            
            print("Masks generation completed")
            return result

        except Exception as e:
            print(f"Error in generate_full_video_masks: {e}")
            raise

    def _generate_masks(self, predictor, video_path, points, bbox, super_mode, method,
                       total_frames, height, width):
        """Non-async version of mask generation to run in thread pool"""
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            state = predictor.init_state(video_path, offload_video_to_cpu=True)
            
            # Prepare points
            all_points = []
            all_labels = []
            
            if points.get('positive'):
                all_points.extend(points['positive'])
                all_labels.extend([1] * len(points['positive']))
                
            if points.get('negative'):
                all_points.extend(points['negative'])
                all_labels.extend([0] * len(points['negative']))

            points_array = np.array(all_points) if all_points else None
            labels_array = np.array(all_labels) if all_points else None

            # Convert to tensors
            if points_array is not None:
                points_tensor = torch.tensor(points_array, dtype=torch.float32, device="cuda")
                labels_tensor = torch.tensor(labels_array, dtype=torch.int64, device="cuda")
            else:
                points_tensor = None
                labels_tensor = None

            # Convert bbox if provided
            bbox_tensor = None
            if bbox and all(x is not None for x in bbox):
                bbox_tensor = torch.tensor(bbox, dtype=torch.float32, device="cuda")

            # Initialize tracking
            frame_idx, obj_ids, masks = predictor.add_new_points_or_box(
                state,
                frame_idx=0,
                obj_id=0,
                points=points_tensor,
                labels=labels_tensor,
                box=bbox_tensor
            )

            
            print("Super mode (preprocess): Processing forward+reverse...")
            all_frame_masks = {}
            
            print("First pass: collecting masks...")
            for frame_idx, object_ids, masks in predictor.propagate_in_video(state):
                frame_masks = {}
                for obj_id, mask in zip(object_ids, masks):
                    mask = mask[0].cpu().numpy() > 0.0
                    frame_masks[obj_id] = mask
                all_frame_masks[frame_idx] = frame_masks
                print(f"Processed frame {frame_idx}/{total_frames}")
            
            print(f"Collected masks for {len(all_frame_masks)} frames")

            print("Generating final masks...")
            final_masks = []
            mask_color = [(255, 255, 255), (255, 255, 255), (255, 255, 255)]

            print("Forward pass...")
            for frame_idx in range(total_frames):
                img = np.zeros((height, width, 3), np.uint8)
                forward_masks = all_frame_masks.get(frame_idx, {})
                
                for obj_id, mask in forward_masks.items():
                    img[mask] = mask_color[(obj_id + 1) % len(mask_color)]
                
                final_masks.append(img)
                print(f"Generated forward mask {frame_idx}/{total_frames}")

            print("Returning final masks")
            result = np.stack(final_masks)  # Keep the RGB format
            print(f"Final masks shape: {result.shape}")
            return result

# Create singleton instance
inference_manager = InferenceManager()
# Set up warmup after instance is created
inference_manager.setup_warmup()
