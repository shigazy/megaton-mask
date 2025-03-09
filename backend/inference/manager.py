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
from typing import Dict, List, Optional
import cv2
import asyncio
import gc

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
        # Track memory usage
        self.log_memory_usage("Initialized InferenceManager")

    def log_memory_usage(self, step_name=""):
        """Log current memory usage for debugging"""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / (1024 ** 3)
            reserved = torch.cuda.memory_reserved() / (1024 ** 3)
            print(f"[MEMORY] {step_name} - CUDA Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")

    def setup_warmup(self):
        """Initialize warmup after instance is created"""
        if self.warmup is None:
            self.warmup = ModelWarmup(self)

    async def initialize(self, is_preview=False):
        """Initialize the appropriate predictor"""
        try:
            # Clear memory before loading model
            self.cleanup()
            torch.cuda.empty_cache()
            gc.collect()
            
            if is_preview:
                # Use tiny model for previews
                model_path = "/home/ec2-user/megaton-roto/backend/inference/sam2/checkpoints/sam2.1_hiera_tiny.pt"  # Adjust path as needed
                if self.preview_predictor is None:
                    model_cfg = determine_model_cfg(model_path)
                    self.preview_predictor = build_sam2_video_predictor(model_cfg, model_path, device="cuda:0")
                self.log_memory_usage("Initialized preview predictor")
                return self.preview_predictor
            else:
                # Use regular model for full masks
                model_path = "/home/ec2-user/megaton-roto/backend/inference/sam2/checkpoints/sam2.1_hiera_large.pt"  # Adjust path as needed
                if self.predictor is None:
                    model_cfg = determine_model_cfg(model_path)
                    self.predictor = build_sam2_video_predictor(model_cfg, model_path, device="cuda:0")
                self.log_memory_usage("Initialized full predictor")
                return self.predictor
        except Exception as e:
            print(f"Error initializing predictor: {e}")
            raise

    async def generate_mask(self, video_path, points, bbox, start_frame=0):
        """Generate preview mask using tiny model"""
        print(f"generate_mask called with video_path: {video_path}, starting from frame: {start_frame}")
        
        # Check cache first
        cache_key = f"{video_path}:{str(points)}:{str(bbox)}:{start_frame}"
        cached_result = self.cache.get(cache_key, None, None)
        if cached_result is not None:
            return cached_result

        try:
            # Use tiny model for previews
            predictor = await self.initialize(is_preview=True)
            
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
                if self.current_state is None:
                    print(f"Initializing state with video_path: {video_path}")
                    self.current_state = predictor.init_state(video_path, offload_video_to_cpu=True)
                    self.log_memory_usage("After init_state")

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

                # Generate mask - now using the provided start_frame
                frame_idx, obj_ids, masks = predictor.add_new_points_or_box(
                    self.current_state,
                    frame_idx=start_frame,  # Use the provided start_frame instead of 0
                    obj_id=0,
                    points=points_array,
                    labels=labels_array,
                    box=bbox if bbox else None
                )
                self.log_memory_usage("After add_new_points_or_box")

                if masks is None or len(masks) == 0:
                    raise ValueError("No masks generated")

                # Process first frame only for preview
                mask = masks[0][0].cpu().numpy()
                result = mask > 0.0

                # Clear GPU memory
                del masks
                torch.cuda.empty_cache()
                self.log_memory_usage("After mask processing")

                # Cache the result
                self.cache.set(cache_key, None, None, result)
                
                return result

        except Exception as e:
            print(f"Error in generate_mask: {e}")
            raise
        finally:
            # Don't fully cleanup here as it might be called in a batch
            torch.cuda.empty_cache()

    def cleanup(self):
        """Cleanup resources"""
        if self.current_state is not None:
            del self.current_state
            self.current_state = None
        
        torch.cuda.empty_cache()
        gc.collect()
        self.log_memory_usage("After cleanup")

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
                
                # Clear cache between requests
                torch.cuda.empty_cache()
            
            return results
        except Exception as e:
            print(f"Error processing batch: {e}")
            raise
        finally:
            # Ensure cleanup after batch processing
            torch.cuda.empty_cache()
            gc.collect()

    async def generate_full_video_masks(
        self,
        video_path: str,
        points: Optional[Dict[str, List[List[float]]]],
        bbox: List[float],
        super_mode: bool = False,
        method: str = "default",
        start_frame: int = 0,
        progress_callback = None
    ) -> np.ndarray:
        """Generate masks for all frames in a video"""
        try:
            # Clear memory before starting
            self.cleanup()
            torch.cuda.empty_cache()
            gc.collect()
            
            # Use regular model for full masks
            predictor = await self.initialize(is_preview=False)
            self.log_memory_usage("After initialize in generate_full_video_masks")
            
            # Get video info
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            
            print(f"Video info: {total_frames} frames, {width}x{height}")
            
            # Add debug print to verify callback is passed correctly
            print(f"Progress callback provided: {progress_callback is not None}")
            
            # Determine optimal batch size based on resolution
            if width * height > 1280 * 720:
                batch_size = 10
            else:
                batch_size = 25
                
            print(f"Using batch size: {batch_size} for resolution {width}x{height}")

            # Run the heavy computation in a thread pool
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, self._generate_masks,
                predictor, video_path, points, bbox, super_mode, method,
                total_frames, height, width, batch_size, start_frame, progress_callback)
            
            print("Masks generation completed")
            self.log_memory_usage("After masks generation")
            return result

        except Exception as e:
            print(f"Error in generate_full_video_masks: {e}")
            raise
        finally:
            self.cleanup()
            torch.cuda.empty_cache()
            gc.collect()

    def _generate_masks(self, predictor, video_path, points, bbox, super_mode, method,
                       total_frames, height, width, batch_size=25, start_frame=0, progress_callback=None):
        """Non-async version of mask generation to run in thread pool"""
        try:
            # Debug print to verify callback is received
            print(f"_generate_masks received progress_callback: {progress_callback is not None}")
            
            # Start with clean memory
            torch.cuda.empty_cache()
            gc.collect()
            self.log_memory_usage("Before mask generation")
            
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
                # Initialize state with video
                state = predictor.init_state(video_path, offload_video_to_cpu=True)
                self.log_memory_usage("After init_state")
                
                # Prepare points
                all_points = []
                all_labels = []
                
                if points.get('positive'):
                    all_points.extend(points['positive'])
                    all_labels.extend([1] * len(points['positive']))
                    
                if points.get('negative'):
                    all_points.extend(points['negative'])
                    all_labels.extend([0] * len(points['negative']))

                # Convert to tensors
                if all_points:
                    points_tensor = torch.tensor(all_points, dtype=torch.float32, device="cuda")
                    labels_tensor = torch.tensor(all_labels, dtype=torch.int64, device="cuda")
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
                    frame_idx=start_frame,
                    obj_id=0,
                    points=points_tensor,
                    labels=labels_tensor,
                    box=bbox_tensor
                )
                self.log_memory_usage("After add_new_points_or_box")
                
                # Clear tensors we don't need anymore
                del points_tensor, labels_tensor, bbox_tensor
                if masks is not None:
                    del masks
                torch.cuda.empty_cache()
                
                print("Processing video frames...")
                # Use a dictionary to store masks by frame index
                # This avoids keeping all masks in memory at once
                all_frame_masks = {}
                
                # Track frames processed for progress reporting
                frames_processed = 0
                
                # Process frames in smaller batches
                print("First pass: collecting masks...")
                for frame_idx, object_ids, masks in predictor.propagate_in_video(state):
                    frame_masks = {}
                    for obj_id, mask in zip(object_ids, masks):
                        # Move to CPU immediately to free GPU memory
                        mask_np = mask[0].cpu().numpy() > 0.0
                        frame_masks[obj_id] = mask_np
                        # Delete the tensor immediately
                        del mask
                    
                    all_frame_masks[frame_idx] = frame_masks
                    frames_processed += 1
                    
                    # Update progress callback with debug prints
                    if progress_callback and frames_processed % 5 == 0:
                        print(f"Calling progress_callback with {frames_processed}/{total_frames}")
                        try:
                            progress_callback(frames_processed, total_frames)
                            print("Progress callback executed successfully")
                        except Exception as e:
                            print(f"Error in progress callback: {e}")
                    
                    print(f"Processed frame {frame_idx}/{total_frames}")
                    
                    # Explicitly clear CUDA cache more frequently
                    if frame_idx % 10 == 0:
                        torch.cuda.empty_cache()
                        self.log_memory_usage(f"After frame {frame_idx}")
                
                # Final progress update for first phase
                if progress_callback:
                    progress_callback(frames_processed, total_frames)
                
                # Clear state to free memory before final processing
                del state
                torch.cuda.empty_cache()
                gc.collect()
                self.log_memory_usage("After collecting all masks")
                
                print(f"Collected masks for {len(all_frame_masks)} frames")
                print("Generating final masks...")
                
                # Process final masks in smaller batches to reduce memory usage
                final_masks = []
                mask_color = [(255, 255, 255), (255, 255, 255), (255, 255, 255)]
                
                # Use even smaller batches for final processing
                final_batch_size = min(batch_size, 20)
                
                for batch_start in range(0, total_frames, final_batch_size):
                    batch_end = min(batch_start + final_batch_size, total_frames)
                    batch_masks = []
                    
                    print(f"Processing batch {batch_start}-{batch_end}...")
                    for frame_idx in range(batch_start, batch_end):
                        # Create empty image for this frame
                        img = np.zeros((height, width, 3), np.uint8)
                        
                        # Get masks for this frame
                        frame_masks = all_frame_masks.get(frame_idx, {})
                        
                        # Apply masks to image
                        for obj_id, mask in frame_masks.items():
                            img[mask] = mask_color[(obj_id + 1) % len(mask_color)]
                            # Delete mask after use
                            del mask
                        
                        # Add to batch
                        batch_masks.append(img)
                        
                        # Clear frame masks to save memory
                        if frame_idx in all_frame_masks:
                            del all_frame_masks[frame_idx]
                    
                    # Append batch to final masks
                    final_masks.extend(batch_masks)
                    
                    # Clear memory
                    del batch_masks
                    gc.collect()
                    torch.cuda.empty_cache()
                    self.log_memory_usage(f"After batch {batch_start}-{batch_end}")
                
                # Clear all remaining frame masks
                del all_frame_masks
                gc.collect()
                
                print("Stacking final masks...")
                # Stack masks in smaller chunks to reduce peak memory
                result = self._stack_masks_in_chunks(final_masks, chunk_size=100)
                
                print(f"Final masks shape: {result.shape}")
                self.log_memory_usage("After stacking final masks")
                
                return result
                
        except Exception as e:
            print(f"Error in _generate_masks: {e}")
            # Clean up on error
            torch.cuda.empty_cache()
            gc.collect()
            raise
        finally:
            # Always clean up
            if 'state' in locals():
                del state
            torch.cuda.empty_cache()
            gc.collect()
            self.log_memory_usage("After _generate_masks cleanup")
    
    def _stack_masks_in_chunks(self, masks_list, chunk_size=100):
        """Stack masks in chunks to reduce peak memory usage"""
        if len(masks_list) <= chunk_size:
            return np.stack(masks_list)
        
        # Process in chunks
        chunks = []
        for i in range(0, len(masks_list), chunk_size):
            chunk = masks_list[i:i+chunk_size]
            stacked_chunk = np.stack(chunk)
            chunks.append(stacked_chunk)
            # Clear original chunk data
            for j in range(i, min(i+chunk_size, len(masks_list))):
                masks_list[j] = None
            gc.collect()
        
        # Concatenate chunks
        result = np.concatenate(chunks, axis=0)
        del chunks
        gc.collect()
        
        return result

# Create singleton instance
inference_manager = InferenceManager()
# Set up warmup after instance is created
inference_manager.setup_warmup()
