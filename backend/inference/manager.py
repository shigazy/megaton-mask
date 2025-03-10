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
import time
import shutil

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
            print("You art here though 0")
            torch.cuda.empty_cache()
            gc.collect()
            print("You art here though 1")
            if is_preview:
                print("You art here though 1.1")
                # Use tiny model for previews
                model_path = "/home/ec2-user/megaton-roto/backend/inference/sam2/checkpoints/sam2.1_hiera_tiny.pt"  # Adjust path as needed
                if self.preview_predictor is None:
                    model_cfg = determine_model_cfg(model_path)
                    self.preview_predictor = build_sam2_video_predictor(model_cfg, model_path, device="cuda:0")
                self.log_memory_usage("Initialized preview predictor")
                return self.preview_predictor
            else:
                print("You art here though 2")
                # Use regular model for full masks
                model_path = "/home/ec2-user/megaton-roto/backend/inference/sam2/checkpoints/sam2.1_hiera_large.pt"  # Adjust path as needed
                if self.predictor is None:
                    model_cfg = determine_model_cfg(model_path)
                    print("You art here though 2.1")
                    self.predictor = build_sam2_video_predictor(model_cfg, model_path, device="cuda:0")
                    print("You art here though 2.2")
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
                    self.log_memory_usage("Before init_state")
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
            print(video_path)
            # Get video info
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print("Error: Unable to open video file")
            else:
                print("Video file opened successfully")
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            print(f"Video info: {total_frames} frames, {width}x{height}")
            cap.release()
            
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
            print("You are here 1")
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
                # Initialize state with video
                print("You are here 1.1")
                start_time = time.time()
                try:
                    self.log_memory_usage("Before init_state")
                    state = predictor.init_state(video_path, offload_video_to_cpu=False)
                    self.log_memory_usage("After init_state")
                except Exception as exc:
                    print(f"Error during predictor.init_state: {exc}")
                    raise
                end_time = time.time()
                print(f"You are here 1.2 -- init_state completed in {end_time - start_time:.2f} seconds")
                self.log_memory_usage("After init_state")
                print("You are here 2")
                # Prepare points
                all_points = []
                all_labels = []
                
                if points.get('positive'):
                    all_points.extend(points['positive'])
                    all_labels.extend([1] * len(points['positive']))
                    
                if points.get('negative'):
                    all_points.extend(points['negative'])
                    all_labels.extend([0] * len(points['negative']))
                print("You are here 3")
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

                # Initialize tracking at the specified start_frame
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


                # Forward propagation: process frames forward from start_frame
                print("First pass (forward propagation): collecting masks...")
                for frame_idx, object_ids, masks in predictor.propagate_in_video(state):
                    # Only process frames with index >= start_frame
                    print(f"Processing frame {frame_idx}")
                    if frame_idx < start_frame:
                        continue
                    frame_masks = {}
                    for obj_id, mask in zip(object_ids, masks):
                        # Move to CPU immediately to free GPU memory
                        mask_np = mask[0].cpu().numpy() > 0.0
                        frame_masks[obj_id] = mask_np
                        # Delete the tensor immediately
                        del mask
                    all_frame_masks[frame_idx] = frame_masks
                    frames_processed += 1
                    print(f"Frames processed: {frames_processed}")
                    # Update progress callback with debug prints
                    if progress_callback and frames_processed % 5 == 0:
                        print(f"Forward propagation: calling progress_callback with {frames_processed}/{total_frames}")
                        try:
                            progress_callback(frames_processed, total_frames)
                            print("Progress callback executed successfully in forward propagation")
                        except Exception as e:
                            print(f"Error in forward progress callback: {e}")
                    
                    print(f"Forward processed frame {frame_idx}/{total_frames}")
                    
                    # Explicitly clear CUDA cache more frequently
                    if frame_idx % 10 == 0:
                        torch.cuda.empty_cache()
                        self.log_memory_usage(f"After forward frame {frame_idx}")
                
                # --- New Code: Backward propagation block ---
                if hasattr(predictor, "propagate_backward"):
                    print("Backward propagation supported; processing frames before start_frame...")
                    for frame_idx, object_ids, masks in predictor.propagate_backward(state):
                        if frame_idx >= start_frame:
                            continue  # Only process frames before start_frame
                        frame_masks = {}
                        for obj_id, mask in zip(object_ids, masks):
                            mask_np = mask[0].cpu().numpy() > 0.0
                            frame_masks[obj_id] = mask_np
                            del mask
                        all_frame_masks[frame_idx] = frame_masks
                        frames_processed += 1
                        if progress_callback and frames_processed % 5 == 0:
                            print(f"Backward propagation: calling progress_callback with {frames_processed}/{total_frames}")
                            try:
                                progress_callback(frames_processed, total_frames)
                                print("Progress callback executed successfully in backward propagation")
                            except Exception as e:
                                print(f"Error in backward propagation progress callback: {e}")
                        print(f"Backward processed frame {frame_idx}")
                        if frame_idx % 10 == 0:
                            torch.cuda.empty_cache()
                    print("Backward propagation completed.")
                else:
                    print("Backward propagation not supported; skipping backward processing.")
                # --- End of backward propagation block ---
                
                # Final progress update for both forward and backward propagation
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
                final_batch_size = min(batch_size, 20)
                
                for batch_start in range(0, total_frames, final_batch_size):
                    batch_end = min(batch_start + final_batch_size, total_frames)
                    batch_masks = []
                    
                    print(f"Processing batch {batch_start}-{batch_end}...")
                    for frame_idx in range(batch_start, batch_end):
                        img = np.zeros((height, width, 3), np.uint8)
                        frame_masks = all_frame_masks.get(frame_idx, {})
                        for obj_id, mask in frame_masks.items():
                            img[mask] = mask_color[(obj_id + 1) % len(mask_color)]
                            del mask
                        batch_masks.append(img)
                        if frame_idx in all_frame_masks:
                            del all_frame_masks[frame_idx]
                    
                    final_masks.extend(batch_masks)
                    del batch_masks
                    gc.collect()
                    torch.cuda.empty_cache()
                    self.log_memory_usage(f"After batch {batch_start}-{batch_end}")
                
                del all_frame_masks
                gc.collect()
                
                print("Stacking final masks...")
                result = self._stack_masks_in_chunks(final_masks, chunk_size=100, return_chunks=True)
                
                # If result is a list of chunks, handle differently
                if isinstance(result, list):
                    print(f"Got {len(result)} chunks instead of a single array. Handling chunks...")
                    # You can either:
                    # 1. Process each chunk separately
                    # 2. Save each chunk to disk separately
                    # 3. Or merge them into a different format/data structure
                
                print(f"Final masks shape: {result.shape}")
                self.log_memory_usage("After stacking final masks")
                
                # After sending results to client
                self.cleanup_saved_chunks(result)
                
                return result
                
        except Exception as e:
            print(f"Error in _generate_masks: {e}")
            torch.cuda.empty_cache()
            gc.collect()
            raise
        finally:
            if 'state' in locals():
                del state
            torch.cuda.empty_cache()
            gc.collect()
            self.log_memory_usage("After _generate_masks cleanup")
    
    def _stack_masks_in_chunks(self, masks_list, chunk_size=100, return_chunks=False):
        """
        Stack masks in chunks to reduce peak memory usage.
        If return_chunks=True, returns a list of chunked arrays instead of one big array.
        """
        print("Starting _stack_masks_in_chunks")
        if len(masks_list) <= chunk_size:
            print(f"Masks list length {len(masks_list)} <= chunk_size {chunk_size}, stacking directly")
            return np.stack(masks_list)
        
        print(f"Processing {len(masks_list)} masks in chunks of {chunk_size}")
        # Process in chunks
        chunks = []
        for i in range(0, len(masks_list), chunk_size):
            print(f"Processing chunk {i} to {min(i+chunk_size, len(masks_list))}")
            chunk = masks_list[i:i+chunk_size]
            stacked_chunk = np.stack(chunk)
            chunks.append(stacked_chunk)
            print(f"Chunk shape: {stacked_chunk.shape}")
            # Clear original chunk data
            for j in range(i, min(i+chunk_size, len(masks_list))):
                masks_list[j] = None
            gc.collect()
            print(f"Memory cleared for chunk {i}")
        
        # If return_chunks=True, return the list of chunks directly
        if return_chunks:
            print(f"Returning {len(chunks)} separate chunks")
            return chunks
        
        # Otherwise try to concatenate but with safety limits
        try:
            # Check total size before concatenating
            total_elements = sum(chunk.size for chunk in chunks)
            element_size = chunks[0].itemsize
            total_size_gb = (total_elements * element_size) / (1024 ** 3)
            
            print(f"Attempting to concatenate {len(chunks)} chunks (estimated {total_size_gb:.2f} GB)")
            if total_size_gb > 4:  # Arbitrary 4GB safety limit
                print("WARNING: Final array would be too large (> 4GB). Returning chunks instead.")
                return chunks
            
            # Proceed with concatenation if size is reasonable
            print(f"Concatenating {len(chunks)} chunks")
            result = np.concatenate(chunks, axis=0)
            print(f"Concatenation complete, result shape: {result.shape}")
            del chunks
            gc.collect()
            print("Chunks deleted and garbage collected")
            return result
            
        except (MemoryError, ValueError) as e:
            print(f"Memory error during concatenation: {e}. Returning chunks instead.")
            return chunks

    def _process_and_save_masks(self, masks_list, output_path, chunk_size=100):
        """Process masks in chunks and save directly to disk"""
        import os
        import cv2  # or use np.save
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        for i in range(0, len(masks_list), chunk_size):
            chunk_end = min(i+chunk_size, len(masks_list))
            print(f"Processing chunk {i} to {chunk_end}")
            
            # Process this chunk
            chunk = masks_list[i:chunk_end]
            stacked_chunk = np.stack(chunk)
            
            # Save this chunk
            chunk_path = f"{output_path}_{i}_{chunk_end}.npy"
            np.save(chunk_path, stacked_chunk)
            
            # Clear memory
            del chunk, stacked_chunk
            for j in range(i, chunk_end):
                masks_list[j] = None
            gc.collect()
            
        print(f"All {len(masks_list)} masks saved in chunks")
        return [f"{output_path}_{i}_{min(i+chunk_size, len(masks_list))}.npy" 
                for i in range(0, len(masks_list), chunk_size)]

    def cleanup_saved_chunks(self, chunk_paths, keep_combined=False):
        """
        Clean up saved chunk files after they've been processed
        
        Args:
            chunk_paths: List of paths to chunk files
            keep_combined: If True, keeps the final combined file
        """
        import os
        
        print(f"Cleaning up {len(chunk_paths)} chunk files")
        
        for path in chunk_paths:
            if os.path.exists(path):
                try:
                    os.remove(path)
                    print(f"Removed chunk file: {path}")
                except Exception as e:
                    print(f"Failed to remove chunk file {path}: {e}")
        
        print("Chunk cleanup complete")

    def check_disk_space(self, min_free_gb=10):
        """Check if there's enough disk space available"""
        # Get the disk where chunks are stored
        chunk_dir = "/path/to/your/chunks/directory"
        
        # Get disk usage statistics
        disk_usage = shutil.disk_usage(chunk_dir)
        
        # Convert to GB
        free_gb = disk_usage.free / (1024**3)
        
        if free_gb < min_free_gb:
            print(f"WARNING: Low disk space! Only {free_gb:.1f}GB free")
            # Trigger emergency cleanup
            self.emergency_cleanup()
        
    def emergency_cleanup(self):
        """Delete old chunks to free space in low-disk situations"""
        # Similar to scheduled_chunk_cleanup but more aggressive
        # Delete oldest files first until enough space is freed

# Create singleton instance
inference_manager = InferenceManager()
# Set up warmup after instance is created
inference_manager.setup_warmup()
