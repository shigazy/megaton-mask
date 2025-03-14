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
import uuid
from .utils.jpg_sequence import download_jpg_sequence, download_jpg, prepare_sam2_jpg_sequence

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
        # Configure temp directory for chunks
        self.temp_dir = os.environ.get('MASK_CHUNKS_DIR', '/home/ec2-user/mask_chunks')
        os.makedirs(self.temp_dir, exist_ok=True)
        print(f"[Manager.py] Using {self.temp_dir} for temporary mask chunks")

    def log_memory_usage(self, step_name=""):
        """Log current memory usage for debugging"""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / (1024 ** 3)
            reserved = torch.cuda.memory_reserved() / (1024 ** 3)
            print(f"[Manager.py] [MEMORY] {step_name} - CUDA Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")

    def setup_warmup(self):
        """Initialize warmup after instance is created"""
        if self.warmup is None:
            self.warmup = ModelWarmup(self)

    async def initialize(self, is_preview=False):
        """Initialize the appropriate predictor"""
        try:
            # Clear memory before loading model
            self.cleanup()
            print("[Manager.py] You art here though 0")
            torch.cuda.empty_cache()
            gc.collect()
            print("[Manager.py] You art here though 1")
            if is_preview:
                print("[Manager.py] You art here though 1.1")
                # Use tiny model for previews
                model_path = "/home/ec2-user/megaton-roto/backend/inference/sam2/checkpoints/sam2.1_hiera_tiny.pt"  # Adjust path as needed
                if self.preview_predictor is None:
                    model_cfg = determine_model_cfg(model_path)
                    self.preview_predictor = build_sam2_video_predictor(model_cfg, model_path, device="cuda:0")
                self.log_memory_usage("Initialized preview predictor")
                return self.preview_predictor
            else:
                print("[Manager.py] You art here though 2")
                # Use regular model for full masks
                model_path = "/home/ec2-user/megaton-roto/backend/inference/sam2/checkpoints/sam2.1_hiera_large.pt"  # Adjust path as needed
                if self.predictor is None:
                    model_cfg = determine_model_cfg(model_path)
                    print("[Manager.py] You art here though 2.1")
                    self.predictor = build_sam2_video_predictor(model_cfg, model_path, device="cuda:0")
                    print("[Manager.py] You art here though 2.2")
                self.log_memory_usage("Initialized full predictor")
                return self.predictor
        except Exception as e:
            print(f"[Manager.py] Error initializing predictor: {e}")
            raise

    async def generate_mask(self, video_path, points, bbox, start_frame, jpg_dir_key: Optional[str] = None):
        """
        Generate preview mask using tiny model.
        If jpg_dir_key is provided, download the JPG file for the specified frame
        for inference instead of using the video.
        """
        print(f"[Manager.py] generate_mask called with video_path: {video_path}, start_frame: {start_frame}")
        print(f"[Manager.py] jpg_dir_key: {jpg_dir_key}")
        temp_dirs = []  # To track temporary directories for cleanup

        try:
            # If a JPG sequence key is provided, use it to get the specific frame image.
            if jpg_dir_key:
                print(f"[Manager.py] Processing preview mask using JPG frame from: {jpg_dir_key}")
                # download_jpg returns the local file path to the specific frame
                process_path = download_jpg(jpg_dir_key, start_frame)
                # Store the temporary directory for cleanup.
                temp_dirs.append(os.path.dirname(process_path))
                print(f"[Manager.py] Using JPG file for inference: {process_path}")
            else:
                process_path = video_path
                print(f"[Manager.py] Using video file for inference: {process_path}")

            # Transform points if they are provided as a list instead of a dict.
            if isinstance(points, list):
                grouped_points = {'positive': [], 'negative': []}
                for p in points:
                    if p.get('type') == 'positive':
                        grouped_points['positive'].append([p['x'], p['y']])
                    elif p.get('type') == 'negative':
                        grouped_points['negative'].append([p['x'], p['y']])
                points = grouped_points

            # Initialize the predictor (using full model for propagation)
            predictor = await self.initialize(is_preview=False)
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
                if self.current_state is None:
                    print(f"[Manager.py] Initializing state with process_path: {process_path}")
                    self.log_memory_usage("Before init_state")
                    # For a JPG file, offloading the video to CPU is not needed.
                    self.current_state = predictor.init_state(process_path, offload_video_to_cpu=(jpg_dir_key is None))
                    self.log_memory_usage("After init_state")

                # Prepare points and labels arrays.
                all_points = []
                all_labels = []
                if points and 'positive' in points and points['positive']:
                    all_points.extend(points['positive'])
                    all_labels.extend([1] * len(points['positive']))
                if points and 'negative' in points and points['negative']:
                    all_points.extend(points['negative'])
                    all_labels.extend([0] * len(points['negative']))
                points_array = np.array(all_points) if all_points else None
                labels_array = np.array(all_labels) if all_labels else None

                print(f"[Manager.py] Points array: {points_array}")
                print(f"[Manager.py] Labels array: {labels_array}")

                # Add points/box to the first frame
                frame_idx, obj_ids, masks = predictor.add_new_points_or_box(
                    self.current_state,
                    frame_idx=0,  # Process a single frame preview.
                    obj_id=0,
                    points=points_array,
                    labels=labels_array,
                    box=bbox if bbox else None
                )
                self.log_memory_usage("After add_new_points_or_box")

                if masks is None or len(masks) == 0:
                    raise ValueError("No masks generated")
                
                print(f"[Manager.py] Initial masks: {masks}, length: {len(masks)}")
                
                # Propagate the mask to at least 3 frames
                print(f"[Manager.py] Propagating mask to frame 3...")
                raw_masks = list(predictor.propagate_in_video(
                    inference_state=self.current_state,
                    start_frame_idx=0,
                    max_frame_num_to_track=4,  # Need at least 4 frames to get to index 3
                    reverse=False
                ))
                
                print(f"[Manager.py] Collected {len(raw_masks)} propagated masks")
                
                # Process the propagated masks to extract frame 3
                masks_list = {}
                for raw_mask_item in raw_masks:
                    if isinstance(raw_mask_item, tuple) and len(raw_mask_item) >= 3:
                        frame_idx = raw_mask_item[0]  # First element is the frame index
                        mask_tensor = raw_mask_item[2]  # THIRD element is the mask tensor
                        print(f"[Manager.py] Unpacked mask tuple for frame {frame_idx}")
                        
                        if hasattr(mask_tensor, 'cpu'):
                            mask_np = mask_tensor.cpu().numpy()
                            
                            # Ensure mask is 2D boolean
                            if len(mask_np.shape) == 4:  # [B, C, H, W]
                                mask_np = mask_np[0, 0] > 0.0  # Take first batch, first channel
                            elif len(mask_np.shape) == 3:  # [C, H, W]
                                mask_np = mask_np[0] > 0.0  # Take first channel
                            else:
                                mask_np = mask_np > 0.0
                            
                            masks_list[frame_idx] = mask_np
                
                # Check if we have frame 3
                if 2 in masks_list:
                    result = masks_list[2]
                    print(f"[Manager.py] Returning mask for frame 2 with shape: {result.shape}")
                else:
                    # Fallback to frame 0 if frame 3 is not available
                    print(f"[Manager.py] Frame 2 not found, using frame 0 instead")
                    if 0 in masks_list:
                        result = masks_list[0]
                    else:
                        print(f"[Manager.py] Frame 0 not found, using frame 1 instead")
                        # Last resort: use the initial mask
                        mask_np = masks[0][0].cpu().numpy()
                        result = mask_np > 0.0

                # Cache the result with a key based on the input parameters.
                cache_key = f"{video_path}:{str(points)}:{str(bbox)}:{start_frame}"
                self.cache.set(cache_key, None, None, result)

                return result

        except Exception as e:
            print(f"[Manager.py] Error in generate_mask: {e}")
            import traceback
            traceback.print_exc()
            raise
        finally:
            # Cleanup temporary directories created by download_jpg.
            for temp_dir in temp_dirs:
                try:
                    if os.path.exists(temp_dir):
                        shutil.rmtree(temp_dir)
                        print(f"[Manager.py] Removed temporary directory: {temp_dir}")
                except Exception as ex:
                    print(f"[Manager.py] Error removing temporary directory {temp_dir}: {ex}")
            torch.cuda.empty_cache()
            gc.collect()

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
            print(f"[Manager.py] Processing batch: {batch}")
            results = []
            for request in batch:
                video_path = request.get('video_path')
                points = request.get('points')
                labels = request.get('labels')
                bbox = request.get('bbox')
                frame = request.get('current_frame')
                jpg_dir_key = request.get('jpg_dir_key')
                
                # Generate mask for each request in batch
                mask = await self.generate_mask(video_path, points, bbox, frame, jpg_dir_key)
                results.append(mask)
                
                # Clear cache between requests
                torch.cuda.empty_cache()
            
            # TODO: if progress updates break in generate full mask, it's here that's the issue
            # if progress_callback:
            #     progress_value = min(len(results), total_frames)
            #     try:
            #         progress_callback(progress_value, total_frames)
            #         print(f"[Manager.py] Updated progress: {progress_value}/{total_frames}")
            #     except Exception as e:
            #         print(f"[Manager.py] Error in progress callback: {e}")
            
            return results
        except Exception as e:
            print(f"[Manager.py] Error processing batch: {e}")
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
        progress_callback = None,
        jpg_dir_key: Optional[str] = None  # New parameter
    ) -> np.ndarray:
        """Generate masks for all frames in a video or JPG sequence"""
        temp_dirs = []  # Track temporary directories to clean up
        
        try:
            # Clear memory before starting
            self.cleanup()
            torch.cuda.empty_cache()
            gc.collect()
            
            # Use regular model for full masks
            predictor = await self.initialize(is_preview=False)
            self.log_memory_usage("After initialize in generate_full_video_masks")
            
            # Get video info (either from video file or first JPG)
            if jpg_dir_key:
                print(f"[Manager.py] Processing from JPG sequence: {jpg_dir_key}")
                # Download the JPG sequence from S3
                jpg_dir = download_jpg_sequence(jpg_dir_key)
                temp_dirs.append(jpg_dir)
                
                # Get the frame count and dimensions from the JPG sequence
                jpg_files = sorted([f for f in os.listdir(jpg_dir) if f.endswith('.jpg')])
                total_frames = len(jpg_files)
                
                if total_frames == 0:
                    raise ValueError(f"No JPG frames found in directory: {jpg_dir}")
                    
                # Get dimensions from the first frame
                first_frame = cv2.imread(os.path.join(jpg_dir, jpg_files[0]))
                height, width = first_frame.shape[:2]
                
                print(f"[Manager.py] JPG sequence info: {total_frames} frames, {width}x{height}")
                
                # Create a reordered sequence for SAM2
                sam2_dir, sam2_to_original = prepare_sam2_jpg_sequence(
                    jpg_dir, 
                    total_frames, 
                    start_frame
                )
                temp_dirs.append(sam2_dir)
                
                # SAM2 will use this directory path instead of a video
                process_path = sam2_dir
                
            else:
                # Fall back to using the video file
                print(f"[Manager.py] Processing from video file: {video_path}")
                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    print("[Manager.py] Error: Unable to open video file")
                else:
                    print("[Manager.py] Video file opened successfully")
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                cap.release()
                
                print(f"[Manager.py] Video info: {total_frames} frames, {width}x{height}")
                process_path = video_path
                
                # Create frame mapping for regular video processing
                _, sam2_to_original = create_sam2_frame_mapping(total_frames, start_frame)
            
            # Add debug print to verify callback is passed correctly
            print(f"[Manager.py] Progress callback provided: {progress_callback is not None}")
            
            # Determine optimal batch size based on resolution
            if width * height > 1280 * 720:
                batch_size = 10
            else:
                batch_size = 25
                
            print(f"[Manager.py] Using batch size: {batch_size} for resolution {width}x{height}")

            # Run the heavy computation in a thread pool
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, self._generate_masks,
                predictor, process_path, points, bbox, super_mode, method,
                total_frames, height, width, batch_size, 0,  # Use 0 as start_frame for SAM2
                progress_callback, sam2_to_original)  # Pass the mapping
            
            print("[Manager.py] Masks generation completed")
            self.log_memory_usage("After masks generation")
            
            # The result will be correctly ordered based on the original frames
            if progress_callback:
                try:
                    progress_callback(total_frames, total_frames)
                    print(f"[Manager.py] Final progress update: {total_frames}/{total_frames}")
                except Exception as e:
                    print(f"[Manager.py] Error in progress callback: {e}")
            
            return result

        except Exception as e:
            print(f"[Manager.py] Error in generate_full_video_masks: {e}")
            raise
        finally:
            # Clean up temporary directories
            for temp_dir in temp_dirs:
                try:
                    if os.path.exists(temp_dir):
                        shutil.rmtree(temp_dir)
                        print(f"[Manager.py] Removed temporary directory: {temp_dir}")
                except Exception as e:
                    print(f"[Manager.py] Error removing temporary directory {temp_dir}: {e}")
            
            self.cleanup()
            torch.cuda.empty_cache()
            gc.collect()

    def _generate_masks(self, predictor, video_path, points, bbox, super_mode, method,
                       total_frames, height, width, batch_size=25, start_frame=0, 
                       progress_callback=None, sam2_to_original=None):
        """Non-async version of mask generation to run in thread pool"""
        try:
            print(f"[Manager.py] _generate_masks received progress_callback: {progress_callback is not None}")
            torch.cuda.empty_cache()
            gc.collect()
            self.log_memory_usage("Before mask generation")
            print(f"[Manager.py] _generate_masks received total_frames: {total_frames}")
            
            # Final masks storage
            final_masks = []
            
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
                # Initialize state with video path (which could be a JPG sequence directory)
                start_time = time.time()
                try:
                    self.log_memory_usage("Before init_state")
                    state = predictor.init_state(video_path, offload_video_to_cpu=False)
                    self.log_memory_usage("After init_state")

                    # After state initialization, add:
                    if progress_callback:
                        try:
                            progress_callback(5, total_frames)
                            print(f"[Manager.py] Updated progress: 5/{total_frames}")
                        except Exception as e:
                            print(f"[Manager.py] Error in progress callback: {e}")
                except Exception as exc:
                    print(f"[Manager.py] Error initializing state: {exc}")
                    import traceback
                    traceback.print_exc()
                    raise

                # Prepare masks_list - a dict for all frame results
                masks_list = {}
                
                # Update progress callback for initialization (5% progress)
                if progress_callback:
                    try:
                        progress_callback(int(total_frames * 0.05), total_frames)
                        print(f"[Manager.py] Updated progress: 5%")
                    except Exception as e:
                        print(f"[Manager.py] Error in progress callback: {e}")
                
                # Prepare points
                all_points = []
                all_labels = []
                
                if points and 'positive' in points and points['positive']:
                    all_points.extend(points['positive'])
                    all_labels.extend([1] * len(points['positive']))
                if points and 'negative' in points and points['negative']:
                    all_points.extend(points['negative'])
                    all_labels.extend([0] * len(points['negative']))

                points_array = np.array(all_points) if all_points else None
                labels_array = np.array(all_labels) if all_labels else None
                
                # Process the initial frame (now always frame 0 in SAM2's perspective)
                try:
                    print(f"[Manager.py] Adding points/box to frame 0")
                    frame_idx, obj_ids, initial_masks = predictor.add_new_points_or_box(
                        state,
                        frame_idx=0,  # Always use frame 0 for SAM2
                        obj_id=0,
                        points=points_array,
                        labels=labels_array,
                        box=bbox if bbox else None
                    )
                    self.log_memory_usage("After add_new_points_or_box")
                    
                    if initial_masks is not None and len(initial_masks) > 0:
                        # Store this in the SAM2 frame index for now
                        masks_list[0] = initial_masks[0][0].cpu().numpy() > 0.0
                        print(f"[Manager.py] Initial mask generated for frame 0")
                    else:
                        print(f"[Manager.py] Warning: No initial mask generated for frame 0")
                except Exception as e:
                    print(f"[Manager.py] Error in add_new_points_or_box: {e}")
                    import traceback
                    traceback.print_exc()
                    raise
                    
                print("[Manager.py] Processing video frames...")
                
                # Simplified propagation - now just a single call since reordering is handled externally
                try:
                    print(f"[Manager.py] Calling propagation with standard parameters for {total_frames} frames")
                    
                    # Call SAM2's propagation with standard parameters
                    raw_masks = list(predictor.propagate_in_video(
                        inference_state=state,
                        start_frame_idx=0,
                        max_frame_num_to_track=total_frames,
                        reverse=False
                    ))
                    
                    print(f"[Manager.py] Collecting propagated masks...")
                    for raw_mask_item in raw_masks:
                        # Handle the case where raw_mask_item is a tuple (which appears to be the case)
                        if isinstance(raw_mask_item, tuple):
                            if len(raw_mask_item) >= 3:  # The tuple has 3 elements (frame_idx, obj_ids, mask_tensor)
                                frame_idx = raw_mask_item[0]  # First element is the frame index
                                mask = raw_mask_item[2]       # THIRD element is the mask tensor
                                print(f"[Manager.py] Unpacked mask tuple for frame {frame_idx}")
                                # Now store the mask after moving it to CPU
                                if hasattr(mask, 'cpu'):
                                    mask_np = mask.cpu().numpy()
                                    print(f"[Manager.py] Mask shape: {mask_np.shape}")
                                    
                                    # Ensure mask is 2D boolean by taking the first channel if needed
                                    if len(mask_np.shape) == 4:  # [B, C, H, W]
                                        mask_np = mask_np[0, 0] > 0.0  # Take first batch, first channel
                                    elif len(mask_np.shape) == 3:  # [C, H, W]
                                        mask_np = mask_np[0] > 0.0  # Take first channel
                                    else:
                                        mask_np = mask_np > 0.0
                                    
                                    masks_list[frame_idx] = mask_np
                                else:
                                    print(f"[Manager.py] Warning: Mask for frame {frame_idx} is not a tensor: {type(mask)}")
                            else:
                                print(f"[Manager.py] Warning: Mask tuple has unexpected length: {len(raw_mask_item)}")
                        else:
                            # Handle the case where it's not a tuple (the original expected format)
                            print(f"[Manager.py] Processing non-tuple mask of type: {type(raw_mask_item)}")
                            # If it's not a tuple, we don't know the frame idx, so skip
                            continue
                    
                    # Update progress after processing all masks
                    if progress_callback:
                        try:
                            progress_callback(min(total_frames, 1), total_frames)
                        except Exception as e:
                            print(f"[Manager.py] Error in progress callback: {e}")
                        
                    # After raw_masks = predictor.propagate_in_video(...), add:
                    if progress_callback:
                        try:
                            progress_callback(total_frames // 2, total_frames)
                            print(f"[Manager.py] Updated progress: {total_frames//2}/{total_frames}")
                        except Exception as e:
                            print(f"[Manager.py] Error in progress callback: {e}")
                    
                except Exception as e:
                    print(f"[Manager.py] Error in propagate_in_video: {e}")
                    import traceback
                    traceback.print_exc()
                
                self.log_memory_usage("After collecting all masks")
                print(f"[Manager.py] Collected masks for {len(masks_list)} frames")
                
                # Generate final colored masks per frame in small batches
                print("[Manager.py] Generating final masks...")
                final_masks = []
                
                # Convert frame-indexed dict to a list in the ORIGINAL video order
                ordered_masks = []
                for sam2_idx in range(total_frames):
                    if sam2_idx in masks_list:
                        # Get the mask for this SAM2 frame
                        mask = masks_list[sam2_idx]
                        
                        # If we have a mapping, convert back to original frame index
                        if sam2_to_original:
                            original_idx = sam2_to_original.get(sam2_idx, sam2_idx)
                        else:
                            original_idx = sam2_idx
                        
                        # Add to ordered list, ensuring the original order is maintained
                        while len(ordered_masks) <= original_idx:
                            ordered_masks.append(None)
                        ordered_masks[original_idx] = mask
                
                # Fill any gaps in ordered_masks
                for i in range(len(ordered_masks)):
                    if ordered_masks[i] is None:
                        print(f"[Manager.py] Warning: Missing mask for original frame {i}")
                        ordered_masks[i] = np.zeros((height, width), dtype=bool)
                
                # Save ordered masks for debugging
                debug_dir = f"/home/ec2-user/megaton-roto-dev/backend/tmp/sam2_debug/ordered_masks_{str(uuid.uuid4())[:8]}"
                os.makedirs(debug_dir, exist_ok=True)
                print(f"[Manager.py] Saving ordered masks for debugging to {debug_dir}")
                for i, mask in enumerate(ordered_masks):
                    if mask is not None:
                        debug_path = f"{debug_dir}/ordered_mask_{i:08d}.npy"
                        np.save(debug_path, mask.astype(np.uint8))
                print(f"[Manager.py] Saved {len(ordered_masks)} ordered masks to {debug_dir}")
                
                # Generate colored masks in the original order
                mask_color = [(255, 255, 255)]  # White color for masks
                
                for batch_start in range(0, total_frames, batch_size):
                    batch_end = min(batch_start + batch_size, total_frames)
                    print(f"[Manager.py] Processing batch {batch_start}-{batch_end}...")
                    
                    batch_masks = []
                    for frame_idx in range(batch_start, batch_end):
                        if frame_idx < len(ordered_masks) and ordered_masks[frame_idx] is not None:
                            img = np.zeros((height, width, 3), np.uint8)
                            mask = ordered_masks[frame_idx]
                            img[mask] = mask_color[0]  # Using first color for now
                            batch_masks.append(img)
                        else:
                            # Create an empty mask if missing
                            img = np.zeros((height, width, 3), np.uint8)
                            batch_masks.append(img)
                    
                    final_masks.extend(batch_masks)
                    del batch_masks
                    gc.collect()
                    self.log_memory_usage(f"After batch {batch_start}-{batch_end}")
                    
                    if progress_callback:
                        progress_value = min(batch_end, total_frames)
                        try:
                            progress_callback(progress_value, total_frames)
                            print(f"[Manager.py] Updated progress: {progress_value}/{total_frames}")
                        except Exception as e:
                            print(f"[Manager.py] Error in progress callback: {e}")
                
                # Clear memory
                del ordered_masks, masks_list
                gc.collect()
                
                print(f"[Manager.py] Generated {len(final_masks)} frame masks")
                self.log_memory_usage("After generating final masks")
                
                # Save ALL final colored masks for debugging (not just every 10th)
                debug_dir = f"/home/ec2-user/megaton-roto-dev/backend/tmp/sam2_debug/final_masks_{str(uuid.uuid4())[:8]}"
                os.makedirs(debug_dir, exist_ok=True)
                print(f"[Manager.py] Saving ALL final colored masks for debugging to {debug_dir}")
                for i, mask in enumerate(final_masks):
                    debug_path = f"{debug_dir}/final_mask_{i:08d}.jpg"
                    cv2.imwrite(debug_path, mask)
                print(f"[Manager.py] Saved {len(final_masks)} final masks to {debug_dir}")
                
                # Save masks as needed
                job_id = str(uuid.uuid4())
                chunk_paths = self._process_and_save_masks(final_masks, job_id)
                
                if len(chunk_paths) == 1:
                    result = np.load(chunk_paths[0])
                    os.remove(chunk_paths[0])
                    print(f"[Manager.py] Returning single numpy array of shape {result.shape}")
                    return result
                else:
                    print(f"[Manager.py] Returning {len(chunk_paths)} chunk paths")
                    return chunk_paths
                
            # After the for loop that processes raw_mask_items, add:
            # Save raw masks for debugging
            debug_dir = f"/home/ec2-user/megaton-roto-dev/backend/tmp/sam2_debug/raw_masks_{str(uuid.uuid4())[:8]}"
            os.makedirs(debug_dir, exist_ok=True)
            print(f"[Manager.py] Saving raw masks for debugging to {debug_dir}")
            for frame_idx, mask in masks_list.items():
                if mask is not None:
                    debug_path = f"{debug_dir}/raw_mask_{frame_idx:08d}.npy"
                    np.save(debug_path, mask.astype(np.uint8))
            print(f"[Manager.py] Saved {len(masks_list)} raw masks to {debug_dir}")
                
        except Exception as e:
            print(f"[Manager.py] Error in _generate_masks: {type(e).__name__}: {str(e)}")
            import traceback
            traceback.print_exc()
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
        print("[Manager.py] Starting _stack_masks_in_chunks")
        if len(masks_list) <= chunk_size:
            print(f"[Manager.py] Masks list length {len(masks_list)} <= chunk_size {chunk_size}, stacking directly")
            return np.stack(masks_list)
        
        print(f"[Manager.py] Processing {len(masks_list)} masks in chunks of {chunk_size}")
        # Process in chunks
        chunks = []
        for i in range(0, len(masks_list), chunk_size):
            print(f"[Manager.py] Processing chunk {i} to {min(i+chunk_size, len(masks_list))}")
            chunk = masks_list[i:i+chunk_size]
            stacked_chunk = np.stack(chunk)
            chunks.append(stacked_chunk)
            print(f"[Manager.py] Chunk shape: {stacked_chunk.shape}")
            # Clear original chunk data
            for j in range(i, min(i+chunk_size, len(masks_list))):
                masks_list[j] = None
            gc.collect()
            print(f"[Manager.py] Memory cleared for chunk {i}")
        
        # If return_chunks=True, return the list of chunks directly
        if return_chunks:
            print(f"[Manager.py] Returning {len(chunks)} separate chunks")
            return chunks
        
        # Otherwise try to concatenate but with safety limits
        try:
            # Check total size before concatenating
            total_elements = sum(chunk.size for chunk in chunks)
            element_size = chunks[0].itemsize
            total_size_gb = (total_elements * element_size) / (1024 ** 3)
            
            print(f"[Manager.py] Attempting to concatenate {len(chunks)} chunks (estimated {total_size_gb:.2f} GB)")
            if total_size_gb > 4:  # Arbitrary 4GB safety limit
                print("[Manager.py] WARNING: Final array would be too large (> 4GB). Returning chunks instead.")
                return chunks
            
            # Proceed with concatenation if size is reasonable
            print(f"[Manager.py] Concatenating {len(chunks)} chunks")
            result = np.concatenate(chunks, axis=0)
            print(f"[Manager.py] Concatenation complete, result shape: {result.shape}")
            del chunks
            gc.collect()
            print("[Manager.py] Chunks deleted and garbage collected")
            return result
            
        except (MemoryError, ValueError) as e:
            print(f"[Manager.py] Memory error during concatenation: {e}. Returning chunks instead.")
            return chunks

    def _process_and_save_masks(self, masks_list, job_id, chunk_size=100):
        """Process masks in chunks and save directly to disk"""
        import os
        import uuid
        import numpy as np
        
        # Create a unique subfolder for this job
        job_dir = os.path.join(self.temp_dir, str(job_id))
        os.makedirs(job_dir, exist_ok=True)
        
        # Check disk space before starting
        try:
            disk_usage = shutil.disk_usage(job_dir)
            free_gb = disk_usage.free / (1024**3)
            print(f"[Manager.py] Available disk space: {free_gb:.2f} GB")
            if free_gb < 5:  # 5GB safety threshold
                print(f"[Manager.py] WARNING: Low disk space! Only {free_gb:.2f} GB available")
        except Exception as e:
            print(f"[Manager.py] Warning: Unable to check disk space: {e}")
        
        chunk_paths = []
        for i in range(0, len(masks_list), chunk_size):
            chunk_end = min(i+chunk_size, len(masks_list))
            print(f"[Manager.py] Processing chunk {i} to {chunk_end}")
            
            try:
                # Process this chunk
                chunk = masks_list[i:chunk_end]
                print(f"[Manager.py] Stacking chunk of size {len(chunk)}")
                stacked_chunk = np.stack(chunk)
                
                # Save this chunk with a unique name
                chunk_filename = f"chunk_{i}_{chunk_end}_{uuid.uuid4().hex[:8]}.npy"
                chunk_path = os.path.join(job_dir, chunk_filename)
                print(f"[Manager.py] Saving chunk to disk: {chunk_path}")
                
                try:
                    np.save(chunk_path, stacked_chunk)
                    print(f"[Manager.py] Chunk saved to disk: {chunk_path}")
                    chunk_paths.append(chunk_path)
                except Exception as save_error:
                    print(f"[Manager.py] ERROR saving chunk to {chunk_path}: {save_error}")
                    import traceback
                    traceback.print_exc()
                    continue  # Skip this chunk but try to process remaining chunks
                
                # Clear memory - using correct index
                print(f"[Manager.py] Clearing memory for chunk {i}-{chunk_end}")
                del chunk, stacked_chunk
                for j in range(i, chunk_end):
                    if j < len(masks_list):  # Guard against index errors
                        masks_list[j] = None
                gc.collect()
                print(f"[Manager.py] Memory cleared for chunk {i}-{chunk_end}")
                
            except Exception as chunk_error:
                print(f"[Manager.py] ERROR processing chunk {i}-{chunk_end}: {chunk_error}")
                import traceback
                traceback.print_exc()
        
        print(f"[Manager.py] All processing complete. Saved {len(chunk_paths)} chunks out of {len(masks_list) // chunk_size + (1 if len(masks_list) % chunk_size else 0)} expected")
        
        # Return even if we have partial results
        return chunk_paths

    def cleanup_saved_chunks(self, chunk_paths, keep_combined=False):
        """
        Clean up saved chunk files after they've been processed
        
        Args:
            chunk_paths: List of paths to chunk files
            keep_combined: If True, keeps the final combined file
        """
        import os
        
        print(f"[Manager.py] Cleaning up {len(chunk_paths)} chunk files")
        
        for path in chunk_paths:
            if os.path.exists(path):
                try:
                    os.remove(path)
                    print(f"[Manager.py] Removed chunk file: {path}")
                except Exception as e:
                    print(f"[Manager.py] Failed to remove chunk file {path}: {e}")
        
        print("[Manager.py] Chunk cleanup complete")

    def check_disk_space(self, min_free_gb=10):
        """Check if there's enough disk space available"""
        # Get the disk where chunks are stored
        chunk_dir = "/path/to/your/chunks/directory"
        
        # Get disk usage statistics
        disk_usage = shutil.disk_usage(chunk_dir)
        
        # Convert to GB
        free_gb = disk_usage.free / (1024**3)
        
        if free_gb < min_free_gb:
            print(f"[Manager.py] WARNING: Low disk space! Only {free_gb:.1f}GB free")
            # Trigger emergency cleanup
            self.emergency_cleanup()
        
    def emergency_cleanup(self):
        """Delete old chunks to free space in low-disk situations"""
        import os
        import time
        import glob
        
        # Get the temporary directory where chunks are stored
        chunk_dir = "/path/to/your/chunks/directory"  # Update this to your actual path
        
        # Find all chunk files
        chunk_files = glob.glob(os.path.join(chunk_dir, "*.npy"))
        
        # Sort by modification time (oldest first)
        chunk_files.sort(key=os.path.getmtime)
        
        # Delete oldest 50% of files
        files_to_delete = chunk_files[:len(chunk_files)//2]
        
        for file_path in files_to_delete:
            try:
                os.remove(file_path)
                print(f"[Manager.py] Emergency cleanup: Removed {file_path}")
            except Exception as e:
                print(f"[Manager.py] Failed to remove {file_path}: {e}")
        
        print(f"[Manager.py] Emergency cleanup deleted {len(files_to_delete)} oldest chunk files")

# Create singleton instance
inference_manager = InferenceManager()
# Set up warmup after instance is created
inference_manager.setup_warmup()
