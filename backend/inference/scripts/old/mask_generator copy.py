import argparse
import os
import os.path as osp
import numpy as np
import cv2
import torch
import sys
sys.path.append("./sam2")
from sam2.build_sam import build_sam2_video_predictor

def generate_single_mask(video_path, points, bbox, model_path):
    """Generate a single mask for the first frame."""
    try:
        print(f"Starting mask generation with video_path: {video_path}")
        print(f"Points: {points}")
        print(f"Bbox: {bbox}")
        print(f"Model path: {model_path}")

        # Determine model configuration
        model_cfg = determine_model_cfg(model_path)
        print(f"Using model config: {model_cfg}")
        
        predictor = build_sam2_video_predictor(model_cfg, model_path, device="cuda:0")
        print("Built predictor successfully")
        
        # Prepare video frames or path
        frames_or_path = prepare_frames_or_path(video_path)
        print(f"Prepared frames/path: {frames_or_path}")
        
        # Read the first frame
        if osp.isdir(video_path):
            frames = sorted([osp.join(video_path, f) for f in os.listdir(video_path) if f.endswith((".jpg", ".jpeg", ".JPG", ".JPEG"))])
            frame = cv2.imread(frames[0])
            print(f"Reading from directory, found {len(frames)} frames")
        else:
            cap = cv2.VideoCapture(video_path)
            ret, frame = cap.read()
            cap.release()
            print(f"Reading from video file, success: {ret}")
            if not ret:
                raise ValueError("Could not read the first frame from the video.")
        
        # Initialize predictor state
        state = predictor.init_state(frames_or_path, offload_video_to_cpu=True)
        print("Initialized predictor state")
        
        # Prepare points
        all_points = []
        all_labels = []
        
        # Add positive points
        for point in points['positive']:
            all_points.append(point)
            all_labels.append(1)
            
        # Add negative points
        for point in points['negative']:
            all_points.append(point)
            all_labels.append(0)
            
        print(f"Prepared {len(all_points)} total points")
        
        # Convert to numpy arrays if we have points
        if all_points:
            points_array = np.array(all_points)
            labels_array = np.array(all_labels)
            print("Converted points to numpy arrays")
        else:
            points_array = None
            labels_array = None
            print("No points provided")
        
        # Generate mask for the first frame
        print("Generating mask...")
        with torch.inference_mode():
            _, _, masks = predictor.add_new_points_or_box(
                state,
                frame_idx=0,  # Only first frame
                obj_id=0,
                points=points_array,
                labels=labels_array,
                box=bbox
            )
        
        print(f"Mask generation complete, got {len(masks) if masks is not None else 0} masks")
        return masks[0] if masks is not None and len(masks) > 0 else None
        
    except Exception as e:
        print(f"Mask generation error: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def determine_model_cfg(model_path):
    print(f"Determining config for model: {model_path}")
    if "large" in model_path:
        return "configs/samurai/sam2.1_hiera_l.yaml"
    elif "base_plus" in model_path:
        return "configs/samurai/sam2.1_hiera_b+.yaml"
    elif "small" in model_path:
        return "configs/samurai/sam2.1_hiera_s.yaml"
    elif "tiny" in model_path:
        return "configs/samurai/sam2.1_hiera_t.yaml"
    else:
        print(f"Model path: {model_path}")
        raise ValueError("Unknown model size in path!")

def prepare_frames_or_path(video_path):
    print(f"Preparing frames/path for: {video_path}")
    if video_path.endswith(".mp4") or osp.isdir(video_path):
        return video_path
    else:
        raise ValueError("Invalid video_path format. Should be .mp4 or a directory of jpg frames.")