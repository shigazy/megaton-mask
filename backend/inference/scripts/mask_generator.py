import argparse
import os
import os.path as osp
import numpy as np
import cv2
import torch
import sys
sys.path.append("./sam2")
from sam2.build_sam import build_sam2_video_predictor

def generate_single_mask(video_path, points, bbox, config_file=None, ckpt_path=None):
    """Generate a single mask for the first frame."""
    try:
        # Get the absolute path to the model files
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        sam2_dir = os.path.join(base_dir, "sam2")
        
        if not ckpt_path:
            ckpt_path = os.path.join(sam2_dir, "checkpoints/sam2.1_hiera_large.pt")
        if not config_file:
            # Use relative path for config file as required by Hydra
            config_file = "configs/samurai/sam2.1_hiera_l.yaml"

        print(f"Using model path: {ckpt_path}")
        print(f"Using config path: {config_file}")
        
        # Add sam2 directory to Python path so Hydra can find configs
        if sam2_dir not in sys.path:
            sys.path.append(sam2_dir)

        # Ensure temp directory exists
        import tempfile
        os.makedirs('/tmp', exist_ok=True)

        predictor = build_sam2_video_predictor(
            config_file=config_file,
            ckpt_path=ckpt_path,
            device="cuda:0"
        )
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

        # Create a temporary file in /tmp directory (not using context manager)
        temp_mask_path = tempfile.mktemp(suffix='.png', dir='/tmp')
            
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
        
        # Save mask to temp file if we got one
        if masks is not None and len(masks) > 0:
            mask = masks[0]
            print(f"Mask type: {type(mask)}, shape: {mask.shape if hasattr(mask, 'shape') else 'unknown'}")
            
            # Convert mask to numpy array
            if isinstance(mask, torch.Tensor):
                mask_np = mask.detach().cpu().numpy()
            else:
                mask_np = np.array(mask)
            
            print(f"Numpy mask shape: {mask_np.shape}, dtype: {mask_np.dtype}")
            
            # Ensure mask is 2D and has correct dimensions
            if len(mask_np.shape) > 2:
                mask_np = mask_np.squeeze()  # Remove singleton dimensions
            
            # Ensure values are in [0, 1] range
            mask_np = (mask_np > 0.5).astype(np.uint8) * 255
            
            # Save the mask
            success = cv2.imwrite(temp_mask_path, mask_np)
            if not success:
                raise ValueError(f"Failed to save mask to {temp_mask_path}")
                
            print(f"Saved mask to {temp_mask_path}")
            return temp_mask_path
        else:
            return None
        
    except Exception as e:
        print(f"Mask generation error: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

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