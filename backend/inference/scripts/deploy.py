import argparse
import os
import os.path as osp
import numpy as np
import cv2
import torch
import gc
import sys
sys.path.append("./sam2")
from sam2.build_sam import build_sam2_video_predictor

color = [(255, 0, 0)]
mask_color = [(255, 255, 255)]

def load_txt(path):
    prompts = []
    with open(path, 'r') as f:
        for line in f:
            # Remove square brackets and split by comma
            line = line.strip().replace('[', '').replace(']', '')
            coords = line.split(',')
            # First 4 values are x,y,w,h, last value is the label
            x, y, w, h = map(float, coords[:4])
            label = int(coords[4])
            prompts.append([[x, y, w, h], label])
    return prompts

def determine_model_cfg(model_path):
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

def prepare_frames_or_path(video_path):
    if video_path.endswith(".mp4") or osp.isdir(video_path):
        return video_path
    else:
        raise ValueError("Invalid video_path format. Should be .mp4 or a directory of jpg frames.")

def main(args):
    model_cfg = determine_model_cfg(args.model_path)
    predictor = build_sam2_video_predictor(model_cfg, args.model_path, device="cuda:0")
    frames_or_path = prepare_frames_or_path(args.video_path)
    prompts = load_txt(args.txt_path)

    frame_rate = 30
    if args.save_to_video or args.save_to_mask:
        if osp.isdir(args.video_path):
            frames = sorted([osp.join(args.video_path, f) for f in os.listdir(args.video_path) if f.endswith((".jpg", ".jpeg", ".JPG", ".JPEG"))])
            loaded_frames = [cv2.imread(frame_path) for frame_path in frames]
            height, width = loaded_frames[0].shape[:2]
        else:
            cap = cv2.VideoCapture(args.video_path)
            frame_rate = cap.get(cv2.CAP_PROP_FPS)
            loaded_frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                loaded_frames.append(frame)
            cap.release()
            height, width = loaded_frames[0].shape[:2]

            if len(loaded_frames) == 0:
                raise ValueError("No frames were loaded from the video.")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.video_output_path, fourcc, frame_rate, (width, height))
    out_mask = cv2.VideoWriter(args.mask_path, fourcc, frame_rate, (width, height))

    # Enter inference mode and use automatic casting for CUDA with float16 precision
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        # Initialize the predictor state with the video frames or path, allowing video data to be offloaded to CPU
        state = predictor.init_state(frames_or_path, offload_video_to_cpu=True)
        # Extract the first bounding box and its associated label from the prompts
        bbox, track_label = prompts[0]
        # Add new points or bounding box to the predictor state for the first frame and object ID 0
        _, _, masks = predictor.add_new_points_or_box(state, box=bbox, frame_idx=0, obj_id=0)

        # Iterate over each frame index, object IDs, and masks as propagated by the predictor in the video
        for frame_idx, object_ids, masks in predictor.propagate_in_video(state):
            # Initialize dictionaries to store masks and bounding boxes for visualization
            mask_to_vis = {}
            bbox_to_vis = {}

            # Iterate over each object ID and its corresponding mask
            for obj_id, mask in zip(object_ids, masks):
                # Convert the first mask to a numpy array on the CPU
                mask = mask[0].cpu().numpy()
                # Create a binary mask where values greater than 0 are set to True
                mask = mask > 0.0
                # Find the indices of non-zero elements in the mask
                non_zero_indices = np.argwhere(mask)
                # If no non-zero indices are found, set the bounding box to zero
                if len(non_zero_indices) == 0:
                    bbox = [0, 0, 0, 0]
                else:
                    # Calculate the minimum and maximum coordinates of the non-zero indices
                    y_min, x_min = non_zero_indices.min(axis=0).tolist()
                    y_max, x_max = non_zero_indices.max(axis=0).tolist()
                    # Define the bounding box as [x_min, y_min, width, height]
                    bbox = [x_min, y_min, x_max - x_min, y_max - y_min]
                # Store the bounding box and mask for the current object ID
                bbox_to_vis[obj_id] = bbox
                mask_to_vis[obj_id] = mask

            # If the save_to_video flag is set, process the current frame for video output
            if args.save_to_video:
                # Retrieve the current frame image from the loaded frames
                img = loaded_frames[frame_idx]
                # Iterate over each object ID and its corresponding mask for visualization
                for obj_id, mask in mask_to_vis.items():
                    # Create an empty image for the mask with the same dimensions as the frame
                    mask_img = np.zeros((height, width, 3), np.uint8)
                    # Apply the color to the mask image where the mask is True
                    mask_img[mask] = color[(obj_id + 1) % len(color)]
                    # Blend the mask image with the original frame image
                    img = cv2.addWeighted(img, 1, mask_img, 0.2, 0)

                # Iterate over each object ID and its corresponding bounding box for visualization
                for obj_id, bbox in bbox_to_vis.items():
                    # Draw the bounding box on the frame image
                    cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), color[obj_id % len(color)], 2)

                # Write the processed frame image to the video output
                out.write(img)
                
            if args.save_to_mask:
                # Retrieve the current frame image from the loaded frames
                img = loaded_frames[frame_idx]
                # Iterate over each object ID and its corresponding mask for visualization
                for obj_id, mask in mask_to_vis.items():
                    # Create an empty image for the mask with the same dimensions as the frame
                    mask_img = np.zeros((height, width, 3), np.uint8)
                    # Apply the color to the mask image where the mask is True
                    mask_img[mask] = mask_color[(obj_id + 1) % len(mask_color)]
                    # Blend the mask image with the original frame image
                    img = mask_img

                # Write the processed frame image to the video output
                out_mask.write(img)
        # Release the video writer if the save_to_video flag is set
        if args.save_to_video:
            out.release()
        if args.save_to_mask:
            out_mask.release()

    # Delete the predictor and state objects to free up memory
    del predictor, state
    # Perform garbage collection to clean up unused objects
    gc.collect()
    # Clear the automatic casting cache for PyTorch
    torch.clear_autocast_cache()
    # Empty the CUDA cache to free up GPU memory
    torch.cuda.empty_cache()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", required=True, help="Input video path or directory of frames.")
    parser.add_argument("--txt_path", required=True, help="Path to ground truth text file.")
    parser.add_argument("--model_path", default="sam2/checkpoints/sam2.1_hiera_base_plus.pt", help="Path to the model checkpoint.")
    parser.add_argument("--video_output_path", default="demo.mp4", help="Path to save the output video.")
    parser.add_argument("--save_to_video", action='store_true', default=True, help="Save results to a video.")
    parser.add_argument("--mask_path", required=True, help="Path to save the mask images.")
    parser.add_argument("--save_to_mask", action='store_true', default=True, help="Save results to mask files.")

    args = parser.parse_args()
    main(args)
