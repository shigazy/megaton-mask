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
    points = {'positive': [], 'negative': []}
    
    with open(path, 'r') as f:
        lines = f.readlines()
        # First line is bbox
        line = lines[0].strip()
        # Remove square brackets and split by comma
        line = line.strip().replace('[', '').replace(']', '')
        coords = line.split(',')
        # First 4 values are x,y,w,h, last value is the label
        x, y, w, h = map(float, coords[:4])
        label = int(coords[4])
        bbox = [x, y, w, h]

        # Look for points after "POINTS" line
        if len(lines) > 1 and lines[1].strip() == "POINTS":
            for line in lines[2:]:
                type_, px, py = line.strip().split(',')
                if type_ == 'p':
                    points['positive'].append([float(px), float(py)])
                elif type_ == 'n':
                    points['negative'].append([float(px), float(py)])

    return bbox, points, label

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

def main(args, bbox=None, points=None):
    model_cfg = determine_model_cfg(args.model_path)
    predictor = build_sam2_video_predictor(model_cfg, args.model_path, device="cuda:0")
    frames_or_path = prepare_frames_or_path(args.video_path)
    
    # Use provided bbox and points if available, otherwise load from file
    if bbox is None or points is None:
        bbox, points, label = load_txt(args.txt_path)
    else:
        label = 0  # Default label if not using a text file

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

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        state = predictor.init_state(frames_or_path, offload_video_to_cpu=True)
        
        all_points = []
        all_labels = []
        
        if points.get('positive'):
            all_points.extend(points['positive'])
            all_labels.extend([1] * len(points['positive']))
            
        if points.get('negative'):
            all_points.extend(points['negative'])
            all_labels.extend([0] * len(points['negative']))

        if all_points:
            points_array = np.array(all_points)
            labels_array = np.array(all_labels)
        else:
            points_array = None
            labels_array = None

        frame_idx, obj_ids, masks = predictor.add_new_points_or_box(
            state,
            frame_idx=0,
            obj_id=0,
            points=points_array,
            labels=labels_array,
            box=bbox
        )

        for frame_idx, object_ids, masks in predictor.propagate_in_video(state):
            mask_to_vis = {}
            bbox_to_vis = {}

            for obj_id, mask in zip(object_ids, masks):
                mask = mask[0].cpu().numpy()
                mask = mask > 0.0
                non_zero_indices = np.argwhere(mask)
                if len(non_zero_indices) == 0:
                    bbox = [0, 0, 0, 0]
                else:
                    y_min, x_min = non_zero_indices.min(axis=0).tolist()
                    y_max, x_max = non_zero_indices.max(axis=0).tolist()
                    bbox = [x_min, y_min, x_max - x_min, y_max - y_min]
                bbox_to_vis[obj_id] = bbox
                mask_to_vis[obj_id] = mask

            if args.save_to_video:
                img = loaded_frames[frame_idx]
                
                original_bbox = bbox
                x, y, w, h = map(int, original_bbox)
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                
                for obj_id, mask in mask_to_vis.items():
                    mask_img = np.zeros((height, width, 3), np.uint8)
                    mask_img[mask] = color[(obj_id + 1) % len(color)]
                    img = cv2.addWeighted(img, 1, mask_img, 0.2, 0)

                for obj_id, bbox in bbox_to_vis.items():
                    cv2.rectangle(img, (bbox[0], bbox[1]), 
                                (bbox[0] + bbox[2], bbox[1] + bbox[3]), 
                                (0, 255, 0), 2)
                    
                    text = f"Original: ({x},{y},{w},{h})"
                    cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                               1, (255, 0, 0), 2)
                    text = f"Model: ({bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]})"
                    cv2.putText(img, text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 
                               1, (0, 255, 0), 2)

                out.write(img)
                
            if args.save_to_mask:
                img = loaded_frames[frame_idx]
                for obj_id, mask in mask_to_vis.items():
                    mask_img = np.zeros((height, width, 3), np.uint8)
                    mask_img[mask] = mask_color[(obj_id + 1) % len(mask_color)]
                    img = mask_img

                out_mask.write(img)
        if args.save_to_video:
            out.release()
        if args.save_to_mask:
            out_mask.release()

    del predictor, state
    gc.collect()
    torch.clear_autocast_cache()
    torch.cuda.empty_cache()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", required=True, help="Input video path or directory of frames.")
    parser.add_argument("--txt_path", required=True, help="Path to ground truth text file.")
    parser.add_argument("--model_path", default="sam2/checkpoints/sam2.1_hiera_large.pt", help="Path to the model checkpoint.")
    parser.add_argument("--video_output_path", default="demo.mp4", help="Path to save the output video.")
    parser.add_argument("--save_to_video", action='store_true', default=True, help="Save results to a video.")
    parser.add_argument("--mask_path", required=True, help="Path to save the mask images.")
    parser.add_argument("--save_to_mask", action='store_true', default=True, help="Save results to mask files.")

    args = parser.parse_args()
    main(args)
