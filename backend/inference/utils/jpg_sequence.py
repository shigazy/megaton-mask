import os
import boto3
import tempfile
import shutil
import re
import json
import uuid
from typing import Dict, List, Tuple, Optional

# Configure S3 client
s3_client = boto3.client('s3')
BUCKET_NAME = os.environ.get('AWS_S3_BUCKET', 'megaton-roto-videos')

def download_jpg_sequence(jpg_dir_key: str, temp_dir: Optional[str] = None) -> str:
    """
    Download a JPG sequence from S3 to a local temporary directory.
    
    Args:
        jpg_dir_key: S3 key for the directory containing JPG sequence
        temp_dir: Optional temporary directory to use (creates one if not provided)
    
    Returns:
        Path to local directory containing the downloaded JPG sequence
    """
    if temp_dir is None:
        temp_dir = tempfile.mkdtemp()
    
    print(f"[JPG Sequence] Downloading JPG sequence from S3: {jpg_dir_key} to {temp_dir}")
    
    # List all objects in the directory
    response = s3_client.list_objects_v2(
        Bucket=BUCKET_NAME,
        Prefix=jpg_dir_key
    )
    
    if 'Contents' not in response:
        raise ValueError(f"No JPG sequence found at {jpg_dir_key}")
    
    # Get file count from S3
    jpg_count = len([obj for obj in response['Contents'] if not obj['Key'].endswith('/')])
    print(f"[JPG Sequence] Found {jpg_count} JPG files in S3 at {jpg_dir_key}")
    
    # Track downloaded files
    downloaded_files = 0
    
    # Download each JPG file
    for obj in response['Contents']:
        # Skip the directory object itself
        if obj['Key'].endswith('/'):
            continue
        
        # Get the filename without the path
        filename = f"{downloaded_files}.jpg"
        local_path = os.path.join(temp_dir, filename)
        
        # Download the file
        s3_client.download_file(
            Bucket=BUCKET_NAME,
            Key=obj['Key'],
            Filename=local_path
        )
        downloaded_files += 1
    
    # Verify we downloaded everything
    actual_files = len([f for f in os.listdir(temp_dir) if f.endswith('.jpg')])
    print(f"[JPG Sequence] Downloaded {downloaded_files} frames, found {actual_files} in local directory")
    
    if downloaded_files != actual_files:
        raise ValueError(f"Downloaded file count mismatch: expected {downloaded_files}, found {actual_files}")
    
    # Save file list for debugging
    debug_dir = "/home/ec2-user/megaton-roto-dev/backend/tmp/sam2_debug"
    os.makedirs(debug_dir, exist_ok=True)
    debug_path = f"{debug_dir}/downloaded_files_{str(uuid.uuid4())[:8]}.txt"
    
    with open(debug_path, 'w') as f:
        f.write(f"S3 Path: {jpg_dir_key}\n")
        f.write(f"Expected files: {jpg_count}\n")
        f.write(f"Downloaded files: {downloaded_files}\n")
        f.write("Files in directory:\n")
        for file in sorted(os.listdir(temp_dir)):
            if file.endswith('.jpg'):
                f.write(f"  {file}\n")
    
    return temp_dir 

def create_sam2_frame_mapping(
    total_frames: int, 
    start_frame: int,
    super_mode: bool
) -> Tuple[Dict[int, int], Dict[int, int], List[Tuple[int, int]]]:
    """
    Create mapping plans for reordering frames for SAM2 processing, returning both
    dictionary and tuple-based mappings.
    
    Args:
        total_frames: Total number of frames in the sequence
        start_frame: The user-specified starting frame for mask propagation
    
    Returns:
        Tuple containing:
            - original_to_sam2: Mapping from original frame indices to SAM2 frame indices (with offset key for duplicates)
            - sam2_to_original: Mapping from SAM2 frame indices to original frame indices
            - original_to_sam2_tuples: List of (orig_idx, sam2_idx) pairs for handling duplicates cleanly
    """
    # Basic validation
    if start_frame < 0 or start_frame >= total_frames:
        raise ValueError(f"start_frame {start_frame} out of bounds (total frames: {total_frames})")
    
    original_to_sam2 = {}  # Dictionary mapping
    sam2_to_original = {}
    original_to_sam2_tuples = []  # Tuple-based mapping
    
    if super_mode:
        sam2_idx = 0
        
        # FIRST PASS: Forward from start_frame, wrapping around to start_frame-1
        # This creates the sequence: start_frame, start_frame+1, ..., total_frames-1, 0, 1, ..., start_frame-1
        for offset in range(total_frames):
            orig_idx = (start_frame + offset) % total_frames
            original_to_sam2[orig_idx] = sam2_idx
            original_to_sam2_tuples.append((orig_idx, sam2_idx))
            sam2_to_original[sam2_idx] = orig_idx
            sam2_idx += 1
        
        # SECOND PASS: Backward from start_frame-1, wrapping around to start_frame
        # This creates the sequence: start_frame-1, start_frame-2, ..., 0, total_frames-1, ..., start_frame+1, start_frame
        for offset in range(total_frames):
            # Calculate backward index, starting from (start_frame-1) and going in reverse
            orig_idx = (start_frame - 1 - offset) % total_frames
            
            # Add to tuples list first (to preserve all mappings)
            original_to_sam2_tuples.append((orig_idx, sam2_idx))
            
            # For dictionary mapping, use offset for duplicates
            if orig_idx in original_to_sam2:
                original_to_sam2[orig_idx + total_frames] = sam2_idx
            else:
                original_to_sam2[orig_idx] = sam2_idx
            
            sam2_to_original[sam2_idx] = orig_idx
            sam2_idx += 1
    else:
        # Regular mode: simple 1:1 mapping
        for i in range(total_frames):
            original_to_sam2[i] = i
            original_to_sam2_tuples.append((i, i))
            sam2_to_original[i] = i
    
    # Save mapping info for debugging
    debug_dir = "/home/ec2-user/megaton-roto-dev/backend/tmp/sam2_debug"
    os.makedirs(debug_dir, exist_ok=True)
    debug_id = str(uuid.uuid4())[:8]
    debug_path = f"{debug_dir}/mapping_{debug_id}.json"
    
    with open(debug_path, 'w') as f:
        json.dump({
            "original_to_sam2": {str(k): v for k, v in original_to_sam2.items()},
            "sam2_to_original": {str(k): v for k, v in sam2_to_original.items()},
            "original_to_sam2_tuples": original_to_sam2_tuples,
            "total_frames": total_frames,
            "start_frame": start_frame
        }, f, indent=2)
    print(f"[JPG Sequence] Saved mapping debug info to {debug_path}")
    
    return original_to_sam2, sam2_to_original, original_to_sam2_tuples

def prepare_sam2_jpg_sequence(
    src_dir: str, 
    total_frames: int, 
    start_frame: int,
    super_mode: bool
) -> Tuple[str, Dict[int, int]]:
    """
    Create a new JPG sequence in the order SAM2 should process it.
    
    Args:
        src_dir: Source directory containing the original JPG sequence
        total_frames: Total number of frames 
        start_frame: The user-specified starting frame for mask propagation
        super_mode: Whether to use super mode (duplicate frames)
    
    Returns:
        Tuple containing:
            - Path to the directory with the reordered sequence
            - Mapping from SAM2 frame indices to original frame indices
    """
    # Create mapping plans
    original_to_sam2, sam2_to_original, original_to_sam2_tuples = create_sam2_frame_mapping(
        total_frames, start_frame, super_mode
    )
    
    # Create a temporary directory for the reordered sequence
    dest_dir = tempfile.mkdtemp()
    
    # Get list of all jpg files in the source directory
    jpg_files = sorted([f for f in os.listdir(src_dir) if f.endswith('.jpg')])
    
    if len(jpg_files) != total_frames:
        print(f"[JPG Sequence] Warning: Found {len(jpg_files)} files but expected {total_frames} frames")
    
    # Create a mapping from frame number to filename
    frame_to_filename = {}
    
    # Try multiple regex patterns to extract frame numbers
    # First try "frame_X.jpg" format
    frame_pattern1 = re.compile(r'frame_(\d+)\.jpg')
    # Then try "X.jpg" format
    frame_pattern2 = re.compile(r'(\d+)\.jpg')
    
    # Print some sample filenames for debugging
    print(f"[JPG Sequence] Sample filenames: {jpg_files[:5]}")
    
    # Try the first pattern (frame_X.jpg)
    for filename in jpg_files:
        match = frame_pattern1.match(filename)
        if match:
            frame_num = int(match.group(1))
            frame_to_filename[frame_num] = filename
            
    # If first pattern didn't work, try the second pattern (X.jpg)
    if not frame_to_filename:
        print("[JPG Sequence] Trying alternative filename pattern (X.jpg)")
        for filename in jpg_files:
            match = frame_pattern2.match(filename)
            if match:
                frame_num = int(match.group(1))
                frame_to_filename[frame_num] = filename
    
    # If neither pattern worked, fall back to sequential ordering
    if not frame_to_filename:
        print("[JPG Sequence] Couldn't extract frame numbers from filenames, assuming sequential ordering")
        for i, filename in enumerate(jpg_files):
            frame_to_filename[i] = filename
    
    # Print frame mapping for debugging
    print(f"[JPG Sequence] Frame mapping sample: {dict(list(frame_to_filename.items())[:5])}")
    
    # Total number of SAM2 frames (double for super_mode)
    total_sam2_frames = len(sam2_to_original)
    print(f"[JPG Sequence] Creating sequence with {total_sam2_frames} frames")
    
    # Create the reordered sequence with sequential numeric filenames (0.jpg, 1.jpg, etc.)
    copied_frames = 0
    print(f"[JPG Sequence] First Frame: {sam2_to_original[0]}")
    for sam2_idx in range(total_sam2_frames):
        orig_idx = sam2_to_original[sam2_idx]
        
        if orig_idx in frame_to_filename:
            src_file = os.path.join(src_dir, frame_to_filename[orig_idx])
            # Create a sequential filename starting from 0.jpg
            dest_file = os.path.join(dest_dir, f"{sam2_idx}.jpg")
            shutil.copy(src_file, dest_file)
            copied_frames += 1
            if sam2_idx % 50 == 0:  # Log every 50 frames to avoid excessive output
                print(f"[JPG Sequence] Copied original frame {orig_idx} to SAM2 frame {sam2_idx}")
        else:
            print(f"[JPG Sequence] Warning: Couldn't find original frame {orig_idx} in frame_to_filename")
    
    print(f"[JPG Sequence] Copied {copied_frames} out of {total_sam2_frames} expected frames")
    
    # Save all frames for debugging
    debug_dir = f"/home/ec2-user/megaton-roto-dev/backend/tmp/sam2_debug/frames_{str(uuid.uuid4())[:8]}"
    os.makedirs(debug_dir, exist_ok=True)
    
    # Save a sample of frames for debugging (first 5, middle 5, last 5)
    sample_indices = list(range(0, min(5, total_sam2_frames)))
    if total_sam2_frames > 10:
        middle = total_sam2_frames // 2
        sample_indices.extend(range(middle-2, middle+3))
    if total_sam2_frames > 5:
        sample_indices.extend(range(max(0, total_sam2_frames-5), total_sam2_frames))
    
    for sam2_idx in sample_indices:
        src_file = os.path.join(dest_dir, f"{sam2_idx}.jpg")
        if os.path.exists(src_file):
            shutil.copy(src_file, f"{debug_dir}/sam2_frame_{sam2_idx:08d}.jpg")
    
    print(f"[JPG Sequence] Saved sample frames to {debug_dir}")
    print(f"[JPG Sequence] Created reordered sequence with {total_sam2_frames} frames in {dest_dir}")
    
    # Save sequence details for debugging
    with open(os.path.join(debug_dir, "sequence_info.json"), 'w') as f:
        json.dump({
            "total_original_frames": total_frames,
            "total_sam2_frames": total_sam2_frames,
            "start_frame": start_frame,
            "super_mode": super_mode,
            "frame_to_filename_sample": {str(k): v for k, v in list(frame_to_filename.items())[:10]},
            "sam2_to_original_sample": {str(k): v for k, v in list(sam2_to_original.items())[:10]}
        }, f, indent=2)
    
    return dest_dir, sam2_to_original

def download_jpg(jpg_dir_key: str, frame: int, temp_dir: Optional[str] = None) -> str:
    """
    Download a single JPG file from S3 corresponding to the specified frame index.
    
    Args:
        jpg_dir_key: S3 key for the directory containing the JPG sequence.
        frame: The 0-indexed frame number to download.
        temp_dir: Optional temporary directory to use (creates one if not provided).
        
    Returns:
        The local file path to the downloaded JPG.
    """
    if temp_dir is None:
        temp_dir = tempfile.mkdtemp()
    
    print(f"[JPG] Downloading JPG from S3: {jpg_dir_key} to {temp_dir} for frame: {frame}")
    
    # List the JPG objects in the specified S3 directory
    response = s3_client.list_objects_v2(
        Bucket=BUCKET_NAME,
        Prefix=jpg_dir_key
    )
    
    if 'Contents' not in response:
        raise ValueError(f"No JPG files found at {jpg_dir_key}")
    
    # Filter out and sort JPG files
    # Filter objects to only include JPG files by checking if the Key (S3 object path) ends with '.jpg'
    jpg_objs = [obj for obj in response['Contents'] if obj['Key'].lower().endswith('.jpg')]
    # Sort the JPG objects alphabetically by their Key names to ensure correct frame order
    jpg_objs = sorted(jpg_objs, key=lambda x: x['Key'])
    
    total_frames = len(jpg_objs)
    print(f"[JPG] Found {total_frames} JPG files at {jpg_dir_key}")
    
    if frame < 0 or frame >= total_frames:
        raise ValueError(f"Frame index {frame} is out of range (total frames: {total_frames})")
    
    target_obj = jpg_objs[frame]
    target_obj_mid = jpg_objs[frame + 1]
    filename = os.path.basename(target_obj['Key'])
    filename_mid = os.path.basename(target_obj_mid['Key'])
    local_path = os.path.join(temp_dir, filename)
    local_path_mid = os.path.join(temp_dir, filename_mid)
    
    # Download the specific JPG file
    s3_client.download_file(
        Bucket=BUCKET_NAME,
        Key=target_obj['Key'],
        Filename=local_path
    )
    # Download the middle frame JPG file
    s3_client.download_file(
        Bucket=BUCKET_NAME,
        Key=target_obj_mid['Key'],
        Filename=local_path_mid
    )
    # Save a copy of the downloaded JPG to a debug directory
    base_debug_dir = "/home/ec2-user/megaton-roto-dev/backend/inference/output_masks"
    
    # Create a unique subfolder using UUID
    unique_id = str(uuid.uuid4())[:8]
    debug_dir = os.path.join(base_debug_dir, unique_id)
    os.makedirs(debug_dir, exist_ok=True)
    
    # Use the original frame number for the filename
    debug_filename = f"frame_{frame}.jpg"
    debug_path = os.path.join(debug_dir, debug_filename)
    
    # Copy the downloaded file to the debug location
    try:
        shutil.copy2(local_path, debug_path)
        print(f"[JPG] Saved debug copy to {debug_path}")
    except Exception as e:
        print(f"[JPG] Failed to save debug copy: {str(e)}")

    # After downloading, compute a new name and rename the file.
    new_filename = f"{int(frame)}.jpg"  # Using the numeric value we computed earlier.
    new_filename_mid = f"{int(frame) + 1}.jpg"
    new_local_path = os.path.join(temp_dir, new_filename)
    new_local_path_mid = os.path.join(temp_dir, new_filename_mid)
    os.rename(local_path, new_local_path)
    os.rename(local_path_mid, new_local_path_mid)
    # Download the specific JPG file
    s3_client.download_file(
        Bucket=BUCKET_NAME,
        Key=target_obj['Key'],
        Filename=local_path
    )

    final_frame = int(frame) + 2
    new_filename = f"{final_frame}.jpg"
    new_local_path = os.path.join(temp_dir, new_filename)
    os.rename(local_path, new_local_path)
    local_path = new_local_path
    
    print(f"[JPG] Downloaded frame {frame} to {local_path}")
    
    return temp_dir 