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
        filename = os.path.basename(obj['Key'])
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
    start_frame: int
) -> Tuple[Dict[int, int], Dict[int, int]]:
    """
    Create mapping plans for reordering frames for SAM2 processing.
    
    Args:
        total_frames: Total number of frames in the sequence
        start_frame: The user-specified starting frame for mask propagation
    
    Returns:
        Tuple containing:
            - original_to_sam2: Mapping from original frame indices to SAM2 frame indices
            - sam2_to_original: Mapping from SAM2 frame indices to original frame indices
    """
    # Basic validation
    if start_frame < 0 or start_frame >= total_frames:
        raise ValueError(f"start_frame {start_frame} out of bounds (total frames: {total_frames})")
    
    original_to_sam2 = {}
    sam2_to_original = {}
    
    if start_frame == 0:
        # Simple case: no reordering needed
        for i in range(total_frames):
            original_to_sam2[i] = i
            sam2_to_original[i] = i
    else:
        # Reordering needed to make start_frame appear as frame 0 to SAM2
        # First: Forward frames (start_frame to end)
        sam2_idx = 0
        for orig_idx in range(start_frame, total_frames):
            original_to_sam2[orig_idx] = sam2_idx
            sam2_to_original[sam2_idx] = orig_idx
            sam2_idx += 1
        
        # Then: Backward frames (start_frame-1 down to 0)
        for orig_idx in range(start_frame - 1, -1, -1):
            original_to_sam2[orig_idx] = sam2_idx
            sam2_to_original[sam2_idx] = orig_idx
            sam2_idx += 1
    
    # Save mapping info for debugging
    debug_dir = "/home/ec2-user/megaton-roto-dev/backend/tmp/sam2_debug"
    os.makedirs(debug_dir, exist_ok=True)
    debug_id = str(uuid.uuid4())[:8]
    debug_path = f"{debug_dir}/mapping_{debug_id}.json"
    with open(debug_path, 'w') as f:
        json.dump({
            "original_to_sam2": {str(k): v for k, v in original_to_sam2.items()},
            "sam2_to_original": {str(k): v for k, v in sam2_to_original.items()},
            "total_frames": total_frames,
            "start_frame": start_frame
        }, f, indent=2)
    print(f"[JPG Sequence] Saved mapping debug info to {debug_path}")
    
    return original_to_sam2, sam2_to_original

def prepare_sam2_jpg_sequence(
    src_dir: str, 
    total_frames: int, 
    start_frame: int
) -> Tuple[str, Dict[int, int]]:
    """
    Create a new JPG sequence in the order SAM2 should process it.
    
    Args:
        src_dir: Source directory containing the original JPG sequence
        total_frames: Total number of frames 
        start_frame: The user-specified starting frame for mask propagation
    
    Returns:
        Tuple containing:
            - Path to the directory with the reordered sequence
            - Mapping from SAM2 frame indices to original frame indices
    """
    # Create mapping plans
    original_to_sam2, sam2_to_original = create_sam2_frame_mapping(total_frames, start_frame)
    
    # Create a temporary directory for the reordered sequence
    dest_dir = tempfile.mkdtemp()
    
    # Get list of all jpg files in the source directory
    jpg_files = sorted([f for f in os.listdir(src_dir) if f.endswith('.jpg')])
    
    if len(jpg_files) != total_frames:
        print(f"Warning: Found {len(jpg_files)} files but expected {total_frames} frames")
    
    # Extract frame numbers using regex
    frame_pattern = re.compile(r'frame_(\d+)\.jpg')
    
    # Create a mapping from frame number to filename
    frame_to_filename = {}
    for filename in jpg_files:
        match = frame_pattern.match(filename)
        if match:
            frame_num = int(match.group(1))
            frame_to_filename[frame_num] = filename
    
    # Create the reordered sequence with numeric filenames (1.jpg, 2.jpg, etc.)
    for sam2_idx in range(len(sam2_to_original)):
        orig_idx = sam2_to_original[sam2_idx]
        if orig_idx in frame_to_filename:
            src_file = os.path.join(src_dir, frame_to_filename[orig_idx])
            # Create a simple numeric filename for SAM2
            dest_file = os.path.join(dest_dir, f"{sam2_idx + 1}.jpg")  # +1 to avoid 0.jpg
            shutil.copy(src_file, dest_file)
    
    # Save some sample frames for debugging
    debug_dir = f"/home/ec2-user/megaton-roto-dev/backend/tmp/sam2_debug/frames_{str(uuid.uuid4())[:8]}"
    os.makedirs(debug_dir, exist_ok=True)
    # Save all frames for inspection
    for sam2_idx in range(len(sam2_to_original)):
        src_file = os.path.join(dest_dir, f"{sam2_idx + 1}.jpg")  # Using the numeric format
        if os.path.exists(src_file):
            shutil.copy(src_file, f"{debug_dir}/sam2_frame_{sam2_idx:08d}.jpg")
    print(f"[JPG Sequence] Saved all frames to {debug_dir}")
    
    print(f"[JPG Sequence] Created reordered sequence with {len(sam2_to_original)} frames in {dest_dir}")
    return dest_dir, sam2_to_original 