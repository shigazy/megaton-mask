from app.db.session import SessionLocal  # Keep this as is
from sqlalchemy.orm import Session  # Add this for type hinting
from app.models import Task, Video
from inference.scripts.process import main as process_video
from app.core.config import get_settings
from app.utils.video import process_video_forward_reverse, convert_to_h264, process_super_video, create_greenscreen_video
from typing import List, Dict, Optional
from inference.manager import inference_manager
import tempfile
import os
from datetime import datetime
import argparse
import boto3
import subprocess
import logging
import cv2
from PIL import Image
import io
import gc
import torch
import numpy as np
import shutil
import uuid

logger = logging.getLogger(__name__)

# Get settings
settings = get_settings()
BUCKET_NAME = settings.AWS_S3_BUCKET

# Initialize S3 client
s3_client = boto3.client(
    's3',
    aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
    aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
    region_name=settings.AWS_REGION,
)

def format_points(points_dict):
    formatted_points = []
    
    # Handle positive points
    if 'positive' in points_dict:
        for point in points_dict['positive']:
            formatted_points.append({
                "x": point[0],
                "y": point[1],
                "type": "positive"
            })
    
    # Handle negative points
    if 'negative' in points_dict:
        for point in points_dict['negative']:
            formatted_points.append({
                "x": point[0],
                "y": point[1],
                "type": "negative"
            })
    
    return formatted_points

async def process_video_masks(
    video_id: str,
    bbox: List[float],
    task_id: str,
    points: Optional[Dict[str, List[List[float]]]] = None,
    super: bool = False,
    method: str = "default",
    start_frame: int = 0
) -> None:
    """
    Process video masks using the optimized InferenceManager
    """
    from inference.manager import inference_manager
    
    temp_files = []
    db = None
    try:
        # Get video and task from database
        db = SessionLocal()
        video = db.query(Video).filter(Video.id == video_id).first()
        task = db.query(Task).filter(Task.id == task_id).first()
        
        # Get fps from video metadata
        fps = video.video_metadata.get('fps', 24) if video.video_metadata else 24
        logger.info(f"Using fps: {fps} from video metadata")
        
        if not video:
            raise ValueError(f"Video {video_id} not found")
        if not task:
            raise ValueError(f"Task {task_id} not found")

        # Check if we have a JPG sequence
        if video.jpg_dir_key:
            logger.info(f"Using JPG sequence from {video.jpg_dir_key}")
            process_video_path = None  # We don't need a local video path
            jpg_dir_key = video.jpg_dir_key
        else:
            # We need to download the video file
            logger.info("No JPG sequence available, using video file")
            process_video_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            temp_files.append(process_video_path.name)
            
            if super and video.forward_reverse_key:
                logger.info("Using forward-reverse video")
                s3_client.download_file(BUCKET_NAME, video.forward_reverse_key, process_video_path.name)
            else:
                logger.info("Using original video")
                s3_client.download_file(BUCKET_NAME, video.s3_key, process_video_path.name)
                
            jpg_dir_key = None

        # Update task status
        task.status = "processing"
        db.commit()

        # Create temporary file for mask output
        temp_mask = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        temp_files.append(temp_mask.name)

        # Generate masks using inference manager
        masks_or_chunk_paths = await inference_manager.generate_full_video_masks(
            video_path=process_video_path.name if process_video_path else None,
            points=points,
            bbox=bbox,
            super_mode=super,
            method=method,
            start_frame=start_frame,
            progress_callback=lambda current, total: update_task_progress(db, task_id, current, total),
            jpg_dir_key=jpg_dir_key
        )
        
        # Debug the received masks
        print(f"[Tasks.py] Received masks of type {type(masks_or_chunk_paths)}")
        if isinstance(masks_or_chunk_paths, np.ndarray):
            print(f"[Tasks.py] Mask array shape: {masks_or_chunk_paths.shape}")
        elif isinstance(masks_or_chunk_paths, list):
            print(f"[Tasks.py] Received {len(masks_or_chunk_paths)} chunk paths")

        # Check if we got chunk paths or actual mask array
        if isinstance(masks_or_chunk_paths, list) and all(isinstance(p, str) for p in masks_or_chunk_paths):
            # We got chunk paths, which means masks were too large and saved to disk
            logger.info(f"Received {len(masks_or_chunk_paths)} mask chunks")
            
            # Create temporary file for combined masks
            temp_combined = tempfile.NamedTemporaryFile(delete=False, suffix='.npy')
            temp_files.append(temp_combined.name)
            
            # Load and combine chunks
            mask_chunks = []
            for chunk_path in masks_or_chunk_paths:
                chunk = np.load(chunk_path)
                mask_chunks.append(chunk)
                os.remove(chunk_path)  # Clean up the chunk file
            
            # Combine chunks
            masks = np.concatenate(mask_chunks, axis=0)
            np.save(temp_combined.name, masks)
            
            # Load the saved combined masks
            masks = np.load(temp_combined.name)
            logger.info(f"Combined masks shape: {masks.shape}")

            # After loading/combining masks:
            debug_dir = f"/home/ec2-user/megaton-roto-dev/backend/tmp/sam2_debug/tasks_masks_{str(uuid.uuid4())[:8]}"
            os.makedirs(debug_dir, exist_ok=True)
            print(f"[Tasks.py] Saving ALL masks before video encoding to {debug_dir}")
            # Save ALL masks before encoding
            for i in range(len(masks)):
                sample_mask = masks[i]
                debug_path = f"{debug_dir}/before_encoding_{i:08d}.jpg"
                cv2.imwrite(debug_path, sample_mask)
            print(f"[Tasks.py] Saved all {len(masks)} masks before encoding")
        else:
            # We got the masks directly
            masks = masks_or_chunk_paths
            logger.info(f"Received masks directly, shape: {getattr(masks, 'shape', 'unknown')}")

        # Memory optimization: Process masks in chunks to avoid OOM
        logger.info(f"Saving masks to video file, shape: {masks.shape}")
        height, width = masks[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_mask.name, fourcc, fps, (width, height))
        
        # Write ALL masks directly - no chunking
        masks_written = 0
        for i, mask in enumerate(masks):
            if mask is not None and mask.size > 0:
                out.write(mask)
                masks_written += 1
                # Log progress
                if i % 50 == 0 or i == len(masks) - 1:
                    print(f"[Tasks.py] Written {i+1}/{len(masks)} masks to video")

        # Release the writer
        out.release()
        print(f"[Tasks.py] Completed writing {masks_written}/{len(masks)} masks to video")

        # Verify the output video
        mask_cap = cv2.VideoCapture(temp_mask.name)
        mask_frames = int(mask_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        mask_cap.release()
        print(f"[Tasks.py] Mask video contains {mask_frames} frames according to OpenCV")

        # Ensure we wrote all frames
        if mask_frames != len(masks):
            print(f"[Tasks.py] WARNING: Mask video frame count ({mask_frames}) doesn't match expected ({len(masks)})")
            print(f"[Tasks.py] This will cause issues in the greenscreen process")

        # Save the mask video to debug for inspection
        debug_dir = "/home/ec2-user/megaton-roto-dev/backend/tmp/sam2_debug"
        os.makedirs(debug_dir, exist_ok=True)
        mask_debug_path = f"{debug_dir}/final_mask_video_{str(uuid.uuid4())[:8]}.mp4"
        shutil.copy(temp_mask.name, mask_debug_path)
        print(f"[Tasks.py] Saved copy of mask video to {mask_debug_path}")


        
        # Convert to H.264 for normal videos
        temp_h264 = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        temp_files.append(temp_h264.name)
        convert_to_h264(temp_mask.name, temp_h264.name)
        upload_path = temp_h264.name

        # Upload results to S3
        logger.info("Starting S3 upload...")
        mask_key = f"users/{video.user_id}/masks/{video_id}_mask.mp4"
        
        # Check if object exists
        try:
            s3_client.head_object(Bucket=BUCKET_NAME, Key=mask_key)
            logger.info(f"Object {mask_key} exists in S3, will overwrite")
        except:
            logger.info(f"Object {mask_key} does not exist in S3")
            
        s3_client.upload_file(
            upload_path,
            BUCKET_NAME,
            mask_key,
            ExtraArgs={'ContentType': 'video/mp4'}
        )
        logger.info(f"S3 upload completed for {mask_key}")

        # Update video record with mask
        video.mask_key = mask_key
        
        # Update task status
        task.status = "processing greenscreen"
        task.completed_at = datetime.utcnow()
        formatted_points = format_points(points)
        video.bbox = bbox
        video.points = formatted_points
        db.commit()

        # Split the greenscreen creation into a separate task to reduce memory pressure
        # First, make sure we have a valid video path for the greenscreen process
        if process_video_path is None:
            # If we used JPG sequence and don't have a local video file
            # We need to download the original video for greenscreen processing
            logger.info("Downloading original video for greenscreen processing")
            temp_video = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            temp_files.append(temp_video.name)
            
            # Download the original video (not forward-reverse)
            s3_client.download_file(BUCKET_NAME, video.s3_key, temp_video.name)
            original_video_path = temp_video.name
        else:
            original_video_path = process_video_path.name

        # Now call greenscreen with the valid path
        await create_greenscreen_async(video_id, original_video_path, upload_path, task_id)

        # NOW we can safely clear masks from memory
        del masks
        gc.collect()
        torch.cuda.empty_cache()
        logger.info("Masks saved to video file, cleared from memory")
        
    except Exception as e:
        logger.error(f"Error occurred: {str(e)}")
        if task and db:
            task.status = "failed"
            task.error_message = str(e)
            db.commit()
        raise
    finally:
        inference_manager.cleanup()  # Clean up GPU resources
        if db:
            db.close()
        # Clean up temporary files
        for temp_file in temp_files:
            try:
                if os.path.exists(temp_file):
                    print(f"Removing temporary file/directory: {temp_file}")
                    if os.path.isdir(temp_file):
                        shutil.rmtree(temp_file)
                    else:
                        os.unlink(temp_file)
            except Exception as e:
                logger.error(f"Error removing temporary file {temp_file}: {str(e)}")

# Add this new function to handle greenscreen creation asynchronously
async def create_greenscreen_async(video_id, original_video_path, mask_video_path, task_id):
    """Create greenscreen video in a separate process to reduce memory pressure"""
    temp_files = []
    db = None
    try:
        db = SessionLocal()
        video = db.query(Video).filter(Video.id == video_id).first()
        task = db.query(Task).filter(Task.id == task_id).first()
        
        if not video or not task:
            logger.error(f"Video or task not found for greenscreen creation: {video_id}, {task_id}")
            return
            
        # Create temporary files for greenscreen output
        temp_greenscreen = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        temp_files.append(temp_greenscreen.name)
        
        # Create greenscreen video
        create_greenscreen_video(
            original_video_path=original_video_path,
            mask_video_path=mask_video_path,
            output_path=temp_greenscreen.name
        )
        
        # Convert to H.264
        temp_greenscreen_h264 = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        temp_files.append(temp_greenscreen_h264.name)
        convert_to_h264(temp_greenscreen.name, temp_greenscreen_h264.name)
        
        # Upload to S3
        greenscreen_key = f"greenscreen/{video_id}_greenscreen.mp4"
        s3_client.upload_file(
            temp_greenscreen_h264.name,
            BUCKET_NAME,
            greenscreen_key,
            ExtraArgs={'ContentType': 'video/mp4'}
        )
        
        # Update video record
        if video.video_keys is None:
            video.video_keys = {}
        
        video.video_keys = {
            **video.video_keys,
            'greenscreen': greenscreen_key
        }
        
        # Update task status
        task.status = "completed"
        task.completed_at = datetime.utcnow()
        db.commit()
        
        logger.info(f"Greenscreen video created and uploaded: {greenscreen_key}")
        
        # Right after creating greenscreen (around line 358):
        # Save greenscreen video to debug folder
        debug_dir = "/home/ec2-user/megaton-roto-dev/backend/tmp/sam2_debug"
        os.makedirs(debug_dir, exist_ok=True)
        debug_gs_path = f"{debug_dir}/greenscreen_raw_{str(uuid.uuid4())[:8]}.mp4"
        debug_gs_h264_path = f"{debug_dir}/greenscreen_h264_{str(uuid.uuid4())[:8]}.mp4"

        # Copy both raw and h264 versions for debug
        if os.path.exists(temp_greenscreen.name):
            shutil.copy(temp_greenscreen.name, debug_gs_path)
            print(f"[Tasks.py] Saved raw greenscreen to {debug_gs_path}")
            
            # Analyze the greenscreen video
            try:
                gs_cap = cv2.VideoCapture(debug_gs_path)
                gs_frame_count = int(gs_cap.get(cv2.CAP_PROP_FRAME_COUNT))
                gs_fps = gs_cap.get(cv2.CAP_PROP_FPS)
                print(f"[Tasks.py] GREENSCREEN RAW: frames={gs_frame_count}, fps={gs_fps}")
                
                # Check specific frames from the greenscreen
                frames_to_check = [0, 50, 100, 150, gs_frame_count-1]
                frames_to_check = [f for f in frames_to_check if f < gs_frame_count]
                
                for idx in frames_to_check:
                    gs_cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                    ret, frame = gs_cap.read()
                    if ret:
                        frame_path = f"{debug_dir}/gs_frame_{idx:08d}.jpg"
                        cv2.imwrite(frame_path, frame)
                        print(f"[Tasks.py] Saved greenscreen frame {idx}")
            
                gs_cap.release()
            except Exception as e:
                print(f"[Tasks.py] Error analyzing greenscreen: {str(e)}")

        # Also save and analyze the h264 version
        if os.path.exists(temp_greenscreen_h264.name):
            shutil.copy(temp_greenscreen_h264.name, debug_gs_h264_path)
            print(f"[Tasks.py] Saved h264 greenscreen to {debug_gs_h264_path}")
            
            try:
                gs_h264_cap = cv2.VideoCapture(debug_gs_h264_path)
                gs_h264_frame_count = int(gs_h264_cap.get(cv2.CAP_PROP_FRAME_COUNT))
                gs_h264_fps = gs_h264_cap.get(cv2.CAP_PROP_FPS)
                print(f"[Tasks.py] GREENSCREEN H264: frames={gs_h264_frame_count}, fps={gs_h264_fps}")
                gs_h264_cap.release()
            except Exception as e:
                print(f"[Tasks.py] Error analyzing h264 greenscreen: {str(e)}")

        # Add debug for mask_video_path
        try:
            mask_cap = cv2.VideoCapture(mask_video_path)
            mask_frame_count = int(mask_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            mask_fps = mask_cap.get(cv2.CAP_PROP_FPS)
            print(f"[Tasks.py] MASK INPUT FOR GREENSCREEN: frames={mask_frame_count}, fps={mask_fps}")
            mask_cap.release()
        except Exception as e:
            print(f"[Tasks.py] Error analyzing mask video input: {str(e)}")

        # Add debug for original_video_path
        try:
            orig_cap = cv2.VideoCapture(original_video_path)
            orig_frame_count = int(orig_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            orig_fps = orig_cap.get(cv2.CAP_PROP_FPS)
            print(f"[Tasks.py] ORIGINAL VIDEO FOR GREENSCREEN: frames={orig_frame_count}, fps={orig_fps}")
            orig_cap.release()
        except Exception as e:
            print(f"[Tasks.py] Error analyzing original video input: {str(e)}")

        # Add S3 upload logging and verification
        print(f"[Tasks.py] About to upload greenscreen to S3: {greenscreen_key}")

        # After uploading to S3, verify the upload:
        try:
            print(f"[Tasks.py] Verifying S3 upload: {greenscreen_key}")
            response = s3_client.head_object(Bucket=BUCKET_NAME, Key=greenscreen_key)
            uploaded_size = response.get('ContentLength', 0)
            original_size = os.path.getsize(temp_greenscreen_h264.name)
            
            print(f"[Tasks.py] S3 upload verification:")
            print(f"[Tasks.py] - Local file size: {original_size} bytes")
            print(f"[Tasks.py] - S3 object size: {uploaded_size} bytes")
            
            if uploaded_size == original_size:
                print(f"[Tasks.py] S3 upload verified successfully - sizes match")
            else:
                print(f"[Tasks.py] WARNING: S3 upload size mismatch!")
        except Exception as e:
            print(f"[Tasks.py] Error verifying S3 upload: {str(e)}")

        # Generate a pre-signed URL to check the uploaded content
        try:
            gs_url = s3_client.generate_presigned_url(
                'get_object',
                Params={'Bucket': BUCKET_NAME, 'Key': greenscreen_key},
                ExpiresIn=3600
            )
            print(f"[Tasks.py] Greenscreen URL for testing: {gs_url[:100]}...")
        except Exception as e:
            print(f"[Tasks.py] Error generating presigned URL: {str(e)}")

    except Exception as e:
        logger.error(f"Error in greenscreen creation: {str(e)}")
        if task and db:
            task.status = "failed"
            task.error_message = f"Mask generation succeeded but greenscreen failed: {str(e)}"
            db.commit()
    finally:
        if db:
            db.close()
        # Clean up temporary files
        for temp_file in temp_files:
            try:
                if os.path.exists(temp_file):
                    print(f"Removing temporary file/directory: {temp_file}")
                    if os.path.isdir(temp_file):
                        shutil.rmtree(temp_file)
                    else:
                        os.unlink(temp_file)
            except Exception as e:
                logger.error(f"Error removing temporary file {temp_file}: {str(e)}")

async def transcode_video(
    input_path: str,
    output_path: str,
    delete_original: bool = True
) -> None:
    """
    Transcode video to MP4 (H.264) format
    """
    try:
        # Use ffmpeg to transcode the video
        command = [
            'ffmpeg',
            '-i', input_path,
            '-c:v', 'libx264',  # H.264 video codec
            '-preset', 'medium',  # Balance between speed and quality
            '-crf', '23',        # Constant Rate Factor (18-28 is good)
            '-c:a', 'aac',       # AAC audio codec
            '-b:a', '128k',      # Audio bitrate
            '-movflags', '+faststart',  # Enable streaming
            '-y',                # Overwrite output file
            output_path
        ]
        
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        stdout, stderr = process.communicate()
        
        if process.returncode != 0:
            raise Exception(f"Transcoding failed: {stderr.decode()}")
            
        # Delete original file if requested
        if delete_original and os.path.exists(input_path):
            os.unlink(input_path)
            
        return output_path
            
    except Exception as e:
        logger.error(f"Transcoding error: {str(e)}")
        raise

async def transcode_to_jpg_sequence(
    input_path: str,
    output_dir: str,
    fps: int = None,
    delete_original: bool = False
) -> str:
    """
    Transcode video to a sequence of JPG images
    
    Args:
        input_path: Path to input video file
        output_dir: Directory to save JPG sequence
        fps: Optional frames per second to extract (if None, uses original video fps)
        delete_original: Whether to delete the original video file
    """
    try:
        # Get original video fps if not specified
        if fps is None:
            cap = cv2.VideoCapture(input_path)
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            cap.release()
            logger.info(f"Using original video fps: {fps}")

        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Created output directory: {output_dir}")
        
        # Build the ffmpeg command
        command = [
            'ffmpeg',
            '-i', input_path,
            '-vf', f'fps={fps}',  # Use original fps or specified fps
            '-frame_pts', '1',
            '-q:v', '2',
            '-f', 'image2',
            os.path.join(output_dir, 'frame_%08d.jpg')
        ]
        
        logger.info(f"Running ffmpeg command: {' '.join(command)}")
        
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        stdout, stderr = process.communicate()
        
        if process.returncode != 0:
            raise Exception(f"JPG sequence conversion failed: {stderr.decode()}")
        
        frame_count = len([f for f in os.listdir(output_dir) if f.endswith('.jpg')])
        logger.info(f"Generated {frame_count} JPG frames in {output_dir}")
            
        return output_dir
            
    except Exception as e:
        logger.error(f"JPG sequence conversion error: {str(e)}")
        raise




async def process_uploaded_video(
    file_path: str,
    video_id: str,
    user_id: str,
    original_filename: str,
    db: Session
) -> dict:
    """
    Process an uploaded video: transcode if needed, generate thumbnail, and upload to S3
    """
    print(f"Starting process_uploaded_video for video_id: {video_id}, user_id: {user_id}")
    temp_files = []
    try:
        # Check if video needs transcoding
        print(f"Checking if video needs transcoding: {file_path}")
        probe = subprocess.run([
            'ffprobe',
            '-v', 'error',
            '-select_streams', 'v:0',
            '-show_entries', 'stream=codec_name',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            file_path
        ], capture_output=True, text=True)
        
        video_codec = probe.stdout.strip()
        print(f"Detected video codec: {video_codec}")
        
        # Transcode if not H.264
        if video_codec != 'h264':
            print(f"Transcoding video from {video_codec} to h264")
            temp_mp4 = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            temp_files.append(temp_mp4.name)
            print(f"Created temporary file for transcoding: {temp_mp4.name}")
            await transcode_video(file_path, temp_mp4.name)
            process_path = temp_mp4.name
            print(f"Transcoding completed, new path: {process_path}")
        else:
            process_path = file_path
            print(f"No transcoding needed, using original file: {process_path}")

        # Get video metadata and thumbnail frame first
        cap = cv2.VideoCapture(process_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        metadata = {
            "fps": fps,
            "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "duration": float(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / float(cap.get(cv2.CAP_PROP_FPS)),
            "file_size": os.path.getsize(process_path),
            "codec": 'h264',
            "uploaded_filename": original_filename,
            "upload_date": datetime.utcnow().isoformat()
        }

        # Generate thumbnail before closing the video capture
        ret, frame = cap.read()
        if not ret:
            raise Exception("Failed to read video frame for thumbnail")

        # Now we can close the capture
        cap.release()

        # Create jpg sequence with the original fps
        jpg_dir = tempfile.mkdtemp()
        temp_files.append(jpg_dir)
        logger.info(f"Converting video to jpg sequence in {jpg_dir}")
        await transcode_to_jpg_sequence(process_path, jpg_dir, fps=fps)

        # Process thumbnail
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        pil_image.thumbnail((320, 180))
        thumb_buffer = io.BytesIO()
        pil_image.save(thumb_buffer, format="WebP", quality=85)
        thumb_buffer.seek(0)

        # Upload to S3
        video_s3_key = f"users/{user_id}/videos/{video_id}.mp4"  # Store in a dedicated variable
        thumbnail_key = f"thumbnails/{video_id}.webp"
        jpg_dir_key = f"users/{user_id}/videos/{video_id}/jpg/"
        print(f"S3 keys prepared - video: {video_s3_key}, thumbnail: {thumbnail_key}, jpg_dir: {jpg_dir_key}")

        # Upload thumbnail
        print(f"Uploading thumbnail to S3: {thumbnail_key}")
        s3_client.upload_fileobj(
            thumb_buffer,
            BUCKET_NAME,
            thumbnail_key,
            ExtraArgs={'ContentType': 'image/webp'}
        )
        print(f"Thumbnail uploaded successfully")

        # Upload video
        print(f"Uploading video to S3: {video_s3_key}")
        s3_client.upload_file(
            process_path,
            BUCKET_NAME,
            video_s3_key,
            ExtraArgs={'ContentType': 'video/mp4'}
        )
        print(f"Video uploaded successfully")

        # Upload jpg sequence - with better error handling
        print(f"Starting upload of JPG sequence to S3")
        jpg_files = sorted([f for f in os.listdir(jpg_dir) if f.endswith('.jpg')])
        print(f"Found {len(jpg_files)} JPG files to upload")
        
        # Check if any files were generated
        if not jpg_files:
            print("WARNING: No JPG files were generated!")
            logger.warning(f"No JPG files were generated in {jpg_dir}")
        
        # Upload in batches to prevent timeouts
        batch_size = 50
        for i in range(0, len(jpg_files), batch_size):
            batch = jpg_files[i:i+batch_size]
            print(f"Uploading batch {i//batch_size + 1}/{(len(jpg_files)-1)//batch_size + 1} ({len(batch)} files)")
            
            for j, jpg_file in enumerate(batch):
                file_path = os.path.join(jpg_dir, jpg_file)
                jpg_s3_key = f"users/{user_id}/videos/{video_id}/jpg/{jpg_file}"  # Use a different variable name
                
                # Verify file exists and has content
                if not os.path.exists(file_path):
                    print(f"WARNING: File doesn't exist: {file_path}")
                    continue
                    
                if os.path.getsize(file_path) == 0:
                    print(f"WARNING: File is empty: {file_path}")
                    continue
                
                try:
                    s3_client.upload_file(
                        file_path,
                        BUCKET_NAME,
                        jpg_s3_key,  # Use jpg_s3_key here
                        ExtraArgs={'ContentType': 'image/jpeg'}
                    )
                    # Print progress every 10 files
                    if (j % 10) == 0:
                        print(f"  Progress: {i+j+1}/{len(jpg_files)} files uploaded")
                except Exception as e:
                    print(f"ERROR uploading {jpg_file}: {str(e)}")
                    logger.error(f"Failed to upload JPG file {jpg_file}: {str(e)}")
                    # Continue with other files
        
        print(f"JPG sequence upload completed")

        # Generate URLs
        print(f"Generating presigned URLs")
        video_url = s3_client.generate_presigned_url(
            'get_object',
            Params={
                'Bucket': BUCKET_NAME,
                'Key': video_s3_key,  # Use video_s3_key here
                'ResponseContentType': 'video/mp4',
                'ResponseContentDisposition': 'inline'
            },
            ExpiresIn=3600
        )
        print(f"Video URL generated: {video_url[:50]}...")

        thumbnail_url = s3_client.generate_presigned_url(
            'get_object',
            Params={'Bucket': BUCKET_NAME, 'Key': thumbnail_key},
            ExpiresIn=3600
        )
        print(f"Thumbnail URL generated: {thumbnail_url[:50]}...")

        result = {
            "s3_key": video_s3_key,  # Use video_s3_key here
            "thumbnail_key": thumbnail_key,
            "video_url": video_url,
            "thumbnail_url": thumbnail_url,
            "jpg_dir_key": jpg_dir_key,
            "metadata": metadata
        }
        print(f"Video processing completed successfully for video_id: {video_id}")
        return result

    finally:
        # Cleanup temp files
        print(f"Cleaning up {len(temp_files)} temporary files")
        for temp_file in temp_files:
            try:
                if os.path.exists(temp_file):
                    print(f"Removing temporary file/directory: {temp_file}")
                    if os.path.isdir(temp_file):
                        shutil.rmtree(temp_file)
                    else:
                        os.unlink(temp_file)
            except Exception as e:
                print(f"Error cleaning up {temp_file}: {str(e)}")
                logger.error(f"Error cleaning up {temp_file}: {str(e)}")

# Add this helper function
def update_task_progress(db, task_id, current, total):
    """Update task progress in the database"""
    print(f"Updating task {task_id} progress: {current}/{total}")
    try:
        print(f"Updating task {task_id} progress: {current}/{total}")
        progress_percent = (current / total) * 100 if total > 0 else 0
        task = db.query(Task).filter(Task.id == task_id).first()
        print(f"Task: {task}")
        if task:
            task.progress = progress_percent
            print(f"Task progress: {task.progress}")
            db.commit()
            logger.info(f"Updated task {task_id} progress: {progress_percent:.1f}%")
    except Exception as e:
        logger.error(f"Error updating task progress: {str(e)}")
