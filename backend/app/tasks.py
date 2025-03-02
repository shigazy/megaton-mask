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
    method: str = "default"
) -> None:
    """
    Process video masks using the optimized InferenceManager
    """
    from inference.manager import inference_manager
    
    temp_files = []
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

        # If super mode and no forward-reverse video exists, create it first
        if super:
            logger.info("Super mode detected - checking for forward-reverse video")
            if not video.forward_reverse_key:
                logger.info("No forward-reverse video found - creating one")
                task.status = "creating_forward_reverse"
                db.commit()
                
                # Create and upload forward-reverse video
                await process_video_forward_reverse(video_id, db)
                logger.info("Forward-reverse video created")

            # Use forward-reverse video for processing
            process_video_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            temp_files.append(process_video_path.name)
            s3_client.download_file(BUCKET_NAME, video.forward_reverse_key, process_video_path.name)
        else:
            # Use original video
            process_video_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            temp_files.append(process_video_path.name)
            s3_client.download_file(BUCKET_NAME, video.s3_key, process_video_path.name)

        # Update task status back to processing
        task.status = "processing"
        db.commit()

        # Create temporary file for mask output
        temp_mask = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        temp_files.append(temp_mask.name)

        # Generate masks using inference manager
        masks = await inference_manager.generate_full_video_masks(
            video_path=process_video_path.name,
            points=points,
            bbox=bbox,
            super_mode=super,
            method=method
        )

        # Save masks to video file
        height, width = masks[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_mask.name, fourcc, fps, (width, height))
        
        for mask in masks:
            out.write(mask)
        out.release()

        # Process the mask video if it's a super video
        if super:
            temp_processed = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            temp_files.append(temp_processed.name)
            process_super_video(temp_mask.name, temp_processed.name)
            upload_path = temp_processed.name
        else:
            # Convert to H.264 for normal videos
            temp_h264 = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            temp_files.append(temp_h264.name)
            convert_to_h264(temp_mask.name, temp_h264.name)
            upload_path = temp_h264.name

        # Upload results to S3
        logger.info("Starting S3 upload...")
        mask_key = f"users/{video.user_id}/masks/{video_id}_mask.mp4"
        # Upload results to S3
        logger.info(f"Starting S3 upload to {mask_key}...")
        
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
        print(formatted_points)
        video.bbox = bbox
        video.points = formatted_points
        db.commit()

        # Create green screen version
        temp_greenscreen = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        temp_files.append(temp_greenscreen.name)
        
        create_greenscreen_video(
            original_video_path=process_video_path.name,
            mask_video_path=upload_path,
            output_path=temp_greenscreen.name
        )
        
        # Convert green screen video to H.264
        temp_greenscreen_h264 = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        temp_files.append(temp_greenscreen_h264.name)
        convert_to_h264(temp_greenscreen.name, temp_greenscreen_h264.name)
        
        # Upload green screen video to S3
        greenscreen_key = f"greenscreen/{video_id}_greenscreen.mp4"
        s3_client.upload_file(
            temp_greenscreen_h264.name,
            BUCKET_NAME,
            greenscreen_key,
            ExtraArgs={'ContentType': 'video/mp4'}
        )
        
        # Update video_keys in the video record
        if video.video_keys is None:
            video.video_keys = {}
        
        video.video_keys = {
            **video.video_keys,
            'greenscreen': greenscreen_key
        }
        # Update task status
        task.status = "completed greenscreen"
        task.completed_at = datetime.utcnow()
        db.commit()

    except Exception as e:
        logger.error(f"Error occurred: {str(e)}")
        if task:
            task.status = "failed"
            task.error_message = str(e)
            db.commit()
        raise
    finally:
        inference_manager.cleanup()  # Clean up GPU resources
        db.close()
        # Clean up temporary files
        for temp_file in temp_files:
            try:
                os.unlink(temp_file)
            except Exception as e:
                logger.error(f"Error removing temporary file {temp_file}: {str(e)}")
        
        if db:
            db.close()

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
    temp_files = []
    try:
        # Check if video needs transcoding
        probe = subprocess.run([
            'ffprobe',
            '-v', 'error',
            '-select_streams', 'v:0',
            '-show_entries', 'stream=codec_name',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            file_path
        ], capture_output=True, text=True)
        
        video_codec = probe.stdout.strip()
        
        # Transcode if not H.264
        if video_codec != 'h264':
            logger.info(f"Transcoding video from {video_codec} to h264")
            temp_mp4 = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            temp_files.append(temp_mp4.name)
            await transcode_video(file_path, temp_mp4.name)
            process_path = temp_mp4.name
        else:
            process_path = file_path

        # Get video metadata
        cap = cv2.VideoCapture(process_path)
        metadata = {
            "fps": cap.get(cv2.CAP_PROP_FPS),
            "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "duration": float(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / float(cap.get(cv2.CAP_PROP_FPS)),
            "file_size": os.path.getsize(process_path),
            "codec": 'h264',
            "uploaded_filename": original_filename,
            "upload_date": datetime.utcnow().isoformat()
        }

        # Generate and upload thumbnail
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            raise Exception("Failed to read video frame for thumbnail")

        # Process thumbnail
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        pil_image.thumbnail((320, 180))
        thumb_buffer = io.BytesIO()
        pil_image.save(thumb_buffer, format="WebP", quality=85)
        thumb_buffer.seek(0)

        # Upload to S3
        s3_key = f"users/{user_id}/videos/{video_id}.mp4"
        thumbnail_key = f"thumbnails/{video_id}.webp"

        # Upload thumbnail
        s3_client.upload_fileobj(
            thumb_buffer,
            BUCKET_NAME,
            thumbnail_key,
            ExtraArgs={'ContentType': 'image/webp'}
        )

        # Upload video
        s3_client.upload_file(
            process_path,
            BUCKET_NAME,
            s3_key,
            ExtraArgs={'ContentType': 'video/mp4'}
        )

        # Generate URLs
        video_url = s3_client.generate_presigned_url(
            'get_object',
            Params={
                'Bucket': BUCKET_NAME,
                'Key': s3_key,
                'ResponseContentType': 'video/mp4',
                'ResponseContentDisposition': 'inline'
            },
            ExpiresIn=3600
        )

        thumbnail_url = s3_client.generate_presigned_url(
            'get_object',
            Params={'Bucket': BUCKET_NAME, 'Key': thumbnail_key},
            ExpiresIn=3600
        )

        return {
            "s3_key": s3_key,
            "thumbnail_key": thumbnail_key,
            "video_url": video_url,
            "thumbnail_url": thumbnail_url,
            "metadata": metadata
        }

    finally:
        # Cleanup temp files
        for temp_file in temp_files:
            try:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
            except Exception as e:
                logger.error(f"Error cleaning up {temp_file}: {str(e)}")
