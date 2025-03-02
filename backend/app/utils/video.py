import cv2
import tempfile
import os
import subprocess
import logging
from typing import Optional
import boto3
from app.core.config import get_settings
from app.db.session import SessionLocal
from app.models import Video
import numpy as np

logger = logging.getLogger(__name__)
settings = get_settings()

s3_client = boto3.client(
    's3',
    aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
    aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
    region_name=settings.AWS_REGION
)

def create_forward_reverse_video(
    input_path: str,
    output_path: str,
    use_ffmpeg: bool = True
) -> None:
    """
    Create a forward-reverse version of a video using either ffmpeg or OpenCV.
    
    Args:
        input_path: Path to input video file
        output_path: Path where the forward-reverse video should be saved
        use_ffmpeg: Whether to use ffmpeg (faster) or OpenCV (more compatible)
    """
    try:
        if use_ffmpeg:
            logger.info("Creating forward-reverse video using ffmpeg...")
            subprocess.run([
                'ffmpeg', '-y',
                '-i', input_path,
                '-filter_complex', '[0:v]split[v1][v2];[v2]reverse[rv];[v1][rv]concat=n=2:v=1[v]',
                '-map', '[v]',
                '-c:v', 'libx264',
                '-preset', 'medium',
                '-movflags', '+faststart',
                '-pix_fmt', 'yuv420p',
                output_path
            ], check=True)
        else:
            logger.info("Creating forward-reverse video using OpenCV...")
            # Open input video
            cap = cv2.VideoCapture(input_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Set up video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            # Read and store forward frames
            frames = []
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)
                out.write(frame)
            
            # Write reverse frames
            for frame in reversed(frames[:-1]):  # Skip last frame to avoid duplicate
                out.write(frame)
            
            out.release()
            cap.release()

        # Verify output file
        if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
            raise Exception("Failed to create forward-reverse video: Output file is empty or missing")

    except Exception as e:
        logger.error(f"Error creating forward-reverse video: {str(e)}")
        raise

async def process_video_forward_reverse(video_id: str, db=None) -> str:
    """
    Process a video to create its forward-reverse version and upload to S3.
    Returns the S3 key of the created video.
    
    Args:
        video_id: ID of the video to process
        db: Optional database session (will create one if not provided)
    
    Returns:
        str: S3 key of the forward-reverse video
    """
    temp_files = []

    
    close_db = False
    
    try:
        # Create DB session if not provided
        if db is None:
            db = SessionLocal()
            close_db = True
            
        # Get video from database
        video = db.query(Video).filter(Video.id == video_id).first()
        if not video:
            raise Exception(f"Video not found: {video_id}")

        # Create temporary files
        temp_input = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        temp_output = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        temp_files.extend([temp_input.name, temp_output.name])

        # Download original video
        logger.info(f"Downloading video from S3: {video.s3_key}")
        s3_client.download_file(settings.AWS_S3_BUCKET, video.s3_key, temp_input.name)

        # Create forward-reverse version
        create_forward_reverse_video(temp_input.name, temp_output.name)

        # Upload to S3
        forward_reverse_key = f"forward_reverse/{video_id}.mp4"
        logger.info(f"Uploading to S3: {forward_reverse_key}")
        s3_client.upload_file(
            temp_output.name,
            settings.AWS_S3_BUCKET,
            forward_reverse_key,
            ExtraArgs={'ContentType': 'video/mp4'}
        )

        # Update database
        video.forward_reverse_key = forward_reverse_key
        db.commit()
        
        return forward_reverse_key

    except Exception as e:
        logger.error(f"Error processing forward-reverse video: {str(e)}")
        if db is not None:
            db.rollback()
        raise

    finally:
        if close_db and db is not None:
            db.close()
        
        for temp_file in temp_files:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
                    logger.info(f"Removed temporary file: {temp_file}")
            except Exception as e:
                logger.error(f"Error removing temporary file {temp_file}: {str(e)}")

def convert_to_h264(input_path: str, output_path: str) -> None:
    """
    Convert a video to H.264 format with standard settings.
    
    Args:
        input_path: Path to input video file
        output_path: Path where the H.264 video should be saved
    
    Raises:
        Exception: If conversion fails or output file is empty
    """
    try:
        logger.info(f"Converting video to H.264: {input_path}")
        subprocess.run([
            'ffmpeg', '-y',
            '-i', input_path,
            '-c:v', 'libx264',
            '-preset', 'medium',
            '-movflags', '+faststart',
            '-pix_fmt', 'yuv420p',
            output_path
        ], check=True)
        
        if os.path.getsize(output_path) == 0:
            raise Exception("Converted file is empty")
            
        logger.info("H.264 conversion completed successfully")
        
    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg conversion failed: {e}")
        raise Exception(f"Failed to convert video to H.264: {e}")
    except Exception as e:
        logger.error(f"Error during H.264 conversion: {e}")
        raise 

def process_super_video(input_path: str, output_path: str):
    """
    Process a super video by:
    1. Reversing the entire video using FFmpeg
    2. Taking only the first half of the reversed frames
    3. Saving as H.264 MP4 with proper encoding settings
    
    Args:
        input_path: Path to input video
        output_path: Path to save processed video
    """
    import subprocess
    import os
    from pathlib import Path
    import tempfile

    try:
        # First, get video duration using ffprobe
        duration = float(subprocess.check_output([
            'ffprobe', '-v', 'error',
            '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            input_path
        ]).decode().strip())

        # Calculate half duration
        half_duration = duration / 2

        logger.info(f"Processing super video: {input_path}")
        # Reverse video and take first half in one command
        subprocess.run([
            'ffmpeg', '-y',
            '-i', input_path,
            '-vf', 'reverse',  # Reverse the entire video
            '-t', str(half_duration),  # Take only first half of reversed video
            '-c:v', 'libx264',  # Use H.264 codec
            '-preset', 'medium',
            '-movflags', '+faststart',
            '-pix_fmt', 'yuv420p',
            output_path
        ], check=True)
        
        if os.path.getsize(output_path) == 0:
            raise Exception("Processed file is empty")
            
        logger.info("Super video processing completed successfully")
        
    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg processing failed: {e}")
        raise Exception(f"Failed to process super video: {e}")
    except Exception as e:
        logger.error(f"Error during super video processing: {e}")
        raise

    return output_path

def create_greenscreen_video(original_video_path: str, mask_video_path: str, output_path: str):
    """
    Creates a green screen video by applying the mask to the original video.
    The masked object will be preserved while the background is replaced with green.
    
    Args:
        original_video_path: Path to the original video
        mask_video_path: Path to the mask video (white object on black background)
        output_path: Path where the green screen video will be saved
    """
    # Open both videos
    original = cv2.VideoCapture(original_video_path)
    mask = cv2.VideoCapture(mask_video_path)
    
    # Get video properties
    width = int(original.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(original.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = original.get(cv2.CAP_PROP_FPS)
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    while True:
        ret1, frame = original.read()
        ret2, mask_frame = mask.read()
        
        if not ret1 or not ret2:
            break
            
        # Convert mask to grayscale if it's RGB
        if len(mask_frame.shape) == 3:
            mask_frame = cv2.cvtColor(mask_frame, cv2.COLOR_BGR2GRAY)
            _, mask_frame = cv2.threshold(mask_frame, 127, 255, cv2.THRESH_BINARY)
        
        # Create green background
        green_bg = np.zeros_like(frame)
        green_bg[:, :] = [0, 255, 0]  # BGR format
        
        # Create the composite frame
        # Keep original where mask is white, use green where mask is black
        mask_3channel = cv2.cvtColor(mask_frame, cv2.COLOR_GRAY2BGR)
        composite = np.where(mask_3channel == [255, 255, 255], frame, green_bg)
        
        out.write(composite)
    
    # Release everything
    original.release()
    mask.release()
    out.release()