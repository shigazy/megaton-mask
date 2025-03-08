import logging
logging.getLogger('botocore').setLevel(logging.ERROR)
logging.getLogger('boto3').setLevel(logging.ERROR)
logging.getLogger('urllib3').setLevel(logging.ERROR)
logging.getLogger('s3transfer').setLevel(logging.ERROR)


from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, status, Form, Body, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import boto3
from botocore.exceptions import ClientError
from uuid import uuid4
from app.core.config import get_settings
import logging
from sqlalchemy.orm import Session
from fastapi.security import OAuth2PasswordRequestForm
from app.core.auth import create_access_token, get_current_user, get_password_hash, verify_password
from app.models import User, Video, Task, AdminAction
from app.db.session import get_db, SessionLocal
from fastapi.middleware import Middleware
from slowapi import Limiter
from slowapi.util import get_remote_address
from pydantic import BaseModel, EmailStr
from datetime import timedelta, datetime
from PIL import Image
import cv2
import io
import numpy as np
import tempfile
import os
from inference.scripts.mask_generator import generate_single_mask
from typing import List, Dict, Optional
from fastapi import BackgroundTasks
from app.tasks import process_video_masks, process_uploaded_video
from inference.manager import inference_manager
from app.core.credits import CreditAction
from app.core.credits_middleware import CreditsManager
from app.core.storage import StorageManager
import uuid
from app.api.stripe.routes import router as stripe_router
from fastapi_mail import FastMail, MessageSchema, ConnectionConfig
import jwt
from passlib.context import CryptContext

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

settings = get_settings()

# Print debug info
print("Verifying AWS credentials...")
print(f"Access Key: {'*' * 16}{settings.AWS_ACCESS_KEY_ID[-4:] if settings.AWS_ACCESS_KEY_ID else 'None'}")
print(f"Secret Key: {'*' * 16}{settings.AWS_SECRET_ACCESS_KEY[-4:] if settings.AWS_SECRET_ACCESS_KEY else 'None'}")
print(f"Region: {settings.AWS_REGION}")
print(f"Bucket: {settings.AWS_S3_BUCKET}")

if not settings.AWS_ACCESS_KEY_ID or not settings.AWS_SECRET_ACCESS_KEY:
    raise ValueError("AWS credentials not properly loaded")

# Initialize S3 client
s3_client = boto3.client(
    's3',
    aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
    aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
    region_name=settings.AWS_REGION
)

# Test S3 connection
try:
    s3_client.list_buckets()
    print("Successfully connected to AWS S3")
except Exception as e:
    print(f"Failed to connect to AWS S3: {str(e)}")
    raise

BUCKET_NAME = settings.AWS_S3_BUCKET

app = FastAPI()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://app.megaton.ai",
        "https://app.megaton.ai"  # Include HTTPS version too
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if not settings.AWS_ACCESS_KEY_ID or not settings.AWS_SECRET_ACCESS_KEY:
    raise ValueError("AWS credentials not properly loaded")

# Initialize S3 client
s3_client = boto3.client(
    's3',
    aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
    aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
    region_name=settings.AWS_REGION
)

BUCKET_NAME = settings.AWS_S3_BUCKET

class UserRegister(BaseModel):
    email: EmailStr
    password: str

class MaskGenerationRequest(BaseModel):
    bbox: List[float]
    points: Dict[str, List[List[float]]]

# Define the request model at the top of the file
class GenerateMasksRequest(BaseModel):
    bbox: List[Optional[float]]  # Changed to allow None values
    points: Dict[str, List[List[float]]]
    super: bool
    method: str
    start_frame: int = 0  # Add start_frame with default value of 0

    class Config:
        arbitrary_types_allowed = True

class CreditPurchase(BaseModel):
    amount: int

class UserUpdate(BaseModel):
    email: str | None = None
    user_credits: int | None = None
    membership: dict | None = None
    super_user: bool | None = None

class UserResponse(BaseModel):
    id: str
    email: str
    created_at: datetime
    user_credits: int
    membership: dict
    super_user: bool
    storage_used: dict

# Email configuration
mail_config = ConnectionConfig(
    MAIL_USERNAME=settings.MAIL_USERNAME,
    MAIL_PASSWORD=settings.MAIL_PASSWORD,
    MAIL_FROM=settings.MAIL_FROM,
    MAIL_PORT=settings.MAIL_PORT,
    MAIL_SERVER=settings.MAIL_SERVER,
    MAIL_STARTTLS=True,
    MAIL_SSL_TLS=False,
    USE_CREDENTIALS=True
)

fastmail = FastMail(mail_config)

# Password hashing configuration
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash."""
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    """Generate password hash."""
    return pwd_context.hash(password)

def authenticate_user(db: Session, email: str, password: str) -> Optional[User]:
    """Authenticate a user by email and password."""
    user = db.query(User).filter(User.email == email).first()
    if not user:
        return None
    if not verify_password(password, user.hashed_password):
        return None
    return user

@app.post("/api/auth/register")
async def register(
    user_data: UserRegister,
    db: Session = Depends(get_db)
):
    # Check if user exists
    if db.query(User).filter(User.email == user_data.email).first():
        raise HTTPException(status_code=400, detail="Email already registered")
    
    # Create confirmation token
    confirmation_token = create_access_token(
        data={"email": user_data.email},
        expires_delta=timedelta(hours=24)
    )
    
    # Create new user (using email as username)
    user = User(
        id=str(uuid4()),
        email=user_data.email,
        username=user_data.email,  # Set username same as email
        hashed_password=get_password_hash(user_data.password),
        confirmation_token=confirmation_token,
        is_confirmed=False
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    
    # Send confirmation email
    await send_confirmation_email(user_data.email, confirmation_token)
    
    return {
        "message": "Registration successful. Please check your email to confirm your account.",
        "needs_confirmation": True
    }

@app.post("/api/auth/login")
async def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db)
):
    user = authenticate_user(db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=401,
            detail="Incorrect email or password"
        )
    
    # Skip email confirmation check for super users
    if not user.is_confirmed and not user.super_user:
        return {
            "email_not_confirmed": True,
            "message": "Please confirm your email to continue"
        }
    
    access_token = create_access_token(
        data={"sub": user.id}
    )
    
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": {
            "id": user.id, 
            "email": user.email,
            "super_user": user.super_user,
            "is_confirmed": user.is_confirmed
        }
    }

@app.post("/api/auth/confirm/{token}")
async def confirm_email(token: str, db: Session = Depends(get_db)):
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        email = payload.get("email")
        if email is None:
            raise HTTPException(status_code=400, detail="Invalid token")
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=400, detail="Token has expired")
    except jwt.JWTError:
        raise HTTPException(status_code=400, detail="Invalid token")

    user = db.query(User).filter(User.email == email).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    user.is_confirmed = True
    user.confirmation_token = None
    db.commit()
    
    return {"message": "Email confirmed successfully"}

@app.post("/api/auth/resend-confirmation")
async def resend_confirmation(email: EmailStr, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == email).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    if user.is_confirmed:
        raise HTTPException(status_code=400, detail="Email already confirmed")
    
    confirmation_token = create_access_token(
        data={"email": email},
        expires_delta=timedelta(hours=24)
    )
    
    user.confirmation_token = confirmation_token
    db.commit()
    
    await send_confirmation_email(email, confirmation_token)
    
    return {"message": "Confirmation email sent"}

async def send_confirmation_email(email: str, token: str):
    confirmation_url = f"{settings.FRONTEND_URL}/confirm-email?token={token}"
    
    message = MessageSchema(
        subject="Confirm your email",
        recipients=[email],
        body=f"""
        Please confirm your email by clicking the following link:
        {confirmation_url}
        
        This link will expire in 24 hours.
        """,
        subtype="html"
    )
    
    await fastmail.send_message(message)

@app.post("/api/videos/upload")
async def upload_video(
    file: UploadFile,
    background_tasks: BackgroundTasks,
    annotations: str = Form(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    temp_file_path = None
    try:
        video_id = str(uuid4())
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp_video:
            temp_file_path = temp_video.name
            content = await file.read()
            temp_video.write(content)
            temp_video.flush()

        # Process video in background
        processed_data = await process_uploaded_video(
            temp_file_path,
            video_id,
            current_user.id,
            file.filename,
            db
        )

        # Create video record
        video = Video(
            id=video_id,
            title=file.filename,
            s3_key=processed_data["s3_key"],
            thumbnail_key=processed_data["thumbnail_key"],
            user_id=current_user.id,
            video_metadata=processed_data["metadata"]
        )
        db.add(video)
        db.commit()

        return {
            "id": video_id,
            "title": file.filename,
            "videoUrl": processed_data["video_url"],
            "thumbnailUrl": processed_data["thumbnail_url"],
            "createdAt": video.created_at,
            "metadata": processed_data["metadata"]
        }

    except Exception as e:
        logger.error(f"Upload failed: {str(e)}")
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
            except Exception as cleanup_error:
                logger.error(f"Failed to clean up temporary file: {str(cleanup_error)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/videos")
async def get_videos(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    videos = db.query(Video).filter(Video.user_id == current_user.id).all()
    
    video_list = []
    for video in videos:
        # Generate presigned URLs for both video and thumbnail
        video_url = s3_client.generate_presigned_url(
            'get_object',
            Params={'Bucket': BUCKET_NAME, 'Key': video.s3_key},
            ExpiresIn=3600
        )
        
        thumbnail_url = None
        if video.thumbnail_key:
            thumbnail_url = s3_client.generate_presigned_url(
                'get_object',
                Params={'Bucket': BUCKET_NAME, 'Key': video.thumbnail_key},
                ExpiresIn=3600
            )

        # Reset mask_url for each video
        mask_url = None
        if video.mask_key:
            mask_url = s3_client.generate_presigned_url(
                'get_object',
                Params={'Bucket': BUCKET_NAME, 'Key': video.mask_key},
                ExpiresIn=3600
            )
        greenscreen_url = None
        if video.video_keys and video.video_keys.get('greenscreen'):
            greenscreen_url = s3_client.generate_presigned_url(
                'get_object',
                Params={'Bucket': BUCKET_NAME, 'Key': video.video_keys['greenscreen']},
                ExpiresIn=3600
            )
        video_list.append({
            "id": video.id,
            "title": video.title,
            "videoUrl": video_url,
            "thumbnailUrl": thumbnail_url,
            "createdAt": video.created_at,
            "bbox": video.bbox,
            "points": video.points,
            "maskUrl": mask_url,
            "greenscreenUrl": greenscreen_url
        })
    
    return {"videos": video_list}

@app.delete("/api/videos/{video_id}")
async def delete_video(
    video_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    video = db.query(Video).filter(
        Video.id == video_id,
        Video.user_id == current_user.id
    ).first()
    
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    
    try:
        # Delete from S3
        s3_client.delete_object(Bucket=BUCKET_NAME, Key=video.s3_key)
        if video.thumbnail_key:
            s3_client.delete_object(Bucket=BUCKET_NAME, Key=video.thumbnail_key)
        
        # Delete from database
        db.delete(video)
        db.commit()
        
        return {"message": "Video deleted successfully"}
        
    except Exception as e:
        logger.error(f"Failed to delete video: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to delete video")

@app.get("/api/videos/{video_id}/refresh-url")
async def refresh_video_url(
    video_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    video = db.query(Video).filter(
        Video.id == video_id,
        Video.user_id == current_user.id
    ).first()
    
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    
    video_url = s3_client.generate_presigned_url(
        'get_object',
        Params={'Bucket': BUCKET_NAME, 'Key': video.s3_key},
        ExpiresIn=3600
    )
    
    thumbnail_url = None
    if video.thumbnail_key:
        thumbnail_url = s3_client.generate_presigned_url(
            'get_object',
            Params={'Bucket': BUCKET_NAME, 'Key': video.thumbnail_key},
            ExpiresIn=3600
        )
    
    return {
        "videoUrl": video_url,
        "thumbnailUrl": thumbnail_url
    }

@app.on_event("startup")
async def startup_event():
    await inference_manager.initialize()

@app.post("/api/videos/{video_id}/preview-mask")
async def generate_preview_mask(
    video_id: str,
    request_data: MaskGenerationRequest,
    background_tasks: BackgroundTasks,
    start_frame: int = Body(0),  # Add start_frame parameter with default value
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    video_path = None
    try:
        # Get video path
        video = db.query(Video).filter(Video.id == video_id).first()
        if not video:
            raise HTTPException(status_code=404, detail="Video not found")
            
        # Download video from S3 to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
            try:
                s3_client.download_file(BUCKET_NAME, video.s3_key, temp_file.name)
                video_path = temp_file.name
            except Exception as e:
                logger.error(f"Failed to download video: {str(e)}")
                raise HTTPException(status_code=500, detail="Failed to download video")
                
        print("request_data.bbox", request_data.bbox)
        print("request_data.points", request_data.points)
        print("video_path", video_path)
        print("request_data.current_frame", request_data.current_frame)

        # Create the batch request data properly
        batch_request = {
            'video_path': video_path,
            'points': request_data.points,
            'bbox': request_data.bbox,
            'current_frame': request_data.current_frame
        }
        
        print("Sending batch request:", batch_request)  # Debug print
        
        # Use the batch processor
        mask = await inference_manager.batch_processor.add_request(batch_request)
        
        if mask is None:
            raise HTTPException(status_code=400, detail="Failed to generate mask")
            
        # Convert mask to response format
        mask_encoded = {
            "shape": list(mask.shape),
            "data": mask.tobytes().hex()
        }

        # Save mask data to database
        video.mask_data = mask_encoded
        db.commit()
        
        return mask_encoded
        
    except Exception as e:
        logger.error(f"Preview mask generation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up temporary files
        if video_path and os.path.exists(video_path):
            try:
                os.remove(video_path)
            except Exception as e:
                logger.error(f"Failed to remove temporary video file: {str(e)}")

        # Cleanup inference manager
        background_tasks.add_task(inference_manager.cleanup)

@app.post("/api/videos/{video_id}/generate-masks")
async def generate_full_masks(
    video_id: str,
    request: Request,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    try:
        print(f"Starting generate_full_masks for video_id: {video_id}")
        body = await request.json()
        print(f"Received request body: {body}")
        
        request_data = GenerateMasksRequest(
            bbox=body.get('bbox'),
            points=body.get('points'),
            super=body.get('super', False),
            method=body.get('method', 'default'),
            start_frame=body.get('start_frame', 0)
        )
        print(f"Parsed request data: {request_data}")
        
        # Get video and check access
        print("Querying video from database...")
        video = db.query(Video).filter(
            Video.id == video_id,
            Video.user_id == current_user.id
        ).first()
        
        if not video:
            print(f"Video not found for id: {video_id}")
            raise HTTPException(status_code=404, detail="Video not found")

        print(f"Found video: {video.id}")
        print(f"Video metadata: {video.video_metadata}")
        if video.video_metadata:
            duration = video.video_metadata.get('duration', 0)
        else:
            duration = 1
        print(f"Video duration: {duration}")
        
        credits_manager = CreditsManager(db)
        print("Created credits manager")
        
        # First get a cost estimate
        print("Getting cost estimate...")
        estimate = await credits_manager.get_cost_estimate(
            action=CreditAction.GENERATE_MASKS.value,
            duration_seconds=duration,
            options={'super': request_data.super}
        )
        print(f"Cost estimate: {estimate}")
        
        # Then attempt to deduct credits
        try:
            print("Attempting to deduct credits...")
            cost = await credits_manager.check_and_deduct_credits(
                user=current_user,
                action=CreditAction.GENERATE_MASKS.value,
                duration_seconds=duration,
                options={'super': request_data.super}
            )
            print(f"Successfully deducted {cost} credits")
        except HTTPException as e:
            print(f"Failed to deduct credits: {e.detail}")
            error_detail = e.detail
            error_detail["estimate"] = estimate
            raise HTTPException(
                status_code=e.status_code,
                detail=error_detail
            )

        print("Creating task...")
        task = Task(
            id=str(uuid4()),
            video_id=video_id,
            status="pending",
            user_id=current_user.id,
            credit_cost=cost,
            credit_action=CreditAction.GENERATE_MASKS.value
        )
        print(f"Created task with id: {task.id}")
        
        db.add(task)
        db.commit()
        print("Saved task to database")

        # Add task to background processing
        logger.info(f"Adding task {task.id} to background processing")
        from app.tasks import process_video_masks
        print("Request data:")
        
        print(request_data)
        start_frame = request_data.start_frame
        print("start_frame", start_frame)

        print("Adding task to background tasks...")
        background_tasks.add_task(
            process_video_masks,
            video_id=video_id,
            bbox=request_data.bbox,
            points=request_data.points,
            task_id=task.id,
            super=request_data.super,
            method=request_data.method,
            start_frame=start_frame
        )
        print("Task added to background tasks")
        
        print("Preparing response...")
        response = {
            "taskId": task.id,
            "creditCost": cost,
            "remainingCredits": current_user.user_credits,
            "estimate": estimate
        }
        print(f"Returning response: {response}")
        return response
        
    except HTTPException:
        print("Re-raising HTTP exception")
        raise
    except Exception as e:
        error_msg = f"Mask generation failed: {str(e)}"
        logger.error(error_msg)
        print(error_msg)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/tasks/{task_id}")
async def get_task_status(
    task_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    task = db.query(Task).filter(
        Task.id == task_id,
        Task.user_id == current_user.id
    ).first()
    
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
        
    response = {
        "status": task.status,
        "createdAt": task.created_at,
        "completedAt": task.completed_at,
        "errorMessage": task.error_message
    }
    
    # Get video for any URLs we need to return
    video = db.query(Video).filter(Video.id == task.video_id).first()
    
    # Return mask URL when initial mask is ready
    if task.status in ["processing greenscreen", "completed greenscreen", "completed"] and video and video.mask_key:
        try:
            mask_url = s3_client.generate_presigned_url(
                'get_object',
                Params={
                    'Bucket': BUCKET_NAME, 
                    'Key': video.mask_key,
                },
                ExpiresIn=3600
            )
            response["maskUrl"] = mask_url
            
        except Exception as e:
            logger.error(f"Error generating mask presigned URL: {str(e)}")
    
    # Return greenscreen URL only when fully complete
    if task.status == "completed greenscreen" and video and video.video_keys:
        try:
            greenscreen_key = video.video_keys.get('greenscreen')
            if greenscreen_key:
                greenscreen_url = s3_client.generate_presigned_url(
                    'get_object',
                    Params={
                        'Bucket': BUCKET_NAME, 
                        'Key': greenscreen_key,
                    },
                    ExpiresIn=3600
                )
                response["greenscreenUrl"] = greenscreen_url
                
        except Exception as e:
            logger.error(f"Error generating greenscreen presigned URL: {str(e)}")
    
    return response

@app.get("/api/test-presigned-url")
async def test_presigned_url():
    try:
        # Try to generate a presigned URL for a test object
        test_key = "test.mp4"
        
        # First verify the object exists
        try:
            s3_client.head_object(Bucket=BUCKET_NAME, Key=test_key)
            print("Test object exists in S3")
        except Exception as e:
            print(f"Test object does not exist, error: {str(e)}")
            # Create a test object if it doesn't exist
            try:
                s3_client.put_object(
                    Bucket=BUCKET_NAME,
                    Key=test_key,
                    Body="This is a test file"
                )
                print("Created test object in S3")
            except Exception as e:
                return JSONResponse(
                    status_code=500,
                    content={"error": f"Failed to create test object: {str(e)}"}
                )

        # Generate presigned URL
        test_url = s3_client.generate_presigned_url(
            'get_object',
            Params={
                'Bucket': BUCKET_NAME,
                'Key': test_key,
            },
            ExpiresIn=3600
        )

        return {
            "message": "Successfully generated presigned URL",
            "url": test_url,
            "bucket": BUCKET_NAME,
            "key": test_key
        }

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to generate presigned URL: {str(e)}"}
        )


@app.patch("/api/videos/{video_id}")
async def update_video(
    video_id: str,
    video_update: dict,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    video = db.query(Video).filter(
        Video.id == video_id,
        Video.user_id == current_user.id
    ).first()
    
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    
    # Update video fields
    for field, value in video_update.items():
        setattr(video, field, value)
    
    try:
        db.commit()
        return {"message": "Video updated successfully"}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

#Unit Test
@app.get("/test-db")
async def test_db(db: Session = Depends(get_db)):
    try:
        # Try to create a table
        Base.metadata.create_all(bind=engine)
        return {"message": "Database connection successful!"}
    except Exception as e:
        logger.error(f"Database connection error: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Database connection failed: {str(e)}"
        )

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

@app.post("/api/credits/purchase")
async def purchase_credits(
    credit_purchase: CreditPurchase,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Add credits to user account"""
    current_user.user_credits += credit_purchase.amount
    db.commit()
    
    return {
        "credits": current_user.user_credits,
        "added": credit_purchase.amount
    }

@app.get("/api/credits")
async def get_user_credits(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get user's current credits and membership info"""
    return {
        "credits": current_user.user_credits,
        "membership": current_user.membership,
    }

@app.get("/api/videos/{video_id}/estimate-cost")
async def estimate_video_cost(
    video_id: str,
    action: CreditAction,
    super_mode: bool = False,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get cost estimate for a specific video operation"""
    try:
        # Get video and verify ownership
        video = db.query(Video).filter(
            Video.id == video_id,
            Video.user_id == current_user.id
        ).first()
        
        if not video:
            raise HTTPException(status_code=404, detail="Video not found")

        # Get video duration from metadata
        duration = video.video_metadata.get('duration', 0) if video.video_metadata else 0
        
        credits_manager = CreditsManager(db)
        estimate = await credits_manager.get_cost_estimate(
            action=action.value,
            duration_seconds=duration,
            options={'super': super_mode}
        )
        
        return {
            "estimate": estimate,
            "currentCredits": current_user.user_credits,
            "canAfford": current_user.user_credits >= estimate["cost"],
            "video": {
                "id": video.id,
                "title": video.title,
                "duration": duration
            }
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to estimate cost: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to calculate estimate")

storage_manager = StorageManager()

@app.get("/api/users/storage")
async def get_user_storage(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get user's storage usage"""
    try:
        # Calculate current storage
        storage_info = await storage_manager.calculate_user_storage(current_user.id)
        
        # Update user's storage information
        current_user.storage_used = {
            **storage_info,
            'last_updated': datetime.utcnow().isoformat()
        }
        db.commit()
        
        return current_user.storage_used
    except Exception as e:
        logger.error(f"Error getting storage info: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/videos/{video_id}/organize-storage")
async def organize_video_storage(
    video_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Organize video files into user-specific directories"""
    try:
        video = db.query(Video).filter(
            Video.id == video_id,
            Video.user_id == current_user.id
        ).first()
        
        if not video:
            raise HTTPException(status_code=404, detail="Video not found")

        # Update file locations
        if video.s3_key:
            video.s3_key = await storage_manager.update_file_location(
                video.s3_key, current_user.id, video_id, 'videos'
            )
        if video.thumbnail_key:
            video.thumbnail_key = await storage_manager.update_file_location(
                video.thumbnail_key, current_user.id, video_id, 'thumbnails'
            )
        if video.mask_key:
            video.mask_key = await storage_manager.update_file_location(
                video.mask_key, current_user.id, video_id, 'masks'
            )

        db.commit()
        
        # Update storage usage
        storage_info = await storage_manager.calculate_user_storage(current_user.id)
        current_user.storage_used = {
            **storage_info,
            'last_updated': datetime.utcnow().isoformat()
        }
        db.commit()

        return {"message": "Storage organized successfully", "storage": storage_info}
    except Exception as e:
        logger.error(f"Error organizing storage: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

def is_admin(user: User) -> bool:
    return user.super_user

@app.get("/api/admin/users")
async def get_all_users(
    skip: int = 0,
    limit: int = 10,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get all users (admin only)"""
    if not current_user.super_user:
        raise HTTPException(status_code=403, detail="Admin access required")
    
    # Get total count
    total = db.query(User).count()
    
    # Get paginated users
    users = db.query(User).offset(skip).limit(limit).all()
    
    # Convert users to dictionary format
    users_data = [{
        "id": user.id,
        "email": user.email,
        "created_at": user.created_at,
        "user_credits": user.user_credits,
        "membership": user.membership,
        "super_user": user.super_user,
        "storage_used": user.storage_used
    } for user in users]
    
    return {
        "users": users_data,
        "total": total
    }

@app.get("/api/admin/users/{user_id}", response_model=UserResponse)
async def get_user(
    user_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get specific user details (admin only)"""
    if not is_admin(current_user):
        raise HTTPException(status_code=403, detail="Admin access required")
        
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user

@app.put("/api/admin/users/{user_id}")
async def update_user(
    user_id: str,
    user_update: UserUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Update user (admin only)"""
    if not current_user.super_user:
        raise HTTPException(status_code=403, detail="Admin access required")
        
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Update user fields
    update_data = user_update.dict(exclude_unset=True)
    for field, value in update_data.items():
        setattr(user, field, value)
    
    try:
        # Log admin action
        admin_action = AdminAction(
            id=str(uuid.uuid4()),
            admin_id=current_user.id,
            action_type="update_user",
            target_user_id=user_id,
            details=update_data
        )
        db.add(admin_action)
        
        # Commit both user update and action log
        db.commit()
        
        # Return updated user data
        return {
            "id": user.id,
            "email": user.email,
            "user_credits": user.user_credits,
            "membership": user.membership,
            "super_user": user.super_user,
            "created_at": user.created_at,
            "storage_used": user.storage_used
        }
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=400, detail=str(e))

@app.delete("/api/admin/users/{user_id}")
async def delete_user(
    user_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Delete user (admin only)"""
    if not is_admin(current_user):
        raise HTTPException(status_code=403, detail="Admin access required")
        
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
        
    # Log admin action before deletion
    admin_action = AdminAction(
        id=str(uuid.uuid4()),
        admin_id=current_user.id,
        action_type="delete_user",
        target_user_id=user_id,
        details={"email": user.email}
    )
    db.add(admin_action)
    
    db.delete(user)
    db.commit()
    return {"message": "User deleted successfully"}

@app.get("/api/admin/actions")
async def get_admin_actions(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    skip: int = 0,
    limit: int = 100
):
    """Get admin action logs (admin only)"""
    if not is_admin(current_user):
        raise HTTPException(status_code=403, detail="Admin access required")
        
    actions = db.query(AdminAction)\
        .order_by(AdminAction.created_at.desc())\
        .offset(skip)\
        .limit(limit)\
        .all()
        
    return actions

@app.get("/api/users/me")
async def get_current_user_profile(current_user: User = Depends(get_current_user)):
    """Get current user profile"""
    if not current_user:
        raise HTTPException(status_code=404, detail="User not found")
        
    return {
        "id": current_user.id,
        "email": current_user.email,
        "super_user": current_user.super_user,
        "created_at": current_user.created_at,
        "user_credits": current_user.user_credits,
        "membership": current_user.membership,
        "storage_used": current_user.storage_used
    }

# Database session middleware
@app.middleware("http")
async def db_session_middleware(request: Request, call_next):
    try:
        request.state.db = SessionLocal()
        response = await call_next(request)
        return response
    except Exception as e:
        logger.error(f"Database session error: {str(e)}")
        raise
    finally:
        request.state.db.close()

# Mount the stripe routes
app.include_router(stripe_router, prefix="/api/stripe", tags=["stripe"])