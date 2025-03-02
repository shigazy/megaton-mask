from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import boto3
from botocore.exceptions import ClientError
import os
from uuid import uuid4
from app.core.config import get_settings, Settings

app = FastAPI()
settings = get_settings()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize S3 client
s3_client = boto3.client(
    's3',
    aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
    aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
    region_name='us-east-1'  # Explicitly set to us-east-1
)

@app.get("/api/health")
async def health_check(settings: Settings = Depends(get_settings)):
    return {
        "status": "healthy",
        "environment": settings.ENV,
        "cors_origins": settings.allowed_origins
    }

@app.post("/api/videos/upload")
async def upload_video(file: UploadFile = File(...)):
    try:
        # Generate unique filename
        file_extension = file.filename.split('.')[-1]
        unique_filename = f"{uuid4()}.{file_extension}"
        
        # Upload to S3
        s3_client.upload_fileobj(
            file.file, 
            settings.AWS_S3_BUCKET, 
            f"videos/{unique_filename}",
            ExtraArgs={"ContentType": file.content_type}
        )
        
        # Generate presigned URL for viewing
        url = s3_client.generate_presigned_url(
            'get_object',
            Params={
                'Bucket': settings.AWS_S3_BUCKET, 
                'Key': f"videos/{unique_filename}"
            },
            ExpiresIn=3600  # URL expires in 1 hour
        )
        
        return JSONResponse({
            "success": True,
            "videoUrl": url,
            "key": f"videos/{unique_filename}"
        })
        
    except ClientError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/videos/{video_key}")
async def get_video_url(video_key: str):
    try:
        url = s3_client.generate_presigned_url(
            'get_object',
            Params={
                'Bucket': settings.AWS_S3_BUCKET, 
                'Key': video_key
            },
            ExpiresIn=3600
        )
        return {"url": url}
    except Exception as e:
        raise HTTPException(status_code=404, detail="Video not found")