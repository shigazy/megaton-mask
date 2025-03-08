# Megaton Roto Project Context

## Project Overview

Megaton Roto is a video processing application focused on rotoscoping and mask generation for video content. The application allows users to upload videos, generate masks for objects in those videos, and create greenscreen effects. It's built with a modern tech stack including FastAPI backend, Next.js frontend, and utilizes advanced machine learning models for video segmentation.

## Architecture

The project follows a client-server architecture with these main components:

1. **Frontend**: Next.js application with TypeScript and Tailwind CSS
2. **Backend**: FastAPI Python application
3. **Database**: SQL database (PostgreSQL) managed via SQLAlchemy ORM
4. **Storage**: AWS S3 for video, mask, and thumbnail storage
5. **ML Inference**: Custom implementation using SAM2 (Segment Anything Model 2) for video segmentation
6. **Authentication**: JWT-based authentication system
7. **Background Processing**: Asynchronous task processing for video processing jobs

## Key Features

1. **Video Upload**: Users can upload videos to the platform
2. **Video Processing**: Automatic transcoding and thumbnail generation
3. **Mask Generation**: AI-powered object masking in videos
4. **Greenscreen Video**: Creating greenscreen effects by applying masks
5. **User Management**: Account creation, login, credits system
6. **Admin Interface**: User management, credit allocation
7. **Storage Management**: Organized user storage with usage tracking
8. **Membership System**: Tiered membership with different capabilities

## Database Schema

The database uses SQLAlchemy ORM with the following models:

### User Model
```python
class User(Base):
    __tablename__ = "users"

    id = Column(String, primary_key=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
    videos = relationship("Video", back_populates="user")
    user_credits = Column(Integer, default=0)
    stripe_customer_id = Column(String, nullable=True)
    is_confirmed = Column(Boolean, default=False)
    confirmation_token = Column(String, nullable=True)
    membership = Column(JSON, default=lambda: {
        "tier": "free",
        "status": "active",
        "auto_renewal": True,
        "stripe_subscription_id": None,
        "updated_at": None,
        "current_period_end": None,
        "cancel_at": None,
        "pending_tier_change": None
    })
    super_user = Column(Boolean, default=False)
    storage_used = Column(JSON, default=lambda: {
        "total_bytes": 0,
        "total_gb": 0,
        "file_count": 0,
        "breakdown": {
            "videos_gb": 0,
            "masks_gb": 0,
            "thumbnails_gb": 0,
            "other_gb": 0
        },
        "last_updated": None
    })
```

### Video Model
```python
class Video(Base):
    __tablename__ = "videos"

    id = Column(String, primary_key=True)
    title = Column(String)
    s3_key = Column(String)
    thumbnail_key = Column(String, nullable=True)
    mask_key = Column(String, nullable=True)
    user_id = Column(String, ForeignKey("users.id"))
    created_at = Column(DateTime, default=datetime.utcnow)
    user = relationship("User", back_populates="videos")
    bbox = Column(JSON, nullable=True)  # Store as {x, y, w, h}
    points = Column(JSON, nullable=True)  # Store as {positive: [[x,y]], negative: [[x,y]]}
    mask_data = Column(JSON, nullable=True)  # Store the mask data
    forward_reverse_key = Column(String, nullable=True)
    video_metadata = Column(JSON, nullable=True)  # Store the metadata
    video_keys = Column(JSON, nullable=True)  # Store the keys
```

### Task Model
```python
class Task(Base):
    __tablename__ = "tasks"

    id = Column(String, primary_key=True)
    video_id = Column(String, ForeignKey("videos.id"))
    status = Column(String)  # pending, processing, completed, failed
    user_id = Column(String, ForeignKey("users.id"))
    created_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)
    error_message = Column(String, nullable=True)
    credit_cost = Column(Float)
    credit_action = Column(String)  # e.g., "generate_masks", "generate_thumbnail", etc.
```

### AdminAction Model
```python
class AdminAction(Base):
    __tablename__ = "admin_actions"
    
    id = Column(String, primary_key=True)
    admin_id = Column(String, ForeignKey("users.id"))
    action_type = Column(String)  # e.g., "update_user", "delete_user", etc.
    target_user_id = Column(String, ForeignKey("users.id"), nullable=True)
    details = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    admin = relationship("User", foreign_keys=[admin_id])
    target_user = relationship("User", foreign_keys=[target_user_id])
```

### GlobalConfig Model
```python
class GlobalConfig(Base):
    __tablename__ = "global_config"

    key = Column(String, primary_key=True)
    value = Column(Float)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
```

## File System Structure

### Key Directories

- **backend/**: FastAPI backend application
  - **app/**: Main application code
    - **main.py**: FastAPI application entrypoint and routes
    - **models.py**: Database models
    - **tasks.py**: Background task handling
    - **core/**: Core application functionality
    - **db/**: Database connection and utilities
    - **api/**: API route definitions
  - **inference/**: Machine learning inference code
    - **manager.py**: ML model management
    - **optimizations/**: Performance optimization code
    - **scripts/**: Processing scripts
  - **alembic/**: Database migration scripts

- **frontend/**: Next.js frontend application
  - **src/**: Source code
    - **app/**: Next.js app router pages
    - **components/**: React components
    - **lib/**: Utility functions
    - **store/**: State management (Zustand)

- **_services/**: System service configuration files
  - **megaton-roto-dev.service**: Development backend service
  - **megaton-roto-frontend-dev.service**: Development frontend service
  - **megaton-roto.service**: Production backend service
  - **megaton-roto-frontend.service**: Production frontend service

## Core Functionality

### Video Upload and Processing

1. User uploads a video through the frontend
2. The backend receives the video, performs transcoding if needed
3. A thumbnail is generated and stored in S3
4. The processed video is stored in S3
5. Metadata is extracted and stored in the database

### Mask Generation

1. User selects points on the video to mark foreground/background
2. User can optionally provide a bounding box
3. The backend generates a preview mask using a lightweight model
4. User confirms and pays credits for full mask generation
5. Full mask generation runs as a background task using SAM2 model
6. Generated masks are stored in S3
7. Optional: Greenscreen video is generated using the masks

### Credits System

The platform uses a credit-based system for mask generation:
1. Credits cost varies based on video duration and quality settings
2. Users can purchase credits or receive them as part of membership tiers
3. Credits are deducted upon successful mask generation
4. Failed tasks may return credits to the user

### Membership Tiers

The system supports different membership tiers:
- **Free**: Basic functionality with limited storage and credits
- **Paid Tiers**: Additional storage, credits, and features
- Integration with Stripe for subscription management

### Storage Management

User storage is organized:
- Files are stored by user ID and video ID
- Storage usage is tracked by category (videos, masks, thumbnails)
- Storage limits may apply based on membership tier

## Machine Learning Infrastructure

### Inference Manager

The system uses a custom InferenceManager to handle ML tasks:
1. Optimizes memory usage and batch processing
2. Handles both preview mask generation (quick, less accurate) and full mask generation (slower, more accurate)
3. Uses different model sizes based on the task
4. Implements caching and memory management strategies

### Model Architecture

The system uses SAM2 (Segment Anything Model 2) with these key components:
- Different model sizes for different tasks (tiny, small, base, large)
- Video-specific adaptations for temporal consistency
- Optimized for both memory usage and inference speed

## API Endpoints

### Authentication
- `POST /api/auth/register`: Register a new user
- `POST /api/auth/login`: User login
- `POST /api/auth/confirm/{token}`: Confirm email
- `POST /api/auth/resend-confirmation`: Resend confirmation email

### Video Management
- `POST /api/videos/upload`: Upload a new video
- `GET /api/videos`: List user's videos
- `GET /api/videos/{video_id}/refresh-url`: Refresh presigned URL
- `DELETE /api/videos/{video_id}`: Delete a video
- `PATCH /api/videos/{video_id}`: Update video metadata

### Mask Generation
- `POST /api/videos/{video_id}/preview-mask`: Generate a preview mask
- `POST /api/videos/{video_id}/generate-masks`: Generate full masks
- `GET /api/tasks/{task_id}`: Get task status

### Credits and Billing
- `POST /api/credits/purchase`: Purchase credits
- `GET /api/credits`: Get user's credits
- `GET /api/videos/{video_id}/estimate-cost`: Estimate cost for processing
- `POST /api/stripe/*`: Stripe integration endpoints

### User and Admin
- `GET /api/users/me`: Get current user profile
- `GET /api/users/storage`: Get user's storage usage
- `GET /api/admin/users`: Admin: List all users
- `GET /api/admin/users/{user_id}`: Admin: Get user details
- `PUT /api/admin/users/{user_id}`: Admin: Update user
- `DELETE /api/admin/users/{user_id}`: Admin: Delete user
- `GET /api/admin/actions`: Admin: View admin action logs

## Development and Deployment

### Development Environment

The development environment uses:
- Conda environment for Python dependencies
- Node.js for frontend development
- Development services run on ports 3000 (frontend) and 8000 (backend)

### Production Environment

The production environment uses:
- Systemd services for application management
- Nginx as a reverse proxy
- AWS S3 for file storage
- Environment variables for configuration

### Service Management

The application includes systemd service files:
- `megaton-roto-frontend-dev.service`: Development frontend service
- `megaton-roto-dev.service`: Development backend service
- `megaton-roto-frontend.service`: Production frontend service
- `megaton-roto.service`: Production backend service

## Configuration

The application is configured through environment variables, including:
- `AWS_ACCESS_KEY_ID`: AWS access key
- `AWS_SECRET_ACCESS_KEY`: AWS secret key
- `AWS_S3_BUCKET`: S3 bucket name
- `AWS_REGION`: AWS region
- `SECRET_KEY`: Application secret key
- `MAIL_*`: Email server configuration
- `FRONTEND_URL`: URL of the frontend

## Troubleshooting

### Common Issues
1. **CORS Issues**: Ensure backend CORS settings match frontend URL
2. **Upload Failures**: Verify AWS credentials and S3 bucket permissions
3. **Port Already in Use**: Use provided commands to kill processes

### Logs
- Frontend logs: `journalctl -u megaton-roto-frontend.service -f`
- Backend logs: `journalctl -u megaton-roto.service -f`
- Development logs: `frontdevlog`, `backdevlog`
- Nginx logs: `nginxlog`

## Future Enhancements

Planned features and improvements:
1. Automated column addition process for database migrations
2. Improved storage management and organization
3. Additional video processing features
4. Enhanced UI/UX for mask editing
5. More membership tiers and features