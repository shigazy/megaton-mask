from sqlalchemy import Column, String, DateTime, ForeignKey, JSON, Integer, Boolean, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime
import uuid


# IMPORTANT: When adding new columns to any table, remember to update the add_columns() function
# in backend/app/db/session.py to include the new column definitions.
# Example:
# ADD COLUMN IF NOT EXISTS new_column_name VARCHAR/JSONB/etc;
# TODO: Automate this process.

Base = declarative_base()

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
    # Add relationship to refresh tokens
    refresh_tokens = relationship("RefreshToken", back_populates="user", cascade="all, delete-orphan")

class Video(Base):
    __tablename__ = "videos"

    id = Column(String, primary_key=True, index=True)
    title = Column(String)
    s3_key = Column(String)
    thumbnail_key = Column(String)
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
    annotation = Column(JSON, nullable=True)  # Store the annotation
    jpg_dir_key = Column(String, nullable=True)

class Task(Base):
    __tablename__ = "tasks"

    id = Column(String, primary_key=True, index=True)
    video_id = Column(String, ForeignKey("videos.id"))
    user_id = Column(String, ForeignKey("users.id"))
    status = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)
    error_message = Column(String, nullable=True)
    credit_cost = Column(Float, default=0)
    credit_action = Column(String, nullable=True)
    progress = Column(Float, default=0)

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

class GlobalConfig(Base):
    __tablename__ = "global_config"

    key = Column(String, primary_key=True)
    value = Column(Float)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class RefreshToken(Base):
    __tablename__ = "refresh_tokens"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, ForeignKey("users.id", ondelete="CASCADE"))
    token = Column(String, unique=True, index=True)
    expires_at = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)
    used = Column(Boolean, default=False)
    
    user = relationship("User", back_populates="refresh_tokens")