from sqlalchemy import Column, String, DateTime, ForeignKey, JSON, Integer, Boolean, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime


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
    annotation = Column(JSON, nullable=True)  # Store the annotation

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