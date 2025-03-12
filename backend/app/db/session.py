from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.core.config import get_settings
from app.models import Base
import logging
from sqlalchemy.sql import text
import psycopg2

logger = logging.getLogger(__name__)
settings = get_settings()

def create_db_engine():
    """Create database engine with error handling"""
    database_url = settings.get_database_url()
    if not database_url:
        raise ValueError(
            "DATABASE_URL is not configured. "
            "Please ensure database credentials are properly loaded"
        )
    
    try:
        engine = create_engine(
            database_url,
            pool_size=5,
            max_overflow=10,
            pool_pre_ping=True
        )
        
        # Create all tables
        Base.metadata.create_all(bind=engine)
        print("Database tables created successfully")
        
        return engine
    except Exception as e:
        print(f"Failed to create database engine: {str(e)}")
        raise

try:
    engine = create_db_engine()
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
except Exception as e:
    logger.error(f"Database initialization failed: {str(e)}")
    raise

def get_db():
    db = SessionLocal()
    try:
        yield db
    except Exception as e:
        logger.error(f"Database session error: {str(e)}")
        raise
    finally:
        db.close()

def add_columns():
    """Add columns if they don't exist, and update foreign key constraints."""
    try:
        conn = psycopg2.connect(settings.DATABASE_URL)
        cursor = conn.cursor()

        # Add columns if they don't exist
        cursor.execute("""
        DO $$ 
        BEGIN 
            -- Existing columns
            BEGIN
                ALTER TABLE users ADD COLUMN IF NOT EXISTS user_credits INTEGER NOT NULL DEFAULT 0;
            EXCEPTION
                WHEN duplicate_column THEN NULL;
            END;
            
            BEGIN
                ALTER TABLE users ADD COLUMN IF NOT EXISTS stripe_customer_id VARCHAR;
            EXCEPTION
                WHEN duplicate_column THEN NULL;
            END;
            
            BEGIN
                ALTER TABLE users ADD COLUMN IF NOT EXISTS membership JSONB NOT NULL DEFAULT '{"tier": "free", "status": "active", "auto_renewal": true, "stripe_subscription_id": null, "updated_at": null, "current_period_end": null, "cancel_at": null, "pending_tier_change": null}';
            EXCEPTION
                WHEN duplicate_column THEN NULL;
            END;
            
            BEGIN
                ALTER TABLE users ADD COLUMN IF NOT EXISTS super_user BOOLEAN NOT NULL DEFAULT false;
            EXCEPTION
                WHEN duplicate_column THEN NULL;
            END;
            
            BEGIN
                ALTER TABLE users ADD COLUMN IF NOT EXISTS storage_used JSONB NOT NULL DEFAULT '{"total_bytes": 0, "total_gb": 0, "file_count": 0, "breakdown": {"videos_gb": 0, "masks_gb": 0, "thumbnails_gb": 0, "other_gb": 0}, "last_updated": null}';
            EXCEPTION
                WHEN duplicate_column THEN NULL;
            END;

            -- New email confirmation columns
            BEGIN
                ALTER TABLE users ADD COLUMN IF NOT EXISTS is_confirmed BOOLEAN NOT NULL DEFAULT false;
            EXCEPTION
                WHEN duplicate_column THEN NULL;
            END;
            
            BEGIN
                ALTER TABLE users ADD COLUMN IF NOT EXISTS confirmation_token VARCHAR;
            EXCEPTION
                WHEN duplicate_column THEN NULL;
            END;

            -- Videos table columns
            BEGIN
                ALTER TABLE videos ADD COLUMN IF NOT EXISTS bbox JSONB;
            EXCEPTION
                WHEN duplicate_column THEN NULL;
            END;
            
            BEGIN
                ALTER TABLE videos ADD COLUMN IF NOT EXISTS points JSONB;
            EXCEPTION
                WHEN duplicate_column THEN NULL;
            END;
            
            BEGIN
                ALTER TABLE videos ADD COLUMN IF NOT EXISTS mask_data JSONB;
            EXCEPTION
                WHEN duplicate_column THEN NULL;
            END;
            
            BEGIN
                ALTER TABLE videos ADD COLUMN IF NOT EXISTS forward_reverse_key VARCHAR;
            EXCEPTION
                WHEN duplicate_column THEN NULL;
            END;
            
            BEGIN
                ALTER TABLE videos ADD COLUMN IF NOT EXISTS video_metadata JSONB;
            EXCEPTION
                WHEN duplicate_column THEN NULL;
            END;
            
            BEGIN
                ALTER TABLE videos ADD COLUMN IF NOT EXISTS video_keys JSONB;
            EXCEPTION
                WHEN duplicate_column THEN NULL;
            END;
            
            BEGIN
                ALTER TABLE videos ADD COLUMN IF NOT EXISTS annotation JSONB;
            EXCEPTION
                WHEN duplicate_column THEN NULL;
            END;
            
            -- Tasks table columns
            BEGIN
                ALTER TABLE tasks ADD COLUMN IF NOT EXISTS credit_cost FLOAT;
            EXCEPTION
                WHEN duplicate_column THEN NULL;
            END;
            
            BEGIN
                ALTER TABLE tasks ADD COLUMN IF NOT EXISTS credit_action VARCHAR;
            EXCEPTION
                WHEN duplicate_column THEN NULL;
            END;
            
            -- Add progress column to tasks table
            BEGIN
                ALTER TABLE tasks ADD COLUMN IF NOT EXISTS progress FLOAT NOT NULL DEFAULT 0;
            EXCEPTION
                WHEN duplicate_column THEN NULL;
            END;
            
            -- Refresh tokens table
            BEGIN
                CREATE TABLE IF NOT EXISTS refresh_tokens (
                    id VARCHAR PRIMARY KEY,
                    user_id VARCHAR NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    token VARCHAR NOT NULL UNIQUE,
                    expires_at TIMESTAMP NOT NULL,
                    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    used BOOLEAN NOT NULL DEFAULT FALSE
                );
            EXCEPTION
                WHEN duplicate_table THEN NULL;
            END;
            
            -- Add index on token if not exists
            BEGIN
                CREATE INDEX IF NOT EXISTS ix_refresh_tokens_token ON refresh_tokens (token);
            EXCEPTION
                WHEN duplicate_table THEN NULL;
            END;

            -- Now update the foreign key constraint for tasks.video_id
            -- This block drops the foreign key (if it exists) and adds a new one with ON DELETE CASCADE.
            BEGIN
                ALTER TABLE tasks DROP CONSTRAINT IF EXISTS tasks_video_id_fkey;
            EXCEPTION
                WHEN undefined_object THEN
                    NULL;
            END;
            
            ALTER TABLE tasks 
              ADD CONSTRAINT tasks_video_id_fkey 
              FOREIGN KEY (video_id) REFERENCES videos(id) ON DELETE CASCADE;
        END $$;
        """)

        conn.commit()
        cursor.close()
        conn.close()
        logger.info("Successfully updated columns and foreign key constraints if needed")
    except Exception as e:
        logger.error(f"Error adding columns or updating constraints: {str(e)}")
        raise

# Add this after creating engine
add_columns()