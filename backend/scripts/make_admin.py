import sys
import os
from pathlib import Path

# Add the backend directory to Python path
backend_dir = Path(__file__).parent.parent
sys.path.append(str(backend_dir))

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.models import User
from app.core.config import get_settings

# Get settings including AWS credentials and database URL
settings = get_settings()
print(f"Environment: {settings.ENV}")

# Create database connection using the URL from settings
DATABASE_URL = settings.get_database_url()  # This will trigger parameter loading
print(f"Using database URL pattern: postgresql://user:****@{settings.DB_HOST}:{settings.DB_PORT}/{settings.DB_NAME}")

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
db = SessionLocal()

def make_user_admin(email: str):
    try:
        # Find user by email
        user = db.query(User).filter(User.email == email).first()
        if not user:
            print(f"No user found with email: {email}")
            return
        
        # Make user an admin
        user.super_user = True
        db.commit()
        print(f"Successfully made {email} an admin")
        
    except Exception as e:
        print(f"Error making user admin: {e}")
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python make_admin.py <email>")
        sys.exit(1)
        
    email = sys.argv[1]
    make_user_admin(email) 