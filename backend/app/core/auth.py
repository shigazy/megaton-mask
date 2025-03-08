from datetime import datetime, timedelta
from typing import Optional
from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session
from app.core.config import get_settings
from app.db.session import get_db
from app.models import User, RefreshToken
import uuid

settings = get_settings()
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: timedelta | None = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, settings.JWT_SECRET_KEY, algorithm=settings.ALGORITHM)

def create_refresh_token(user_id: str, db: Session) -> str:
    """Create a new refresh token and store it in the database"""
    # Generate a secure token
    token_value = str(uuid.uuid4())
    
    # Calculate expiration (30 days)
    expires_at = datetime.utcnow() + timedelta(days=30)
    
    # Store in database
    db_token = RefreshToken(
        user_id=user_id,
        token=token_value,
        expires_at=expires_at
    )
    db.add(db_token)
    db.commit()
    
    return token_value

def validate_refresh_token(token: str, db: Session) -> Optional[str]:
    """Validate a refresh token and return the user_id if valid"""
    # Find token in database
    db_token = db.query(RefreshToken).filter(
        RefreshToken.token == token,
        RefreshToken.expires_at > datetime.utcnow(),
        RefreshToken.used == False
    ).first()
    
    if not db_token:
        return None
    
    return db_token.user_id

def invalidate_refresh_token(token: str, db: Session) -> bool:
    """Mark a refresh token as used"""
    db_token = db.query(RefreshToken).filter(RefreshToken.token == token).first()
    if db_token:
        db_token.used = True
        db.commit()
        return True
    return False

async def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db)
) -> User:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, settings.JWT_SECRET_KEY, algorithms=[settings.ALGORITHM])
        user_id: str = payload.get("sub")
        if user_id is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception

    user = db.query(User).filter(User.id == user_id).first()
    if user is None:
        raise credentials_exception
    return user