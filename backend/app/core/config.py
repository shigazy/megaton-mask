from pydantic_settings import BaseSettings
from pydantic import validator
from typing import List, Dict, Optional
import os
import boto3
from functools import lru_cache
import json
import logging

logger = logging.getLogger(__name__)

class ParameterStoreClient:
    def __init__(self):
        self.client = boto3.client('ssm', region_name='us-east-1')
        self.secrets_client = boto3.client('secretsmanager', region_name='us-east-1')

    def get_parameters_by_path(self, path: str = "/", decrypt: bool = True) -> Dict[str, str]:
        """Get all parameters under a path"""
        parameters = {}
        try:
            response = self.client.get_parameters(
                Names=[
                    'ACCESS_AWS_ROTO',
                    'SECRET_AWS_ROTO',
                    'JWT_SECRET_KEY',
                    '/megaton/stripe_test_key',
                    '/megaton/stripe_test_secret'
                ],
                WithDecryption=True
            )
            
            for param in response['Parameters']:
                name = param['Name']
                parameters[name] = param['Value']
            
            print(f"Retrieved parameters: {list(parameters.keys())}")
            return parameters
            
        except Exception as e:
            print(f"Error fetching parameters: {str(e)}")
            return {}

    def get_database_secret(self) -> Dict[str, str]:
        """Get database credentials from Secrets Manager"""
        try:
            secret_id = "rds!db-b6e814ea-dcfb-4201-aa97-0c7f0dad8b81"  # Your secret ID
            response = self.secrets_client.get_secret_value(SecretId=secret_id)
            if 'SecretString' in response:
                return json.loads(response['SecretString'])
            return {}
        except Exception as e:
            print(f"Error fetching database secret: {str(e)}")
            return {}

class Settings(BaseSettings):
    # Environment
    ENV: str = os.getenv("ENV", "development")

    # Database connection (non-sensitive)
    DB_HOST: str
    DB_PORT: str = "5432"
    DB_NAME: str = "postgres"
    
    # Development database fallback credentials
    POSTGRES_USER: str = "postgres"
    POSTGRES_PASSWORD: str = ""
    
    # AWS Settings
    AWS_ACCESS_KEY_ID: str = ""
    JWT_SECRET_KEY: str = ""
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 7  # 7 days

    AWS_SECRET_ACCESS_KEY: str = ""
    AWS_REGION: str = "us-east-1"
    AWS_S3_BUCKET: str = "megaton-roto-videos"
    
    DATABASE_URL: str = ""

    # Add Stripe settings
    STRIPE_SECRET_KEY: str = ""
    STRIPE_PUBLISHABLE_KEY: str = ""
    STRIPE_WEBHOOK_SECRET: str = ""

    NEXT_PUBLIC_URL: str = "https://app.megaton.ai"  

    # Email settings with default values
    MAIL_USERNAME: Optional[str] = None
    MAIL_PASSWORD: Optional[str] = None
    MAIL_FROM: Optional[str] = None
    MAIL_PORT: Optional[int] = None
    MAIL_SERVER: Optional[str] = None
    FRONTEND_URL: str = "https://app.megaton.ai"  # This is the default value

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Load mail settings from AWS Parameter Store
        try:
            self.MAIL_USERNAME = "apikey"  # For SES, this is fixed
            self.MAIL_FROM = "contact@megaton.ai"
            self.MAIL_SERVER = "email-smtp.us-east-1.amazonaws.com"  # SES SMTP endpoint
            self.MAIL_PORT = 587
            self.MAIL_PASSWORD = self.AWS_SECRET_ACCESS_KEY  # Use AWS credentials for SMTP authentication
            self.MAIL_STARTTLS = True  # Required for AWS SES
            self.MAIL_SSL_TLS = False  # Don't use SSL/TLS (we're using STARTTLS instead)
        except Exception as e:
            logger.error(f"Failed to load mail settings: {str(e)}")
            # Allow the application to start even if mail settings fail to load
            pass

    def get_database_url(self) -> str:
        """Get database URL, loading parameters if needed"""
        if not self.DATABASE_URL:
            self._load_parameters()
        return self.DATABASE_URL

    def _load_parameters(self):
        """Load credentials from AWS and combine with env variables"""
        try:
            print("Loading AWS credentials and secrets...")
            param_store = ParameterStoreClient()
            
            # Load AWS credentials from Parameter Store
            try:
                params = param_store.get_parameters_by_path()
                self.AWS_ACCESS_KEY_ID = params.get('ACCESS_AWS_ROTO')
                self.AWS_SECRET_ACCESS_KEY = params.get('SECRET_AWS_ROTO')
                self.JWT_SECRET_KEY = params.get('JWT_SECRET_KEY')
                
                # Add Stripe parameters
                self.STRIPE_SECRET_KEY = params.get('/megaton/stripe_test_secret')
                self.STRIPE_PUBLISHABLE_KEY = params.get('/megaton/stripe_test_key')
                
            except Exception as e:
                print(f"Warning: Failed to load parameters from Parameter Store: {e}")
                if self.ENV != "development":
                    raise
            
            # Load database credentials from Secrets Manager
            try:
                db_secrets = param_store.get_database_secret()
                if db_secrets and 'username' in db_secrets and 'password' in db_secrets:
                    # Construct DATABASE_URL using both sources
                    self.DATABASE_URL = (
                        f"postgresql://{db_secrets['username']}:{db_secrets['password']}"
                        f"@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"
                    )
                    print("Database URL constructed successfully")
                else:
                    print("Warning: Database secrets incomplete, using development credentials")
                    if self.ENV == "development":
                        # Use development credentials from .env
                        self.DATABASE_URL = (
                            f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}"
                            f"@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"
                        )
                        print("Using development database credentials")
            except Exception as e:
                print(f"Warning: Failed to load database secrets: {e}")
                if self.ENV == "development":
                    # Use development credentials from .env
                    self.DATABASE_URL = (
                        f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}"
                        f"@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"
                    )
                    print("Using development database credentials")
                else:
                    raise
            
            # Debug prints
            print(f"Environment: {self.ENV}")
            print(f"Database host: {self.DB_HOST}")
            print(f"Database name: {self.DB_NAME}")
            print(f"Database URL constructed: {'Yes' if self.DATABASE_URL else 'No'}")
            
        except Exception as e:
            print(f"Error loading parameters: {str(e)}")
            if self.ENV != "development":
                raise

    def get_mail_config(self) -> dict:
        """Get mail configuration as a dictionary"""
        return {
            'MAIL_USERNAME': self.MAIL_USERNAME,
            'MAIL_PASSWORD': self.MAIL_PASSWORD,  # This should not be None
            'MAIL_FROM': self.MAIL_FROM,
            'MAIL_PORT': self.MAIL_PORT,
            'MAIL_SERVER': self.MAIL_SERVER,
            'MAIL_STARTTLS': True,  # Required for AWS SES
            'MAIL_SSL_TLS': False,  # Don't use SSL/TLS
            'USE_CREDENTIALS': True  # Required when using authentication
        }

    class Config:
        env_file = ".env"

    @validator("AWS_S3_BUCKET")
    def validate_bucket_name(cls, v: str, values: dict) -> str:
        env = values.get("ENV", "development")
        if env == "production" and not v.startswith("prod-"):
            raise ValueError("Production bucket must start with 'prod-'")
        return v

    @property
    def allowed_origins(self) -> List[str]:
        return self.BACKEND_CORS_ORIGINS

    @property
    def aws_frontend_url(self) -> str:
        """
        Return the frontend URL fetched from AWS Parameter Store.
        This property does not conflict with the FRONTEND_URL field.
        """
        return self.get_aws_parameter('/megaton/frontend_url')

    def get_aws_parameter(self, parameter_name: str) -> str:
        """Get parameter from AWS Parameter Store"""
        try:
            ssm = boto3.client(
                'ssm',
                aws_access_key_id=self.ACCESS_AWS_ROTO,
                aws_secret_access_key=self.SECRET_AWS_ROTO,
                region_name='us-east-1'
            )
            response = ssm.get_parameter(
                Name=parameter_name,
                WithDecryption=True
            )
            return response['Parameter']['Value']
        except Exception as e:
            logger.error(f"Failed to get parameter {parameter_name}: {str(e)}")
            raise

@lru_cache()
def get_settings() -> Settings:
    return Settings()