import boto3
from botocore.exceptions import ClientError
from app.core.config import get_settings
import logging

logger = logging.getLogger(__name__)
settings = get_settings()

def send_ses_email(to_email: str, subject: str, body_html: str):
    """Send email using AWS SES with your verified sender"""
    
    mail_config = settings.get_mail_config()
    client = boto3.client(
        'ses',
        aws_access_key_id=mail_config['AWS_ACCESS_KEY'],
        aws_secret_access_key=mail_config['AWS_SECRET_KEY'],
        region_name=mail_config['AWS_REGION']
    )
    
    try:
        response = client.send_email(
            Source=settings.MAIL_CONFIG['MAIL_FROM'],
            Destination={
                'ToAddresses': [to_email]
            },
            Message={
                'Subject': {
                    'Data': subject,
                    'Charset': 'UTF-8'
                },
                'Body': {
                    'Html': {
                        'Data': body_html,
                        'Charset': 'UTF-8'
                    }
                }
            },
            ConfigurationSetName='megaton-auth-emails'  # Optional: Create this in SES for tracking
        )
        logger.info(f"Email sent successfully to {to_email}")
        return response['MessageId']
    except ClientError as e:
        logger.error(f"Failed to send email to {to_email}: {str(e)}")
        raise

def get_confirmation_template(confirmation_url: str) -> str:
    """HTML template for confirmation emails"""
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            body {{
                font-family: Arial, sans-serif;
                line-height: 1.6;
                color: #333;
            }}
            .container {{
                max-width: 600px;
                margin: 0 auto;
                padding: 20px;
            }}
            .button {{
                display: inline-block;
                padding: 12px 24px;
                background-color: #6D28D9;
                color: white;
                text-decoration: none;
                border-radius: 5px;
                margin: 20px 0;
            }}
            .footer {{
                margin-top: 30px;
                font-size: 12px;
                color: #666;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h2>Welcome to Megaton!</h2>
            <p>Please confirm your email address to complete your registration.</p>
            <a href="{confirmation_url}" class="button">Confirm Email</a>
            <p>Or copy and paste this link into your browser:</p>
            <p>{confirmation_url}</p>
            <div class="footer">
                <p>This link will expire in 24 hours.</p>
                <p>If you didn't create an account with Megaton, please ignore this email.</p>
            </div>
        </div>
    </body>
    </html>
    """

async def send_confirmation_email(email: str, token: str) -> bool:
    """Send confirmation email to user"""
    try:
        confirmation_url = f"{settings.FRONTEND_URL}/confirm-email?token={token}"
        subject = "Confirm Your Megaton Account"
        body_html = get_confirmation_template(confirmation_url)
        
        message_id = send_ses_email(
            to_email=email,
            subject=subject,
            body_html=body_html
        )
        
        logger.info(f"Confirmation email sent to {email}, MessageId: {message_id}")
        return True
    except Exception as e:
        logger.error(f"Failed to send confirmation email to {email}: {str(e)}")
        return False 