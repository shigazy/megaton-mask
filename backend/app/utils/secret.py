import boto3
import json
from app.core.config import get_settings

settings = get_settings()

def get_secret():
    secret_name = "rds!db-b6e814ea-dcfb-4201-aa97-0c7f0dad8b81"  # Your secret name
    region_name = "us-east-1"  # Your region

    session = boto3.session.Session()
    client = boto3.client(
        service_name='secretsmanager',
        region_name=region_name
    )

    try:
        get_secret_value_response = client.get_secret_value(
            SecretId=secret_name
        )
    except Exception as e:
        raise e
    else:
        if 'SecretString' in get_secret_value_response:
            secret = json.loads(get_secret_value_response['SecretString'])
            return secret
