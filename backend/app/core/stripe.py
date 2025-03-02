import stripe
from app.core.config import get_settings

settings = get_settings()

# Initialize the Stripe library with your secret key
stripe.api_key = settings.STRIPE_SECRET_KEY

# Expose the stripe module (or any helper functions) for your routes 