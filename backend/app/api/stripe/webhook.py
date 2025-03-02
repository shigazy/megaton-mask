from fastapi import APIRouter, Request, HTTPException
from app.core.stripe import stripe
from app.models import User
from app.db.session import get_db

router = APIRouter()

@router.post("/webhook")
async def stripe_webhook(request: Request):
    payload = await request.body()
    sig_header = request.headers.get('stripe-signature')
    
    try:
        event = stripe.Webhook.construct_event(
            payload, sig_header, process.env.STRIPE_WEBHOOK_SECRET
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    if event.type == 'checkout.session.completed':
        session = event.data.object
        
        # Handle subscription
        if session.mode == 'subscription':
            # Update user's subscription status
            db = next(get_db())
            user = db.query(User).filter(User.id == session.metadata.user_id).first()
            if user:
                user.membership = {
                    "tier": "basic" if "basic" in session.metadata.get('plan', '') else "enterprise",
                    "status": "active",
                    "subscription_id": session.subscription
                }
                db.commit()
        
        # Handle one-time credit purchase
        elif session.mode == 'payment':
            db = next(get_db())
            user = db.query(User).filter(User.id == session.metadata.user_id).first()
            if user:
                user.user_credits += int(session.metadata.credits)
                db.commit()

    return {"status": "success"} 