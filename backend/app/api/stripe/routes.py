from fastapi import APIRouter, Depends, HTTPException, Query, Body, Request
from sqlalchemy.orm import Session
from app.core.stripe import stripe
from app.core.auth import get_current_user
from app.models import User
from app.core.config import get_settings
from app.db.session import get_db
import os
import logging
from datetime import datetime
import boto3
from botocore.exceptions import ClientError

# Import the central membership config values
from app.config.membership_config import DEFAULT_MEMBERSHIP,TIER_FEATURES, TIER_CREDITS, SUBSCRIPTION_PRICES

router = APIRouter()
settings = get_settings()
logger = logging.getLogger(__name__)

stripe.api_key = settings.STRIPE_SECRET_KEY

# Create reverse mapping dynamically from the shared config
PRICE_TO_TIER = {price_id: tier for tier, price_id in SUBSCRIPTION_PRICES.items()}
# After your imports, add:
def get_user_instance(user, db: Session):
    """
    Given a user from get_current_user (which might be a dict or ORM instance),
    ensure we have a proper SQLAlchemy User instance.
    """
    if hasattr(user, "id"):
        return user
    # Assume it's a dictionary and re-query the DB.
    return db.query(User).filter(User.id == user.get("id")).first()
# Get the webhook secret from AWS Parameter Store
def get_webhook_secret():
    try:
        ssm = boto3.client('ssm')
        response = ssm.get_parameter(
            Name='/megaton/stripe_webhook_secret',
            WithDecryption=True
        )
        return response['Parameter']['Value']
    except ClientError as e:
        logger.error(f"Error fetching webhook secret: {str(e)}")
        raise HTTPException(status_code=500, detail="Could not fetch webhook secret")

CREDIT_PACKAGES = {
    100: {
        "price_id": "price_1Qq6ufCx0G73BQ1DoOP38jbq",  # Your 100 credits price ID
        "name": "100 Credits Package"
    },
    500: {
        "price_id": "price_1Qq6ufCx0G73BQ1DSDLg2VsT",  # Your 500 credits price ID
        "name": "500 Credits Package"
    }
}

# Reverse mapping to get tier from price ID
PRICE_TO_TIER = {price_id: tier for tier, price_id in SUBSCRIPTION_PRICES.items()}

@router.post("/create-subscription-session")
async def create_subscription_session(
    tier: str = Body(..., embed=True),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    try:
        # Log start of subscription creation
        logger.info(f"Starting subscription session creation for user {current_user.id}")
        logger.info(f"Requested tier: {tier}")
        
        # Step 1: Validate tier
        logger.info("Step 1: Validating tier")
        if tier not in SUBSCRIPTION_PRICES:
            logger.error(f"Invalid tier requested: {tier}")
            raise HTTPException(
                status_code=400,
                detail=f"Invalid tier. Available options: {list(SUBSCRIPTION_PRICES.keys())}"
            )
        logger.info("Tier validation successful")

        try:
            # Step 2: Handle Stripe customer creation/retrieval
            logger.info("Step 2: Managing Stripe customer")
            if current_user.stripe_customer_id:
                logger.info(f"Retrieving existing Stripe customer: {current_user.stripe_customer_id}")
                customer = stripe.Customer.retrieve(current_user.stripe_customer_id)
            else:
                logger.info("Creating new Stripe customer")
                customer = stripe.Customer.create(
                    email=current_user.email,
                    metadata={
                        'user_id': str(current_user.id)
                    }
                )
                logger.info(f"Created new Stripe customer with ID: {customer.id}")
                current_user.stripe_customer_id = customer.id
                db.commit()
                logger.info("Stored new Stripe customer ID in database")

            # Step 3: Create checkout session
            logger.info("Step 3: Creating Stripe checkout session")
            session = stripe.checkout.Session.create(
                payment_method_types=['card'],
                line_items=[{
                    'price': SUBSCRIPTION_PRICES[tier],
                    'quantity': 1,
                }],
                mode='subscription',
                success_url=f"{settings.NEXT_PUBLIC_URL}/?status=success&tier={tier}",
                cancel_url=f"{settings.NEXT_PUBLIC_URL}/?status=error",
                customer=customer.id,
                metadata={
                    'user_id': str(current_user.id),
                    'tier': tier
                }
            )
            
            logger.info(f"Successfully created Stripe session with ID: {session.id}")
            return {"sessionId": session.id}

        except stripe.error.StripeError as e:
            logger.error(f"Stripe API error occurred: {str(e)}")
            raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        logger.error(f"Unexpected error during subscription creation: {str(e)}")
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/create-credits-session")
async def create_credits_session(
    amount: int = Query(..., description="Amount of credits to purchase (100 or 500)"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    # Convert current_user to an ORM instance if necessary.
    user = get_user_instance(current_user, db)
    
    try:
        logger.info(f"Creating credits session for user {user.id}, amount: {amount}")
        
        # Validate amount
        amount = int(amount)
        if amount not in CREDIT_PACKAGES:
            logger.error(f"Invalid credit amount requested: {amount}")
            raise HTTPException(
                status_code=400,
                detail=f"Invalid credit amount. Available options: {list(CREDIT_PACKAGES.keys())}"
            )

        package = CREDIT_PACKAGES[amount]
        logger.info(f"Selected package: {package['name']}")

        try:
            # Create Stripe checkout session
            session = stripe.checkout.Session.create(
                payment_method_types=['card'],
                line_items=[{
                    'price': package["price_id"],
                    'quantity': 1,
                }],
                mode='payment',
                success_url=f"{settings.NEXT_PUBLIC_URL}/?status=success&amount={amount}",
                cancel_url=f"{settings.NEXT_PUBLIC_URL}/?status=error",
                client_reference_id=str(user.id)
            )
            
            # If session created successfully, add credits immediately
            user.user_credits += amount
            db.commit()
            
            logger.info(f"Successfully created session and added {amount} credits for user {user.id}")
            return {
                "sessionId": session.id,
                "success": True,
                "creditsAdded": amount,
                "totalCredits": user.user_credits
            }

        except stripe.error.StripeError as e:
            logger.error(f"Stripe error: {str(e)}")
            db.rollback()
            raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/config")
async def get_stripe_config():
    settings = get_settings()
    return {
        "publishableKey": settings.STRIPE_PUBLISHABLE_KEY
    }

@router.post("/verify-session")
async def verify_session(
    session_id: str = Body(..., embed=True),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    # Ensure current_user is an ORM instance
    user = get_user_instance(current_user, db)
    try:
        session = stripe.checkout.Session.retrieve(session_id)
        if session.client_reference_id != str(user.id):
            raise HTTPException(status_code=403, detail="Unauthorized")
        if session.payment_status != 'paid':
            raise HTTPException(status_code=400, detail="Payment not completed")
        credits_amount = int(session.metadata.get('credits_amount', 0))
        user.user_credits += credits_amount
        db.commit()            
        return {
            "success": True,
            "credits_added": credits_amount,
            "total_credits": user.user_credits
        }            
    except stripe.error.StripeError as e:
        logger.error(f"Stripe error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error verifying session: {str(e)}")
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/cancel-subscription")
async def cancel_subscription(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    try:
        logger.info(f"Attempting to cancel subscription for user {current_user.id}")
        
        # Check if user has an active subscription
        if not current_user.membership.get("stripe_subscription_id"):
            logger.error("No active subscription found")
            raise HTTPException(
                status_code=400, 
                detail="No active subscription found"
            )

        try:
            # Retrieve the subscription from Stripe
            subscription = stripe.Subscription.retrieve(
                current_user.membership["stripe_subscription_id"]
            )
            
            # Cancel the subscription at period end
            cancelled_subscription = stripe.Subscription.modify(
                subscription.id,
                cancel_at_period_end=True
            )
            
            logger.info(f"Subscription {subscription.id} cancelled successfully")

            # Explicitly copy the membership object and update
            membership = dict(current_user.membership)
            membership.update({
                "auto_renewal": False,
                "status": "active",  # Keep active until the end of period
                "cancel_at": (
                    datetime.fromtimestamp(cancelled_subscription.cancel_at).isoformat()
                    if cancelled_subscription.cancel_at else None
                ),
                "current_period_end": datetime.fromtimestamp(
                    cancelled_subscription.current_period_end
                ).isoformat(),
                "updated_at": datetime.utcnow().isoformat(),
                "pending_tier_change": {
                    "tier": "free",
                    "effective_date": datetime.fromtimestamp(
                        cancelled_subscription.current_period_end
                    ).isoformat()
                }
            })
            # Reassign the updated membership
            current_user.membership = membership
            db.add(current_user)
            db.commit()
            logger.info("User membership updated to reflect cancellation")

            return {
                "status": "success",
                "message": "Subscription cancelled successfully",
                "details": {
                    "current_period_end": datetime.fromtimestamp(
                        cancelled_subscription.current_period_end
                    ).isoformat(),
                    "cancel_at": (
                        datetime.fromtimestamp(cancelled_subscription.cancel_at).isoformat()
                        if cancelled_subscription.cancel_at else None
                    )
                }
            }

        except stripe.error.StripeError as e:
            logger.error(f"Stripe error while cancelling subscription: {str(e)}")
            raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        logger.error(f"Error cancelling subscription: {str(e)}")
        db.rollback()
        raise HTTPException(
            status_code=500,
            detail="An error occurred while cancelling the subscription"
        )

@router.post("/downgrade-subscription")
async def downgrade_subscription(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    try:
        # Get the user's active subscription
        subscription = await get_active_subscription(current_user.stripe_customer_id)
        if not subscription:
            raise HTTPException(status_code=404, detail="No active subscription found")

        # Schedule the update to take effect at the end of the current period
        updated_subscription = stripe.Subscription.modify(
            subscription.id,
            items=[{
                'id': subscription['items']['data'][0].id,
                'price': SUBSCRIPTION_PRICES['basic'],
            }],
            proration_behavior='none',  # Don't prorate, wait for next cycle
            billing_cycle_anchor='unchanged',  # Keep the existing billing cycle
        )

        # Update user's subscription status in database
        current_user.membership = {
            **current_user.membership,
            "pending_tier_change": {
                "tier": "basic",
                "effective_date": datetime.fromtimestamp(updated_subscription.current_period_end).isoformat()
            }
        }
        db.commit()

        return {
            "status": "success",
            "message": "Plan will be downgraded at the end of the current billing period",
            "effective_date": datetime.fromtimestamp(updated_subscription.current_period_end).isoformat()
        }

    except stripe.error.StripeError as e:
        logger.error(f"Stripe error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error downgrading subscription: {str(e)}")
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/toggle-auto-renewal")
async def toggle_auto_renewal(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    try:
        subscription = await get_active_subscription(current_user.stripe_customer_id)
        if not subscription:
            raise HTTPException(status_code=404, detail="No active subscription found")

        current_auto_renewal = current_user.membership.get('auto_renewal', True)
        new_auto_renewal = not current_auto_renewal

        if new_auto_renewal:
            # Remove cancel_at_period_end to enable auto-renewal
            stripe.Subscription.modify(
                subscription.id,
                cancel_at_period_end=False
            )
        else:
            # Set subscription to cancel at period end to disable auto-renewal
            stripe.Subscription.modify(
                subscription.id,
                cancel_at_period_end=True
            )

        # Reassign the membership dictionary so the change is tracked
        membership = dict(current_user.membership)
        membership['auto_renewal'] = new_auto_renewal
        current_user.membership = membership  # Reassign to trigger SQLAlchemy detection
        
        db.add(current_user)
        db.commit()

        return {
            "status": "success",
            "auto_renewal": new_auto_renewal,
            "message": f"Auto-renewal {'enabled' if new_auto_renewal else 'disabled'}"
        }

    except Exception as e:
        logger.error(f"Error toggling auto-renewal: {str(e)}")
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

# Helper function to get active subscription
async def get_active_subscription(user_id: str):
    subscriptions = stripe.Subscription.list(
        customer=user_id,
        status='active',
        limit=1
    )
    return subscriptions.data[0] if subscriptions.data else None

@router.post("/webhook")
async def stripe_webhook(request: Request, db: Session = Depends(get_db)):
    try:
        webhook_secret = get_webhook_secret()
        stripe_signature = request.headers.get('stripe-signature')
        payload = await request.body()
        
        event = stripe.Webhook.construct_event(
            payload,
            stripe_signature,
            webhook_secret
        )

        logger.info(f"Received Stripe event: {event['type']}")

        if event['type'] == 'checkout.session.completed':
            session = event['data']['object']
            logger.info(f"Processing completed checkout session: {session.get('id')}")
            
            metadata = session.get('metadata', {})
            user_id = metadata.get('user_id')
            tier_from_metadata = metadata.get('tier')
            
            if not user_id:
                logger.error("No user_id in session metadata")
                return {"error": "No user_id found"}

            # Retrieve subscription details from Stripe
            subscription = stripe.Subscription.retrieve(session.get('subscription'))
            logger.info(f"Retrieved subscription: {subscription.id}")

            # Retrieve customer details from the session
            customer_id = session.get('customer')
            logger.info(f"Customer ID from session: {customer_id}")

            # Retrieve the price_id from the session line items
            line_items = stripe.checkout.Session.list_line_items(session['id'])
            if not line_items['data']:
                logger.error(f"No line items found in session: {session['id']}")
                return {"error": "No line items found"}
            
            price_id = line_items['data'][0]['price']['id']
            logger.info(f"Processing price_id: {price_id}")

            new_tier = PRICE_TO_TIER.get(price_id, tier_from_metadata)
            if not new_tier:
                logger.error(f"Unknown price_id: {price_id}")
                return {"error": "Unknown price_id"}

            # Query the user in the database
            user = db.query(User).filter(User.id == user_id).first()
            if user:
                # Update the user's Stripe customer id
                user.stripe_customer_id = customer_id

                # Create an updated membership object based on the central default
                updated_membership = {
                    **DEFAULT_MEMBERSHIP,
                    "tier": new_tier,
                    "status": "active",
                    "auto_renewal": True,
                    "stripe_subscription_id": session.get('subscription'),
                    "stripe_customer_id": customer_id,
                    "updated_at": datetime.utcnow().isoformat(),
                    "current_period_end": datetime.fromtimestamp(subscription.current_period_end).isoformat(),
                    "cancel_at": datetime.fromtimestamp(subscription.cancel_at).isoformat() if subscription.cancel_at else None,
                    "pending_tier_change": None
                }
                user.membership = updated_membership

                # Add credits based on the tier using central config mapping
                credits_to_add = TIER_CREDITS.get(new_tier, 0)
                if credits_to_add > 0:
                    user.user_credits = (user.user_credits or 0) + credits_to_add
                    logger.info(f"Added {credits_to_add} credits to user {user_id}")
                
                db.commit()
                logger.info(f"Updated user {user_id} to {new_tier} tier with subscription {subscription.id}")
            else:
                logger.error(f"User not found: {user_id}")
                return {"error": "User not found"}

        elif event['type'] == 'customer.subscription.updated':
            subscription = event['data']['object']
            logger.info(f"Subscription updated: {subscription.id}")
            
            # Find the user by Stripe customer id
            user = db.query(User).filter(
                User.stripe_customer_id == subscription.customer
            ).first()
            if user:
                # Update membership with new subscription details
                user.membership.update({
                    "current_period_end": datetime.fromtimestamp(subscription.current_period_end).isoformat(),
                    "cancel_at": datetime.fromtimestamp(subscription.cancel_at).isoformat() if subscription.cancel_at else None,
                    "updated_at": datetime.utcnow().isoformat()
                })
                db.commit()
                logger.info(f"Updated subscription details for user {user.id}")

        elif event['type'] == 'customer.subscription.deleted':
            subscription = event['data']['object']
            logger.info(f"Subscription deleted: {subscription.id}")
            
            # Find the user by Stripe customer id
            user = db.query(User).filter(
                User.stripe_customer_id == subscription.customer
            ).first()
            if user:
                user.membership = {
                    **DEFAULT_MEMBERSHIP,
                    "tier": "free",
                    "status": "inactive",
                    "auto_renewal": False,
                    "stripe_subscription_id": None,
                    "stripe_customer_id": user.stripe_customer_id,  # keep the customer id
                    "updated_at": datetime.utcnow().isoformat(),
                    "current_period_end": None,
                    "cancel_at": None,
                    "pending_tier_change": None
                }
                db.commit()
                logger.info(f"Reverted user {user.id} to free tier")

        return {"status": "success"}

    except stripe.error.SignatureVerificationError as e:
        logger.error(f"Invalid signature: {str(e)}")
        raise HTTPException(status_code=400, detail="Invalid signature")
    except Exception as e:
        logger.error(f"Error processing webhook: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/cancel-auto-renewal")
async def cancel_auto_renewal(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    try:
        if not current_user.membership.get("stripe_subscription_id"):
            raise HTTPException(status_code=400, detail="No active subscription found")

        # Cancel the subscription at period end in Stripe
        subscription = stripe.Subscription.modify(
            current_user.membership["stripe_subscription_id"],
            cancel_at_period_end=True
        )

        # Update user membership info
        current_user.membership.update({
            "auto_renewal": False,
            "cancel_at": datetime.fromtimestamp(subscription.cancel_at).isoformat() if subscription.cancel_at else None,
            "updated_at": datetime.utcnow().isoformat()
        })
        
        db.commit()
        
        return {
            "message": "Auto-renewal disabled successfully",
            "cancel_at": current_user.membership["cancel_at"]
        }
    except stripe.error.StripeError as e:
        logger.error(f"Stripe error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error canceling auto-renewal: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.post("/schedule-downgrade")
async def schedule_downgrade(
    new_tier: str = Body(..., embed=True),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    try:
        # Validate the new tier is in our allowed tiers list
        if new_tier not in TIER_FEATURES:
            raise HTTPException(status_code=400, detail=f"Invalid tier selected. Must be one of: {', '.join(TIER_FEATURES.keys())}")
        
        # Don't allow scheduling a downgrade to the current tier
        if new_tier == current_user.membership.get('tier'):
            raise HTTPException(status_code=400, detail="Cannot downgrade to current tier")

        subscription = await get_active_subscription(current_user.stripe_customer_id)
        if not subscription:
            raise HTTPException(status_code=404, detail="No active subscription found")

        # Determine when the downgrade should take effect (end of current period)
        effective_date = datetime.fromtimestamp(subscription.current_period_end).isoformat()

        # Record the pending downgrade
        membership = dict(current_user.membership)
        membership["pending_tier_change"] = {
            "tier": new_tier,
            "effective_date": effective_date
        }
        membership["updated_at"] = datetime.utcnow().isoformat()
        current_user.membership = membership
        
        db.add(current_user)
        db.commit()

        return {
            "message": f"Plan will be downgraded to {new_tier} at the end of the current billing period.",
            "effective_date": effective_date
        }

    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error scheduling downgrade: {str(e)}")
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/invoices")
async def get_invoices(
    current_user: User = Depends(get_current_user),
    limit: int = 10,
    starting_after: str = None
):
    try:
        if not current_user.stripe_customer_id:
            return {
                "invoices": [],
                "has_more": False
            }

        # Fetch invoices from Stripe
        invoices = stripe.Invoice.list(
            customer=current_user.stripe_customer_id,
            limit=limit,
            starting_after=starting_after if starting_after else None,
            expand=['data.subscription']
        )

        # Format the invoice data
        formatted_invoices = [{
            'id': invoice.id,
            'amount_paid': invoice.amount_paid / 100,  # Convert cents to dollars
            'currency': invoice.currency,
            'status': invoice.status,
            'created': invoice.created,
            'period_start': invoice.period_start,
            'period_end': invoice.period_end,
            'invoice_pdf': invoice.invoice_pdf,
            'hosted_invoice_url': invoice.hosted_invoice_url,
            'subscription_id': invoice.subscription.id if invoice.subscription else None,
            'tier': invoice.lines.data[0].metadata.get('tier', 'unknown') if invoice.lines.data else 'unknown'
        } for invoice in invoices.data]

        return {
            "invoices": formatted_invoices,
            "has_more": invoices.has_more
        }

    except stripe.error.StripeError as e:
        logger.error(f"Stripe error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error fetching invoices: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error") 