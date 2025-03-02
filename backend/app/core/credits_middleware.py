from fastapi import HTTPException
from sqlalchemy.orm import Session
from app.models import User
from .credits import (
    calculate_video_cost,
    InsufficientCreditsError,
    get_action_cost_info,
    is_action_refundable,
    CreditAction
)

class CreditsManager:
    def __init__(self, db: Session):
        self.db = db

    async def check_and_deduct_credits(
        self, 
        user: User, 
        action: str, 
        duration_seconds: float = 0,
        options: dict = None
    ) -> float:
        """
        Check credits and deduct if sufficient
        Returns the cost if successful, raises HTTPException if insufficient
        """
        try:
            cost = calculate_video_cost(duration_seconds, action, options)
            
            if user.user_credits < cost:
                raise InsufficientCreditsError(cost, user.user_credits)

            # Deduct credits
            user.user_credits -= cost
            self.db.commit()
            
            return cost

        except InsufficientCreditsError as e:
            raise HTTPException(
                status_code=402,
                detail={
                    "error": "Insufficient credits",
                    "required": e.required,
                    "available": e.available,
                    "action": action,
                    "actionInfo": get_action_cost_info(action)
                }
            )
        except ValueError as e:
            raise HTTPException(
                status_code=400,
                detail={"error": str(e)}
            )
        except Exception as e:
            self.db.rollback()
            raise HTTPException(
                status_code=500,
                detail={"error": f"Credit operation failed: {str(e)}"}
            )

    async def refund_credits(self, user: User, action: str, amount: float):
        """Refund credits to user if action is refundable"""
        try:
            if not is_action_refundable(action):
                return False

            user.user_credits += amount
            self.db.commit()
            return True

        except Exception as e:
            self.db.rollback()
            raise HTTPException(
                status_code=500,
                detail={"error": f"Credit refund failed: {str(e)}"}
            )

    async def get_cost_estimate(
        self,
        action: str,
        duration_seconds: float = 0,
        options: dict = None
    ) -> dict:
        """
        Get cost estimate for an action without deducting credits
        """
        try:
            cost = calculate_video_cost(duration_seconds, action, options)
            action_info = get_action_cost_info(action)
            
            return {
                "cost": cost,
                "action": action,
                "actionInfo": action_info,
                "duration": duration_seconds,
                "options": options
            }
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail={"error": f"Failed to calculate cost estimate: {str(e)}"}
            ) 