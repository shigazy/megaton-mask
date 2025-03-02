from enum import Enum
from typing import Dict, Any

class CreditAction(Enum):
    GENERATE_MASKS = "generate_masks"
    GENERATE_GREENSCREEN = "generate_greenscreen"
    PREVIEW_MASK = "preview_mask"
    PROCESS_VIDEO = "process_video"
    # Add more actions as needed

CREDIT_COSTS = {
    CreditAction.GENERATE_MASKS.value: {
        'base': 1,
        'super': 2,
        'per_minute': 0.5,
        'description': 'Generate video masks',
        'refundable': True,
    },
    CreditAction.GENERATE_GREENSCREEN.value: {
        'base': 2,
        'per_minute': 1,
        'description': 'Generate greenscreen video',
        'refundable': True,
    },
    CreditAction.PREVIEW_MASK.value: {
        'base': 0.2,
        'description': 'Preview mask generation',
        'refundable': False,
    },
    CreditAction.PROCESS_VIDEO.value: {
        'base': 0.5,
        'per_minute': 0.25,
        'description': 'Process video',
        'refundable': True,
    }
}

class InsufficientCreditsError(Exception):
    def __init__(self, required: float, available: float):
        self.required = required
        self.available = available
        super().__init__(f"Insufficient credits. Required: {required}, Available: {available}")

def calculate_video_cost(duration_seconds: float, action: str, options: dict = None) -> float:
    """
    Calculate the credit cost for a video operation
    """
    options = options or {}
    
    if action not in CREDIT_COSTS:
        raise ValueError(f"Unknown action: {action}")
    
    costs = CREDIT_COSTS[action]
    total_cost = costs['base']
    
    # Add super cost if applicable
    if options.get('super') and 'super' in costs:
        total_cost += costs['super']
    
    # Add duration-based cost if applicable
    if 'per_minute' in costs and duration_seconds > 0:
        minutes = duration_seconds / 60
        total_cost += costs['per_minute'] * minutes
    
    return round(total_cost, 2)

def get_action_cost_info(action: str) -> Dict[str, Any]:
    """
    Get detailed cost information for an action
    """
    if action not in CREDIT_COSTS:
        raise ValueError(f"Unknown action: {action}")
    
    return CREDIT_COSTS[action]

def is_action_refundable(action: str) -> bool:
    """
    Check if an action is refundable
    """
    return CREDIT_COSTS.get(action, {}).get('refundable', False)

def has_sufficient_credits(user_credits: float, cost: float) -> bool:
    """Check if user has enough credits"""
    return user_credits >= cost 