import json
import os

# Load membership config from absolute path
CONFIG_PATH = '/home/ec2-user/megaton-roto/shared/membershipConfig.json'

with open(CONFIG_PATH, 'r') as config_file:
    config = json.load(config_file)

# Export the membership defaults and related config.
DEFAULT_MEMBERSHIP = config.get("defaultMembership")
TIER_CREDITS = config.get("TIER_CREDITS")
SUBSCRIPTION_PRICES = config.get("SUBSCRIPTION_PRICES")
TIER_FEATURES = config.get("TIER_FEATURES")
TIER_ORDER = config.get("TIER_ORDER")