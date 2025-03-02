import { loadStripe } from '@stripe/stripe-js';
import { getStripeConfig } from './stripeConfig';

let stripePromise: Promise<any> | null = null;

export const getStripe = async () => {
    if (!stripePromise) {
        const config = await getStripeConfig();
        stripePromise = loadStripe(config.publishableKey);
    }
    return stripePromise;
};
