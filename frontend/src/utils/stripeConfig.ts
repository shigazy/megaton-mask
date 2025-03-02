export const getStripeConfig = async () => {
    const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/api/stripe/config`);
    if (!response.ok) {
        throw new Error('Failed to load Stripe configuration');
    }
    return response.json();
}; 