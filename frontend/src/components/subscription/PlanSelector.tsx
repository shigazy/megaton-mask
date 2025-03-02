'use client';

import { useState } from 'react';
import { useAuth } from '@/contexts/AuthContext';
import { useCredits } from '@/contexts/CreditsContext';
import { getStripe } from '@/lib/stripe';

const plans = [
    {
        name: 'Free',
        price: 0,
        credits: 10,
        features: ['Basic access', '10 credits/month', 'Standard support'],
        priceId: null
    },
    {
        name: 'Basic',
        price: 29,
        credits: 100,
        features: ['Everything in Free', '100 credits/month', 'Priority support'],
        priceId: 'price_XXXXX' // Replace with your Stripe price ID
    },
    {
        name: 'Enterprise',
        price: 99,
        credits: 500,
        features: ['Everything in Basic', '500 credits/month', '24/7 support'],
        priceId: 'price_YYYY' // Replace with your Stripe price ID
    }
];

export const PlanSelector = () => {
    const [isLoading, setIsLoading] = useState(false);
    const { user } = useAuth();
    const { fetchCredits } = useCredits();

    const handleSubscribe = async (priceId: string) => {
        try {
            setIsLoading(true);
            const response = await fetch('/api/stripe/create-subscription-session', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${localStorage.getItem('token')}`
                },
                body: JSON.stringify({ priceId })
            });

            const { sessionId } = await response.json();
            const stripe = await getStripe();
            await stripe?.redirectToCheckout({ sessionId });
        } catch (error) {
            console.error('Error:', error);
        } finally {
            setIsLoading(false);
        }
    };

    const handleBuyCredits = async (amount: number) => {
        try {
            setIsLoading(true);
            const response = await fetch('/api/stripe/create-credits-session', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${localStorage.getItem('token')}`
                },
                body: JSON.stringify({ amount })
            });

            const { sessionId } = await response.json();
            const stripe = await getStripe();
            await stripe?.redirectToCheckout({ sessionId });
        } catch (error) {
            console.error('Error:', error);
        } finally {
            setIsLoading(false);
        }
    };

    return (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 p-6">
            {plans.map((plan) => (
                <div key={plan.name} className="border rounded-lg p-6 bg-[var(--card-background)]">
                    <h3 className="text-xl font-bold">{plan.name}</h3>
                    <p className="text-2xl font-bold mt-2">
                        ${plan.price}<span className="text-sm font-normal">/month</span>
                    </p>
                    <ul className="mt-4 space-y-2">
                        {plan.features.map((feature) => (
                            <li key={feature} className="flex items-center">
                                <span className="mr-2">✓</span>
                                {feature}
                            </li>
                        ))}
                    </ul>
                    {plan.priceId && (
                        <button
                            onClick={() => handleSubscribe(plan.priceId!)}
                            disabled={isLoading}
                            className="w-full mt-6 px-4 py-2 bg-[var(--accent-purple)] text-white rounded-full 
                                     hover:bg-[var(--accent-purple-hover)] transition-colors"
                        >
                            {isLoading ? 'Processing...' : 'Subscribe'}
                        </button>
                    )}
                </div>
            ))}

            {/* Credits Purchase Section */}
            <div className="col-span-full mt-8">
                <h2 className="text-2xl font-bold mb-4">Buy Additional Credits</h2>
                <div className="flex gap-4">
                    <button
                        onClick={() => handleBuyCredits(100)}
                        disabled={isLoading}
                        className="px-6 py-3 bg-[var(--accent-purple)] text-white rounded-full 
                                 hover:bg-[var(--accent-purple-hover)] transition-colors"
                    >
                        Buy 100 Credits
                    </button>
                    <button
                        onClick={() => handleBuyCredits(500)}
                        disabled={isLoading}
                        className="px-6 py-3 bg-[var(--accent-purple)] text-white rounded-full 
                                 hover:bg-[var(--accent-purple-hover)] transition-colors"
                    >
                        Buy 500 Credits
                    </button>
                </div>
            </div>
        </div>
    );
}; 