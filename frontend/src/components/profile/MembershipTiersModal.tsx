'use client';

import React from 'react';
import { SUBSCRIPTION_PRICES, TIER_FEATURES, TIER_ORDER } from '@/config/membershipConfig';
import { Dialog } from '@/components/ui/Dialog';
import { FaCheck } from 'react-icons/fa';

interface MembershipTiersModalProps {
    open: boolean;
    currentTier: string;
    onSelectTier: (selectedTier: string, priceId: string) => void;
    onClose: () => void;
}

export const MembershipTiersModal: React.FC<MembershipTiersModalProps> = ({
    open,
    currentTier,
    onSelectTier,
    onClose
}) => {
    // Include all tiers from TIER_FEATURES instead of SUBSCRIPTION_PRICES
    const availableTiers = Object.keys(TIER_FEATURES);

    return (
        <Dialog open={open} onClose={onClose}>
            <div className="fixed inset-0 flex items-center justify-center p-4">
                <div className="p-8 w-[95vw] max-w-[1400px] mx-auto bg-[var(--card-background)] rounded-xl">
                    <h2 className="text-2xl font-bold text-[var(--text-primary)] mb-8">
                        Choose Your Plan
                    </h2>
                    <div className="grid grid-cols-1 md:grid-cols-4 gap-12">
                        {availableTiers.map((tier) => {
                            const tierInfo = TIER_FEATURES[tier];
                            const isDowngrade = TIER_ORDER.indexOf(tier) < TIER_ORDER.indexOf(currentTier);
                            const isCurrentTier = tier === currentTier;
                            const priceId = SUBSCRIPTION_PRICES[tier] || null; // Handle free tier with no price ID

                            return (
                                <div
                                    key={tier}
                                    className={`bg-[var(--background)] rounded-xl p-8 border-2 transition-all duration-200
                                        ${isCurrentTier
                                            ? 'border-[var(--accent-purple)]'
                                            : 'border-[var(--border-color)] hover:border-[var(--accent-purple)]'
                                        }`}
                                >
                                    <h3 className="text-xl font-semibold text-[var(--text-primary)] mb-2">
                                        {tierInfo.name}
                                    </h3>
                                    <p className="text-3xl font-bold text-[var(--accent-purple)] mb-6">
                                        {tierInfo.price}
                                    </p>
                                    <ul className="space-y-4 mb-8">
                                        {tierInfo.features.map((feature, index) => (
                                            <li key={index} className="flex items-center gap-3 text-[var(--text-secondary)]">
                                                <FaCheck
                                                    size={16}
                                                    color="var(--accent-purple)"
                                                />
                                                <span>{feature}</span>
                                            </li>
                                        ))}
                                    </ul>
                                    <button
                                        onClick={() => onSelectTier(tier, priceId)}
                                        disabled={isCurrentTier}
                                        className={`w-full px-6 py-3 rounded-full transition-colors ${isCurrentTier
                                            ? 'bg-[var(--background)] text-[var(--text-secondary)] cursor-not-allowed border-2 border-[var(--border-color)]'
                                            : isDowngrade
                                                ? 'border-2 border-[var(--accent-purple)] text-[var(--accent-purple)] hover:bg-[var(--accent-purple)] hover:text-white'
                                                : 'bg-[var(--accent-purple)] text-white hover:bg-[var(--accent-purple-hover)]'
                                            }`}
                                    >
                                        {isCurrentTier ? 'Current Plan' : isDowngrade ? 'Downgrade' : 'Upgrade'}
                                    </button>
                                </div>
                            );
                        })}
                    </div>
                    <div className="mt-8 flex justify-end">
                        <button
                            onClick={onClose}
                            className="px-6 py-2 text-[var(--text-secondary)] hover:text-[var(--accent-purple)] transition-colors"
                        >
                            Close
                        </button>
                    </div>
                </div>
            </div>
        </Dialog>
    );
}; 