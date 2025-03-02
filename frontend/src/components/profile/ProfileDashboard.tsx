'use client';

import React, { useState, useEffect } from 'react';
import { useAuth } from '@/contexts/AuthContext';
import { useStatus } from '@/contexts/StatusContext';
import { getStripe } from '@/utils/stripeClient';
import {
    FaCrown,
    FaGem,
    FaAward,
    FaCoins,
    FaEnvelope,
    FaKey,
    FaDatabase,
    FaCreditCard,
    FaTimes,
    FaArrowDown,
    FaDiceOne
} from 'react-icons/fa';
// Add to your imports
import { MembershipTiersModal } from './MembershipTiersModal';
import { DEFAULT_MEMBERSHIP, TIER_CREDITS, SUBSCRIPTION_PRICES } from '@/config/membershipConfig';
import { Dialog } from '@/components/ui/Dialog';
import { BillingHistory } from './BillingHistory';

interface UserProfile {
    credits: number;
    membership: {
        tier: string;
        expires_at: string | null;
        status: string;
        auto_renewal: boolean;
        current_period_end: string | null;
        cancel_at: string | null;
        pending_tier_change: {
            tier: string;
            effective_date: string;
        } | null;
    };
    storage_used: {
        total_bytes: number;
        total_gb: number;
        file_count: number;
        breakdown: {
            videos_gb: number;
            masks_gb: number;
            thumbnails_gb: number;
            other_gb: number;
        };
        last_updated: string;
    };
    email: string;
}

const creditPackages = [
    { id: 'package_small', credits: 100, price_id: 'price_small' },
    { id: 'package_large', credits: 500, price_id: 'price_large' }
];

// Helper function to pick the correct icon based on the membership tier.
// For our purposes, we use FaCrown for both free and basic (but with different colors),
// FaGem for Pro and FaAward for Enterprise.
const getMembershipIcon = (tier?: string) => {
    switch (tier?.toLowerCase()) {
        case 'free':
            return <FaCrown className="text-2xl text-gray-400" />;
        case 'basic':
            return <FaCrown className="text-2xl text-blue-500" />;
        case 'pro':
            return <FaGem className="text-2xl text-purple-500" />;
        case 'enterprise':
            return <FaCrown className="text-2xl text-yellow-500" />;
        default:
            return <FaCrown className="text-2xl text-gray-400" />;
    }
};

interface ProfileDashboardProps {
    // define any expected props here
}

const ProfileDashboard: React.FC<ProfileDashboardProps> = () => {
    const { user } = useAuth();
    const { setStatus } = useStatus();
    const [profile, setProfile] = useState<UserProfile | null>(null);
    const [isAddingCredits, setIsAddingCredits] = useState(false);
    const [isOrganizing, setIsOrganizing] = useState(false);
    const [isLoading, setIsLoading] = useState(true);
    const [isUpgrading, setIsUpgrading] = useState(false);
    const [selectedPackage, setSelectedPackage] = useState(creditPackages[0]);
    const [isProcessingPayment, setIsProcessingPayment] = useState(false);
    const [showTiersModal, setShowTiersModal] = useState(false);
    const [showCancelDialog, setShowCancelDialog] = useState(false);
    const [isCancelling, setIsCancelling] = useState(false);
    const [showAutoRenewalDialog, setShowAutoRenewalDialog] = useState(false);
    const [isTogglingAutoRenewal, setIsTogglingAutoRenewal] = useState(false);
    const [showBillingHistory, setShowBillingHistory] = useState(false);

    const fetchProfile = async () => {
        try {
            setIsLoading(true);
            const token = localStorage.getItem('token');
            console.log('Fetching profile data...');

            const [creditsResponse, storageResponse] = await Promise.all([
                fetch(`${process.env.NEXT_PUBLIC_API_URL}/api/credits`, {
                    headers: {
                        'Authorization': `Bearer ${token}`
                    }
                }),
                fetch(`${process.env.NEXT_PUBLIC_API_URL}/api/users/storage`, {
                    headers: {
                        'Authorization': `Bearer ${token}`
                    }
                })
            ]);

            console.log('Credits Response:', creditsResponse.status);
            console.log('Storage Response:', storageResponse.status);

            const creditsData = await creditsResponse.json();
            const storageData = await storageResponse.json();

            console.log('Credits Data:', creditsData);
            console.log('Storage Data:', storageData);

            setProfile({
                credits: creditsData.credits,
                membership: creditsData.membership,
                storage_used: storageData,
                email: user?.email || ''
            });

        } catch (error) {
            console.error('Error fetching profile:', error);
        } finally {
            setIsLoading(false);
        }
    };

    useEffect(() => {
        if (user) {
            fetchProfile();
        }
    }, [user]);

    useEffect(() => {
        // Check for session_id in URL when component mounts
        const queryParams = new URLSearchParams(window.location.search);
        const sessionId = queryParams.get('session_id');

        if (sessionId) {
            handlePaymentSuccess(sessionId);
        }
    }, []);

    const handlePaymentSuccess = async (sessionId: string) => {
        try {
            setIsProcessingPayment(true);
            const token = localStorage.getItem('token');

            // Call your backend to verify the payment and add credits
            const response = await fetch(
                `${process.env.NEXT_PUBLIC_API_URL}/api/stripe/verify-session`,
                {
                    method: 'POST',
                    headers: {
                        'Authorization': `Bearer ${token}`,
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({ session_id: sessionId })
                }
            );

            if (!response.ok) {
                throw new Error('Failed to verify payment');
            }

            // Refresh the profile to show updated credits
            await fetchProfile();

            // Clear the session_id from URL
            window.history.replaceState({}, '', '/profile');

        } catch (error) {
            console.error('Error processing payment:', error);
        } finally {
            setIsProcessingPayment(false);
        }
    };

    // Add useEffect to check URL parameters when component mounts
    useEffect(() => {
        const queryParams = new URLSearchParams(window.location.search);
        const status = queryParams.get('status');
        const amount = queryParams.get('amount');

        if (status === 'success' && amount) {
            setStatus(`Successfully added ${amount} credits!`, 'success');
            // Clean up URL parameters
            window.history.replaceState({}, '', window.location.pathname);
        } else if (status === 'error') {
            setStatus('Error processing credit purchase', 'error');
            window.history.replaceState({}, '', window.location.pathname);
        }
    }, []);

    console.log('Render state:', { user, profile, isLoading });

    if (isLoading) {
        return (
            <div className="flex items-center justify-center h-[calc(100vh-4rem)]">
                <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-[var(--accent-purple)]"></div>
            </div>
        );
    }

    if (!user) {
        return (
            <div className="flex items-center justify-center h-[calc(100vh-4rem)]">
                <div className="text-center">
                    <p className="text-[var(--text-secondary)]">Please log in to view your profile.</p>
                </div>
            </div>
        );
    }

    if (!profile) {
        return (
            <div className="flex items-center justify-center h-[calc(100vh-4rem)]">
                <div className="text-center">
                    <p className="text-[var(--text-secondary)]">Failed to load profile data. Please try again.</p>
                </div>
            </div>
        );
    }

    const organizeStorage = async () => {
        try {
            setIsOrganizing(true);
            const token = localStorage.getItem('token');

            // First get all videos
            const videosResponse = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/api/videos`, {
                headers: {
                    'Authorization': `Bearer ${token}`
                }
            });

            if (!videosResponse.ok) throw new Error('Failed to fetch videos');
            const { videos } = await videosResponse.json();

            console.log(`Starting to organize ${videos.length} videos...`);

            // Organize each video's storage
            for (const video of videos) {
                const response = await fetch(
                    `${process.env.NEXT_PUBLIC_API_URL}/api/videos/${video.id}/organize-storage`,
                    {
                        method: 'POST',
                        headers: {
                            'Authorization': `Bearer ${token}`
                        }
                    }
                );

                if (!response.ok) {
                    console.error(`Failed to organize storage for video ${video.id}`);
                    continue;
                }

                const result = await response.json();
                console.log(`Organized storage for video ${video.id}:`, result);
            }

            // Refresh profile to get updated storage info
            await fetchProfile();

        } catch (error) {
            console.error('Error organizing storage:', error);
        } finally {
            setIsOrganizing(false);
        }
    };

    const handleBuyCredits = async () => {
        try {
            const token = localStorage.getItem('token');
            const response = await fetch(
                `${process.env.NEXT_PUBLIC_API_URL}/api/stripe/create-credits-session?amount=${selectedPackage.credits}`,
                {
                    method: 'POST',
                    headers: {
                        'Authorization': `Bearer ${token}`,
                        'Content-Type': 'application/json'
                    }
                }
            );

            if (!response.ok) {
                const errorData = await response.json();
                setStatus('Error initiating credit purchase', 'error');
                throw new Error(`Failed to create credits session: ${JSON.stringify(errorData)}`);
            }

            const data = await response.json();

            // Redirect to Stripe checkout
            const stripe = await getStripe();
            if (stripe) {
                const { error } = await stripe.redirectToCheckout({ sessionId: data.sessionId });
                if (error) {
                    setStatus('Error redirecting to checkout', 'error');
                    console.error('Stripe checkout error:', error);
                }
            }

        } catch (error) {
            console.error('Error purchasing credits:', error);
            setStatus('Error purchasing credits', 'error');
        }
    };

    const handlePlanChange = async (selectedTier: string, priceId: string) => {
        console.log("Plan selected:", selectedTier, priceId); // Debug log
        try {
            setIsUpgrading(true);
            // Define our tier priority (lowest to highest)
            const tierOrder = ["free", "basic", "pro", "enterprise"];
            const currentTier = profile.membership.tier;

            // Determine if this is a downgrade
            if (tierOrder.indexOf(selectedTier) < tierOrder.indexOf(currentTier)) {
                // Downgrade scenario: confirm with the user
                const confirmed = window.confirm(
                    `Are you sure you want to downgrade from ${currentTier} to ${selectedTier}? ` +
                    `This change will take effect at the end of your current billing period (${formatDate(profile.membership.current_period_end)}).`
                );
                if (!confirmed) {
                    return;
                }

                // Call the scheduling endpoint for downgrade (no redirect required)
                const token = localStorage.getItem('token');
                const response = await fetch(
                    `${process.env.NEXT_PUBLIC_API_URL}/api/stripe/schedule-downgrade`,
                    {
                        method: 'POST',
                        headers: {
                            'Authorization': `Bearer ${token}`,
                            'Content-Type': 'application/json'
                        },
                        body: JSON.stringify({ new_tier: selectedTier })
                    }
                );

                if (!response.ok) {
                    const errorData = await response.json();
                    throw new Error(`Failed to schedule downgrade: ${errorData.detail || 'Unknown error'}`);
                }

                const data = await response.json();
                setStatus(`Plan will be downgraded to ${selectedTier} at ${formatDate(data.effective_date)}`, "success");
                await fetchProfile();
            } else {
                // Upgrade scenario: create a subscription session in Stripe
                const token = localStorage.getItem('token');

                const response = await fetch(
                    `${process.env.NEXT_PUBLIC_API_URL}/api/stripe/create-subscription-session`,
                    {
                        method: 'POST',
                        headers: {
                            'Authorization': `Bearer ${token}`,
                            'Content-Type': 'application/json'
                        },
                        body: JSON.stringify({
                            tier: selectedTier,
                            priceId: priceId
                        })
                    }
                );

                if (!response.ok) {
                    const errorData = await response.json();
                    throw new Error(`Failed to create subscription session: ${errorData.detail || 'Unknown error'}`);
                }

                const data = await response.json();

                // Redirect to Stripe checkout
                const stripe = await getStripe();
                if (stripe) {
                    const { error } = await stripe.redirectToCheckout({
                        sessionId: data.sessionId
                    });

                    if (error) {
                        setStatus("Error redirecting to checkout", "error");
                        console.error("Stripe checkout error:", error);
                    } else {
                        // Success message will be shown after successful redirect and webhook processing
                        setStatus("Membership updated successfully!", "success");
                        await fetchProfile();
                    }
                }
            }
        } catch (error) {
            console.error("Error changing plan:", error);
            setStatus("Error changing plan. Please try again.", "error");
        } finally {
            setIsUpgrading(false);
            setShowTiersModal(false);
        }
    };

    const handleCancelMembership = () => {
        setShowCancelDialog(true);
    };

    const confirmCancelSubscription = async () => {
        setIsCancelling(true);
        try {
            const token = localStorage.getItem('token');
            console.log('Cancelling subscription...');
            const response = await fetch(
                `${process.env.NEXT_PUBLIC_API_URL}/api/stripe/cancel-subscription`,
                {
                    method: 'POST',
                    headers: {
                        'Authorization': `Bearer ${token}`,
                        'Content-Type': 'application/json',
                    },
                }
            );

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.detail || 'Failed to cancel subscription');
            }

            console.log('Subscription cancelled successfully');
            setStatus('Subscription cancellation scheduled.', 'success');
            setShowCancelDialog(false);

            // Wait a brief moment before refreshing the profile
            // This gives Stripe webhook a chance to process
            setTimeout(() => {
                fetchProfile();
            }, 1000);

        } catch (error) {
            console.error('Error cancelling subscription:', error);
            setStatus(error instanceof Error ? error.message : 'An error occurred while cancelling your subscription.', 'error');
        } finally {
            setIsCancelling(false);
        }
    };

    const handleDowngradeMembership = async () => {
        try {
            const token = localStorage.getItem('token');
            const response = await fetch(
                `${process.env.NEXT_PUBLIC_API_URL}/api/stripe/downgrade-subscription`,
                {
                    method: 'POST',
                    headers: {
                        'Authorization': `Bearer ${token}`,
                        'Content-Type': 'application/json'
                    }
                }
            );

            if (!response.ok) {
                const errorData = await response.json();
                setStatus('Error downgrading membership', 'error');
                throw new Error(`Failed to downgrade subscription: ${JSON.stringify(errorData)}`);
            }

            setStatus('Successfully downgraded membership', 'success');
            await fetchProfile(); // Refresh profile data
        } catch (error) {
            console.error('Error downgrading membership:', error);
            setStatus('Error downgrading membership', 'error');
        }
    };

    const handleToggleAutoRenewal = () => {
        // Only show confirmation dialog when disabling
        if (profile?.membership?.auto_renewal) {
            setShowAutoRenewalDialog(true);
        } else {
            // Enable without confirmation
            confirmToggleAutoRenewal();
        }
    };

    const confirmToggleAutoRenewal = async () => {
        setIsTogglingAutoRenewal(true);
        try {
            const token = localStorage.getItem('token');
            const response = await fetch(
                `${process.env.NEXT_PUBLIC_API_URL}/api/stripe/toggle-auto-renewal`,
                {
                    method: 'POST',
                    headers: {
                        'Authorization': `Bearer ${token}`,
                        'Content-Type': 'application/json'
                    }
                }
            );

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.detail || 'Failed to toggle auto-renewal');
            }

            const action = profile?.membership?.auto_renewal ? 'disabled' : 'enabled';
            setStatus(`Auto-renewal ${action} successfully`, 'success');
            setShowAutoRenewalDialog(false);

            // Wait a brief moment before refreshing the profile
            setTimeout(() => {
                fetchProfile();
            }, 1000);

        } catch (error) {
            console.error('Error toggling auto-renewal:', error);
            setStatus(
                error instanceof Error ? error.message : 'An error occurred while updating auto-renewal.',
                'error'
            );
        } finally {
            setIsTogglingAutoRenewal(false);
        }
    };

    const formatDate = (dateString: string | null) => {
        return dateString ? new Date(dateString).toLocaleDateString('en-US', {
            year: 'numeric',
            month: 'long',
            day: 'numeric'
        }) : 'N/A';
    };

    const getMainButtonLabel = (tier: string) => {
        switch (tier.toLowerCase()) {
            case 'free':
                return "Upgrade Plan";
            case 'enterprise':
                return "Downgrade Plan";
            default:
                return "Change Plan";
        }
    };

    return (
        <div className="bg-[var(--card-background)] rounded-lg p-6 shadow-sm">
            <h1 className="text-2xl font-bold mb-6">User Profile</h1>

            {isProcessingPayment && (
                <div className="fixed top-0 left-0 right-0 bg-green-500 text-white p-4 text-center">
                    Processing your payment...
                </div>
            )}

            <div className="space-y-6">
                {/* Credits Section */}
                <div className="flex items-center justify-between p-4 bg-[var(--background)] rounded-lg">
                    <div className="flex items-center gap-3">
                        <FaCoins className="text-2xl text-yellow-500" />
                        <div>
                            <h2 className="font-medium">Buy Credits</h2>
                            <p className="text-[var(--text-secondary)]">
                                Current Balance: {profile?.credits || 0} credits
                            </p>
                        </div>
                    </div>
                    <div className="flex flex-col items-end gap-2">
                        <div className="flex gap-2 mb-2">
                            {creditPackages.map((pkg) => (
                                <label
                                    key={pkg.id}
                                    className={`cursor-pointer px-4 py-2 rounded-full border 
                                        ${selectedPackage.id === pkg.id
                                            ? 'border-[var(--accent-purple)] text-[var(--accent-purple)]'
                                            : 'border-[var(--border-color)] hover:border-[var(--accent-purple)] hover:text-[var(--accent-purple)]'
                                        } transition-colors`}
                                >
                                    <input
                                        type="radio"
                                        name="creditPackage"
                                        value={pkg.id}
                                        checked={selectedPackage.id === pkg.id}
                                        onChange={() => setSelectedPackage(pkg)}
                                        className="hidden"
                                    />
                                    {pkg.credits} Credits
                                </label>
                            ))}
                        </div>
                        <button
                            onClick={handleBuyCredits}
                            className="w-full px-4 py-2 bg-[var(--accent-purple)] text-white rounded-full 
                                hover:bg-[var(--accent-purple-hover)] transition-colors"
                        >
                            Buy Credits
                        </button>
                    </div>
                </div>

                {/* Membership Section - Updated for consistent width */}
                <div className="flex items-center justify-between p-4 bg-[var(--background)] rounded-lg mt-4">
                    <div className="flex items-center gap-3">
                        {getMembershipIcon(profile.membership.tier)}
                        <div>
                            <h2 className="font-medium">Membership</h2>
                            <div className="space-y-1">
                                <p className="text-[var(--text-secondary)] capitalize">
                                    {profile.membership.tier} Plan
                                </p>
                                {profile.membership.tier !== 'free' && profile.membership.auto_renewal && (
                                    <p className="text-xs text-[var(--accent-purple)]">
                                        Auto-renewal active
                                    </p>
                                )}
                            </div>
                        </div>
                    </div>
                    <div className="flex flex-col items-end gap-2">

                        <button
                            onClick={() => setShowTiersModal(true)}
                            className="w-full px-4 py-2 bg-[var(--accent-purple)] text-white rounded-full 
                    hover:bg-[var(--accent-purple-hover)] transition-colors"
                        >
                            {getMainButtonLabel(profile.membership.tier)}
                        </button>


                        {profile.membership.tier !== 'free' && (
                            <>
                                <button
                                    onClick={handleToggleAutoRenewal}
                                    className="w-full px-4 py-2 border border-[var(--border-color)] rounded-full 
                        hover:border-[var(--accent-purple)] hover:text-[var(--accent-purple)] 
                        transition-colors text-sm"
                                    disabled={isTogglingAutoRenewal}
                                >
                                    {isTogglingAutoRenewal
                                        ? 'Updating...'
                                        : profile?.membership?.auto_renewal
                                            ? 'Disable Auto-Renewal'
                                            : 'Enable Auto-Renewal'
                                    }
                                </button>
                            </>
                        )}
                    </div>

                    {showTiersModal && (
                        <MembershipTiersModal
                            open={showTiersModal}
                            currentTier={profile.membership.tier}
                            onSelectTier={handlePlanChange}
                            onClose={() => setShowTiersModal(false)}
                        />
                    )}
                </div>

                {/* Account Settings */}
                <div className="space-y-4">
                    <h2 className="font-medium text-lg">Account Settings</h2>

                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                        <button className="flex items-center gap-3 p-4 bg-[var(--background)] rounded-lg hover:bg-[var(--border-color)] transition-colors">
                            <FaEnvelope className="text-xl text-[var(--text-secondary)]" />
                            <div className="text-left">
                                <h3 className="font-medium">Change Email</h3>
                                <p className="text-sm text-[var(--text-secondary)]">{user.email}</p>
                            </div>
                        </button>

                        <button className="flex items-center gap-3 p-4 bg-[var(--background)] rounded-lg hover:bg-[var(--border-color)] transition-colors">
                            <FaKey className="text-xl text-[var(--text-secondary)]" />
                            <div className="text-left">
                                <h3 className="font-medium">Change Password</h3>
                                <p className="text-sm text-[var(--text-secondary)]">Update your password</p>
                            </div>
                        </button>
                    </div>
                </div>

                {/* Storage Usage */}
                <div className="space-y-4">
                    <div className="flex items-center justify-between p-4 bg-[var(--background)] rounded-lg">
                        <div className="flex items-center gap-3">
                            <FaDatabase className="text-2xl text-[var(--text-secondary)]" />
                            <div>
                                <h2 className="font-medium">Storage Used</h2>
                                <p className="text-[var(--text-secondary)]">
                                    {profile?.storage_used ? (
                                        <>
                                            {profile.storage_used.total_gb.toFixed(2)} GB Total
                                            <br />
                                            <span className="text-sm">
                                                Videos: {profile.storage_used.breakdown.videos_gb.toFixed(2)} GB
                                                <br />
                                                Masks: {profile.storage_used.breakdown.masks_gb.toFixed(2)} GB
                                                <br />
                                                Files: {profile.storage_used.file_count}
                                            </span>
                                        </>
                                    ) : 'Calculating...'}
                                </p>
                            </div>
                        </div>
                    </div>
                    <p className="text-xs text-[var(--text-secondary)]">
                        Last updated: {profile?.storage_used?.last_updated ?
                            new Date(profile.storage_used.last_updated).toLocaleString() :
                            'Never'}
                    </p>
                </div>

                {/* Billing History */}
                <div className="flex items-center justify-between p-4 bg-[var(--background)] rounded-lg">
                    <div className="flex items-center gap-3">
                        <FaCreditCard className="text-2xl text-[var(--text-secondary)]" />
                        <div>
                            <h2 className="font-medium">Billing History</h2>
                            <p className="text-[var(--text-secondary)]">View your past transactions</p>
                            {profile.membership.auto_renewal ? (
                                <div className="space-y-1">
                                    <p className="text-[var(--text-secondary)] capitalize">
                                        Current Plan
                                    </p>
                                    <p className="text-xs text-[var(--accent-purple)]">
                                        Next billing date: {formatDate(profile.membership.current_period_end)}
                                    </p>
                                </div>
                            ) : (
                                <p className="text-xs text-[var(--accent-purple)]">
                                    Subscription ends on: {formatDate(profile.membership.cancel_at)}
                                </p>
                            )}

                            {profile.membership.pending_tier_change && (
                                <p className="text-xs text-[var(--accent-purple)]">
                                    Your plan will be downgraded to {profile.membership.pending_tier_change.tier} on {formatDate(profile.membership.pending_tier_change.effective_date)}
                                </p>
                            )}
                        </div>
                    </div>
                    <button
                        onClick={() => setShowBillingHistory(true)}
                        className="px-4 py-2 border border-[var(--border-color)] rounded-full hover:border-[var(--accent-purple)] hover:text-[var(--accent-purple)] transition-colors"
                    >
                        View History
                    </button>
                </div>
                <div className="flex justify-end">
                    {profile.membership.auto_renewal && (
                        <button
                            onClick={handleCancelMembership}
                            className="text-red-500 text-sm hover:text-red-600 transition-colors mt-1"
                        >
                            <FaTimes className="inline mr-1" />
                            Cancel Subscription
                        </button>
                    )}
                </div>
            </div>

            {/* Cancellation Confirmation Dialog */}
            {showCancelDialog && (
                <Dialog open={showCancelDialog} onClose={() => setShowCancelDialog(false)}>
                    <div className="bg-[var(--card-background)] p-6 rounded-lg max-w-md mx-auto">
                        <h3 className="text-lg font-semibold mb-4 text-[var(--text-primary)]">
                            Cancel Subscription?
                        </h3>
                        <p className="text-[var(--text-secondary)] mb-6">
                            Are you sure you want to cancel your subscription? You'll retain access until the end of your current billing period.
                        </p>
                        <div className="flex justify-end gap-3">
                            <button
                                onClick={() => setShowCancelDialog(false)}
                                className="px-4 py-2 bg-[var(--accent-purple)] text-white rounded-lg 
                                  hover:bg-[var(--accent-purple-hover)] transition-colors"
                                disabled={isCancelling}
                            >
                                Keep Subscription
                            </button>
                            <button
                                onClick={confirmCancelSubscription}
                                className="px-3 py-1.5 text-sm text-red-500 rounded-lg 
                                  hover:text-red-600 transition-colors disabled:opacity-50"
                                disabled={isCancelling}
                            >
                                {isCancelling ? 'Cancelling...' : 'Yes, Cancel'}
                            </button>
                        </div>
                    </div>
                </Dialog>
            )}

            {showAutoRenewalDialog && (
                <Dialog open={showAutoRenewalDialog} onClose={() => setShowAutoRenewalDialog(false)}>
                    <div className="bg-[var(--card-background)] p-6 rounded-lg max-w-md mx-auto">
                        <h3 className="text-lg font-semibold mb-4 text-[var(--text-primary)]">
                            Disable Auto-Renewal?
                        </h3>
                        <p className="text-[var(--text-secondary)] mb-6">
                            Are you sure you want to disable auto-renewal? You'll lose access to:
                            <br /><br />
                            • Premium features<br />
                            • Higher quality exports<br />
                            • Priority support
                            <br /><br />
                            Your subscription will remain active until the end of your current billing period.
                        </p>
                        <div className="flex justify-end gap-3">
                            <button
                                onClick={() => setShowAutoRenewalDialog(false)}
                                className="px-4 py-2 bg-[var--accent-purple)] text-white rounded-lg 
                                  hover:bg-[var(--accent-purple-hover)] transition-colors"
                                disabled={isTogglingAutoRenewal}
                            >
                                Keep Auto-Renewal
                            </button>
                            <button
                                onClick={confirmToggleAutoRenewal}
                                className="px-3 py-1.5 text-sm text-red-500 rounded-lg 
                                  hover:text-red-600 transition-colors disabled:opacity-50"
                                disabled={isTogglingAutoRenewal}
                            >
                                {isTogglingAutoRenewal ? 'Disabling...' : 'Yes, Disable'}
                            </button>
                        </div>
                    </div>
                </Dialog>
            )}

            <BillingHistory
                isOpen={showBillingHistory}
                onClose={() => setShowBillingHistory(false)}
            />
        </div>
    );
};

export default ProfileDashboard;