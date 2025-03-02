'use client';

import { useState } from 'react';
import { FaTimes } from 'react-icons/fa';
import { SUBSCRIPTION_PRICES } from '@/config/membershipConfig';

interface User {
    id: string;
    email: string;
    user_credits: number;
    membership: {
        tier: string;
        status: string;
    };
    super_user: boolean;
}

interface UserEditModalProps {
    user: User;
    onClose: () => void;
    onUpdate: () => void;
}

export const UserEditModal = ({ user, onClose, onUpdate }: UserEditModalProps) => {
    const [formData, setFormData] = useState({
        email: user.email,
        user_credits: user.user_credits,
        membership: user.membership,
        super_user: user.super_user
    });
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);

    const membershipTiers = ["free", ...Object.keys(SUBSCRIPTION_PRICES)];

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        setIsLoading(true);
        setError(null);

        try {
            const token = localStorage.getItem('token');
            const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/api/admin/users/${user.id}`, {
                method: 'PUT',
                headers: {
                    'Authorization': `Bearer ${token}`,
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(formData)
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || 'Failed to update user');
            }

            const updatedUser = await response.json();
            console.log('User updated successfully:', updatedUser);
            onUpdate();
            onClose();
        } catch (error) {
            console.error('Error updating user:', error);
            setError(error instanceof Error ? error.message : 'Failed to update user');
        } finally {
            setIsLoading(false);
        }
    };

    return (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
            <div className="bg-[var(--card-background)] rounded-lg p-6 max-w-md w-full">
                <div className="flex justify-between items-center mb-4">
                    <h2 className="text-xl font-semibold">Edit User</h2>
                    <button onClick={onClose} className="p-2 hover:text-[var(--accent-purple)]">
                        <FaTimes />
                    </button>
                </div>

                {error && (
                    <div className="mb-4 p-3 bg-red-100 border border-red-400 text-red-700 rounded">
                        {error}
                    </div>
                )}

                <form onSubmit={handleSubmit} className="space-y-4">
                    <div>
                        <label className="block text-sm font-medium mb-1">Email</label>
                        <input
                            type="email"
                            value={formData.email}
                            onChange={(e) => setFormData(prev => ({ ...prev, email: e.target.value }))}
                            className="w-full p-2 rounded-lg bg-[var(--background)] border border-[var(--border-color)]"
                        />
                    </div>

                    <div>
                        <label className="block text-sm font-medium mb-1">Credits</label>
                        <input
                            type="number"
                            value={formData.user_credits}
                            onChange={(e) => setFormData(prev => ({ ...prev, user_credits: parseInt(e.target.value) }))}
                            className="w-full p-2 rounded-lg bg-[var(--background)] border border-[var(--border-color)]"
                        />
                    </div>

                    <div>
                        <label className="block text-sm font-medium mb-1">Membership Tier</label>
                        <select
                            value={formData.membership.tier}
                            onChange={(e) =>
                                setFormData(prev => ({
                                    ...prev,
                                    membership: { ...prev.membership, tier: e.target.value }
                                }))
                            }
                            className="w-full p-2 rounded-lg bg-[var(--background)] border border-[var(--border-color)]"
                        >
                            {membershipTiers.map(tier => (
                                <option key={tier} value={tier}>
                                    {tier.charAt(0).toUpperCase() + tier.slice(1)}
                                </option>
                            ))}
                        </select>
                    </div>

                    <div className="flex items-center gap-2">
                        <input
                            type="checkbox"
                            id="super_user"
                            checked={formData.super_user}
                            onChange={(e) => setFormData(prev => ({ ...prev, super_user: e.target.checked }))}
                            className="rounded border-[var(--border-color)]"
                        />
                        <label htmlFor="super_user" className="text-sm font-medium">Admin Access</label>
                    </div>

                    <div className="flex justify-end gap-2 mt-6">
                        <button
                            type="button"
                            onClick={onClose}
                            className="px-4 py-2 border border-[var(--border-color)] rounded-lg hover:border-[var(--accent-purple)] transition-colors"
                        >
                            Cancel
                        </button>
                        <button
                            type="submit"
                            disabled={isLoading}
                            className="px-4 py-2 bg-[var(--accent-purple)] text-white rounded-lg hover:bg-[var(--accent-purple-hover)] transition-colors disabled:opacity-50"
                        >
                            {isLoading ? 'Saving...' : 'Save Changes'}
                        </button>
                    </div>
                </form>
            </div>
        </div>
    );
}; 