'use client';

import { useState, useEffect } from 'react';
import { format } from 'date-fns';

interface AdminAction {
    id: string;
    admin_id: string;
    action_type: string;
    target_user_id: string;
    details: any;
    created_at: string;
    admin?: {
        email?: string;
    };
    target_user?: {
        email?: string;
    };
}

export const AdminActionLog = () => {
    const [actions, setActions] = useState<AdminAction[]>([]);
    const [isLoading, setIsLoading] = useState(true);
    const [page, setPage] = useState(1);
    const [hasMore, setHasMore] = useState(true);
    const ITEMS_PER_PAGE = 20;

    const fetchActions = async (pageNum: number) => {
        try {
            const token = localStorage.getItem('token');
            const response = await fetch(
                `${process.env.NEXT_PUBLIC_API_URL}/api/admin/actions?skip=${(pageNum - 1) * ITEMS_PER_PAGE}&limit=${ITEMS_PER_PAGE}`,
                {
                    headers: {
                        'Authorization': `Bearer ${token}`
                    }
                }
            );

            if (!response.ok) throw new Error('Failed to fetch actions');

            const data = await response.json();
            if (pageNum === 1) {
                setActions(data);
            } else {
                setActions(prev => [...prev, ...data]);
            }
            setHasMore(data.length === ITEMS_PER_PAGE);
        } catch (error) {
            console.error('Error fetching actions:', error);
        } finally {
            setIsLoading(false);
        }
    };

    useEffect(() => {
        fetchActions(page);
    }, [page]);

    const getActionDescription = (action?: AdminAction) => {
        if (!action) return 'Unknown Action';

        // Safely retrieve target email with fallback
        const targetEmail = action.target_user?.email || 'Unknown User';

        switch (action.action_type) {
            case 'update_user':
                return `Updated user ${targetEmail}`;
            case 'delete_user':
                return `Deleted user ${targetEmail}`;
            case 'cancel_subscription':
                return `Cancelled subscription for ${targetEmail}`;
            case 'update_subscription':
                return `Updated subscription for ${targetEmail}`;
            default:
                return `${action.action_type} - ${targetEmail}`;
        }
    };

    if (isLoading && page === 1) {
        return (
            <div className="flex justify-center items-center h-64">
                <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-[var(--accent-purple)]"></div>
            </div>
        );
    }

    return (
        <div className="space-y-4">
            <h2 className="text-xl font-semibold mb-4">Admin Actions</h2>
            <div className="space-y-2">
                {actions
                    .filter((action) => action !== undefined)
                    .map((action) => (
                        <div
                            key={action.id}
                            className="p-4 bg-[var(--card-background)] rounded-lg border border-[var(--border-color)]"
                        >
                            <div className="flex justify-between items-start">
                                <div>
                                    <p className="font-medium">
                                        {getActionDescription(action)}
                                    </p>
                                    <p className="text-sm text-[var(--text-secondary)]">
                                        By {action.admin?.email || 'Unknown Admin'}
                                    </p>
                                </div>
                                <span className="text-sm text-[var(--text-secondary)]">
                                    {format(new Date(action.created_at), 'MMM d, yyyy HH:mm')}
                                </span>
                            </div>
                            {action.details && (
                                <pre className="mt-2 p-2 bg-[var(--background)] rounded text-sm overflow-x-auto">
                                    {JSON.stringify(action.details, null, 2)}
                                </pre>
                            )}
                        </div>
                    ))}
            </div>

            {hasMore && (
                <div className="flex justify-center mt-4">
                    <button
                        onClick={() => setPage(prev => prev + 1)}
                        className="px-4 py-2 border border-[var(--border-color)] rounded-lg
                                 hover:border-[var(--accent-purple)] transition-colors"
                    >
                        Load More
                    </button>
                </div>
            )}
        </div>
    );
}; 