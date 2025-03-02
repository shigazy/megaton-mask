'use client';

import { useAuth } from '@/contexts/AuthContext';
import { useRouter } from 'next/navigation';
import { useEffect } from 'react';

export const AdminRoute = ({ children }: { children: React.ReactNode }) => {
    const { user, isLoading } = useAuth();
    const router = useRouter();

    useEffect(() => {
        console.log('AdminRoute - Current user:', user);
        console.log('AdminRoute - Is loading:', isLoading);
        console.log('AdminRoute - Is admin:', user?.super_user);

        if (!isLoading && (!user || !user.super_user)) {
            console.log('AdminRoute - Redirecting to homepage');
            router.push('/');
        }
    }, [user, isLoading, router]);

    if (isLoading) {
        console.log('AdminRoute - Showing loading state');
        return (
            <div className="flex items-center justify-center h-screen">
                <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-[var(--accent-purple)]"></div>
            </div>
        );
    }

    if (!user?.super_user) {
        console.log('AdminRoute - User is not admin');
        return null;
    }

    console.log('AdminRoute - Rendering admin content');
    return <>{children}</>;
}; 