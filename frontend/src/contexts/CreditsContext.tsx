'use client';

import React, { createContext, useContext, useState, useCallback } from 'react';

interface CreditsContextType {
    credits: number;
    fetchCredits: () => Promise<void>;
}

const CreditsContext = createContext<CreditsContextType | undefined>(undefined);

export function CreditsProvider({ children }: { children: React.ReactNode }) {
    const [credits, setCredits] = useState<number>(0);

    const fetchCredits = useCallback(async () => {
        try {
            const token = localStorage.getItem('token');
            const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/api/credits`, {
                headers: {
                    'Authorization': `Bearer ${token}`
                }
            });
            const data = await response.json();
            setCredits(data.credits);
        } catch (error) {
            console.error('Error fetching credits:', error);
        }
    }, []);

    return (
        <CreditsContext.Provider value={{ credits, fetchCredits }}>
            {children}
        </CreditsContext.Provider>
    );
}

export function useCredits() {
    const context = useContext(CreditsContext);
    if (context === undefined) {
        throw new Error('useCredits must be used within a CreditsProvider');
    }
    return context;
} 