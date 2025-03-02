'use client';

import { createContext, useContext, useState, useEffect } from 'react';

type StatusType = 'success' | 'error' | 'info' | 'processing';

interface StatusContextType {
    status: string;
    type: StatusType;
    setStatus: (message: string, type?: StatusType) => void;
    clearStatus: () => void;
}

const StatusContext = createContext<StatusContextType | undefined>(undefined);

export function StatusProvider({ children }: { children: React.ReactNode }) {
    const [status, setStatusMessage] = useState('');
    const [type, setType] = useState<StatusType>('processing');
    const [timeoutId, setTimeoutId] = useState<NodeJS.Timeout | null>(null);

    const clearStatus = () => {
        if (timeoutId) {
            clearTimeout(timeoutId);
            setTimeoutId(null);
        }
        setStatusMessage('');
    };

    const setStatus = (message: string, newType: StatusType = 'processing') => {
        // Clear any existing timeout
        if (timeoutId) {
            clearTimeout(timeoutId);
        }

        setStatusMessage(message);
        setType(newType);

        // Auto-dismiss after 3 seconds unless it's a processing message
        if (newType !== 'processing') {
            const id = setTimeout(() => {
                setStatusMessage('');
            }, 3000);
            setTimeoutId(id);
        }
    };

    // Cleanup on unmount
    useEffect(() => {
        return () => {
            if (timeoutId) {
                clearTimeout(timeoutId);
            }
        };
    }, [timeoutId]);

    return (
        <StatusContext.Provider value={{ status, type, setStatus, clearStatus }}>
            {children}
        </StatusContext.Provider>
    );
}

export function useStatus() {
    const context = useContext(StatusContext);
    if (context === undefined) {
        throw new Error('useStatus must be used within a StatusProvider');
    }
    return context;
}