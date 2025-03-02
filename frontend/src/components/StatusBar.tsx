'use client';

import { FaSpinner, FaTimes } from 'react-icons/fa';
import { useStatus } from '@/contexts/StatusContext';

const STATUS_COLORS = {
    success: '#56dd92',  // Green
    error: '#dd6456',    // Red
    info: '#71a4f3',     // Blue
    processing: 'var(--accent-purple)', // Purple
} as const;

type StatusType = 'success' | 'error' | 'info' | 'processing';

type StatusBarProps = {
    status: string;
    type?: StatusType;
};

export const StatusBar = ({ status, type = 'info' }: StatusBarProps) => {
    const { setStatus } = useStatus();

    if (!status) return null;

    return (
        <div
            className="w-full border-b border-[var(--border-color)] transition-colors duration-200"
            style={{
                backgroundColor: `${STATUS_COLORS[type]}20`,
            }}
        >
            <div className="max-w-7xl mx-auto px-4 h-8 flex items-center justify-between">
                <div className="flex items-center gap-2">
                    {type === 'processing' && (
                        <FaSpinner className="animate-spin text-[var(--accent-purple)]" />
                    )}
                    <span className="text-sm text-[var(--text-secondary)]">
                        {status}
                    </span>
                </div>
                <button
                    onClick={() => setStatus('')}
                    className="p-1 rounded-full hover:bg-black/10 transition-colors"
                    aria-label="Close status message"
                >
                    <FaTimes className="text-sm text-[var(--text-secondary)]" />
                </button>
            </div>
        </div>
    );
}; 