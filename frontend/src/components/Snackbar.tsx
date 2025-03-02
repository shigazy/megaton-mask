'use client';

import { useStatus } from '@/contexts/StatusContext';
import { motion, AnimatePresence } from 'framer-motion';
import { FaTimes } from 'react-icons/fa';

const STATUS_COLORS = {
    success: '#56dd92',  // Green
    error: '#dd6456',    // Red
    info: '#71a4f3',     // Blue
    processing: 'var(--accent-purple)', // Purple
} as const;

export const Snackbar = () => {
    const { status, type, clearStatus } = useStatus();

    return (
        <AnimatePresence>
            {status && (
                <motion.div
                    initial={{ y: -100, opacity: 0 }}
                    animate={{ y: 0, opacity: 1 }}
                    exit={{ y: -100, opacity: 0 }}
                    transition={{ type: "spring", stiffness: 300, damping: 30 }}
                    className="fixed top-4 left-1/2 z-50"
                    style={{ transform: 'translateX(-50%)' }}
                >
                    <div
                        className="px-6 py-3 rounded-full shadow-lg flex items-center gap-2"
                        style={{
                            backgroundColor: `${STATUS_COLORS[type]}`,
                            color: 'white'
                        }}
                    >
                        <span className="text-sm font-medium">
                            {status}
                        </span>
                        <button
                            onClick={clearStatus}
                            className="ml-2 hover:opacity-80 transition-opacity"
                            aria-label="Close notification"
                        >
                            <FaTimes size={14} />
                        </button>
                    </div>
                </motion.div>
            )}
        </AnimatePresence>
    );
}; 