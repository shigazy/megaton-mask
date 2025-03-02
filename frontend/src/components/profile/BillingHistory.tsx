'use client';

import React, { useState, useEffect } from 'react';
import { format } from 'date-fns';
import { FaFileInvoice, FaDownload, FaSpinner } from 'react-icons/fa';

interface Invoice {
    id: string;
    amount_paid: number;
    currency: string;
    status: string;
    created: number;
    period_start: number;
    period_end: number;
    invoice_pdf: string;
    hosted_invoice_url: string;
    subscription_id: string;
    tier: string;
}

interface BillingHistoryProps {
    isOpen: boolean;
    onClose: () => void;
}

export const BillingHistory: React.FC<BillingHistoryProps> = ({ isOpen, onClose }) => {
    const [invoices, setInvoices] = useState<Invoice[]>([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);
    const [hasMore, setHasMore] = useState(false);

    const fetchInvoices = async (startingAfter?: string) => {
        try {
            const token = localStorage.getItem('token');
            const response = await fetch(
                `${process.env.NEXT_PUBLIC_API_URL}/api/stripe/invoices${startingAfter ? `?starting_after=${startingAfter}` : ''}`,
                {
                    headers: {
                        'Authorization': `Bearer ${token}`
                    }
                }
            );

            if (!response.ok) {
                throw new Error('Failed to fetch invoices');
            }

            const data = await response.json();
            setInvoices(prev => startingAfter ? [...prev, ...data.invoices] : data.invoices);
            setHasMore(data.has_more);
        } catch (err) {
            setError('Failed to load billing history');
            console.error('Error fetching invoices:', err);
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => {
        if (isOpen) {
            fetchInvoices();
        }
    }, [isOpen]);

    if (!isOpen) return null;

    return (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50">
            <div className="bg-[var(--card-background)] rounded-xl p-6 max-w-4xl w-full max-h-[80vh] overflow-y-auto">
                <div className="flex justify-between items-center mb-6">
                    <h2 className="text-2xl font-bold text-[var(--text-primary)]">Billing History</h2>
                    <button
                        onClick={onClose}
                        className="text-[var(--text-secondary)] hover:text-[var(--text-primary)] transition-colors"
                    >
                        Close
                    </button>
                </div>

                {loading ? (
                    <div className="flex justify-center py-8">
                        <FaSpinner className="animate-spin h-8 w-8 text-[var(--accent-purple)]" />
                    </div>
                ) : error ? (
                    <div className="text-red-500 text-center py-4">{error}</div>
                ) : invoices.length === 0 ? (
                    <div className="text-[var(--text-secondary)] text-center py-4">
                        No billing history available
                    </div>
                ) : (
                    <>
                        <div className="space-y-4">
                            {invoices.map((invoice) => (
                                <div
                                    key={invoice.id}
                                    className="bg-[var(--background)] p-4 rounded-lg border border-[var(--border-color)] flex items-center justify-between"
                                >
                                    <div className="flex items-center gap-4">
                                        <FaFileInvoice
                                            className="h-5 w-5 text-[var(--accent-purple)]"
                                        />
                                        <div>
                                            <div className="font-medium text-[var(--text-primary)]">
                                                {invoice.tier.charAt(0).toUpperCase() + invoice.tier.slice(1)} Plan
                                            </div>
                                            <div className="text-sm text-[var(--text-secondary)]">
                                                {format(invoice.created * 1000, 'MMM d, yyyy')}
                                            </div>
                                        </div>
                                    </div>
                                    <div className="flex items-center gap-6">
                                        <div className="text-right">
                                            <div className="font-medium text-[var(--text-primary)]">
                                                ${invoice.amount_paid.toFixed(2)}
                                            </div>
                                            <div className="text-sm text-[var(--text-secondary)] capitalize">
                                                {invoice.status}
                                            </div>
                                        </div>
                                        <a
                                            href={invoice.invoice_pdf}
                                            target="_blank"
                                            rel="noopener noreferrer"
                                            className="p-2 hover:bg-[var(--background-hover)] rounded-full transition-colors"
                                        >
                                            <FaDownload
                                                className="h-5 w-5 text-[var(--text-secondary)] hover:text-[var(--accent-purple)]"
                                            />
                                        </a>
                                    </div>
                                </div>
                            ))}
                        </div>

                        {hasMore && (
                            <div className="mt-6 text-center">
                                <button
                                    onClick={() => fetchInvoices(invoices[invoices.length - 1].id)}
                                    className="px-4 py-2 text-[var(--text-secondary)] hover:text-[var(--accent-purple)] transition-colors"
                                >
                                    Load More
                                </button>
                            </div>
                        )}
                    </>
                )}
            </div>
        </div>
    );
}; 