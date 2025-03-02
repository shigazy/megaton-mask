'use client';

import { AdminRoute } from '@/components/auth/AdminRoute';
import { UserList } from '@/components/admin/UserList';
import { AdminActionLog } from '@/components/admin/AdminActionLog';
import { Header } from '@/components/Header';
import { useState } from 'react';

export default function AdminPage() {
    const [activeTab, setActiveTab] = useState<'users' | 'actions'>('users');

    return (
        <AdminRoute>
            <div className="min-h-screen bg-[var(--background)]">
                <Header />
                <main className="max-w-7xl mx-auto py-8 px-4">
                    <div className="mb-6">
                        <h1 className="text-2xl font-bold">Admin Dashboard</h1>
                        <p className="text-[var(--text-secondary)]">Manage users and system settings</p>
                    </div>

                    <div className="mb-6">
                        <div className="border-b border-[var(--border-color)]">
                            <nav className="-mb-px flex space-x-8">
                                <button
                                    onClick={() => setActiveTab('users')}
                                    className={`py-4 px-1 border-b-2 font-medium text-sm
                                        ${activeTab === 'users'
                                            ? 'border-[var(--accent-purple)] text-[var(--accent-purple)]'
                                            : 'border-transparent text-[var(--text-secondary)] hover:text-[var(--text-primary)] hover:border-[var(--border-color)]'
                                        }`}
                                >
                                    Users
                                </button>
                                <button
                                    onClick={() => setActiveTab('actions')}
                                    className={`py-4 px-1 border-b-2 font-medium text-sm
                                        ${activeTab === 'actions'
                                            ? 'border-[var(--accent-purple)] text-[var(--accent-purple)]'
                                            : 'border-transparent text-[var(--text-secondary)] hover:text-[var(--text-primary)] hover:border-[var(--border-color)]'
                                        }`}
                                >
                                    Action Log
                                </button>
                            </nav>
                        </div>
                    </div>

                    {activeTab === 'users' ? <UserList /> : <AdminActionLog />}
                </main>
            </div>
        </AdminRoute>
    );
} 