'use client';

import { useState, useEffect } from 'react';
import { FaEdit, FaTrash, FaSearch, FaChevronLeft, FaChevronRight } from 'react-icons/fa';
import { UserEditModal } from './UserEditModal';
import { ConfirmDialog } from './ConfirmDialog';

interface User {
    id: string;
    email: string;
    created_at: string;
    user_credits: number;
    membership: {
        tier: string;
        status: string;
    };
    super_user: boolean;
    storage_used: {
        total_gb: number;
    };
}

export const UserList = () => {
    const [users, setUsers] = useState<User[]>([]);
    const [isLoading, setIsLoading] = useState(true);
    const [searchTerm, setSearchTerm] = useState('');
    const [selectedUser, setSelectedUser] = useState<User | null>(null);
    const [userToDelete, setUserToDelete] = useState<User | null>(null);
    const [currentPage, setCurrentPage] = useState(1);
    const [totalPages, setTotalPages] = useState(1);
    const [itemsPerPage] = useState(10);

    const fetchUsers = async (page: number) => {
        try {
            setIsLoading(true);
            const token = localStorage.getItem('token');
            const skip = (page - 1) * itemsPerPage;
            console.log('Fetching users with:', { page, skip, limit: itemsPerPage });

            const response = await fetch(
                `${process.env.NEXT_PUBLIC_API_URL}/api/admin/users?skip=${skip}&limit=${itemsPerPage}`,
                {
                    headers: {
                        'Authorization': `Bearer ${token}`
                    }
                }
            );

            if (!response.ok) {
                const errorData = await response.json();
                console.error('Error response:', errorData);
                throw new Error(`Failed to fetch users: ${response.status}`);
            }

            const data = await response.json();
            console.log('Fetched users:', data);

            setUsers(data.users || []);
            setTotalPages(Math.ceil((data.total || 0) / itemsPerPage));
        } catch (error) {
            console.error('Error fetching users:', error);
            setUsers([]);
        } finally {
            setIsLoading(false);
        }
    };

    useEffect(() => {
        fetchUsers(currentPage);
    }, [currentPage]);

    const handleDeleteUser = async (user: User) => {
        try {
            const token = localStorage.getItem('token');
            const response = await fetch(
                `${process.env.NEXT_PUBLIC_API_URL}/api/admin/users/${user.id}`,
                {
                    method: 'DELETE',
                    headers: {
                        'Authorization': `Bearer ${token}`
                    }
                }
            );

            if (!response.ok) throw new Error('Failed to delete user');

            // Refresh the current page
            fetchUsers(currentPage);
        } catch (error) {
            console.error('Error deleting user:', error);
        } finally {
            setUserToDelete(null);
        }
    };

    // Only filter users if they exist
    const filteredUsers = users && users.length > 0
        ? users.filter(user =>
            user.email.toLowerCase().includes(searchTerm.toLowerCase())
        )
        : [];

    const handlePageChange = (newPage: number) => {
        if (newPage >= 1 && newPage <= totalPages) {
            setCurrentPage(newPage);
        }
    };

    return (
        <div className="space-y-4">
            <div className="flex items-center gap-4 mb-6">
                <div className="relative flex-1">
                    <FaSearch className="absolute left-3 top-1/2 transform -translate-y-1/2 text-[var(--text-secondary)]" />
                    <input
                        type="text"
                        placeholder="Search users..."
                        value={searchTerm}
                        onChange={(e) => setSearchTerm(e.target.value)}
                        className="w-full pl-10 pr-4 py-2 rounded-lg bg-[var(--background)] border border-[var(--border-color)] focus:outline-none focus:border-[var(--accent-purple)]"
                    />
                </div>
            </div>

            {isLoading ? (
                <div className="flex justify-center items-center h-64">
                    <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-[var(--accent-purple)]"></div>
                </div>
            ) : (
                <>
                    <div className="overflow-x-auto">
                        <table className="w-full">
                            <thead className="bg-[var(--background)]">
                                <tr>
                                    <th className="px-4 py-2 text-left">Email</th>
                                    <th className="px-4 py-2 text-left">Credits</th>
                                    <th className="px-4 py-2 text-left">Membership</th>
                                    <th className="px-4 py-2 text-left">Storage</th>
                                    <th className="px-4 py-2 text-left">Admin</th>
                                    <th className="px-4 py-2 text-left">Actions</th>
                                </tr>
                            </thead>
                            <tbody>
                                {filteredUsers.map(user => (
                                    <tr key={user.id} className="border-t border-[var(--border-color)]">
                                        <td className="px-4 py-2">{user.email}</td>
                                        <td className="px-4 py-2">{user.user_credits}</td>
                                        <td className="px-4 py-2 capitalize">{user.membership.tier}</td>
                                        <td className="px-4 py-2">{user.storage_used.total_gb.toFixed(2)} GB</td>
                                        <td className="px-4 py-2">{user.super_user ? 'Yes' : 'No'}</td>
                                        <td className="px-4 py-2">
                                            <div className="flex items-center gap-2">
                                                <button
                                                    onClick={() => setSelectedUser(user)}
                                                    className="p-2 hover:text-[var(--accent-purple)] transition-colors"
                                                >
                                                    <FaEdit />
                                                </button>
                                                <button
                                                    onClick={() => setUserToDelete(user)}
                                                    className="p-2 hover:text-red-500 transition-colors"
                                                >
                                                    <FaTrash />
                                                </button>
                                            </div>
                                        </td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>

                    {filteredUsers.length === 0 && !isLoading && (
                        <div className="text-center py-8 text-[var(--text-secondary)]">
                            {searchTerm ? 'No users found matching your search' : 'No users available'}
                        </div>
                    )}
                </>
            )}

            <div className="flex justify-between items-center mt-4">
                <div className="text-sm text-[var(--text-secondary)]">
                    Page {currentPage} of {totalPages}
                </div>
                <div className="flex gap-2">
                    <button
                        onClick={() => handlePageChange(currentPage - 1)}
                        disabled={currentPage === 1}
                        className="p-2 rounded-lg border border-[var(--border-color)] 
                                 disabled:opacity-50 disabled:cursor-not-allowed
                                 hover:border-[var(--accent-purple)]"
                    >
                        <FaChevronLeft />
                    </button>
                    <button
                        onClick={() => handlePageChange(currentPage + 1)}
                        disabled={currentPage === totalPages}
                        className="p-2 rounded-lg border border-[var(--border-color)]
                                 disabled:opacity-50 disabled:cursor-not-allowed
                                 hover:border-[var(--accent-purple)]"
                    >
                        <FaChevronRight />
                    </button>
                </div>
            </div>

            {selectedUser && (
                <UserEditModal
                    user={selectedUser}
                    onClose={() => setSelectedUser(null)}
                    onUpdate={() => {
                        fetchUsers(currentPage);
                        setSelectedUser(null);
                    }}
                />
            )}

            <ConfirmDialog
                isOpen={!!userToDelete}
                title="Delete User"
                message={`Are you sure you want to delete ${userToDelete?.email}? This action cannot be undone.`}
                confirmText="Delete"
                cancelText="Cancel"
                onConfirm={() => userToDelete && handleDeleteUser(userToDelete)}
                onCancel={() => setUserToDelete(null)}
                isDestructive={true}
            />
        </div>
    );
}; 