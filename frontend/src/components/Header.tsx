'use client';

import { useAuth } from '@/contexts/AuthContext';
import { useCredits } from '@/contexts/CreditsContext';
import Link from 'next/link';
import Image from 'next/image';
import { FaSignOutAlt, FaCoins, FaCrown, FaGem, FaDiceOne, FaAward } from 'react-icons/fa';
import { useState, useEffect } from 'react';

interface HeaderProps {
  onProfileClick?: () => void;
}

export const Header = ({ onProfileClick }: HeaderProps) => {
  const { user, logout } = useAuth();
  const { credits, fetchCredits } = useCredits();
  const [membershipTier, setMembershipTier] = useState('free');

  const getMembershipIcon = (tier?: string) => {
    switch (tier?.toLowerCase()) {
      case 'free':
        return <FaDiceOne className="text-gray-400" />;
      case 'basic':
        return <FaCrown className="text-blue-500" />;
      case 'pro':
        return <FaGem className="text-purple-500" />;
      case 'enterprise':
        return <FaAward className="text-yellow-500" />;
      default:
        return <FaDiceOne className="text-gray-400" />;
    }
  };

  useEffect(() => {
    const fetchMembership = async () => {
      if (user) {
        try {
          const token = localStorage.getItem('token');
          const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/api/users/me`, {
            headers: {
              'Authorization': `Bearer ${token}`
            }
          });
          const data = await response.json();
          setMembershipTier(data.membership.tier);
        } catch (error) {
          console.error('Error fetching membership:', error);
        }
      }
    };

    fetchMembership();
    if (user) {
      fetchCredits();
    }
  }, [user, fetchCredits]);

  return (
    <header className="bg-[var(--card-background)] border-b border-[var(--border-color)]">
      <div className="max-w-7xl mx-auto px-4 h-16 flex items-center justify-between">
        <Link href="/" className="flex items-center">
          <Image
            src="https://megaton-storage.s3.us-east-1.amazonaws.com/logo-roto-128px-transparent.webp"
            alt="Megaton Roto Logo"
            width={48}
            height={48}
            className="h-12 w-auto"
          />
        </Link>

        {user && (
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-2 px-3 py-1.5 rounded-full border border-[var(--border-color)] bg-[var(--background)] tooltip-wrapper">
              {getMembershipIcon(membershipTier)}
              <span className="capitalize text-sm font-medium">{membershipTier}</span>
              <span className="tooltip">Current Membership</span>
            </div>

            <div className="flex items-center gap-2 px-3 py-1.5 rounded-full border border-[var(--border-color)] bg-[var(--background)]">
              <FaCoins className="text-yellow-500" />
              <span className="text-sm font-medium">{credits}</span>
            </div>

            <button
              onClick={onProfileClick}
              className="flex items-center gap-2 hover:opacity-80 transition-opacity tooltip-wrapper"
            >
              <Image
                src="https://megaton-storage.s3.us-east-1.amazonaws.com/profile-pic.webp"
                alt="Profile"
                width={32}
                height={32}
                className="rounded-full"
              />
              <span className="tooltip">View Profile</span>
            </button>

            <button
              onClick={logout}
              className="flex items-center gap-2 px-3 py-1.5 rounded-full border border-[var(--border-color)] 
                       text-[var(--text-secondary)] hover:border-[var(--accent-purple)] 
                       hover:text-[var(--accent-purple)] transition-colors tooltip-wrapper"
            >
              <FaSignOutAlt className="text-sm" />
              <span className="tooltip">Sign Out</span>
            </button>
          </div>
        )}
      </div>
    </header>
  );
};