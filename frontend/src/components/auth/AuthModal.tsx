'use client';

import { useState } from 'react';
import axios from 'axios';
import { FaSignInAlt, FaUserPlus } from 'react-icons/fa';
import Image from 'next/image';
import { useAuth } from '@/contexts/AuthContext';

interface AuthModalProps {
  isOpen: boolean;
  onClose: () => void;
  embedded?: boolean;
}

export const AuthModal = ({ isOpen, onClose, embedded = false }: AuthModalProps) => {
  const { login } = useAuth();
  const [isLogin, setIsLogin] = useState(true);
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [error, setError] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [needsConfirmation, setNeedsConfirmation] = useState(false);
  const [showConfirmationPrompt, setShowConfirmationPrompt] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');
    setIsLoading(true);
    console.log('Attempting authentication...');

    try {
      if (!isLogin && password !== confirmPassword) {
        setError('Passwords do not match');
        return;
      }

      const endpoint = isLogin ? '/api/auth/login' : '/api/auth/register';
      const data = isLogin
        ? new URLSearchParams({ username: email, password })
        : { email, password };

      console.log('Making request to:', endpoint);

      const response = await axios.post(
        `${process.env.NEXT_PUBLIC_API_URL}${endpoint}`,
        data,
        {
          headers: isLogin
            ? { 'Content-Type': 'application/x-www-form-urlencoded' }
            : { 'Content-Type': 'application/json' }
        }
      );

      console.log('Auth response:', response.data);

      if (response.data.needs_confirmation || response.data.email_not_confirmed) {
        setNeedsConfirmation(true);
        setShowConfirmationPrompt(true);
        setError('Please confirm your email to continue');
        return;
      }

      if (isLogin) {
        // Use the auth context login function to handle token storage and state
        await login(
          response.data.access_token,
          response.data.refresh_token,
          response.data.user
        );
        console.log('Logged in successfully');
      } else {
        // For registration, we don't get tokens yet as email confirmation is required
        console.log('Registered successfully, confirmation required');
      }

      console.log('Closing modal...');
      onClose();

      // Only reload if we were logging in and succeeded
      if (isLogin) {
        window.location.reload();
      }

    } catch (err: any) {
      console.error('Auth error:', err);
      if (err.response?.data?.detail === 'Email not confirmed') {
        setNeedsConfirmation(true);
        setShowConfirmationPrompt(true);
        setError('Please confirm your email to continue');
      } else {
        setError(err.response?.data?.detail || 'An error occurred');
      }
    } finally {
      setIsLoading(false);
    }
  };

  const handleResendConfirmation = async () => {
    try {
      await axios.post(
        `${process.env.NEXT_PUBLIC_API_URL}/api/auth/resend-confirmation`,
        { email },
        { headers: { 'Content-Type': 'application/json' } }
      );
      setError('Confirmation email has been resent');
    } catch (err: any) {
      setError('Failed to resend confirmation email');
    }
  };

  const inputClasses = `
    w-full px-4 py-3 rounded-lg
    bg-[var(--background-darker)]
    border border-[var(--border-color)]
    text-[var(--text-color)]
    placeholder-[var(--text-color-secondary)]
    focus:outline-none focus:border-[var(--accent-purple)]
    transition-colors duration-200
  `;

  const buttonClasses = `
    w-full py-3 px-4
    bg-[var(--accent-purple)]
    hover:bg-[var(--accent-purple-hover)]
    text-white font-medium
    rounded-lg
    transition-all duration-200
    disabled:opacity-50 disabled:cursor-not-allowed
    focus:outline-none focus:ring-2 focus:ring-[var(--accent-purple)] focus:ring-opacity-50
  `;

  const content = (
    <div className={`
      ${embedded ? "" : "bg-[var(--background-darker)] rounded-xl p-8 max-w-md w-full"}
      shadow-2xl border border-[var(--border-color)]
    `}>
      <div className="flex flex-col items-center mb-8">
        <div className="w-40 h-40 relative mb-6">
          <Image
            src="https://megaton-storage.s3.us-east-1.amazonaws.com/logo-roto-128px-transparent.webp"
            alt="Megaton Roto"
            fill
            className="object-contain"
            priority
            sizes="(max-width: 768px) 100vw, (max-width: 1200px) 50vw, 33vw"
          />
        </div>
        <div className="inline-flex items-center gap-3 px-6 py-3 rounded-full bg-[var(--background-darker)] border border-[var(--border-color)]">
          {isLogin ? (
            <FaSignInAlt className="text-[var(--accent-purple)] text-xl" />
          ) : (
            <FaUserPlus className="text-[var(--accent-purple)] text-xl" />
          )}
          <span className="text-base font-medium text-[var(--accent-purple)] tracking-wide">
            {isLogin ? 'LOGIN' : 'REGISTER'}
          </span>
        </div>
      </div>

      {error && (
        <div className="mb-6 p-4 rounded-lg bg-red-500/10 border border-red-500/20">
          <p className="text-red-500 text-sm">
            {error}
            {needsConfirmation && (
              <button
                onClick={handleResendConfirmation}
                className="ml-2 text-[var(--accent-purple)] hover:underline focus:outline-none"
              >
                Resend confirmation email
              </button>
            )}
          </p>
        </div>
      )}

      <form onSubmit={handleSubmit} className="space-y-5">
        <div>
          <label className="block text-sm font-medium text-[var(--text-color-secondary)] mb-2">
            Email Address
          </label>
          <input
            type="email"
            placeholder="you@example.com"
            className={inputClasses}
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            required
          />
        </div>

        <div>
          <label className="block text-sm font-medium text-[var(--text-color-secondary)] mb-2">
            Password
          </label>
          <input
            type="password"
            placeholder="••••••••"
            className={inputClasses}
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            required
          />
        </div>

        {!isLogin && (
          <div>
            <label className="block text-sm font-medium text-[var(--text-color-secondary)] mb-2">
              Confirm Password
            </label>
            <input
              type="password"
              placeholder="••••••••"
              className={inputClasses}
              value={confirmPassword}
              onChange={(e) => setConfirmPassword(e.target.value)}
              required
            />
          </div>
        )}

        <button
          type="submit"
          disabled={isLoading}
          className={buttonClasses}
        >
          {isLoading ? (
            <span className="flex items-center justify-center">
              <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
              </svg>
              Processing...
            </span>
          ) : (
            isLogin ? 'Sign In' : 'Create Account'
          )}
        </button>

        <p className="text-center text-sm text-[var(--text-color-secondary)] mt-6">
          {isLogin ? "Don't have an account? " : "Already have an account? "}
          <button
            type="button"
            className="text-[var(--accent-purple)] hover:underline focus:outline-none"
            onClick={() => {
              setIsLogin(!isLogin);
              setError('');
              setNeedsConfirmation(false);
            }}
          >
            {isLogin ? 'Sign up' : 'Sign in'}
          </button>
        </p>
      </form>
    </div>
  );

  const confirmationPrompt = (
    <div className="fixed inset-0 bg-black/50 backdrop-blur-sm flex items-center justify-center z-50 p-4">
      <div className="bg-[var(--background-darker)] rounded-xl p-8 max-w-md w-full shadow-2xl border border-[var(--border-color)]">
        <h2 className="text-xl font-semibold text-[var(--text-color)] mb-4">Email Confirmation Required</h2>
        <p className="text-[var(--text-color-secondary)] mb-6">
          Your email address needs to be confirmed before you can continue. Please check your inbox for the confirmation link.
        </p>
        <div className="space-y-4">
          <button
            onClick={handleResendConfirmation}
            className={buttonClasses}
          >
            Resend Confirmation Email
          </button>
          <button
            onClick={() => setShowConfirmationPrompt(false)}
            className="w-full py-3 px-4 bg-transparent border border-[var(--border-color)] text-[var(--text-color)] rounded-lg hover:bg-[var(--background-lighter)]"
          >
            Close
          </button>
        </div>
      </div>
    </div>
  );

  if (!isOpen) return null;

  if (showConfirmationPrompt) {
    return confirmationPrompt;
  }

  if (embedded) {
    return content;
  }

  return (
    <div className="fixed inset-0 bg-black/50 backdrop-blur-sm flex items-center justify-center z-50 p-4">
      <div className="animate-in fade-in duration-300">
        {content}
      </div>
    </div>
  );
};