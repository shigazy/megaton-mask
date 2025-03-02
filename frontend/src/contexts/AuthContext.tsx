'use client';

import { createContext, useContext, useState, useEffect } from 'react';
import axios from 'axios';

interface User {
  id: string;
  email: string;
  super_user?: boolean;
  membership?: {
    tier: string;
    status: string;
  };
  storage_used?: {
    total_gb: number;
  };
  user_credits?: number;
}

interface AuthContextType {
  user: User | null;
  isLoading: boolean;
  login: (token: string, user: User) => void;
  logout: () => void;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [user, setUser] = useState<User | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  const fetchUserProfile = async (token: string) => {
    try {
      setIsLoading(true);  // Set loading true while fetching
      const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/api/users/me`, {
        headers: {
          'Authorization': `Bearer ${token}`
        }
      });

      if (!response.ok) throw new Error('Failed to fetch user profile');

      const userData = await response.json();
      console.log('Fetched user profile:', userData);

      // Update stored user data with complete profile
      localStorage.setItem('user', JSON.stringify(userData));
      setUser(userData);
    } catch (error) {
      console.error('Error fetching user profile:', error);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    const initializeAuth = async () => {
      const token = localStorage.getItem('token');
      const storedUser = localStorage.getItem('user');

      if (token) {
        // Set initial user data from localStorage
        if (storedUser) {
          setUser(JSON.parse(storedUser));
        }

        // Set up axios default header
        axios.defaults.headers.common['Authorization'] = `Bearer ${token}`;

        // Fetch complete profile
        await fetchUserProfile(token);
      } else {
        setIsLoading(false);
      }
    };

    initializeAuth();
  }, []);

  const login = async (token: string, initialUser: User) => {
    localStorage.setItem('token', token);
    localStorage.setItem('user', JSON.stringify(initialUser));
    axios.defaults.headers.common['Authorization'] = `Bearer ${token}`;
    setUser(initialUser);
    // Fetch complete user profile after login
    await fetchUserProfile(token);
  };

  const logout = () => {
    localStorage.removeItem('token');
    localStorage.removeItem('user');
    delete axios.defaults.headers.common['Authorization'];
    setUser(null);
  };

  return (
    <AuthContext.Provider value={{ user, isLoading, login, logout }}>
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
}