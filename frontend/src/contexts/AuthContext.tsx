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
  login: (token: string, refreshToken: string, user: User) => void;
  logout: () => void;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [user, setUser] = useState<User | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [refreshingToken, setRefreshingToken] = useState(false);

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

  // Function to refresh token
  const refreshToken = async () => {
    try {
      setRefreshingToken(true);
      const storedRefreshToken = localStorage.getItem('refreshToken');

      if (!storedRefreshToken) {
        throw new Error('No refresh token found');
      }

      const response = await window.fetch(`${process.env.NEXT_PUBLIC_API_URL}/api/auth/refresh`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ refresh_token: storedRefreshToken }),
      });

      if (!response.ok) {
        throw new Error('Failed to refresh token');
      }

      const data = await response.json();

      // Update tokens in storage
      localStorage.setItem('token', data.access_token);
      localStorage.setItem('refreshToken', data.refresh_token);

      // Update axios header
      axios.defaults.headers.common['Authorization'] = `Bearer ${data.access_token}`;

      return data.access_token;
    } catch (error) {
      console.error('Error refreshing token:', error);
      // Clear auth data on refresh failure
      logout();
      throw error;
    } finally {
      setRefreshingToken(false);
    }
  };

  // Setup fetch interceptor
  useEffect(() => {
    // Store the original fetch function
    const originalFetch = window.fetch;

    // Override the global fetch function
    window.fetch = async (input: RequestInfo | URL, init?: RequestInit) => {
      // Create a new init object to avoid modifying the original
      const modifiedInit = init ? { ...init } : {};

      // Add authorization header if a token exists
      const token = localStorage.getItem('token');
      if (token) {
        modifiedInit.headers = {
          ...modifiedInit.headers,
          'Authorization': `Bearer ${token}`
        };
      }

      // Make the initial fetch request
      let response = await originalFetch(input, modifiedInit);

      // If we get a 401 and we're not currently refreshing tokens and we have a refresh token
      if (response.status === 401 && !refreshingToken && localStorage.getItem('refreshToken')) {
        try {
          // Try to refresh the token
          const newToken = await refreshToken();

          // Create new headers with the new token
          const newInit = {
            ...modifiedInit,
            headers: {
              ...modifiedInit.headers,
              'Authorization': `Bearer ${newToken}`
            }
          };

          // Retry the request with the new token
          response = await originalFetch(input, newInit);
        } catch (error) {
          console.error('Error refreshing token for fetch:', error);
          // Let the 401 response pass through if refresh failed
        }
      }

      return response;
    };

    // Cleanup function to restore original fetch
    return () => {
      window.fetch = originalFetch;
    };
  }, [refreshingToken]);

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

  // Setup axios interceptor to handle 401 responses
  useEffect(() => {
    const interceptor = axios.interceptors.response.use(
      (response) => response,
      async (error) => {
        const originalRequest = error.config;

        // If error is 401 and we haven't already tried to refresh
        if (error.response?.status === 401 && !originalRequest._retry && !refreshingToken) {
          originalRequest._retry = true;

          try {
            // Try to refresh the token
            const newToken = await refreshToken();

            // Update the original request with new token
            originalRequest.headers['Authorization'] = `Bearer ${newToken}`;

            // Retry the original request
            return axios(originalRequest);
          } catch (refreshError) {
            // If refresh fails, pass through to the next error handler
            return Promise.reject(refreshError);
          }
        }

        // For other errors, just pass them through
        return Promise.reject(error);
      }
    );

    // Clean up interceptor on unmount
    return () => {
      axios.interceptors.response.eject(interceptor);
    };
  }, [refreshingToken]);

  const login = async (token: string, refreshToken: string, initialUser: User) => {
    localStorage.setItem('token', token);
    localStorage.setItem('refreshToken', refreshToken);
    localStorage.setItem('user', JSON.stringify(initialUser));
    axios.defaults.headers.common['Authorization'] = `Bearer ${token}`;
    setUser(initialUser);
    // Fetch complete user profile after login
    await fetchUserProfile(token);
  };

  const logout = () => {
    localStorage.removeItem('token');
    localStorage.removeItem('refreshToken');
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