'use client';

import { createContext, useContext, useState, useEffect, useRef } from 'react';
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

// Create a centralized API client
const apiClient = axios.create({
  baseURL: process.env.NEXT_PUBLIC_API_URL
});

// Add this function to decode and check token expiration
const getTokenExpiration = (token: string) => {
  try {
    // JWT tokens are base64 encoded with 3 parts separated by dots
    const base64Url = token.split('.')[1];
    const base64 = base64Url.replace(/-/g, '+').replace(/_/g, '/');
    const payload = JSON.parse(window.atob(base64));
    
    // exp is in seconds, convert to milliseconds for JS Date
    if (payload.exp) {
      const expiryDate = new Date(payload.exp * 1000);
      const now = new Date();
      console.log('Token expires at:', expiryDate.toISOString());
      console.log('Current time:', now.toISOString());
      console.log('Time until expiry:', (expiryDate.getTime() - now.getTime()) / 1000, 'seconds');
      return payload.exp;
    }
    return null;
  } catch (error) {
    console.error('Error decoding token:', error);
    return null;
  }
};

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [user, setUser] = useState<User | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [refreshingToken, setRefreshingToken] = useState(false);
  const refreshPromise = useRef<Promise<string> | null>(null);

  const fetchUserProfile = async (token: string) => {
    try {
      setIsLoading(true);  // Set loading true while fetching
      const response = await apiClient.get('/api/users/me', {
        headers: {
          'Authorization': `Bearer ${token}`
        }
      });

      // Update stored user data with complete profile
      localStorage.setItem('user', JSON.stringify(response.data));
      setUser(response.data);
      console.log('Fetched user profile:', response.data);
    } catch (error) {
      console.error('Error fetching user profile:', error);
    } finally {
      setIsLoading(false);
    }
  };

  // Function to refresh token with mutex pattern to prevent race conditions
  const refreshToken = async () => {
    console.log('Attempting to refresh token...');
    // If there's already a refresh in progress, return that promise
    if (refreshPromise.current) {
      return refreshPromise.current;
    }

    try {
      setRefreshingToken(true);
      const storedRefreshToken = localStorage.getItem('refreshToken');

      if (!storedRefreshToken) {
        throw new Error('No refresh token found');
      }

      // Create a new promise for this refresh operation
      refreshPromise.current = (async () => {
        try {
          const response = await apiClient.post('/api/auth/refresh', {
            refresh_token: storedRefreshToken
          });

          const data = response.data;

          // Update tokens in storage
          localStorage.setItem('token', data.access_token);
          localStorage.setItem('refreshToken', data.refresh_token);

          // Update axios default headers
          apiClient.defaults.headers.common['Authorization'] = `Bearer ${data.access_token}`;

          // After successful refresh
          console.log('Token refreshed successfully');
          const newToken = localStorage.getItem('token');
          if (newToken) {
            getTokenExpiration(newToken);
          }

          return data.access_token;
        } catch (error) {
          console.error('Error refreshing token:', error);
          // Clear auth data on refresh failure
          logout();
          throw error;
        }
      })();

      return await refreshPromise.current;
    } finally {
      setRefreshingToken(false);
      refreshPromise.current = null;
    }
  };

  // Initialize authentication state
  useEffect(() => {
    const initializeAuth = async () => {
      const token = localStorage.getItem('token');
      
      if (token) {
        console.log('Initial token check:');
        // Check if token is expired
        const isExpired = checkIfTokenExpired(token);
        
        if (isExpired) {
          console.log('Token is expired, refreshing immediately...');
          // Try to refresh the token before proceeding
          try {
            await refreshToken();
            // After refresh, proceed with fetching user profile
            await fetchUserProfile(token);
          } catch (error) {
            console.error('Failed to refresh expired token:', error);
            // Clear auth state since we couldn't refresh
            setUser(null);
            localStorage.removeItem('token');
          }
        } else {
          // Token is still valid, proceed normally
          await fetchUserProfile(token);
        }
      } else {
        setIsLoading(false);
      }
    };

    initializeAuth();
  }, []);

  // Helper function to check if token is expired
  const checkIfTokenExpired = (token: string): boolean => {
    try {
      const base64Url = token.split('.')[1];
      const base64 = base64Url.replace(/-/g, '+').replace(/_/g, '/');
      const payload = JSON.parse(window.atob(base64));
      
      if (payload.exp) {
        const expiryTime = payload.exp * 1000; // Convert to milliseconds
        const currentTime = Date.now();
        const timeUntilExpiry = expiryTime - currentTime;
        
        console.log('Token expires at:', new Date(expiryTime).toISOString());
        console.log('Current time:', new Date(currentTime).toISOString());
        console.log('Time until expiry:', timeUntilExpiry / 1000, 'seconds');
        
        return timeUntilExpiry <= 0;
      }
      return false;
    } catch (error) {
      console.error('Error checking token expiration:', error);
      return true; // Assume expired if we can't parse it
    }
  };

  // Setup axios interceptor to handle 401 responses
  useEffect(() => {
    const interceptor = apiClient.interceptors.response.use(
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
            return apiClient(originalRequest);
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
      apiClient.interceptors.response.eject(interceptor);
    };
  }, [refreshingToken]);

  const login = async (token: string, refreshToken: string, initialUser: User) => {
    localStorage.setItem('token', token);
    localStorage.setItem('refreshToken', refreshToken);
    localStorage.setItem('user', JSON.stringify(initialUser));
    
    // Set authorization header for all future requests
    apiClient.defaults.headers.common['Authorization'] = `Bearer ${token}`;
    
    setUser(initialUser);
    
    // Fetch complete user profile after login
    await fetchUserProfile(token);
  };

  const logout = () => {
    localStorage.removeItem('token');
    localStorage.removeItem('refreshToken');
    localStorage.removeItem('user');
    delete apiClient.defaults.headers.common['Authorization'];
    setUser(null);
  };

  useEffect(() => {
    // Check token expiration every 60 seconds and refresh if needed
    const tokenCheckInterval = setInterval(() => {
      const token = localStorage.getItem('token');
      if (!token) return;
      
      try {
        const base64Url = token.split('.')[1];
        const base64 = base64Url.replace(/-/g, '+').replace(/_/g, '/');
        const payload = JSON.parse(window.atob(base64));
        
        if (payload.exp) {
          const expiryTime = payload.exp * 1000; // Convert to milliseconds
          const currentTime = Date.now();
          const timeUntilExpiry = expiryTime - currentTime;
          
          console.log(`Token expires in: ${timeUntilExpiry / 1000} seconds`);
          
          // If token will expire in less than 5 minutes (300 seconds), refresh it
          if (timeUntilExpiry > 0 && timeUntilExpiry < 300000) {
            console.log('Token expiring soon, refreshing preemptively');
            refreshToken().catch(err => {
              console.error('Preemptive token refresh failed:', err);
            });
          }
        }
      } catch (error) {
        console.error('Error checking token expiration:', error);
      }
    }, 60000); // Check every 60 seconds
    
    return () => {
      clearInterval(tokenCheckInterval);
    };
  }, []);

  return (
    <AuthContext.Provider value={{ user, isLoading, login, logout }}>
      {children}
    </AuthContext.Provider>
  );
}

// Export apiClient for use throughout the application
export const api = apiClient;

export function useAuth() {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
}