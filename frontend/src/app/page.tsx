'use client';

import { useAuth } from '@/contexts/AuthContext';
import { Header } from '@/components/Header';
import { AuthModal } from '@/components/auth/AuthModal';
import { VideoUpload } from '@/components/upload/Video';
import { VideoGallery } from '@/components/video/VideoGallery';
import ProfileDashboard from '@/components/profile/ProfileDashboard';
import { useState, useEffect, Suspense } from 'react';
import { Video } from '@/lib/types/video';
import { FaArrowLeft } from 'react-icons/fa';
import { useSearchParams } from 'next/navigation';
import { useStatus } from '@/contexts/StatusContext';

// Create a separate component for the part that uses useSearchParams
function PaymentStatusHandler() {
  const searchParams = useSearchParams();
  const { setStatus } = useStatus();

  useEffect(() => {
    const status = searchParams.get('status');
    const amount = searchParams.get('amount');

    if (status === 'success' && amount) {
      setStatus(`Successfully added ${amount} credits!`, 'success');
      // Clean up URL parameters
      window.history.replaceState({}, '', window.location.pathname);
    } else if (status === 'error') {
      setStatus('Error processing credit purchase', 'error');
      window.history.replaceState({}, '', window.location.pathname);
    }
  }, [searchParams, setStatus]);

  return null;
}

export default function Home() {
  const { user, isLoading } = useAuth();
  const [videos, setVideos] = useState<Video[]>([]);
  const [activeVideoId, setActiveVideoId] = useState<string | null>(null);
  const [selectedVideo, setSelectedVideo] = useState<Video | null>(null);
  const [showProfile, setShowProfile] = useState(false);
  const { setStatus } = useStatus();

  const fetchVideos = async () => {
    try {
      // Get token from localStorage
      const token = localStorage.getItem('token');

      if (!token) {
        console.log('No token found, user might need to login');
        return;
      }

      const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/api/videos`, {
        credentials: 'include',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`
        }
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      console.log('Raw API response:', data);

      if (!data.videos) {
        console.error('No videos array in response:', data);
        setVideos([]);
        return;
      }

      // Sort videos by date in descending order (newest first)
      const sortedVideos = data.videos.sort((a, b) => {
        return new Date(b.createdAt).getTime() - new Date(a.createdAt).getTime();
      });

      setVideos(sortedVideos);
      console.log('Fetched videos:', data.videos);
    } catch (error) {
      console.error('Error fetching videos:', error);
      setVideos([]);
    }
  };

  const fetchAndReturn = async () => {
    try {
      // Get token from localStorage
      const token = localStorage.getItem('token');

      if (!token) {
        console.log('No token found, user might need to login');
        return;
      }

      const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/api/videos`, {
        credentials: 'include',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`
        }
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      console.log('Raw API response:', data);

      if (!data.videos) {
        console.error('No videos array in response:', data);
        setVideos([]);
        return;
      }

      // Sort videos by date in descending order (newest first)
      const sortedVideos = data.videos.sort((a, b) => {
        return new Date(b.createdAt).getTime() - new Date(a.createdAt).getTime();
      });

      setVideos(sortedVideos);
      console.log('Fetched videos:', data.videos);
      return sortedVideos;
    } catch (error) {
      console.error('Error fetching videos:', error);
      setVideos([]);
    }
  };

  useEffect(() => {
    const token = localStorage.getItem('token');
    if (user && token) {
      fetchVideos();
    }
  }, [user]);

  const handleVideoSelect = async (videoId: string) => {
    console.log('Video selected:', videoId);
    const sortedVideos = await fetchAndReturn();

    const video = sortedVideos.find(v => v.id === videoId);
    if (video) {
      console.log('Setting selected video:', video); // Already has correct S3 URLs
      setActiveVideoId(videoId);
      setSelectedVideo(video);
    }
  };

  const handleDeleteVideo = async (videoId: string) => {
    try {
      const token = localStorage.getItem('token');
      if (!token) return;

      const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/api/videos/${videoId}`, {
        method: 'DELETE',
        headers: {
          'Authorization': `Bearer ${token}`
        },
        credentials: 'include'
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      // Refresh videos list
      fetchVideos();
    } catch (error) {
      console.error('Error deleting video:', error);
    }
  };

  if (isLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-blue-500"></div>
      </div>
    );
  }

  if (!user) {
    return (
      <div className="min-h-screen bg-[var(--background)]">
        <Header />
        <main className="max-w-md mx-auto mt-20 p-6 bg-[var(--card-background)] rounded-lg shadow-lg">
          <AuthModal isOpen={true} onClose={() => { }} embedded />
        </main>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-[var(--background)]">
      <Header onProfileClick={() => setShowProfile(true)} />
      <Suspense fallback={null}>
        <PaymentStatusHandler />
      </Suspense>
      <main className="max-w-7xl mx-auto py-6 px-4">
        {showProfile ? (
          <>
            <div className="mb-6">
              <button
                onClick={() => setShowProfile(false)}
                className="flex items-center gap-2 px-4 py-2 text-[var(--text-secondary)] hover:text-[var(--accent-purple)] transition-colors"
              >
                <FaArrowLeft className="text-sm" />
                <span>Back to Masks</span>
              </button>
            </div>
            <ProfileDashboard />
          </>
        ) : (
          <>
            <VideoUpload
              onUploadSuccess={fetchVideos}
              fetchVideos={fetchVideos}
              initialVideo={selectedVideo}
              setInitialVideo={setSelectedVideo}
              key={selectedVideo?.id}
            />
            <div className="mt-8">
              <VideoGallery
                videos={videos}
                onSelectVideo={handleVideoSelect}
                onDeleteVideo={handleDeleteVideo}
                activeVideoId={activeVideoId}
                onUploadSuccess={fetchVideos}
                fetchVideos={fetchVideos}
              />
            </div>
          </>
        )}
      </main>
    </div>
  );
}
