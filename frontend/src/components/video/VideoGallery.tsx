'use client';

import { useState, useEffect, useCallback, useRef } from 'react';
import { FaTrash, FaPlay, FaVideo, FaCloudUploadAlt, FaFilm, FaDownload, FaSync, FaMask, FaCheckSquare, FaSquare, FaCheck, FaThLarge, FaList, FaChevronLeft, FaChevronRight, FaPlus } from 'react-icons/fa';
import { Video } from '@/lib/types/video';
import { UPLOAD_LIMITS } from '@/lib/constants';  // Update this import
import { format } from 'date-fns';
import { useStatus } from '@/contexts/StatusContext';
import axios from 'axios';

type ViewMode = 'grid' | 'list';

// Add type for sort options
type SortOption = 'newest' | 'oldest' | 'name-asc' | 'name-desc';

interface VideoGalleryProps {
  videos: Video[];
  onSelectVideo: (videoId: string, videoUrl?: string) => void;
  onDeleteVideo: (videoId: string) => void;
  activeVideoId?: string;
  onUploadSuccess?: () => void;
}

export const VideoGallery = ({
  videos = [],
  onSelectVideo,
  onDeleteVideo,
  activeVideoId,
  onUploadSuccess
}: VideoGalleryProps) => {
  const [refreshedUrls, setRefreshedUrls] = useState<Record<string, string>>({});
  const [isUploading, setIsUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState<number>(0);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const { setStatus } = useStatus();
  const [selectedVideos, setSelectedVideos] = useState<Set<string>>(new Set());
  const [viewMode, setViewMode] = useState<ViewMode>('list');
  const [currentPage, setCurrentPage] = useState(1);
  const [isHovered, setIsHovered] = useState(false);
  const itemsPerPage = viewMode === 'grid' ? 9 : 10;
  
  // Add state for delete confirmation
  const [showDeleteConfirm, setShowDeleteConfirm] = useState(false);
  const [videoToDelete, setVideoToDelete] = useState<string | null>(null);

  // Add state for sorting
  const [sortBy, setSortBy] = useState<SortOption>('newest');

  // Function to sort videos based on selected option
  const getSortedVideos = useCallback((videoList: Video[]) => {
    const sortedVideos = [...videoList];
    
    switch (sortBy) {
      case 'newest':
        return sortedVideos.sort((a, b) => 
          new Date(b.createdAt).getTime() - new Date(a.createdAt).getTime()
        );
      case 'oldest':
        return sortedVideos.sort((a, b) => 
          new Date(a.createdAt).getTime() - new Date(b.createdAt).getTime()
        );
      case 'name-asc':
        return sortedVideos.sort((a, b) => 
          a.title.localeCompare(b.title)
        );
      case 'name-desc':
        return sortedVideos.sort((a, b) => 
          b.title.localeCompare(a.title)
        );
      default:
        return sortedVideos;
    }
  }, [sortBy]);

  // Update pagination to use sorted videos
  const sortedVideos = getSortedVideos(videos);
  const totalPages = Math.ceil(sortedVideos.length / itemsPerPage);
  const paginatedVideos = sortedVideos.slice(
    (currentPage - 1) * itemsPerPage,
    currentPage * itemsPerPage
  );

  // Reuse file validation from VideoUpload
  const validateFile = (file: File): string | null => {
    if (file.size > UPLOAD_LIMITS.MAX_FILE_SIZE) {
      return `File size must be less than ${UPLOAD_LIMITS.MAX_FILE_SIZE / 1024 / 1024}MB`;
    }
    if (file.type !== 'video/mp4') {
      return 'Only MP4 videos are allowed';
    }
    return null;
  };

  const handleFileSelect = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    await handleUpload(file);
  };

  const handleDrop = async (event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    event.stopPropagation();

    const file = event.dataTransfer.files[0];
    if (!file) return;

    const error = validateFile(file);
    if (error) {
      alert(error);
      return;
    }

    await handleUpload(file);
  };

  const handleUpload = async (file: File) => {
    try {
      // Validate file size and type
      if (file.size > UPLOAD_LIMITS.MAX_FILE_SIZE) {
        setStatus(`${file.name} is too large. Maximum size is ${UPLOAD_LIMITS.MAX_FILE_SIZE / (1024 * 1024)}MB.`, 'error');
        return;
      }
      if (!file.type.startsWith('video/')) {
        setStatus(`${file.name} is not a video file.`, 'error');
        return;
      }

      setStatus(`Uploading ${file.name}...`, 'processing');
      setIsUploading(true);
      setUploadProgress(0);

      const token = localStorage.getItem('token');
      if (!token) throw new Error('No token found');

      const formData = new FormData();
      formData.append('file', file);
      formData.append('annotations', JSON.stringify({}));  // Required by backend

      const response = await axios.post(
        `${process.env.NEXT_PUBLIC_API_URL}/api/videos/upload`,
        formData,
        {
          headers: {
            'Authorization': `Bearer ${token}`,
            // Let axios set the correct Content-Type with boundary
          },
          onUploadProgress: (progressEvent) => {
            const percentCompleted = Math.round(
              (progressEvent.loaded * 100) / (progressEvent.total || 1)
            );
            setUploadProgress(percentCompleted);
            setStatus(`Uploading: ${percentCompleted}%`, 'loading');
            console.log("Upload Progress:", percentCompleted, "%");
          },
        }
      );

      console.log('Upload Response:', response.data);

      // Handle successful upload
      if (response.data) {
        const { id, videoUrl } = response.data;
        // Call onSelectVideo to set the active video
        if (onSelectVideo && id) {
          onSelectVideo(id, videoUrl);
        }
        onUploadSuccess?.();
        setStatus(`Successfully uploaded ${file.name}`, 'success');
      }

    } catch (error) {
      console.error('Upload error:', error);
      setStatus(`Failed to upload video: ${error instanceof Error ? error.message : 'Unknown error'}`, 'error');
    } finally {
      setIsUploading(false);
      // Reset file input
      if (fileInputRef.current) {
        fileInputRef.current.value = '';
      }
    }
  };

  const refreshVideoUrl = async (videoId: string) => {
    try {
      const token = localStorage.getItem('token');
      if (!token) return;
      console.log('Refreshing video URL for [VideoGallery.tsx]:', videoId);
      const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/api/videos/${videoId}/refresh-url`, {
        headers: {
          'Authorization': `Bearer ${token}`
        },
        credentials: 'include'
      });

      if (!response.ok) throw new Error('Failed to refresh URL');

      const data = await response.json();
      setRefreshedUrls(prev => ({
        ...prev,
        [videoId]: data.videoUrl
      }));
    } catch (error) {
      console.error('Error refreshing video URL:', error);
    }
  };

  // Add this function to handle presigned URL refresh
  const refreshPresignedUrl = async (videoId: string) => {
    try {
      const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/api/videos/${videoId}/refresh-mask-url`, {
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('token')}`
        }
      });
      const data = await response.json();
      return data.maskUrl;
    } catch (error) {
      console.error('Error refreshing presigned URL:', error);
      return null;
    }
  };

  // Refresh URL when video is selected
  useEffect(() => {
    if (activeVideoId) {
      console.log('Refreshing video URL for [VideoGallery.tsx, UseEffect - 2]:', activeVideoId);
      refreshVideoUrl(activeVideoId);
    }
  }, [activeVideoId]);

  const handleVideoSelect = useCallback((video: Video) => {
    console.log('Selecting video:', video.id); // Debug log
    onSelectVideo(video.id, refreshedUrls[video.id] || video.videoUrl);
  }, [onSelectVideo, refreshedUrls]);

  const handleSelectVideo = (e: React.MouseEvent, videoId: string) => {
    e.stopPropagation();
    setSelectedVideos(prev => {
      const newSet = new Set(prev);
      if (newSet.has(videoId)) {
        newSet.delete(videoId);
      } else {
        newSet.add(videoId);
      }
      return newSet;
    });
  };

  const handleBulkDownload = async () => {
    setStatus('Preparing downloads...', 'processing');
    const selectedVideosList = videos.filter(v => selectedVideos.has(v.id) && v.maskUrl);

    try {
      for (const video of selectedVideosList) {
        let url = video.maskUrl;
        try {
          // Fetch the actual file instead of just checking headers
          const response = await fetch(url!);
          if (!response.ok) {
            const newUrl = await refreshPresignedUrl(video.id);
            if (newUrl) {
              url = newUrl;
              const newResponse = await fetch(newUrl);
              if (!newResponse.ok) throw new Error('Failed to refresh URL');
              response = newResponse;
            }
          }

          // Get the blob from the response
          const blob = await response.blob();
          // Create object URL for the blob
          const objectUrl = window.URL.createObjectURL(blob);

          // Create and trigger download
          const a = document.createElement('a');
          a.href = objectUrl;
          a.download = `${video.title}_mask.mp4`; // This forces download instead of opening
          document.body.appendChild(a);
          a.click();

          // Cleanup
          window.URL.revokeObjectURL(objectUrl);
          document.body.removeChild(a);

          await new Promise(resolve => setTimeout(resolve, 1000)); // Increased delay between downloads
        } catch (error) {
          console.error(`Error downloading mask for ${video.title}:`, error);
          setStatus(`Failed to download ${video.title}`, 'error');
        }
      }
      setStatus('Downloads completed', 'success');
    } catch (error) {
      setStatus('Some downloads failed', 'error');
    }
  };

  // Update the handleDelete function to show confirmation instead of directly deleting
  const handleDelete = (videoId: string, e: React.MouseEvent) => {
    e.stopPropagation();
    setVideoToDelete(videoId);
    setShowDeleteConfirm(true);
  };

  // Function to handle confirmed deletion
  const confirmDelete = () => {
    if (videoToDelete) {
      onDeleteVideo(videoToDelete);
      setStatus('Video deleted successfully', 'success');
      
      // If we're deleting the active video, clear it
      if (videoToDelete === activeVideoId) {
        onSelectVideo('');
      }
      
      // Clear the selected videos set if it contains the deleted video
      if (selectedVideos.has(videoToDelete)) {
        const newSelectedVideos = new Set(selectedVideos);
        newSelectedVideos.delete(videoToDelete);
        setSelectedVideos(newSelectedVideos);
      }
      
      // Reset states
      setVideoToDelete(null);
      setShowDeleteConfirm(false);
    }
  };

  // Cancel deletion
  const cancelDelete = () => {
    setVideoToDelete(null);
    setShowDeleteConfirm(false);
  };

  // UI for the sort dropdown
  const renderSortDropdown = () => {
    return (
      <div className="relative">
        <select
          value={sortBy}
          onChange={(e) => setSortBy(e.target.value as SortOption)}
          className="appearance-none bg-[var(--bg-color)] border border-[var(--border-color)] rounded-lg py-2 pl-3 pr-8 text-sm focus:outline-none focus:ring-1 focus:ring-[var(--accent-purple)]"
        >
          <option value="newest">Newest First</option>
          <option value="oldest">Oldest First</option>
          <option value="name-asc">Name (A-Z)</option>
          <option value="name-desc">Name (Z-A)</option>
        </select>
        <div className="pointer-events-none absolute inset-y-0 right-0 flex items-center px-2 text-[var(--text-secondary)]">
          <svg className="fill-current h-4 w-4" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20">
            <path d="M9.293 12.95l.707.707L15.657 8l-1.414-1.414L10 10.828 5.757 6.586 4.343 8z" />
          </svg>
        </div>
      </div>
    );
  };

  const renderGridView = () => (
    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
      {/* Existing grid view code ... */}
      {paginatedVideos.map((video) => (
        <div
          key={video.id}
          className={`card relative group cursor-pointer transition-all
            ${activeVideoId === video.id ? 'ring-2 ring-[var(--accent-purple)]' : ''}
          `}
          onClick={() => handleVideoSelect(video)}
        >
          <div className="relative group">
            <div className="absolute top-2 right-2 z-10 opacity-0 group-hover:opacity-100 transition-opacity flex gap-2">


              {video.maskUrl && (
                <button
                  onClick={async (e) => {
                    e.preventDefault();
                    e.stopPropagation();
                    let url = video.maskUrl;

                    // Try to download, if fails, refresh URL
                    try {
                      const response = await fetch(url, { method: 'HEAD' });
                      if (!response.ok) {
                        const newUrl = await refreshPresignedUrl(video.id);
                        if (newUrl) url = newUrl;
                      }
                      window.open(url, '_blank');
                    } catch (error) {
                      console.error('Error downloading mask:', error);
                    }
                  }}
                  className="p-2 rounded-full bg-[var(--background)] hover:bg-[var(--accent-purple)] 
                             transition-colors shadow-lg tooltip-wrapper"
                  aria-label="Download mask"
                >
                  <FaDownload
                    className="h-5 w-5 text-[var(--text-secondary)] group-hover:text-white"
                  />
                  <span className="tooltip">Download Mask</span>
                </button>
              )}

              <button
                onClick={(e) => {
                  e.preventDefault();
                  e.stopPropagation();
                  handleDelete(video.id, e);
                }}
                className="p-2 rounded-full bg-[var(--background)] hover:bg-[var(--accent-purple)] 
                           transition-colors shadow-lg tooltip-wrapper"
                aria-label="Delete video"
              >
                <svg
                  xmlns="http://www.w3.org/2000/svg"
                  className="h-5 w-5 text-[var(--text-secondary)] group-hover:text-white"
                  fill="none"
                  viewBox="0 0 24 24"
                  stroke="currentColor"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"
                  />
                </svg>
                <span className="tooltip">Delete Video</span>
              </button>
            </div>

            {video.maskUrl && (
              <div
                className={`absolute top-2 left-2 z-20 transition-opacity p-2
                  ${selectedVideos.has(video.id) ? 'opacity-100' : 'opacity-0 group-hover:opacity-100'}`}
                onClick={(e) => handleSelectVideo(e, video.id)}
              >
                <div
                  className={`w-5 h-5 border-2 border-white rounded flex items-center justify-center
                    ${selectedVideos.has(video.id) ? 'bg-transparent' : 'bg-transparent'}`}
                >
                  {selectedVideos.has(video.id) && (
                    <FaCheck className="text-sm text-white" />
                  )}
                </div>
              </div>
            )}

            {video.maskUrl && !selectedVideos.has(video.id) && (
              <div className="absolute top-2 left-2 z-10 p-2 tooltip-wrapper">
                <FaMask
                  className="h-5 w-5 text-white bg-[var(--accent-purple)] rounded-full p-1"
                  style={{ width: '20px', height: '20px' }}
                />
                <span className="tooltip">Mask Generated</span>
              </div>
            )}

            <div className="aspect-video relative bg-gray-100">
              {video.thumbnailUrl ? (
                <img
                  src={video.thumbnailUrl}
                  alt={video.title}
                  className="w-full h-full object-cover"
                />
              ) : (
                <div className="w-full h-full flex items-center justify-center">
                  <FaVideo className="text-gray-400 text-4xl" />
                </div>
              )}
              <div className="absolute inset-0 bg-black bg-opacity-0 group-hover:bg-opacity-30 transition-all flex items-center justify-center">
                <FaPlay className="text-white text-3xl opacity-0 group-hover:opacity-100 transition-all" />
              </div>
            </div>
            <div className="absolute bottom-2 left-2 z-10">
              <span className="px-2 py-1 rounded-full bg-black/50 text-white text-sm">
                {format(new Date(video.createdAt), 'M/d/yyyy')}
              </span>
            </div>
          </div>
          <div className="p-2">
            <h4 className="font-medium truncate text-[var(--foreground)]">{video.title}</h4>
          </div>
        </div>
      ))}
    </div>
  );

  const renderListView = () => (
    <div className="space-y-2">
      {paginatedVideos.map((video) => (
        <div
          key={video.id}
          className={`flex items-center gap-4 p-4 rounded-lg bg-[var(--background)] border border-[var(--border-color)]
            hover:border-[var(--accent-purple)] transition-all cursor-pointer group
            ${activeVideoId === video.id ? 'ring-2 ring-[var(--accent-purple)]' : ''}`}
          onClick={() => handleVideoSelect(video)}
        >
          {/* Thumbnail */}
          <div className="relative w-48 aspect-video flex-shrink-0">
            {video.thumbnailUrl ? (
              <img
                src={video.thumbnailUrl}
                alt={video.title}
                className="w-full h-full object-cover rounded-lg"
              />
            ) : (
              <div className="w-full h-full flex items-center justify-center bg-gray-100 rounded-lg">
                <FaVideo className="text-gray-400 text-2xl" />
              </div>
            )}
            <div className="absolute inset-0 bg-black bg-opacity-0 group-hover:bg-opacity-30 transition-all 
                          flex items-center justify-center rounded-lg">
              <FaPlay className="text-white text-2xl opacity-0 group-hover:opacity-100 transition-all" />
            </div>
          </div>

          {/* Video Info */}
          <div className="flex-grow">
            <h4 className="font-medium text-[var(--foreground)]">{video.title}</h4>
            <p className="text-sm text-[var(--text-secondary)]">
              {format(new Date(video.createdAt), 'MMM d, yyyy')}
            </p>
          </div>

          {/* Status Icons */}
          <div className="flex items-center gap-4">
            {video.maskUrl && (
              <div className="tooltip-wrapper">
                <FaMask className="h-5 w-5 text-[var(--accent-purple)]" />
                <span className="tooltip">Mask Generated</span>
              </div>
            )}
          </div>

          {/* Action Buttons */}
          <div className="flex items-center gap-2 opacity-0 group-hover:opacity-100 transition-opacity">
            {video.maskUrl && (
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  handleSelectVideo(e, video.id);
                }}
                className="p-2 rounded-full hover:bg-[var(--background-hover)] transition-colors"
              >
                {selectedVideos.has(video.id) ? (
                  <FaCheck className="h-5 w-5 text-[var(--accent-purple)]" />
                ) : (
                  <div className="w-5 h-5 border-2 border-[var(--text-secondary)] rounded" />
                )}
              </button>
            )}

            {video.maskUrl && (
              <button
                onClick={async (e) => {
                  e.stopPropagation();
                  // ... existing mask download logic ...
                }}
                className="p-2 rounded-full hover:bg-[var(--background-hover)] transition-colors tooltip-wrapper"
              >
                <FaDownload className="h-5 w-5 text-[var(--text-secondary)]" />
                <span className="tooltip">Download Mask</span>
              </button>
            )}

            <button
              onClick={(e) => {
                e.stopPropagation();
                handleDelete(video.id, e);
              }}
              className="p-2 rounded-full hover:bg-[var(--background-hover)] transition-colors tooltip-wrapper"
            >
              <FaTrash className="h-5 w-5 text-[var(--text-secondary)]" />
              <span className="tooltip">Delete Video</span>
            </button>
          </div>
        </div>
      ))}
    </div>
  );

  const renderPagination = () => {
    // Generate array of page numbers to show
    const getPageNumbers = () => {
      const pageNumbers = [];
      const maxPagesToShow = 5;

      if (totalPages <= maxPagesToShow) {
        // Show all pages if total pages is less than max
        for (let i = 1; i <= totalPages; i++) {
          pageNumbers.push(i);
        }
      } else {
        // Always show first page
        pageNumbers.push(1);

        // Calculate start and end of page range around current page
        let start = Math.max(2, currentPage - 1);
        let end = Math.min(totalPages - 1, currentPage + 1);

        // Add ellipsis after first page if needed
        if (start > 2) {
          pageNumbers.push('...');
        }

        // Add pages around current page
        for (let i = start; i <= end; i++) {
          pageNumbers.push(i);
        }

        // Add ellipsis before last page if needed
        if (end < totalPages - 1) {
          pageNumbers.push('...');
        }

        // Always show last page
        pageNumbers.push(totalPages);
      }

      return pageNumbers;
    };

    return (
      <div className="mt-6 flex justify-center items-center gap-4">
        <div className="flex items-center gap-2 border border-[var(--border-color)] rounded-lg">
          <button
            onClick={() => setCurrentPage(prev => Math.max(prev - 1, 1))}
            disabled={currentPage === 1}
            className={`p-2 transition-colors ${currentPage === 1
              ? 'text-[var(--text-secondary)] cursor-not-allowed'
              : 'text-[var(--text-primary)] hover:text-[var(--accent-purple)]'
              }`}
            aria-label="Previous page"
          >
            <FaChevronLeft className="h-4 w-4" />
          </button>

          {getPageNumbers().map((pageNum, index) => (
            <button
              key={index}
              onClick={() => typeof pageNum === 'number' && setCurrentPage(pageNum)}
              disabled={pageNum === '...' || pageNum === currentPage}
              className={`px-3 py-2 transition-colors border-l border-[var(--border-color)] ${pageNum === currentPage
                ? 'text-[var(--accent-purple)] font-medium'
                : pageNum === '...'
                  ? 'text-[var(--text-secondary)] cursor-default'
                  : 'text-[var(--text-primary)] hover:text-[var(--accent-purple)]'
                }`}
            >
              {pageNum}
            </button>
          ))}

          <button
            onClick={() => setCurrentPage(prev => Math.min(prev + 1, totalPages))}
            disabled={currentPage === totalPages}
            className={`p-2 transition-colors border-l border-[var(--border-color)] ${currentPage === totalPages
              ? 'text-[var(--text-secondary)] cursor-not-allowed'
              : 'text-[var(--text-primary)] hover:text-[var(--accent-purple)]'
              }`}
            aria-label="Next page"
          >
            <FaChevronRight className="h-4 w-4" />
          </button>
        </div>
      </div>
    );
  };

  return (
    <div className="container mx-auto p-4">
      {/* Bulk download button */}
      {selectedVideos.size > 0 && (
        <div className="fixed bottom-6 right-6 z-50">
          <button onClick={handleBulkDownload} className="flex items-center gap-2 px-6 py-3 bg-[var(--accent-purple)] 
            text-white rounded-full shadow-lg hover:bg-[var(--accent-purple-hover)] transition-colors">
            <FaDownload className="h-5 w-5" />
            <span>Download {selectedVideos.size} mask{selectedVideos.size > 1 ? 's' : ''}</span>
          </button>
        </div>
      )}

      {/* Updated Floating Upload Button using absolute positioning */}
      <div
        className={`fixed bottom-6 ${selectedVideos.size > 0 ? 'right-44' : 'right-6'} z-50`}
        onMouseEnter={() => setIsHovered(true)}
        onMouseLeave={() => setIsHovered(false)}
      >
        <button
          onClick={() => fileInputRef.current?.click()}
          disabled={isUploading}
          className="relative rounded-full bg-[var(--accent-purple)] text-white shadow-lg hover:bg-[var(--accent-purple-hover)] transition-all duration-300 disabled:opacity-50 disabled:cursor-not-allowed"
          style={{
            width: isHovered ? '160px' : '56px',
            height: '56px',
            overflow: 'hidden'
          }}
        >
          <div className={`flex items-center transition-all duration-300 ${isHovered ? 'justify-start pl-4' : 'justify-center'}`}>
            <FaPlus className="h-6 w-6" />
            {isHovered && (
              <span className="ml-2 text-base">
                {isUploading ? `${uploadProgress}%` : 'Upload Video'}
              </span>
            )}
          </div>
        </button>
        <input
          type="file"
          onChange={handleFileSelect}
          accept="video/mp4"
          className="hidden"
          ref={fileInputRef}
        />
      </div>

      <div className="card p-6">
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center gap-2">
            <div className="flex items-center gap-2 px-3 py-1.5 rounded-full border border-[var(--border-color)]">
              <FaFilm className="text-[var(--text-secondary)] text-sm" />
              <h3 className="text-sm text-[var(--text-secondary)]">Videos</h3>
            </div>
          </div>

          {/* Controls Group - Now includes Sort Dropdown */}
          <div className="flex items-center gap-3">
            {/* Sort Dropdown */}
            {renderSortDropdown()}
            
            {/* View Mode Toggle */}
            <div className="flex items-center gap-2 border border-[var(--border-color)] rounded-lg">
              <button
                onClick={() => setViewMode('grid')}
                className={`p-2 ${viewMode === 'grid' ? 'text-[var(--accent-purple)]' : 'text-[var(--text-secondary)]'}`}
              >
                <FaThLarge className="h-4 w-4" />
              </button>
              <button
                onClick={() => setViewMode('list')}
                className={`p-2 ${viewMode === 'list' ? 'text-[var(--accent-purple)]' : 'text-[var(--text-secondary)]'}`}
              >
                <FaList className="h-4 w-4" />
              </button>
            </div>
          </div>
        </div>

        {/* Upload input */}
        <input
          type="file"
          onChange={handleFileSelect}
          accept="video/*"
          className="hidden"
          ref={fileInputRef}
        />

        {/* Content */}
        {!videos || videos.length === 0 ? (
          <p className="text-[var(--text-secondary)] text-center py-8">
            No videos uploaded yet. Upload your first video to get started!
          </p>
        ) : (
          <>
            {viewMode === 'grid' ? renderGridView() : renderListView()}
            {renderPagination()}
          </>
        )}
      </div>

      {/* Delete Confirmation Modal */}
      {showDeleteConfirm && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
          <div className="bg-[var(--bg-color)] rounded-lg p-6 max-w-md w-full shadow-xl">
            <h3 className="text-lg font-medium mb-4">Confirm Deletion</h3>
            <p className="text-[var(--text-secondary)] mb-6">
              Are you sure you want to delete this video? This action cannot be undone.
            </p>
            <div className="flex justify-end gap-3">
              <button
                onClick={cancelDelete}
                className="px-4 py-2 border border-[var(--border-color)] rounded-lg text-sm font-medium hover:bg-[var(--hover-color)]"
              >
                Cancel
              </button>
              <button
                onClick={confirmDelete}
                className="px-4 py-2 bg-red-500 text-white rounded-lg text-sm font-medium hover:bg-red-600"
              >
                Delete
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};