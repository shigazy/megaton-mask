import { forwardRef, useRef, useState } from 'react';
import { FaPlay, FaPause, FaExpand } from 'react-icons/fa';
import { IoMdVolumeHigh, IoMdVolumeOff } from 'react-icons/io';

interface CustomVideoPlayerProps extends React.VideoHTMLAttributes<HTMLVideoElement> {
  onTimeUpdate?: (e: React.SyntheticEvent<HTMLVideoElement>) => void;
  onLoadedMetadata?: (e: React.SyntheticEvent<HTMLVideoElement>) => void;
}

export const CustomVideoPlayer = forwardRef<HTMLVideoElement, CustomVideoPlayerProps>(
  ({ className, onTimeUpdate, onLoadedMetadata, ...props }, ref) => {
    const [isPlaying, setIsPlaying] = useState(false);
    const [isMuted, setIsMuted] = useState(false);
    const [progress, setProgress] = useState(0);
    const progressBarRef = useRef<HTMLDivElement>(null);

    const togglePlay = () => {
      const video = (ref as React.RefObject<HTMLVideoElement>)?.current;
      if (video) {
        if (video.paused) {
          video.play();
          setIsPlaying(true);
        } else {
          video.pause();
          setIsPlaying(false);
        }
      }
    };

    const toggleMute = () => {
      const video = (ref as React.RefObject<HTMLVideoElement>)?.current;
      if (video) {
        video.muted = !video.muted;
        setIsMuted(video.muted);
      }
    };

    const handleTimeUpdate = (e: React.SyntheticEvent<HTMLVideoElement>) => {
      const video = e.currentTarget as HTMLVideoElement;
      if (video && !isNaN(video.duration)) {
        const progress = (video.currentTime / video.duration) * 100;
        setProgress(progress);
      }
      // Call the parent's onTimeUpdate if provided
      onTimeUpdate?.(e);
    };

    const handleLoadedMetadata = (e: React.SyntheticEvent<HTMLVideoElement>) => {
      const video = e.currentTarget as HTMLVideoElement;
      if (video && !isNaN(video.duration)) {
        // Handle loaded metadata if needed
      }
      // Call the parent's onLoadedMetadata if provided
      onLoadedMetadata?.(e);
    };

    const handleProgressBarClick = (e: React.MouseEvent<HTMLDivElement>) => {
      const video = (ref as React.RefObject<HTMLVideoElement>)?.current;
      const progressBar = progressBarRef.current;
      if (video && progressBar) {
        const rect = progressBar.getBoundingClientRect();
        const pos = (e.clientX - rect.left) / rect.width;
        video.currentTime = pos * video.duration;
      }
    };

    return (
      <div className="relative group">
        <video
          ref={ref}
          className={`w-full ${className}`}
          onTimeUpdate={handleTimeUpdate}
          onLoadedMetadata={handleLoadedMetadata}
          controls={false}
          {...props}
        />

        {/* Custom Controls */}
        <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black/70 to-transparent p-4 opacity-0 group-hover:opacity-100 transition-opacity">
          {/* Progress Bar */}
          <div
            ref={progressBarRef}
            className="w-full h-1.5 bg-gray-400 cursor-pointer mb-4 rounded-full"
            onClick={handleProgressBarClick}
          >
            <div
              className="h-full bg-blue-500 rounded-full"
              style={{ width: `${progress}%` }}
            />
          </div>

          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              <button
                onClick={togglePlay}
                className="text-white hover:text-blue-400 transition-colors"
                aria-label={isPlaying ? 'Pause' : 'Play'}
              >
                {isPlaying ? <FaPause size={20} /> : <FaPlay size={20} />}
              </button>

              <button
                onClick={toggleMute}
                className="text-white hover:text-blue-400 transition-colors"
                aria-label={isMuted ? 'Unmute' : 'Mute'}
              >
                {isMuted ? <IoMdVolumeOff size={20} /> : <IoMdVolumeHigh size={20} />}
              </button>
            </div>

            <button
              onClick={() => {
                const video = (ref as React.RefObject<HTMLVideoElement>)?.current;
                video?.requestFullscreen();
              }}
              className="text-white hover:text-blue-400 transition-colors"
              aria-label="Full screen"
            >
              <FaExpand size={20} />
            </button>
          </div>
        </div>
      </div>
    );
  }
);

CustomVideoPlayer.displayName = 'CustomVideoPlayer';