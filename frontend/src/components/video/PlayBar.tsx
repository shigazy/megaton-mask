import React, { useEffect, useCallback } from 'react';
import { FaPlay, FaPause, FaExpand } from 'react-icons/fa';
import { IoMdVolumeHigh, IoMdVolumeOff } from 'react-icons/io';
import { useVideoStore } from '@/store/videoStore';

interface PlayBarProps {
    videoRef: React.RefObject<HTMLVideoElement>;
    videoId: string;
    onFullscreen?: () => void;
    className?: string;
    FPS?: number;
}

export const PlayBar: React.FC<PlayBarProps> = ({
    videoRef,
    videoId,
    onFullscreen,
    className = '',
    FPS = 24,
}) => {
    const videoStore = useVideoStore();
    const instance = videoStore.getInstance(videoId);

    // Register instance only once on mount
    useEffect(() => {
        videoStore.registerInstance(videoId);
        return () => videoStore.removeInstance(videoId);
    }, [videoId]); // Remove videoStore from dependencies

    // Memoize the update handlers
    const handleTimeUpdate = useCallback(() => {
        const video = videoRef.current;
        if (!video || isNaN(video.duration)) return;

        videoStore.setProgress(videoId, (video.currentTime / video.duration) * 100);
        videoStore.setCurrentTime(videoId, video.currentTime);
    }, [videoId, videoRef]);

    const handlePlay = useCallback(() => {
        videoStore.setIsPlaying(videoId, true);
    }, [videoId]);

    const handlePause = useCallback(() => {
        videoStore.setIsPlaying(videoId, false);
    }, [videoId]);

    const handleVolumeChange = useCallback(() => {
        const video = videoRef.current;
        if (!video) return;
        videoStore.setIsMuted(videoId, video.muted);
    }, [videoId, videoRef]);

    const handleDurationChange = useCallback(() => {
        const video = videoRef.current;
        if (!video || isNaN(video.duration)) return;
        videoStore.setDuration(videoId, video.duration);
    }, [videoId, videoRef]);

    // Set up video event listeners
    useEffect(() => {
        const video = videoRef.current;
        if (!video) return;

        video.addEventListener('timeupdate', handleTimeUpdate);
        video.addEventListener('play', handlePlay);
        video.addEventListener('pause', handlePause);
        video.addEventListener('volumechange', handleVolumeChange);
        video.addEventListener('durationchange', handleDurationChange);

        // Set initial duration if available
        if (video.duration && !isNaN(video.duration)) {
            handleDurationChange();
        }

        return () => {
            video.removeEventListener('timeupdate', handleTimeUpdate);
            video.removeEventListener('play', handlePlay);
            video.removeEventListener('pause', handlePause);
            video.removeEventListener('volumechange', handleVolumeChange);
            video.removeEventListener('durationchange', handleDurationChange);
        };
    }, [handleTimeUpdate, handlePlay, handlePause, handleVolumeChange, handleDurationChange]);

    if (!instance) return null;

    const handlePlayPause = () => {
        const video = videoRef.current;
        if (!video) return;

        if (video.paused) {
            video.play();
        } else {
            video.pause();
        }
    };

    const handleMute = () => {
        const video = videoRef.current;
        if (!video) return;
        video.muted = !video.muted;
    };

    const handleSeek = (e: React.MouseEvent<HTMLDivElement>) => {
        const video = videoRef.current;
        if (!video) return;

        const rect = e.currentTarget.getBoundingClientRect();
        const percent = ((e.clientX - rect.left) / rect.width) * 100;
        video.currentTime = (percent / 100) * video.duration;
    };

    const formatTime = (seconds: number): string => {
        if (!seconds || isNaN(seconds)) return '00:00';
        const minutes = Math.floor(seconds / 60);
        const remainingSeconds = Math.floor(seconds % 60);
        return `${minutes.toString().padStart(2, '0')}:${remainingSeconds.toString().padStart(2, '0')}`;
    };

    const getFrameCount = (seconds: number): number => {
        if (!seconds || isNaN(seconds)) return 0;
        return Math.floor(seconds * FPS);
    };

    return (
        <div className={`bg-gray-800 rounded-lg p-4 shadow-lg ${className}`}>
            <div
                className="h-2 bg-gray-600 cursor-pointer w-full rounded-full overflow-hidden mb-3"
                onClick={handleSeek}
            >
                <div
                    className="h-full bg-blue-500 transition-all duration-100"
                    style={{ width: `${instance.progress}%` }}
                />
            </div>

            <div className="flex items-center justify-between">
                <div className="flex items-center space-x-4">
                    <button
                        onClick={handlePlayPause}
                        className="text-white hover:text-blue-400 transition-colors"
                    >
                        {instance.isPlaying ? <FaPause size={20} /> : <FaPlay size={20} />}
                    </button>

                    <button
                        onClick={handleMute}
                        className="text-white hover:text-blue-400 transition-colors"
                    >
                        {instance.isMuted ? <IoMdVolumeOff size={20} /> : <IoMdVolumeHigh size={20} />}
                    </button>
                </div>

                <div className="text-white text-sm font-mono">
                    {instance.showFrames ? (
                        <button
                            onClick={() => videoStore.setShowFrames(videoId, false)}
                            title={`${FPS} FPS`}
                        >
                            {getFrameCount(instance.currentTime)}/{getFrameCount(instance.duration)}
                        </button>
                    ) : (
                        <button
                            onClick={() => videoStore.setShowFrames(videoId, true)}
                        >
                            {formatTime(instance.currentTime)} / {formatTime(instance.duration)}
                        </button>
                    )}
                </div>

                <button
                    onClick={onFullscreen}
                    className="text-white hover:text-blue-400 transition-colors"
                >
                    <FaExpand size={20} />
                </button>
            </div>
        </div>
    );
}; 