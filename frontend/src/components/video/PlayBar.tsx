import React, { useEffect, useCallback } from 'react';
import { FaPlay, FaPause, FaExpand } from 'react-icons/fa';
import { IoMdVolumeHigh, IoMdVolumeOff } from 'react-icons/io';
import { useVideoStore } from '@/store/videoStore';

interface FrameAnnotation {
    frame: number;
    points?: Point[];
    bbox?: BBox | null;
}

interface PlayBarProps {
    videoRef: React.RefObject<HTMLVideoElement>;
    videoId: string;
    onFullscreen?: () => void;
    className?: string;
    FPS?: number;
    frameAnnotations?: FrameAnnotation[];
    onAnnotationClick?: (annotation: FrameAnnotation) => void;
}

export const PlayBar: React.FC<PlayBarProps> = ({
    videoRef,
    videoId,
    onFullscreen,
    className = '',
    FPS = 24,
    frameAnnotations = [],
    onAnnotationClick,
}) => {
    const videoStore = useVideoStore();
    const instance = videoStore.getInstance(videoId);

    // Register instance only once on mount
    useEffect(() => {
        console.debug(`[PlayBar] Registering video instance for id: ${videoId}`);
        videoStore.registerInstance(videoId);
        return () => {
            console.debug(`[PlayBar] Removing video instance for id: ${videoId}`);
            videoStore.removeInstance(videoId);
        };
    }, [videoId]);

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
        console.debug(`[PlayBar] Duration set to ${video.duration} seconds`);
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
            console.debug('[PlayBar] Playing video');
            video.play();
        } else {
            console.debug('[PlayBar] Pausing video');
            video.pause();
        }
    };

    const handleMute = () => {
        const video = videoRef.current;
        if (!video) return;
        video.muted = !video.muted;
        console.debug(`[PlayBar] Video muted: ${video.muted}`);
    };


    const handleSeek = (e: React.MouseEvent<HTMLDivElement>) => {
        const video = videoRef.current;
        if (!video) return;

        const rect = e.currentTarget.getBoundingClientRect();
        const percent = ((e.clientX - rect.left) / rect.width) * 100;
        video.currentTime = (percent / 100) * video.duration;
        console.debug(`[PlayBar] Seek clicked at ${percent.toFixed(2)}% (Time: ${video.currentTime.toFixed(2)} sec)`);
    };

    // Add function to handle annotation marker clicks with logging
    const handleAnnotationClick = (e: React.MouseEvent<HTMLDivElement>, annotation: FrameAnnotation) => {
        e.stopPropagation(); // Prevent triggering the regular seek
        const video = videoRef.current;
        if (!video) return;
        console.debug(`[PlayBar] Annotation marker clicked. Frame: ${annotation.frame}, Calculated time: ${(annotation.frame / FPS).toFixed(2)} seconds`);
        // Seek to the appropriate time based on the frame
        video.currentTime = annotation.frame / FPS;
        if (onAnnotationClick) {
            onAnnotationClick(annotation);
        }
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
                className="h-2 bg-gray-600 cursor-pointer w-full rounded-full overflow-hidden mb-3 relative"
                onClick={handleSeek}
            >
                <div
                    className="h-full bg-blue-500 transition-all duration-100"
                    style={{ width: `${instance.progress}%` }}
                />

                {/* Annotation markers */}
                {frameAnnotations.map((annotation, index) => {
                    // Calculate position as percentage of video duration
                    const position = instance.duration > 0
                        ? (annotation.frame / (FPS * instance.duration)) * 100
                        : 0;
                    console.debug(`[PlayBar] Rendering marker ${index} - Frame: ${annotation.frame}, Position: ${position.toFixed(2)}%`);
                    return (
                        <div
                            key={`annotation-${index}`}
                            className="absolute top-0 h-full w-1 bg-white cursor-pointer hover:bg-yellow-300"
                            style={{
                                left: `${position}%`,
                                zIndex: 10,
                            }}
                            onClick={(e) => handleAnnotationClick(e, annotation)}
                            title={`Frame ${annotation.frame}: ${annotation.bbox ? 'Box' : ''} ${annotation.points?.length ? 'Points' : ''}`}
                        />
                    );
                })}
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