import React, { useEffect, useCallback } from 'react';
import { FaPlay, FaPause, FaExpand } from 'react-icons/fa';
import { IoMdVolumeHigh, IoMdVolumeOff } from 'react-icons/io';
import { useVideoStore } from '@/store/videoStore';

interface Point {
    x: number;
    y: number;
    type: 'positive' | 'negative';
}

interface AnnotationData {
    [frame: string]: {
        bbox: number[] | null;       // [x, y, width, height] as array
        points: Point[] | null;
        mask_data: any | null;
    };
}

interface FrameAnnotation {
    frame: number; // Index of the frame in the video
    points?: Point[]; // Points for this frame index
    bbox?: BBox | null; // Bounding box for this frame index
}

interface PlayBarProps {
    videoRef: React.RefObject<HTMLVideoElement>;
    videoId: string;
    onFullscreen?: () => void;
    className?: string;
    FPS?: number;
    frameAnnotations?: FrameAnnotation[];
    onAnnotationClick?: (annotation: FrameAnnotation) => void;
    annotation?: AnnotationData;
}

export const PlayBar: React.FC<PlayBarProps> = ({
    videoRef,
    videoId,
    onFullscreen,
    className = '',
    FPS = 24,
    frameAnnotations = [],
    onAnnotationClick,
    annotation = {},
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

    // Update this code inside your PlayBar component
    useEffect(() => {
        // Debug what annotation data we're receiving
        console.log("Annotation data received:", annotation);
        if (annotation) {
            console.log("Number of frames with annotations:", Object.keys(annotation).length);
        }
    }, [annotation]);

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

    // Updated to handle frame numbers directly
    const handleAnnotationClick = (e: React.MouseEvent<HTMLDivElement>, frame: number) => {
        e.stopPropagation(); // Prevent triggering the regular seek
        const video = videoRef.current;
        if (!video) return;

        // Seek to the frame
        video.currentTime = frame / FPS;
        console.debug(`[PlayBar] Jumped to frame ${frame} (${(frame / FPS).toFixed(2)}s)`);

        if (onAnnotationClick && annotation && annotation[frame]) {
            onAnnotationClick({
                frame,
                points: annotation[frame].points || [],
                bbox: annotation[frame].bbox || null
            });
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
                {annotation && Object.keys(annotation).map((frameStr) => {
                    const frame = parseInt(frameStr, 10);
                    const frameData = annotation[frameStr];

                    // Debug each frame's data
                    console.log(`Frame ${frame} data:`, frameData);

                    // Only show markers for frames that have annotations
                    // IMPORTANT: Check the exact structure that your data has
                    const hasBbox = frameData.bbox && (
                        Array.isArray(frameData.bbox) ? frameData.bbox.length > 0 : true
                    );
                    const hasPoints = frameData.points && Array.isArray(frameData.points) && frameData.points.length > 0;

                    // Skip frames without annotations
                    if (!frameData || (!hasBbox && !hasPoints)) {
                        console.log(`Skipping frame ${frame} - no valid annotations`);
                        return null;
                    }

                    // Calculate position as percentage of video duration
                    const totalFrames = instance.duration ? Math.floor(instance.duration * FPS) : 0;
                    console.log(`Total frames: ${totalFrames}, Current frame: ${frame}`);

                    // THIS CALCULATION IS CRITICAL - adjust if markers aren't positioned correctly
                    const position = totalFrames > 0
                        ? (frame / totalFrames) * 100
                        : 0;

                    console.log(`Frame ${frame} position: ${position.toFixed(2)}%`);

                    // Skip markers that would be off-screen
                    if (position < 0 || position > 100) {
                        console.log(`Skipping frame ${frame} - position out of range (${position}%)`);
                        return null;
                    }

                    // Choose color based on annotation type
                    const markerColor = hasBbox && hasPoints
                        ? 'bg-yellow-400' // Both box and points
                        : hasBbox
                            ? 'bg-blue-400'  // Only box
                            : 'bg-green-400'; // Only points

                    const tooltipText = `Frame ${frame}: ${hasBbox ? 'Box' : ''} ${hasPoints ? `${frameData.points.length} Points` : ''}`;

                    return (
                        <div
                            key={`annotation-${frame}`}
                            className={`absolute top-0 h-full w-1 ${markerColor} hover:bg-white cursor-pointer`}
                            style={{
                                left: `${position}%`,
                                zIndex: 20, // Increased z-index to make sure markers are visible
                            }}
                            onClick={(e) => {
                                e.stopPropagation();
                                const video = videoRef.current;
                                if (!video) return;
                                video.currentTime = frame / FPS;
                                console.log(`Jumped to frame ${frame} (${(frame / FPS).toFixed(2)}s)`);
                                if (onAnnotationClick) {
                                    onAnnotationClick({ frame, points: frameData.points, bbox: frameData.bbox });
                                }
                            }}
                            title={tooltipText}
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