'use client';

import { useState, useRef, useEffect, useMemo, useCallback } from 'react';
import axios from 'axios';
import { CustomVideoPlayer } from '../video/VideoPlayer';
import { MaskOverlay } from './MaskOverlay';
import debounce from 'lodash/debounce';
import { FaCloudUploadAlt, FaTrash, FaTerminal, FaBorderNone, FaDownload, FaPlus, FaMinus, FaUndo, FaPencilAlt, FaMask, FaSquare, FaDotCircle, FaPlay, FaStar, FaEye } from 'react-icons/fa';
import { UPLOAD_LIMITS } from '@/lib/constants';
import { useStatus } from '@/contexts/StatusContext';
import { Video } from '@/lib/types/video';
import { useVideoStore } from '@/store/videoStore';
import { PlayBar } from '../video/PlayBar';
import { handleDownload } from '@/utils/functions';
import { useCredits } from '@/contexts/CreditsContext';

// Types and Interfaces
interface Point {
  x: number;
  y: number;
  type: 'positive' | 'negative';
}

interface BBox {
  x: number;
  y: number;
  w: number;
  h: number;
}

interface AnnotationLayerProps {
  videoWidth: number;
  videoHeight: number;
  drawMode: 'bbox' | 'points';
  pointType: 'positive' | 'negative';
  points: Point[];
  bbox: BBox | null;
  onBboxChange: (bbox: BBox | null) => void;
  onPointsChange: (points: Point[]) => void;
  onPointClick: (point: Point, index: number) => void;
  className?: string;
  forceRedrawRef: MutableRefObject<(() => void) | undefined>;
  redrawTrigger: number;
}

interface UploadResponse {
  videoUrl: string;
  processedVideoUrl: string;
  maskVideoUrl: string;
}

interface VideoUploadProps {
  onUploadSuccess: () => void;
  fetchVideos: () => void;
  initialVideo?: Video;
  fps?: number;
}

interface MaskData {
  shape: number[];
  data: string;
}

interface VideoMetadata {
  fps: number;
  frame_count: number;
  width: number;
  height: number;
  duration: number;
  file_size: number;
  codec: number;
  codec_name?: string;
  uploaded_filename: string;
  upload_date: string;
}

interface FrameAnnotation {
  [currentFrame: number]: {
    points: Point[];
    bbox: BBox | null;
    mask_data: MaskData | null;
  };
}

declare global {
  namespace NodeJS {
    interface ProcessEnv {
      NEXT_PUBLIC_API_URL: string;
    }
  }
}

const COLORS = {
  POSITIVE_POINT: '#56dd92',
  NEGATIVE_POINT: '#dd6456',
  BBOX: '#56dd92',
} as const;



const AnnotationLayer = ({
  videoWidth,
  videoHeight,
  drawMode,
  pointType,
  points,
  bbox,
  getCurrentFrame,
  onBboxChange,
  onPointsChange,
  onPointClick,
  className,
  forceRedrawRef,
  redrawTrigger
}: AnnotationLayerProps) => {
  //console.log('=== AnnotationLayer Render ===');
  //console.log('Received bbox prop:', bbox);
  // console.log('Current dimensions:', { videoWidth, videoHeight });

  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const needsRedraw = useRef(true);
  const startPoint = useRef<{ x: number, y: number } | null>(null);

  const [isDrawing, setIsDrawing] = useState(false);
  const [localBbox, setLocalBbox] = useState<BBox | null>(bbox);
  const [hoveredPointIndex, setHoveredPointIndex] = useState<number | null>(null);
  const [scale, setScale] = useState({ x: 1, y: 1 });
  const [frameAnnotations, setFrameAnnotations] = useState<FrameAnnotation[]>([]);
  // Update effect to prevent loops
  useEffect(() => {
    if (bbox && (!localBbox || JSON.stringify(bbox) !== JSON.stringify(localBbox))) {
      // console.log('Updating localBbox from bbox prop:', bbox);
      setLocalBbox(bbox);
      needsRedraw.current = true;
    }
  }, [bbox]); // Only depend on bbox prop

  const handleMouseDown = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const rect = canvas.getBoundingClientRect();
    const x = (e.clientX - rect.left) * (videoWidth / rect.width);
    const y = (e.clientY - rect.top) * (videoHeight / rect.height);

    // Get current frame information (you need to track this)
    const currentFrame = getCurrentFrame(); // You need to implement this function

    if (drawMode === 'bbox') {
      startPoint.current = { x, y };
      setIsDrawing(true);
    } else if (drawMode === 'points') {
      // console.log('Drawing points', console.log(points));
      const pointIndex = points.findIndex(point => {
        const dx = point.x - x;
        const dy = point.y - y;
        return Math.sqrt(dx * dx + dy * dy) < 5;
      });
      setHoveredPointIndex(pointIndex);

      if (pointIndex !== -1) {
        const newPoints = points.filter((_, i) => i !== pointIndex);
        onPointsChange(newPoints);
      } else {
        const newPoint = { x, y, type: pointType };
        onPointsChange([...points, newPoint]);
      }
    }
  }, [drawMode, pointType, points, onPointsChange, videoWidth, videoHeight]);

  const handleMouseMove = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const rect = canvas.getBoundingClientRect();
    const x = (e.clientX - rect.left) * (videoWidth / rect.width);
    const y = (e.clientY - rect.top) * (videoHeight / rect.height);

    if (isDrawing && drawMode === 'bbox' && startPoint.current) {
      const newBbox = {
        x: Math.min(startPoint.current.x, x),
        y: Math.min(startPoint.current.y, y),
        w: Math.abs(x - startPoint.current.x),
        h: Math.abs(y - startPoint.current.y)
      };
      setLocalBbox(newBbox);
      onBboxChange(newBbox);
      needsRedraw.current = true;
    }

    const pointIndex = points.findIndex(point => {
      const dx = point.x - x;
      const dy = point.y - y;
      return Math.sqrt(dx * dx + dy * dy) < 5;
    });
    setHoveredPointIndex(pointIndex);

    if (canvas) {
      canvas.style.cursor = pointIndex !== -1 ? 'pointer' : 'default';
    }
  }, [isDrawing, drawMode, videoWidth, videoHeight, onBboxChange, points]);

  const handleMouseUp = useCallback(() => {
    if (isDrawing && startPoint.current) {
      const finalBbox = localBbox;
      console.log('Final bbox:', finalBbox);
      if (finalBbox && finalBbox.w > 0 && finalBbox.h > 0) {
        console.log('Saving final bbox:', finalBbox);
        onBboxChange(finalBbox);  // This should trigger the save
      }
    }
    setIsDrawing(false);
    startPoint.current = null;
    needsRedraw.current = true;
  }, [isDrawing, localBbox, onBboxChange]);

  // Log when canvas is set up
  useEffect(() => {
    console.log('=== Canvas Setup ===');
    const canvas = canvasRef.current;
    if (!canvas) {
      console.log('No canvas in setup');
      return;
    }

    const updateCanvasSize = () => {
      const rect = canvas.getBoundingClientRect();
      console.log('Setting canvas size:', rect);
      const dpr = window.devicePixelRatio || 1;
      canvas.width = rect.width * dpr;
      canvas.height = rect.height * dpr;

      const ctx = canvas.getContext('2d');
      if (ctx) {
        ctx.scale(dpr, dpr);
      }
      needsRedraw.current = true;
    };

    updateCanvasSize();
    const resizeObserver = new ResizeObserver(() => {
      console.log('Canvas resized');
      updateCanvasSize();
    });
    resizeObserver.observe(canvas);

    return () => resizeObserver.disconnect();
  }, [videoWidth, videoHeight]);

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const rect = canvas.getBoundingClientRect();
    const scaleX = rect.width / videoWidth;
    const scaleY = rect.height / videoHeight;

    ctx.clearRect(0, 0, rect.width, rect.height);

    // Draw bbox
    const bboxToDraw = localBbox || bbox;
    if (bboxToDraw) {
      const [x, y, w, h] = Array.isArray(bboxToDraw) ? bboxToDraw : [bboxToDraw.x, bboxToDraw.y, bboxToDraw.w, bboxToDraw.h];

      // Only proceed if we have valid dimensions
      if (w > 0 && h > 0) {
        const canvasX = x * scaleX;
        const canvasY = y * scaleY;
        const canvasW = w * scaleX;
        const canvasH = h * scaleY;

        ctx.strokeStyle = COLORS.BBOX;
        ctx.lineWidth = 4;
        ctx.beginPath();
        ctx.roundRect(canvasX, canvasY, canvasW, canvasH, 8);
        ctx.stroke();
      }
    }

    console.log('Drawing points:', points);

    // Convert points from {positive: [], negative: []} format to [{x, y, type}] format
    // TO DO: Deprecate this once backend is standardized. 
    if (points && typeof points === 'object' && ('positive' in points || 'negative' in points)) {
      const formattedPoints = [];

      if (points.positive) {
        formattedPoints.push(...points.positive.map(point => ({
          x: point[0],
          y: point[1],
          type: 'positive'
        })));
      }

      if (points.negative) {
        formattedPoints.push(...points.negative.map(point => ({
          x: point[0],
          y: point[1],
          type: 'negative'
        })));
      }

      points = formattedPoints;
    }

    // Draw points
    points.forEach((point, index) => {
      const canvasX = point.x * scaleX;
      const canvasY = point.y * scaleY;

      // Add shadow
      ctx.shadowColor = 'rgba(0, 0, 0, 0.3)';
      ctx.shadowBlur = 4;
      ctx.shadowOffsetX = 1;
      ctx.shadowOffsetY = 1;

      ctx.beginPath();
      ctx.arc(canvasX, canvasY, 5, 0, Math.PI * 2);
      ctx.fillStyle = point.type === 'positive' ?
        COLORS.POSITIVE_POINT :
        COLORS.NEGATIVE_POINT;
      ctx.fill();

      if (index === hoveredPointIndex) {
        ctx.strokeStyle = 'var(--foreground)';
        ctx.lineWidth = 2;
        ctx.stroke();

        // Reset shadow for text
        ctx.shadowColor = 'transparent';
        ctx.shadowBlur = 0;
        ctx.shadowOffsetX = 0;
        ctx.shadowOffsetY = 0;

        ctx.font = '12px var(--font-geist-sans)';
        ctx.fillStyle = 'var(--foreground)';
        ctx.textAlign = 'center';
        ctx.fillText(
          `(${Math.round(point.x)}, ${Math.round(point.y)})`,
          canvasX,
          canvasY - 10
        );
      }

      // Reset shadow after each point
      ctx.shadowColor = 'transparent';
      ctx.shadowBlur = 0;
      ctx.shadowOffsetX = 0;
      ctx.shadowOffsetY = 0;
    });

    needsRedraw.current = false;
  }, [points, localBbox, bbox, hoveredPointIndex, videoWidth, videoHeight]);

  // Animation and redraw effects
  useEffect(() => {
    let animationFrame: number;
    const animate = () => {
      if (needsRedraw.current) {
        draw();
      }
      animationFrame = requestAnimationFrame(animate);
    };
    animate();
    return () => cancelAnimationFrame(animationFrame);
  }, [draw]);

  useEffect(() => {
    if (forceRedrawRef) {
      forceRedrawRef.current = () => {
        needsRedraw.current = true;
        draw();
      };
    }
  }, [draw]);

  useEffect(() => {
    needsRedraw.current = true;
    draw();
  }, [redrawTrigger, draw]);

  return (
    <div
      ref={containerRef}
      className={`absolute top-0 left-0 w-full h-full ${className || ''}`}
    >
      <canvas
        ref={canvasRef}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
        className="absolute top-0 left-0 w-full h-full"
        style={{ zIndex: 30 }}
      />
    </div>
  );
};

export const VideoUpload = ({ onUploadSuccess, fetchVideos, initialVideo, fps }: VideoUploadProps) => {
  const { setStatus } = useStatus();
  const [uploading, setUploading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [videoUrl, setVideoUrl] = useState<string | null>(initialVideo?.videoUrl || null);
  const [processedVideoUrl, setProcessedVideoUrl] = useState<string | null>(null);
  const [maskVideoUrl, setMaskVideoUrl] = useState<string | null>(null);
  const [drawMode, setDrawMode] = useState<'bbox' | 'points'>('bbox');
  const [pointType, setPointType] = useState<'positive' | 'negative'>('positive');
  const [bbox, setBbox] = useState<BBox | null>(initialVideo?.bbox || null);
  const [points, setPoints] = useState<Point[]>(initialVideo?.points || []);
  const [pointHistory, setPointHistory] = useState<Point[][]>([]);
  const videoRef = useRef<HTMLVideoElement>(null);
  const [videoDimensions, setVideoDimensions] = useState({ width: 0, height: 0 });
  const [isProcessing, setIsProcessing] = useState(false);
  const [previewMask, setPreviewMask] = useState<string | null>(null);
  const [isGenerating, setIsGenerating] = useState(false);
  const [maskData, setMaskData] = useState<MaskData | null>(null);
  const [videoSize, setVideoSize] = useState({ width: 0, height: 0 });
  const abortControllerRef = useRef<AbortController | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const needsRedraw = useRef(true);
  const forceRedrawRef = useRef<() => void>();
  const [redrawTrigger, setRedrawTrigger] = useState(0);
  const pendingRequestRef = useRef<AbortController | null>(null);
  const [activeTab, setActiveTab] = useState<'annotation' | 'mask' | 'greenscreen'>('annotation');
  const [isLoadingVideo, setIsLoadingVideo] = useState(false);
  const [videoError, setVideoError] = useState<string | null>(null);
  const [superMasks, setSuperMasks] = useState<boolean>(true);
  const [previewBackend, setPreviewBackend] = useState<boolean>(false);
  const [method, setMethod] = useState<'dual_process' | 'preprocess'>('preprocess');
  const progressBarRef = useRef<HTMLDivElement>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [isMuted, setIsMuted] = useState(false);
  const [showFrames, setShowFrames] = useState(false);
  const DEFAULT_FPS = 24;
  const [currentTime, setCurrentTime] = useState(0);
  const [greenscreenUrl, setGreenscreenUrl] = useState<string | null>(null);
  const [duration, setDuration] = useState(0);
  const [isPolling, setIsPolling] = useState(false);
  const [pollCount, setPollCount] = useState(0);
  const [annotation, setAnnotation] = useState<FrameAnnotation[]>(initialVideo?.annotation || []);

  const MAX_POLLS = 300; // Safety limit (5 minutes at 1 second intervals)

  const videoStore = useVideoStore();
  const { fetchCredits } = useCredits();
  // Add this with your other refs
  const maskVideoRef = useRef<HTMLVideoElement>(null);
  const greenscreenVideoRef = useRef<HTMLVideoElement>(null);

  // Modify the refreshUrls useEffect
  useEffect(() => {
    if (!initialVideo?.id || isGenerating) return; // Don't refresh during generation

    let intervalId: NodeJS.Timeout;
    const refreshUrls = async () => {
      try {
        const response = await fetch(
          `${process.env.NEXT_PUBLIC_API_URL}/api/videos/${initialVideo.id}/refresh-url`,
          {
            headers: {
              'Authorization': `Bearer ${localStorage.getItem('token')}`
            }
          }
        );

        if (!response.ok) {
          throw new Error('Failed to refresh URLs');
        }

        const data = await response.json();

        // Only update if URLs have changed
        if (data.videoUrl !== videoUrl) {
          setVideoUrl(data.videoUrl);
        }

        // Only update if mask URL exists and has changed
        if (data.maskUrl && data.maskUrl !== maskVideoUrl) {
          setMaskVideoUrl(data.maskUrl);
        }

      } catch (error) {
        console.error('Error refreshing URLs:', error);
        // If we get an auth error, clear the interval to stop refreshing
        if (error instanceof Error && error.message.includes('401')) {
          clearInterval(intervalId);
        }
      }
    };

    // Initial refresh
    refreshUrls();

    // Refresh every 45 minutes (presigned URLs expire after 1 hour)
    intervalId = setInterval(refreshUrls, 45 * 60 * 1000);

    return () => clearInterval(intervalId);
  }, [initialVideo?.id, videoUrl, maskVideoUrl, isGenerating]);


  // Create a function to force redraw that can be called from anywhere
  const forceRedraw = useCallback(() => {
    if (forceRedrawRef.current) {
      console.log('Forcing canvas redraw...');
      forceRedrawRef.current();
    }
  }, []);
  // MODIFY EXISTING - Update the saveChanges function to include current frame
  const saveChanges = useCallback(async (
    newPoints?: Point[],
    newBbox?: BBox | null,
    newMaskData?: ImageData | null,
    currentFrame?: number
  ) => {
    if (!initialVideo?.id) return;

    try {
      const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
      const url = `${apiUrl}/api/videos/${initialVideo.id}`;

      // Ensure we're using the new bbox if provided
      const bboxToSave = newBbox !== undefined ? newBbox : bbox;
      console.log('Preparing to save bbox:', bboxToSave);

      // Convert BBox to array format
      const bboxArray = bboxToSave ? [
        bboxToSave.x,
        bboxToSave.y,
        bboxToSave.w,
        bboxToSave.h
      ] : null;

      // Calculate current frame if not provided
      const frameToSave = currentFrame !== undefined ? currentFrame : getCurrentFrame();
      console.log('Saving at frame:', frameToSave);

      const annotationToSave = {
        ...annotation,
        [frameToSave]: {
          points: newPoints || points,
          bbox: bboxArray,
          mask_data: newMaskData || maskData
        },
      }

      const response = await fetch(url, {
        method: 'PATCH',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${localStorage.getItem('token')}`
        },
        body: JSON.stringify({
          points: newPoints || points,
          bbox: bboxArray,
          mask_data: newMaskData || maskData,
          current_frame: frameToSave,
          annotation: annotationToSave
        })
      });

      setAnnotation(annotationToSave);

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      console.log('Save response from backend:', data);

    } catch (error) {
      console.error('Error saving changes:', error);
      setStatus('Error saving changes');
    }
  }, [initialVideo?.id, points, bbox, maskData]);

  const getCurrentFrame = useCallback((): number => {
    const video = videoRef.current;
    if (!video || isNaN(video.currentTime)) return 0;

    // Get FPS from video metadata or use default
    const fps = initialVideo?.video_metadata?.fps || DEFAULT_FPS;
    return Math.floor(video.currentTime * fps);
  }, [videoRef, initialVideo?.video_metadata?.fps]);

  // Update handlers to use the improved saveChanges
  const handlePointsChange = useCallback((newPoints: Point[]) => {
    setPointHistory(prev => [...prev, points]);
    setPoints(newPoints);
    // Get current frame when setting points
    const currentFrame = getCurrentFrame();
    console.log('Setting points at frame:', currentFrame);

    if (initialVideo?.id) {  // Only save if we have a video ID
      saveChanges(newPoints, undefined, undefined, currentFrame);
    }
  }, [points, saveChanges, initialVideo?.id]);

  // In main VideoUpload component
  const handleBboxChange = useCallback((newBbox: BBox | null) => {
    console.log('handleBboxChange called with:', newBbox);
    setBbox(newBbox);
    // Get current frame when setting bbox
    const video = videoRef.current;
    const currentFrame = getCurrentFrame();
    console.log('Setting bbox at frame:', currentFrame);

    if (initialVideo?.id && newBbox) {  // Only save if we have both video ID and valid bbox
      console.log('Triggering saveChanges with new bbox');
      saveChanges(undefined, newBbox, undefined, currentFrame);
    }
  }, [saveChanges, initialVideo?.id, getCurrentFrame]);
  // Add this new function
  const handleAnnotationClick = useCallback((annotation: FrameAnnotation) => {
    const currentFrame = getCurrentFrame();
    console.log('Loading annotation from frame:', annotation[currentFrame]);
    // Set the bbox and points from the annotation
    if (annotation[currentFrame]) {
      setBbox(annotation[currentFrame].bbox || null);
    } else {
      setBbox(null);
    }

    if (annotation[currentFrame]?.points && annotation[currentFrame].points.length > 0) {
      setPoints(annotation[currentFrame].points);
    } else {
      setPoints([]);
    }

    // Force redraw to show the loaded annotations
    forceRedraw();
  }, [forceRedraw]);

  const handleUndo = useCallback(() => {
    setStatus('Undoing last action...');
    if (pointHistory.length > 0) {
      const previousPoints = pointHistory[pointHistory.length - 1];
      console.log('Undoing to previous points:', previousPoints);

      // Update points state
      setPoints(previousPoints);
      // Update history
      setPointHistory(prev => prev.slice(0, -1));
      const currentFrame = getCurrentFrame();
      // Immediately save changes with the new points
      saveChanges(previousPoints, bbox, maskData, currentFrame);

      // Force redraw if using AnnotationLayer
      if (forceRedrawRef.current) {
        forceRedrawRef.current();
      }
      forceRedraw(); // Force canvas update

      console.log('After undo:', {
        newPoints: previousPoints,
        remainingHistory: pointHistory.slice(0, -1)
      });
    }
    setStatus('Action undone');
  }, [pointHistory, bbox, maskData, saveChanges, getCurrentFrame]);

  const handlePointInteraction = useCallback((point: Point, index: number) => {
    const newPoints = points.filter((_, i) => i !== index);
    handlePointsChange(newPoints);
  }, [points, handlePointsChange]);

  // Annotation Layer - IF not thing breaks, delete it since it's not being called
  /*
  const annotationLayer = useMemo(() => (
    videoDimensions.width > 0 && (
      <AnnotationLayer
        videoWidth={videoDimensions.width}
        videoHeight={videoDimensions.height}
        drawMode={drawMode}
        pointType={pointType}
        points={points}
        bbox={bbox}
        onBboxChange={handleBboxChange}
        onPointsChange={handlePointsChange}
        onPointClick={handlePointInteraction}
        className="absolute top-0 left-0 w-full h-full"
        forceRedrawRef={forceRedrawRef}
        redrawTrigger={redrawTrigger}
      />
    )
  ), [videoDimensions, drawMode, pointType, handleBboxChange, handlePointsChange, handlePointInteraction, redrawTrigger]);
  */

  const handleDrop = useCallback((e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();

    const files = Array.from(e.dataTransfer.files);
    files.forEach(file => handleUpload(file));
  }, []);

  const handleFileSelect = useCallback((event: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(event.target.files || []);
    files.forEach(file => handleUpload(file));
  }, []);

  const handleUpload = async (file: File) => {
    try {
      if (file.size > UPLOAD_LIMITS.MAX_FILE_SIZE) {
        setStatus(`${file.name} is too large. Maximum size is 100MB.`, 'error');
        return;
      }

      if (!file.type.startsWith('video/')) {
        setStatus(`${file.name} is not a video file.`, 'error');
        return;
      }

      setStatus(`Uploading ${file.name}...`, 'processing');
      setIsProcessing(true);
      setUploading(true);
      setProgress(0);

      const formData = new FormData();
      formData.append('file', file);
      formData.append('filename', file.name);

      const annotationData = {
        bbox: bbox,
        points: points.map(p => ({
          type: p.type === 'positive' ? 'p' : 'n',
          x: p.x,
          y: p.y
        }))
      };
      formData.append('annotations', JSON.stringify(annotationData));

      const response = await axios.post(
        `${process.env.NEXT_PUBLIC_API_URL}/api/videos/upload`,
        formData,
        {
          headers: {
            'Content-Type': 'multipart/form-data',
            'Authorization': `Bearer ${localStorage.getItem('token')}`
          },
          onUploadProgress: (progressEvent) => {
            const percentCompleted = Math.round(
              (progressEvent.loaded * 100) / (progressEvent.total || 1)
            );
            setProgress(percentCompleted);
          },
        }
      );

      console.log('Upload response:', response.data);

      // Update initialVideo with the newly uploaded video data
      setInitialVideo({
        id: response.data.id,
        title: response.data.title,
        videoUrl: response.data.videoUrl,
        thumbnailUrl: response.data.thumbnailUrl,
        createdAt: response.data.createdAt,
        metadata: response.data.metadata
      });

      setVideoUrl(response.data.videoUrl);
      setProcessedVideoUrl(response.data.processedVideoUrl);
      setMaskVideoUrl(response.data.maskVideoUrl);
      setProgress(100);
      onUploadSuccess();
      setStatus(`Successfully uploaded ${file.name}`, 'success');
    } catch (error) {
      console.error('Upload failed:', error);
      setStatus(
        `Failed to upload ${file.name}: ${error instanceof Error ? error.message : 'Unknown error'}`,
        'error'
      );
    } finally {
      setIsProcessing(false);
      setUploading(false);
    }
  };

  useEffect(() => {
    // Cleanup URLs on component unmount
    return () => {
      if (videoUrl) URL.revokeObjectURL(videoUrl);
      if (processedVideoUrl) URL.revokeObjectURL(processedVideoUrl);
      if (maskVideoUrl) URL.revokeObjectURL(maskVideoUrl);
    };
  }, [videoUrl, processedVideoUrl, maskVideoUrl]);

  // Update the useEffect for initialVideo to prevent loops
  useEffect(() => {
    if (initialVideo?.videoUrl) {
      console.log('Setting video URL from initialVideo:', initialVideo.videoUrl);
      setVideoUrl(initialVideo.videoUrl);
      setMaskVideoUrl(initialVideo.maskUrl || null);
      setGreenscreenUrl(initialVideo.greenscreenUrl || null);
      // Only set bbox if it's different from current
      if (initialVideo.bbox && JSON.stringify(initialVideo.bbox) !== JSON.stringify(bbox)) {
        console.log('Setting bbox from initialVideo:', initialVideo.bbox);
        setBbox(initialVideo.annotation.bbox);
      }
      setPoints(initialVideo.annotation.points || []);
    }
  }, [initialVideo]); // Remove bbox from dependencies

  const handleVideoLoad = (e: React.SyntheticEvent<HTMLVideoElement>) => {
    const video = e.currentTarget;
    console.log('Video loaded with dimensions:', {
      width: video.videoWidth,
      height: video.videoHeight
    });
    setVideoDimensions({
      width: video.videoWidth,
      height: video.videoHeight
    });
  };

  // Modified preview mask generation with better abort handling
  const generatePreviewMask = useCallback(
    debounce(async (currentPoints: Point[], currentBbox: BBox | null) => {
      if (!currentPoints.length && !currentBbox) return;
      console.log('=== Generate Preview Mask ===');
      console.log('Current Points:', currentPoints);
      console.log('Current BBox:', currentBbox);
      const currentFrame = getCurrentFrame();
      console.log('Generating preview mask at frame:', currentFrame);

      try {
        if (abortControllerRef.current) {
          abortControllerRef.current.abort();
          abortControllerRef.current = null;
        }

        const controller = new AbortController();
        abortControllerRef.current = controller;

        // Format the request data
        const requestData = {
          points: {
            positive: currentPoints
              .filter(p => p.type === 'positive')
              .map(p => [p.x, p.y]),
            negative: currentPoints
              .filter(p => p.type === 'negative')
              .map(p => [p.x, p.y])
          },
          bbox: currentBbox,
          current_frame: currentFrame
        };

        console.log('Sending request with data:', JSON.stringify(requestData, null, 2));

        const response = await fetch(
          `${process.env.NEXT_PUBLIC_API_URL}/api/videos/${initialVideo?.id}/preview-mask`,
          {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
              'Authorization': `Bearer ${localStorage.getItem('token')}`
            },
            body: JSON.stringify(requestData),
            signal: controller.signal
          }
        );

        if (!response.ok) {
          const errorText = await response.text();
          console.error('Server response:', errorText);
          throw new Error(`Failed to generate preview: ${errorText}`);
        }

        const data = await response.json();
        console.log('Received mask data:', data);
        setMaskData(data);
        setStatus('Preview mask generated');
      } catch (error) {
        if (error instanceof Error && error.name === 'AbortError') {
          console.log('Request aborted as new request started');
        } else {
          console.error('Error generating preview:', error);
        }
      }
    }, 500),
    [initialVideo?.id]
  );

  // Cleanup abort controller on unmount
  useEffect(() => {
    return () => {
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
        abortControllerRef.current = null;
      }
    };
  }, []);

  const pollTaskStatus = async (taskId: string) => {
    let isPolling = true;
    let pollCount = 0;
    const MAX_POLLS = 300; // Safety limit (5 minutes at 1 second intervals)

    while (isPolling && pollCount < MAX_POLLS) {
      pollCount++;

      try {
        const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/api/tasks/${taskId}`, {
          headers: {
            'Authorization': `Bearer ${localStorage.getItem('token')}`
          }
        });

        if (!response.ok) {
          console.error('Failed to fetch task status');
          setVideoError('Failed to check processing status');
          setIsProcessing(false);
          isPolling = false;
          break;
        }

        const data = await response.json();
        console.log(`Poll #${pollCount} - Task status:`, data.status);

        // Show mask as soon as it's available
        if (data.maskUrl) {
          setMaskVideoUrl(data.maskUrl);
        }

        // Handle different status cases
        switch (data.status) {
          case 'pending':
          case 'processing':
            // Continue polling after delay
            await new Promise(resolve => setTimeout(resolve, 1000));
            break;

          case 'completed':
            console.log('Task completed successfully');
            fetchVideos();
            await fetchCredits();
            setStatus('Masks generated successfully', 'success');
            setIsProcessing(false);
            isPolling = false; // Stop polling
            return data;

          case 'failed':
            console.error('Task failed:', data.errorMessage || 'Processing failed');
            setVideoError(data.errorMessage || 'Processing failed');
            setIsProcessing(false);
            isPolling = false; // Stop polling
            return data;

          default:
            console.warn(`Unknown task status: ${data.status}`);
            // For unknown status, poll a few more times then stop
            if (pollCount > 5) {
              console.log('Stopping polling after 5 attempts with unknown status');
              setIsProcessing(false);
              isPolling = false;
            }
            await new Promise(resolve => setTimeout(resolve, 1000));
            break;
        }
      } catch (error) {
        console.error('Error polling task status:', error);
        setVideoError('Failed to check processing status');
        setIsProcessing(false);
        isPolling = false; // Stop polling on error
      }
    }

    // If we reached the maximum number of polls
    if (pollCount >= MAX_POLLS) {
      console.warn('Reached maximum number of status polls');
      setVideoError('Processing is taking longer than expected');
      setIsProcessing(false);
    }
  };

  const generateFullMasks = async () => {
    console.log('Step 1: Starting mask generation');
    setStatus('Generating masks...', 'processing');
    console.log('Step 2: Current bbox:', bbox);
    console.log('Step 3: Current points:', points);
    console.log('Step 3.5: Current initialVideo:', initialVideo);

    if (!videoUrl || !bbox || !points.length) {
      console.log('Step 4: Missing required data, returning early');
      if (!videoUrl) console.log('Missing videoUrl');
      if (!bbox) console.log('Missing bbox');
      if (!points.length) console.log('Missing points');
      setStatus('Missing required data for mask generation', 'error');
      return;
    }

    if (!initialVideo?.id) {
      console.log('Step 4.5: Missing video ID, cannot generate masks');
      setStatus('Video ID not found. Please try uploading again.', 'error');
      return;
    }

    try {
      console.log('Step 5: Setting generating state to true');
      setIsGenerating(true);

      console.log('Step 6: Getting auth token');
      const token = localStorage.getItem('token');
      const startFrame = getCurrentFrame();

      console.log('Step 7: Preparing request body');
      const requestBody = {
        bbox: Array.isArray(bbox) ? bbox : [bbox.x, bbox.y, bbox.w, bbox.h],
        points: 'positive' in points ? points : {
          positive: points.filter(p => p.type === 'positive').map(p => [p.x, p.y]),
          negative: points.filter(p => p.type === 'negative').map(p => [p.x, p.y])
        },
        super: superMasks,
        method: method,
        start_frame: startFrame
      };
      console.log('Step 8: Request body prepared:', requestBody);

      console.log('Step 9: Making API request');
      const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/api/videos/${initialVideo?.id}/generate-masks`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`
        },
        body: JSON.stringify(requestBody)
      });

      if (!response.ok) {
        console.log('Step 10: API request failed');
        throw new Error('Failed to start generation');
      }

      console.log('Step 11: Getting task ID from response');
      const { taskId } = await response.json();

      console.log('Step 12: Starting task status polling');
      await pollTaskStatus(taskId);

    } catch (error) {
      console.log('Step 14: Error caught:', error);
      console.error('Error generating masks:', error);
      setStatus('Error generating masks', 'error');
      await fetchCredits();
    } finally {
      console.log('Step 15: Cleanup - resetting states');
      setIsGenerating(false);
    }
  };

  // Add this with your other handlers
  const handleTimeUpdate = (e: React.SyntheticEvent<HTMLVideoElement>) => {
    try {
      if (!e?.currentTarget) return;

      const video = e.currentTarget as HTMLVideoElement;
      if (video && !isNaN(video.duration)) {
        const currentProgress = (video.currentTime / video.duration) * 100;
        setProgress(currentProgress);
        setCurrentTime(video.currentTime);
        setDuration(video.duration);
      }
    } catch (error) {
      console.error('Error updating video time:', error);
    }
  };

  const handleSeek = (e: React.MouseEvent<HTMLDivElement>) => {
    const video = videoRef.current;
    const progressBar = progressBarRef.current;

    if (video && progressBar) {
      const rect = progressBar.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const percent = x / rect.width;
      video.currentTime = percent * video.duration;
    }
  };

  const handleLoadedMetadata = (e: React.SyntheticEvent<HTMLVideoElement>) => {
    try {
      if (!e?.currentTarget) return;

      const video = e.currentTarget as HTMLVideoElement;
      if (video && !isNaN(video.duration)) {
        setDuration(video.duration);
      }
    } catch (error) {
      console.error('Error loading video metadata:', error);
    }
  };

  // Add these handlers with your other functions
  const handlePlayPause = () => {
    if (videoRef.current) {
      if (videoRef.current.paused) {
        videoRef.current.play();
        setIsPlaying(true);
      } else {
        videoRef.current.pause();
        setIsPlaying(false);
      }
    }
  };

  const handleMute = () => {
    if (videoRef.current) {
      videoRef.current.muted = !videoRef.current.muted;
      setIsMuted(!isMuted);
    }
  };

  const handleFullscreen = () => {
    if (videoRef.current) {
      videoRef.current.requestFullscreen();
    }
  };

  // Add this helper function to format time
  const formatTime = (seconds: number): string => {
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = Math.floor(seconds % 60);
    return `${minutes.toString().padStart(2, '0')}:${remainingSeconds.toString().padStart(2, '0')}`;
  };

  // Add helper function to convert time to frames
  const getFrameCount = (seconds: number): number => {
    return Math.floor(seconds * DEFAULT_FPS);
  };

  // Add this component for the time display
  const TimeDisplay = () => {
    const formattedCurrentTime = formatTime(currentTime);
    const formattedDuration = formatTime(duration);

    if (showFrames) {
      const currentFrame = getFrameCount(currentTime);
      const totalFrames = getFrameCount(duration);
      return (
        <button
          onClick={() => setShowFrames(false)}
          className="text-white hover:text-blue-400 transition-colors text-sm font-mono"
          title={`${DEFAULT_FPS} FPS`}
        >
          {currentFrame}/{totalFrames}
        </button>
      );
    }

    return (
      <button
        onClick={() => setShowFrames(true)}
        className="text-white hover:text-blue-400 transition-colors text-sm font-mono"
      >
        {formattedCurrentTime} / {formattedDuration}
      </button>
    );
  };

  // Use consistent IDs for your videos
  const MAIN_VIDEO_ID = 'main-video';
  const MASK_VIDEO_ID = 'mask-video';
  const GREENS_VIDEO_ID = 'greenscreen-video';

  // Now you can check any video's state
  const isMainVideoPlaying = videoStore.getInstance(MAIN_VIDEO_ID)?.isPlaying;
  const isMaskVideoPlaying = videoStore.getInstance(MASK_VIDEO_ID)?.isPlaying;

  // Get FPS from metadata or use default
  const getFPS = (video?: Video): number => {
    if (video?.video_metadata) {
      const metadata = video.video_metadata as VideoMetadata;
      return metadata.fps || DEFAULT_FPS;
    }
    return DEFAULT_FPS;
  };

  // Update the loading screen
  if (isLoadingVideo) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-[var(--background)]">
        <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-[var(--accent-purple)]"></div>
      </div>
    );
  }

  return (
    <div className="card p-6">
      {!initialVideo && !videoUrl && (
        <div
          className="border-2 border-dashed border-[var(--border-color)] rounded-lg p-12 text-center
                     hover:border-[var(--accent-purple)] transition-colors cursor-pointer
                     bg-[var(--card-background)]"
          onDragOver={(e) => {
            e.preventDefault();
            e.stopPropagation();
          }}
          onDrop={handleDrop}
          onClick={() => fileInputRef.current?.click()}
        >
          <input
            type="file"
            ref={fileInputRef}
            onChange={handleFileSelect}
            accept="video/*"
            multiple
            className="hidden"
          />
          <div className="space-y-4">
            <div className="text-6xl text-[var(--text-secondary)] flex justify-center">
              <FaCloudUploadAlt className="hover:text-[var(--accent-purple)] transition-colors" />
            </div>
            <p className="text-[var(--text-secondary)]">
              Drag and drop video files here or click to browse
            </p>
            <p className="text-sm text-[var(--text-secondary)] mt-2">
              Supports multiple files up to 100MB each
            </p>
          </div>
        </div>
      )}

      {(initialVideo || videoUrl) && (
        <div className="space-y-6">
          <div className="flex items-center gap-2">
            <div>
              <input
                type="file"
                ref={fileInputRef}
                onChange={handleFileSelect}
                accept="video/*"
                multiple
                className="hidden"
              />
              <button
                onClick={() => fileInputRef.current?.click()}
                className={`flex items-center gap-2 px-3 py-1.5 rounded-full transition-colors text-sm
                  border border-[var(--border-color)] text-[var(--text-secondary)] 
                  hover:border-[var(--accent-purple)] hover:text-[var(--accent-purple)]`}
              >
                <FaCloudUploadAlt className="text-sm" />
                Upload
              </button>
            </div>
            <button
              onClick={() => setActiveTab('annotation')}
              className={`flex items-center gap-2 px-3 py-1.5 rounded-full transition-colors text-sm
                ${activeTab === 'annotation'
                  ? 'bg-[var(--accent-purple)] text-white border border-[var(--accent-purple)]'
                  : 'border border-[var(--border-color)] text-[var(--text-secondary)] hover:border-[var(--accent-purple)] hover:text-[var(--accent-purple)]'
                }`}
            >
              <FaPencilAlt className="text-sm" />
              Annotation
            </button>
            {maskVideoUrl && (
              <button
                onClick={() => setActiveTab('mask')}
                className={`flex items-center gap-2 px-3 py-1.5 rounded-full transition-colors text-sm
                  ${activeTab === 'mask'
                    ? 'bg-[var(--accent-purple)] text-white border border-[var(--accent-purple)]'
                    : 'border border-[var(--border-color)] text-[var(--text-secondary)] hover:border-[var(--accent-purple)] hover:text-[var(--accent-purple)]'
                  }`}
              >
                <FaMask className="text-sm" />
                Mask
              </button>
            )}
            {greenscreenUrl && (
              <button
                onClick={() => setActiveTab('greenscreen')}
                className={`flex items-center gap-2 px-3 py-1.5 rounded-full transition-colors text-sm
                  ${activeTab === 'greenscreen'
                    ? 'bg-[var(--accent-purple)] text-white border border-[var(--accent-purple)]'
                    : 'border border-[var(--border-color)] text-[var(--text-secondary)] hover:border-[var(--accent-purple)] hover:text-[var(--accent-purple)]'
                  }`}
              >
                <FaBorderNone className="text-sm" />
                Green Screen
              </button>
            )}
          </div>

          {activeTab === 'annotation' && (
            <div className="relative bg-[var(--card-background)] rounded-lg overflow-hidden">
              <video
                ref={videoRef}
                className="w-full h-full object-contain"
                src={videoUrl || initialVideo?.videoUrl}
                onLoadedMetadata={(e) => {
                  handleVideoLoad(e);
                  const video = e.currentTarget;
                  if (video && !isNaN(video.duration)) {
                    videoStore.setDuration(MAIN_VIDEO_ID, video.duration);
                  }
                }}
                onTimeUpdate={(e) => {
                  const video = e.currentTarget;
                  if (video && !isNaN(video.duration)) {
                    videoStore.setProgress(MAIN_VIDEO_ID, (video.currentTime / video.duration) * 100);
                    videoStore.setCurrentTime(MAIN_VIDEO_ID, video.currentTime);
                  }
                  handleTimeUpdate(e);  // Keep your existing handler if needed
                }}
                onPlay={() => videoStore.setIsPlaying(MAIN_VIDEO_ID, true)}
                onPause={() => videoStore.setIsPlaying(MAIN_VIDEO_ID, false)}
                controls={false}
              />

              {maskData && !videoStore.getInstance(MAIN_VIDEO_ID)?.isPlaying && (
                <MaskOverlay
                  maskData={annotation.mask_data}
                  videoWidth={videoDimensions.width}
                  videoHeight={videoDimensions.height}
                  className="absolute top-0 left-0 w-full h-full"
                  style={{ zIndex: 20 }}
                />
              )}

              {/* Existing annotation layer with isPlaying condition */}
              {!videoStore.getInstance(MAIN_VIDEO_ID)?.isPlaying && (
                <AnnotationLayer
                  videoWidth={videoDimensions.width}
                  videoHeight={videoDimensions.height}
                  drawMode={drawMode}
                  pointType={pointType}
                  points={annotation.points}
                  bbox={annotation.bbox}
                  onPointsChange={handlePointsChange}
                  onBboxChange={handleBboxChange}
                  onPointClick={handlePointInteraction}
                  getCurrentFrame={getCurrentFrame}
                />
              )}
            </div>
          )}

          {activeTab === 'mask' && (
            <div className="bg-[var(--card-background)] rounded-lg overflow-hidden">
              {maskVideoUrl && (
                <>
                  <div className="relative">
                    <video
                      ref={maskVideoRef}
                      className="w-full h-full object-contain"
                      playsInline
                      key={maskVideoUrl}
                      onError={(e) => {
                        console.error('Video error:', e.currentTarget.error);
                      }}
                      controls={false}
                    >
                      <source src={maskVideoUrl} type="video/mp4" />
                    </video>
                  </div>

                  <PlayBar
                    videoRef={videoRef}
                    videoId={MAIN_VIDEO_ID}
                    onFullscreen={() => videoRef.current?.requestFullscreen()}
                    className="rounded-lg"
                    FPS={videoMetadata?.fps || 30}
                    frameAnnotations={frameAnnotations}  // Add this line
                    onAnnotationClick={handleAnnotationClick}  // Add this line
                  />

                  <div className="p-2 border-t border-[var(--border-color)]">
                    <button
                      onClick={() => handleDownload(maskVideoUrl, `${initialVideo?.title || 'video'}_mask.mp4`)}
                      className="text-[var(--accent-purple)] hover:text-[var(--accent-purple-hover)] flex items-center gap-2"
                    >
                      <FaDownload className="text-sm" />
                      Download
                    </button>
                  </div>
                </>
              )}
            </div>
          )}

          {activeTab === 'greenscreen' && (
            <div className="bg-[var(--card-background)] rounded-lg overflow-hidden">
              {greenscreenUrl && (
                <>
                  <div className="relative">
                    <video
                      ref={greenscreenVideoRef}
                      className="w-full h-full object-contain"
                      playsInline
                      key={greenscreenUrl}
                      onError={(e) => {
                        console.error('Video error:', e.currentTarget.error);
                      }}
                      controls={false}
                    >
                      <source src={greenscreenUrl} type="video/mp4" />
                    </video>
                  </div>

                  <PlayBar
                    videoRef={greenscreenVideoRef}
                    videoId={GREENS_VIDEO_ID}
                    onFullscreen={() => greenscreenVideoRef.current?.requestFullscreen()}
                    FPS={getFPS(initialVideo)}
                    className="border-t border-[var(--border-color)]"
                  />

                  <div className="p-2 border-t border-[var(--border-color)]">
                    <button
                      onClick={() => handleDownload(greenscreenUrl, `${initialVideo?.title || 'video'}_greenscreen.mp4`)}
                      className="text-[var(--accent-purple)] hover:text-[var(--accent-purple-hover)] flex items-center gap-2"
                    >
                      <FaDownload className="text-sm" />
                      Download
                    </button>
                  </div>
                </>
              )}
            </div>
          )}

          {activeTab === 'annotation' && (
            <div className="flex flex-col gap-2">
              <div className="space-y-4">
                {/* Add PlayBar separately */}
                <PlayBar
                  videoRef={videoRef}
                  videoId={MAIN_VIDEO_ID}
                  onFullscreen={() => videoRef.current?.requestFullscreen()}
                  FPS={getFPS(initialVideo)}
                />
              </div>

              <div className="flex items-center gap-2">
                <button
                  onClick={() => setDrawMode('bbox')}
                  className={`flex items-center gap-2 px-3 py-1.5 rounded-full transition-colors text-sm
                    ${drawMode === 'bbox'
                      ? 'bg-[var(--accent-purple)] text-white border border-[var(--accent-purple)]'
                      : 'border border-[var(--border-color)] text-[var(--text-secondary)] hover:border-[var(--accent-purple)] hover:text-[var(--accent-purple)]'
                    }`}
                >
                  <FaSquare className="text-sm" />
                  Box
                </button>
                <button
                  onClick={() => setDrawMode('points')}
                  className={`flex items-center gap-2 px-3 py-1.5 rounded-full transition-colors text-sm
                    ${drawMode === 'points'
                      ? 'bg-[var(--accent-purple)] text-white border border-[var(--accent-purple)]'
                      : 'border border-[var(--border-color)] text-[var(--text-secondary)] hover:border-[var(--accent-purple)] hover:text-[var(--accent-purple)]'
                    }`}
                >
                  <FaDotCircle className="text-sm" />
                  Points
                </button>

                {drawMode === 'points' && (
                  <>
                    <button
                      onClick={() => setPointType('positive')}
                      className={`flex items-center gap-2 px-3 py-1.5 rounded-full transition-colors text-sm tooltip-wrapper
                        ${pointType === 'positive'
                          ? 'bg-[#56dd92] text-white border border-[#56dd92]'
                          : 'border border-[var(--border-color)] text-[var(--text-secondary)] hover:border-[#56dd92] hover:text-[#56dd92]'
                        }`}
                    >
                      <FaPlus className="text-sm" />
                      <span className="tooltip">Positive Points</span>
                    </button>
                    <button
                      onClick={() => setPointType('negative')}
                      className={`flex items-center gap-2 px-3 py-1.5 rounded-full transition-colors text-sm tooltip-wrapper
                        ${pointType === 'negative'
                          ? 'bg-[#dd6456] text-white border border-[#dd6456]'
                          : 'border border-[var(--border-color)] text-[var(--text-secondary)] hover:border-[#dd6456] hover:text-[#dd6456]'
                        }`}
                    >
                      <FaMinus className="text-sm" />
                      <span className="tooltip">Negative Points</span>
                    </button>
                  </>
                )}

                <button
                  onClick={handleUndo}
                  disabled={pointHistory.length === 0}
                  className={`flex items-center gap-2 px-3 py-1.5 rounded-full transition-colors text-sm tooltip-wrapper
                    ${pointHistory.length > 0
                      ? 'border border-[var(--accent-purple)] text-[var(--accent-purple)] hover:bg-[var(--accent-purple)] hover:text-white'
                      : 'border border-[var(--border-color)] text-[var(--text-secondary)] opacity-50 cursor-not-allowed'
                    }`}
                >
                  <FaUndo className="text-sm" />
                  <span className="tooltip">Undo</span>
                </button>

                <button
                  className={`flex items-center gap-2 px-3 py-1.5 rounded-full transition-colors text-sm tooltip-wrapper
                    ${superMasks
                      ? 'border border-[var(--accent-purple)] text-[var(--accent-purple)]'
                      : 'border border-[var(--border-color)] text-[var(--text-secondary)] hover:border-[var(--accent-purple)] hover:text-[var(--accent-purple)]'
                    }`}
                  onClick={(e) => setSuperMasks(!superMasks)}
                >
                  <FaStar className="text-sm" />
                  Super
                  <span className="tooltip">Enable Super Mode</span>
                </button>
                <button
                  onClick={() => {
                    setBbox(null);
                    setPoints([]);
                    setPointHistory([]);
                    forceRedraw();
                  }}
                  className="flex items-center gap-2 px-3 py-1.5 rounded-full transition-colors text-sm tooltip-wrapper
                    border border-[var(--border-color)] text-[var(--text-secondary)] hover:border-[#dd6456] hover:text-[#dd6456]"
                >
                  <FaTrash className="text-sm" />
                  Clear All
                  <span className="tooltip">Clear All Annotations</span>
                </button>

                <button
                  className={`flex items-center gap-2 px-3 py-1.5 rounded-full transition-colors text-sm tooltip-wrapper
                    ${previewBackend
                      ? 'border border-[var(--accent-purple)] text-[var(--accent-purple)]'
                      : 'border border-[var(--border-color)] text-[var(--text-secondary)] hover:border-[var(--accent-purple)] hover:text-[var(--accent-purple)]'
                    }`}
                  onClick={(e) => generatePreviewMask(points, bbox)}
                >
                  <FaEye className="text-sm" />
                  Preview
                  <span className="tooltip">Generate Preview Mask</span>
                </button>

                <select
                  value={method}
                  onChange={(e) => setMethod(e.target.value as 'dual_process' | 'preprocess')}
                  className="px-3 py-1.5 rounded-full text-sm bg-[var(--card-background)] text-[var(--text-secondary)]
                           border border-[var(--border-color)] hover:border-[var(--accent-purple)]"
                >
                  <option value="dual_process" className="bg-[var(--card-background)] text-[var(--text-primary)]">Dual Process</option>
                  <option value="preprocess" className="bg-[var(--card-background)] text-[var(--text-primary)]">Preprocess</option>
                </select>
                <button
                  onClick={generateFullMasks}
                  disabled={points.length === 0}
                  className={`flex items-center gap-2 px-3 py-1.5 rounded-full transition-colors text-sm
                    ${points.length > 0
                      ? 'bg-[var(--accent-purple)] text-white border border-[var(--accent-purple)]'
                      : 'border border-[var(--border-color)] text-[var(--text-secondary)] opacity-50 cursor-not-allowed'
                    }`}
                >
                  <FaPlay className="text-sm" />
                  Generate
                  <span className="tooltip">Generate Masks</span>
                </button>
                <button
                  onClick={() => console.log('Current Points:', points, points.length)}
                  className="flex items-center gap-2 px-3 py-1.5 rounded-full transition-colors text-sm tooltip-wrapper
                    border border-[var(--border-color)] text-[var(--text-secondary)] hover:border-[var(--accent-purple)] hover:text-[var(--accent-purple)]"
                >
                  <FaTerminal className="text-sm" />
                  Log Points
                  <span className="tooltip">Log Points to Console</span>
                </button>

              </div>
            </div>
          )}
        </div>
      )}


    </div>
  );
}

export default VideoUpload;

// Update the tooltip styles
export const tooltipStyles = `
  .tooltip {
    @apply invisible absolute -top-8 left-1/2 -translate-x-1/2 px-2 py-1 rounded text-xs 
    bg-[var(--card-background)] text-[var(--text-primary)] border border-[var(--border-color)]
    whitespace-nowrap;
  }
  
  .tooltip-wrapper:hover .tooltip {
    @apply visible;
  }
`;