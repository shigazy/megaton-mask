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
  annotation: FrameAnnotation;
  forceRedrawRef: MutableRefObject<(() => void) | undefined>;
  redrawTrigger: number;
  getCurrentFrame: () => number;
  onBboxDragStart: () => void;
  onBboxDragEnd: () => void;
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
  [frameNumber: string]: {
    points: Point[];
    bbox: number[] | null;
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
  setDrawMode,
  getCurrentFrame,
  onBboxChange,
  onPointsChange,
  onPointClick,
  annotation,
  className,
  forceRedrawRef,
  redrawTrigger,
  onBboxDragStart,
  onBboxDragEnd
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

    // Get current frame information
    const currentFrame = getCurrentFrame();
    console.log('Current Points on Click:', points);
    // Ensure points is an array before using findIndex
    const pointsArray = Array.isArray(points) ? points : [];

    if (drawMode === 'bbox') {
      startPoint.current = { x, y };
      setIsDrawing(true);
      onBboxDragStart && onBboxDragStart();
      const pointIndex = pointsArray.findIndex(point => {
        const dx = point.x - x;
        const dy = point.y - y;
        return Math.sqrt(dx * dx + dy * dy) < 5;
      });
      setHoveredPointIndex(pointIndex);
      if (pointIndex !== -1) {
        const newPoints = pointsArray.filter((_, i) => i !== pointIndex);
        onPointsChange(newPoints);
        setDrawMode('points');
      }
    } else if (drawMode === 'points') {
      // console.log('Drawing points', console.log(points));
      const pointIndex = pointsArray.findIndex(point => {
        const dx = point.x - x;
        const dy = point.y - y;
        return Math.sqrt(dx * dx + dy * dy) < 5;
      });
      setHoveredPointIndex(pointIndex);

      if (pointIndex !== -1) {
        const newPoints = pointsArray.filter((_, i) => i !== pointIndex);
        onPointsChange(newPoints);
      } else {
        const newPoint = { x, y, type: pointType };
        onPointsChange([...pointsArray, newPoint]);
      }
    }
  }, [drawMode, pointType, points, onPointsChange, videoWidth, videoHeight, onBboxDragStart]);

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

    //console.log('Points on Move:', points);

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
      // console.log('Setting canvas size:', rect);
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
      // console.log('Canvas resized');
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

    // Debug the current frame and annotation data
    const currentFrame = getCurrentFrame();
    // console.log('Drawing points for frame:', currentFrame, points, annotation);

    const rect = canvas.getBoundingClientRect();
    const scaleX = rect.width / videoWidth;
    const scaleY = rect.height / videoHeight;

    ctx.clearRect(0, 0, rect.width, rect.height);

    // Draw bbox
    const bboxToDraw = localBbox || bbox;
    if (bboxToDraw) {
      // Handle both array and object formats for bbox
      const [x, y, w, h] = Array.isArray(bboxToDraw)
        ? bboxToDraw
        : [bboxToDraw.x, bboxToDraw.y, bboxToDraw.w, bboxToDraw.h];

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

    // Ensure points is always an array
    let pointsToRender = [];

    if (Array.isArray(points)) {
      pointsToRender = points;
    } else {
      // If points is not an array, log the issue and use an empty array
      console.error('Points is not an array:', points, typeof points);

      // Try to get points from annotation if available
      if (annotation && annotation[currentFrame.toString()] && Array.isArray(annotation[currentFrame.toString()].points)) {
        console.log('Using points from annotation instead');
        pointsToRender = annotation[currentFrame.toString()].points;
      }
    }

    // Draw points
    pointsToRender.forEach((point, index) => {
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
  }, [points, localBbox, bbox, hoveredPointIndex, videoWidth, videoHeight, annotation, getCurrentFrame]);

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

export const VideoUpload = ({ onUploadSuccess, fetchVideos, initialVideo, setInitialVideo, fps }: VideoUploadProps) => {
  const { setStatus } = useStatus();
  const [uploading, setUploading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [videoUrl, setVideoUrl] = useState<string | null>(initialVideo?.videoUrl || null);
  const [processedVideoUrl, setProcessedVideoUrl] = useState<string | null>(null);
  const [maskVideoUrl, setMaskVideoUrl] = useState<string | null>(null);
  const [drawMode, setDrawMode] = useState<'bbox' | 'points'>('bbox');
  const [pointType, setPointType] = useState<'positive' | 'negative'>('positive');
  const [annotation, setAnnotation] = useState<FrameAnnotation>(initialVideo?.annotation || {});
  const [bbox, setBbox] = useState<BBox | null>(initialVideo?.bbox || (annotation["0"]?.bbox || null));
  const [points, setPoints] = useState<Point[]>(initialVideo?.points || (annotation["0"]?.points || []));
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
  const [isDraggingBbox, setIsDraggingBbox] = useState(false);
  const [maskGenerationProgress, setMaskGenerationProgress] = useState(0);
  const [maskGenerationTaskId, setMaskGenerationTaskId] = useState<string | null>(null);
  const [isMaskGenerating, setIsMaskGenerating] = useState(false);

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
      // console.log('Forcing canvas redraw...');
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
      const frameKey = frameToSave.toString();
      console.log('Saving at frame:', frameKey);

      // Create new annotation object with updated frame data
      const updatedAnnotation = {
        ...annotation,
        [frameKey]: {
          points: newPoints || points,
          bbox: bboxArray,
          mask_data: newMaskData || maskData
        },
      };

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
          annotation: updatedAnnotation
        })
      });

      // Update local state with the new annotation
      setAnnotation(updatedAnnotation);

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      console.log('Save response from backend:', data);

    } catch (error) {
      console.error('Error saving changes:', error);
      setStatus('Error saving changes');
    }
  }, [initialVideo?.id, points, bbox, maskData, annotation]);

  const getCurrentFrame = useCallback((): number => {
    const video = videoRef.current;
    if (!video || isNaN(video.currentTime)) return 0;

    // Get FPS from video metadata or use default
    const fps = initialVideo?.video_metadata?.fps || DEFAULT_FPS;

    // Use floor to ensure consistent frame calculation
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

  // Create a debounced save function (outside of any other functions)
  const debouncedSaveChanges = useMemo(
    () => debounce((frame: number, newBbox: BBox | null, newPoints: Point[]) => {
      console.log('Debounced save triggered');
      saveChanges(newPoints, newBbox, undefined, frame);
    }, 500),
    []
  );

  // Modify your handleBboxChange function
  const handleBboxChange = useCallback((newBbox: BBox | null) => {
    console.log('handleBboxChange called with:', newBbox);
    const frame = getCurrentFrame();

    // Update local state immediately for responsive UI
    setBbox(newBbox);

    // Update the annotation for the current frame
    setAnnotation(prev => {
      const newAnnotation = { ...prev };
      if (!newAnnotation[frame.toString()]) {
        newAnnotation[frame.toString()] = {
          points: [],
          bbox: null,
          mask_data: null
        };
      }
      newAnnotation[frame.toString()].bbox = newBbox ? [newBbox.x, newBbox.y, newBbox.w, newBbox.h] : null;
      return newAnnotation;
    });

    console.log('Setting bbox at frame:', frame);

    // Use the debounced save instead of immediate save
    debouncedSaveChanges(frame, newBbox, points);

  }, [getCurrentFrame, points, debouncedSaveChanges]);

  // Add these handlers to your AnnotationLayer component
  const handleBboxDragStart = useCallback(() => {
    setIsDraggingBbox(true);
  }, []);

  const handleBboxDragEnd = useCallback(() => {
    setIsDraggingBbox(false);
    // Force a final save when dragging ends
    const frame = getCurrentFrame();
    saveChanges(points, bbox, undefined, frame);
  }, [getCurrentFrame, bbox, points]);

  // Update the handleAnnotationClick function to use the exact frame
  const handleAnnotationClick = useCallback((data: { frame: number, exactFrame?: string }) => {
    // Use the exactFrame string key if provided, otherwise convert frame to string
    const frameKey = data.exactFrame || data.frame.toString();
    console.log(`Clicked on marker - using frame key: ${frameKey}`);

    // Check if annotation data exists for this exact frame key
    if (annotation && annotation[frameKey]) {
      console.log(`Loading annotation data for frame ${frameKey}:`, annotation[frameKey]);

      // Set bbox and points directly from the stored annotation
      if (annotation[frameKey].bbox) {
        // Handle array format for bbox
        const bboxArray = annotation[frameKey].bbox;
        if (Array.isArray(bboxArray) && bboxArray.length === 4) {
          const newBbox = {
            x: bboxArray[0],
            y: bboxArray[1],
            w: bboxArray[2],
            h: bboxArray[3]
          };
          console.log('Setting bbox to:', newBbox);
          setBbox(newBbox);
        }
      }

      if (annotation[frameKey].points && annotation[frameKey].points.length > 0) {
        console.log('Setting points to:', annotation[frameKey].points);
        setPoints(annotation[frameKey].points);
      }

      // Seek to the exact frame with a slight offset to prevent rounding issues
      if (videoRef.current) {
        const fps = initialVideo?.video_metadata?.fps || DEFAULT_FPS;
        const exactTime = (data.frame / fps) + 0.001;
        videoRef.current.currentTime = exactTime;
        console.log(`Setting video time to ${exactTime.toFixed(3)}s for frame ${data.frame}`);
      }

      // Force redraw to ensure annotations are displayed
      setTimeout(forceRedraw, 50); // Small delay to ensure video has updated
    } else {
      console.warn(`No annotation data found for frame key ${frameKey}`);
    }
  }, [annotation, forceRedraw, initialVideo?.video_metadata?.fps]);

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

      // Create custom axios instance with progress tracking
      const uploadInstance = axios.create({
        baseURL: process.env.NEXT_PUBLIC_API_URL,
        headers: {
          'Content-Type': 'multipart/form-data',
          'Authorization': `Bearer ${localStorage.getItem('token')}`
        }
      });

      // Add upload progress event listener
      const response = await uploadInstance.post(
        '/api/videos/upload',
        formData,
        {
          onUploadProgress: (progressEvent) => {
            // Enhanced progress calculation
            const total = progressEvent.total || file.size;
            const loaded = progressEvent.loaded;

            // Calculate percentage - divide into phases:
            // - 0-90%: Actual upload
            // - 90-100%: Server processing
            const uploadPercentage = Math.min(90, Math.round((loaded * 90) / total));

            setProgress(uploadPercentage);

            if (uploadPercentage === 90) {
              setStatus(`Processing ${file.name}...`, 'processing');
            } else {
              setStatus(`Uploading ${file.name} (${uploadPercentage}%)...`, 'processing');
            }
          },
        }
      );

      // After upload completes, show processing
      setStatus(`Finalizing ${file.name}...`, 'processing');
      setProgress(95);

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

      // Handle authentication errors specifically
      if (axios.isAxiosError(error) && error.response?.status === 401) {
        setStatus(`Session expired. Please refresh the page and try again.`, 'error');

        // Force authentication refresh
        try {
          await refreshToken();
          setStatus(`Authentication refreshed. Please try uploading again.`, 'warning');
        } catch (refreshError) {
          setStatus(`Authentication failed. Please log in again.`, 'error');
        }
      } else {
        setStatus(
          `Failed to upload ${file.name}: ${error instanceof Error ? error.message : 'Unknown error'}`,
          'error'
        );
      }
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

      // Check if annotation exists
      if (initialVideo.annotation) {
        // Set the full annotation object
        setAnnotation(initialVideo.annotation);

        // Get the first frame (or frame 0 if it exists)
        const frameKey = initialVideo.annotation["0"] ? "0" : Object.keys(initialVideo.annotation)[0];

        if (frameKey) {
          // Only set bbox if it's different from current
          if (initialVideo.annotation[frameKey]?.bbox &&
            JSON.stringify(initialVideo.annotation[frameKey].bbox) !== JSON.stringify(bbox)) {
            console.log('Setting bbox from initialVideo:', initialVideo.annotation[frameKey].bbox);
            setBbox(initialVideo.annotation[frameKey].bbox ? {
              x: initialVideo.annotation[frameKey].bbox[0],
              y: initialVideo.annotation[frameKey].bbox[1],
              w: initialVideo.annotation[frameKey].bbox[2],
              h: initialVideo.annotation[frameKey].bbox[3]
            } : null);
          }

          // Set points from the frame
          if (initialVideo.annotation[frameKey]?.points) {
            setPoints(initialVideo.annotation[frameKey].points || []);
          }
        }
      }
    }
  }, [initialVideo]); // Remove bbox from dependencies

  const handleVideoLoad = (e: React.SyntheticEvent<HTMLVideoElement>) => {
    const video = e.currentTarget;
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
          points: annotation[currentFrame].points,
          bbox: annotation[currentFrame].bbox,
          current_frame: currentFrame
        };

        console.log('Sending request with data [preview-mask]:', JSON.stringify(requestData, null, 2));

        try {
          // Proper axios response handling
          const response = await axios.post(
            `${process.env.NEXT_PUBLIC_API_URL}/api/videos/${initialVideo?.id}/preview-mask`,
            requestData,
            {
              headers: {
                'Authorization': `Bearer ${localStorage.getItem('token')}`
              },
              signal: controller.signal
            }
          );

          // With axios, data is already parsed 
          console.log('Received mask data:', response.data);
          // Store the mask in the annotation object for the current frame
          const currentFrame = getCurrentFrame();
          setAnnotation(prev => {
            const updatedAnnotation = { ...prev };
            if (!updatedAnnotation[currentFrame]) {
              updatedAnnotation[currentFrame] = { points: [], bbox: null, mask_data: null };
            }
            updatedAnnotation[currentFrame].mask_data = response.data;
            return updatedAnnotation;
          });
          setMaskData(response.data);
          setStatus('Preview mask generated');
        } catch (error) {
          if (axios.isCancel(error)) {
            console.log('Request aborted as new request started');
          } else {
            console.error('Error generating preview:', error);
            setStatus('Failed to generate preview mask', 'error');
          }
        }
      } catch (error) {
        console.error('Error in preview mask generation:', error);
        setStatus('Error processing mask request', 'error');
      }
    }, 500),
    [initialVideo?.id, annotation]
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

  const pollTaskStatus = useCallback(async (taskId: string) => {
    console.log('Polling task status for taskId:', taskId);
    if (!taskId) return;

    setIsMaskGenerating(true);
    setMaskGenerationTaskId(taskId);
    setMaskGenerationProgress(0);
    console.log('Starting to poll task status for taskId:', taskId);

    const pollInterval = setInterval(async () => {
      try {
        console.log('Polling task status for taskId:', taskId);
        const response = await axios.get(
          `${process.env.NEXT_PUBLIC_API_URL}/api/tasks/${taskId}`,
          {
            headers: {
              'Authorization': `Bearer ${localStorage.getItem('token')}`
            }
          }
        );

        const data = response.data;
        console.log('Task status response:', data);

        // Extract progress information from the response
        if (data.progress !== undefined) {
          const newProgress = Math.round(data.progress);
          console.log('Setting progress from data.progress:', newProgress);
          setMaskGenerationProgress(newProgress);
          setStatus(`Generating masks: ${newProgress}%`, 'processing');
        } else if (data.status === 'processing' && data.message) {
          // Try to extract progress from message like "Processed frame 50/110"
          const match = data.message.match(/Processed frame (\d+)\/(\d+)/);
          if (match && match[1] && match[2]) {
            const current = parseInt(match[1]);
            const total = parseInt(match[2]);
            const progress = Math.round((current / total) * 100);
            console.log('Setting progress from message parsing:', progress, 'Current:', current, 'Total:', total);
            setMaskGenerationProgress(progress);
            setStatus(`Generating masks: ${progress}%`, 'processing');
          }
        }

        // Check if task is complete
        if (data.status === 'completed' || data.status === 'failed') {
          console.log('Task completed with status:', data.status);
          clearInterval(pollInterval);
          setIsMaskGenerating(false);

          if (data.status === 'completed') {
            console.log('Mask generation completed successfully');
            setStatus('Mask generation completed!', 'success');
            // Refresh video data if needed
            fetchVideos && fetchVideos();
          } else {
            console.log('Mask generation failed:', data.message || 'Unknown error');
            setStatus('Mask generation failed: ' + (data.message || 'Unknown error'), 'error');
          }
        }
      } catch (error) {
        console.error('Error polling task status:', error);
        clearInterval(pollInterval);
        setIsMaskGenerating(false);
        setStatus('Error checking mask generation status', 'error');
      }
    }, 1000); // Poll every second

    // Clean up interval when component unmounts
    return () => {
      console.log('Cleaning up poll interval for taskId:', taskId);
      clearInterval(pollInterval);
    };
  }, [setStatus, fetchVideos]);

  const generateFullMasks = async () => {
    console.log('Step 1: Starting mask generation');
    setStatus('Generating masks...', 'processing');
    const currentFrame = getCurrentFrame();
    const mask_bbox = annotation[currentFrame.toString()].bbox;
    const mask_points = annotation[currentFrame.toString()].points;
    console.log('Step 2: Current bbox:', mask_bbox);
    console.log('Step 3: Current points:', mask_points);
    console.log('Step 3.5: Current initialVideo:', initialVideo);

    if (!videoUrl || !mask_bbox || !mask_points.length) {
      console.log('Step 4: Missing required data, returning early');
      if (!videoUrl) {
        console.log('Missing videoUrl');
        setStatus('Missing video URL for mask generation', 'error');
      } else if (!mask_bbox) {
        console.log('Missing bbox');
        setStatus('Missing bounding box for mask generation. Please draw a bounding box on the video.', 'error');
      } else if (!mask_points.length) {
        console.log('Missing points');
        setStatus('Missing annotation points for mask generation. Please place positive and negative points on the video.', 'error');
      }
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
        bbox: Array.isArray(mask_bbox) ? mask_bbox : [mask_bbox.x, mask_bbox.y, mask_bbox.w, mask_bbox.h],
        points: 'positive' in mask_points ? mask_points : {
          positive: mask_points.filter(p => p.type === 'positive').map(p => [p.x, p.y]),
          negative: mask_points.filter(p => p.type === 'negative').map(p => [p.x, p.y])
        },
        super: superMasks,
        //TO DO: Deprecate method
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
      const data = await response.json();
      console.log('Step 12: Task data:', data);
      console.log('Step 12: Task ID:', data.taskId);

      // Start polling if we have a task ID
      if (data.taskId) {
        pollTaskStatus(data.taskId);
      }

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

  // Add this useEffect to debug annotation data
  useEffect(() => {
    console.log('Current annotation data:', annotation);
    console.log('Number of frames with annotations:', Object.keys(annotation).length);

    // Log details of each frame's annotation
    Object.keys(annotation).forEach(frame => {
      console.log(`Frame ${frame} data:`, annotation[frame]);
    });
  }, [annotation]);

  // Add this useEffect to debug initial annotation loading
  useEffect(() => {
    if (initialVideo?.annotation) {
      console.log('Initial annotation data:', initialVideo.annotation);
      console.log('Frames with annotations:', Object.keys(initialVideo.annotation));

      // Ensure annotation state is properly set from initialVideo
      setAnnotation(initialVideo.annotation);
    }
  }, [initialVideo?.annotation]);

  // Replace the direct addEventListener with this useEffect
  useEffect(() => {
    // Make sure video timeupdate properly updates the current frame display
    const videoElement = videoRef.current;

    if (!videoElement) return;

    const handleTimeUpdate = () => {
      const frame = getCurrentFrame();
      const frameStr = frame.toString();

      // Log when we're at a frame that has annotation
      if (annotation && annotation[frameStr]) {
        console.log(`At frame ${frame} with annotation:`, annotation[frameStr]);
      }
    };

    videoElement.addEventListener('timeupdate', handleTimeUpdate);

    // Clean up function to remove the event listener when component unmounts
    return () => {
      videoElement.removeEventListener('timeupdate', handleTimeUpdate);
    };
  }, [videoRef, annotation, getCurrentFrame]); // Add dependencies

  // Add this effect to enhance the snackbar with progress information
  useEffect(() => {
    // Skip if not generating masks or if we're at 0% progress
    if (!isMaskGenerating || maskGenerationProgress <= 0) return;

    // Find the active snackbar in the DOM
    const snackbars = document.querySelectorAll('.snackbar');
    const activeSnackbar = Array.from(snackbars).find(
      snackbar => snackbar.classList.contains('processing')
    );

    if (activeSnackbar) {
      // Check if we already added a progress bar
      let progressBar = activeSnackbar.querySelector('.mask-gen-progress');

      if (!progressBar) {
        // Create progress bar if it doesn't exist
        const progressBarContainer = document.createElement('div');
        progressBarContainer.className = 'mask-gen-progress w-full mt-2';

        progressBarContainer.innerHTML = `
          <div class="w-full bg-gray-700 rounded-full h-1">
            <div class="bg-blue-400 h-1 rounded-full transition-all duration-300" style="width: ${maskGenerationProgress}%"></div>
          </div>
          <div class="text-xs mt-1 text-gray-200">${maskGenerationProgress}% complete</div>
        `;

        // Find the content area of the snackbar and append the progress bar
        const contentArea = activeSnackbar.querySelector('.snackbar-content');
        if (contentArea) {
          contentArea.appendChild(progressBarContainer);
        }
      } else {
        // Update existing progress bar
        const progressFill = progressBar.querySelector('.bg-blue-400');
        const progressText = progressBar.querySelector('.text-xs');

        if (progressFill && progressText) {
          progressFill.setAttribute('style', `width: ${maskGenerationProgress}%`);
          progressText.textContent = `${maskGenerationProgress}% complete`;
        }
      }
    }
  }, [isMaskGenerating, maskGenerationProgress]);

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
                  maskData={annotation[getCurrentFrame().toString()]?.mask_data}
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
                  annotation={annotation}
                  points={annotation[getCurrentFrame().toString()]?.points || []}
                  bbox={annotation[getCurrentFrame().toString()]?.bbox || null}
                  onPointsChange={handlePointsChange}
                  onBboxChange={handleBboxChange}
                  onPointClick={handlePointInteraction}
                  onBboxDragStart={handleBboxDragStart}
                  onBboxDragEnd={handleBboxDragEnd}
                  setDrawMode={setDrawMode}
                  getCurrentFrame={getCurrentFrame}
                  forceRedrawRef={forceRedrawRef}
                  redrawTrigger={redrawTrigger}
                  className="absolute top-0 left-0 w-full h-full"
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
                    videoRef={maskVideoRef}
                    videoId={MASK_VIDEO_ID}
                    onFullscreen={() => maskVideoRef.current?.requestFullscreen()}
                    className="rounded-lg"
                    FPS={getFPS(initialVideo)}
                    annotation={annotation}
                    onAnnotationClick={handleAnnotationClick}
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
                    annotation={annotation}
                    onAnnotationClick={handleAnnotationClick}
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
                  annotation={annotation}
                  onAnnotationClick={handleAnnotationClick}
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

                {/* TODO: Deprecate method selector 
                <select
                  value={method}
                  onChange={(e) => setMethod(e.target.value as 'dual_process' | 'preprocess')}
                  className="px-3 py-1.5 rounded-full text-sm bg-[var(--card-background)] text-[var(--text-secondary)]
                           border border-[var(--border-color)] hover:border-[var(--accent-purple)]"
                >
                  <option value="dual_process" className="bg-[var(--card-background)] text-[var(--text-primary)]">Dual Process</option>
                  <option value="preprocess" className="bg-[var(--card-background)] text-[var(--text-primary)]">Preprocess</option>
                </select>
                */}
                <button
                  onClick={generateFullMasks}
                  disabled={points.length === 0 || isMaskGenerating}
                  className={`flex items-center gap-2 px-3 py-1.5 rounded-full transition-colors text-sm tooltip-wrapper
                    ${points.length > 0 && !isMaskGenerating
                      ? 'bg-[var(--accent-purple)] text-white border border-[var(--accent-purple)]'
                      : 'border border-[var(--border-color)] text-[var(--text-secondary)] opacity-50 cursor-not-allowed'
                    }`}
                >
                  {isMaskGenerating ? (
                    <>
                      <div className="animate-spin rounded-full h-3 w-3 border-t-2 border-white mr-1"></div>
                      {console.log('Rendering progress:', maskGenerationProgress)}
                      {maskGenerationProgress}%
                    </>
                  ) : (
                    <>
                      <FaPlay className="text-sm" />
                      Generate
                    </>
                  )}
                  <span className="tooltip">Generate Masks</span>
                </button>
                <button
                  onClick={() => console.log('Current Points:', points, points.length, 'Annotation:', annotation)}
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