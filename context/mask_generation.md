 [video.tsx]
 
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


//tasks.py
  async def process_video_masks(
    video_id: str,
    bbox: List[float],
    task_id: str,
    points: Optional[Dict[str, List[List[float]]]] = None,
    super: bool = False,
    method: str = "default",
    start_frame: int = 0
) -> None:
    """
    Process video masks using the optimized InferenceManager
    """
    from inference.manager import inference_manager
    
    temp_files = []
    db = None
    try:
        # Get video and task from database
        db = SessionLocal()
        video = db.query(Video).filter(Video.id == video_id).first()
        task = db.query(Task).filter(Task.id == task_id).first()

        # If we used JPG sequence and don't have a local video file
        # We need to download the original video for greenscreen processing
        logger.info("Downloading original video for greenscreen processing")
        temp_video = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        temp_files.append(temp_video.name)
        
        s3_client.download_file(BUCKET_NAME, video.s3_key, temp_video.name)
        original_video_path = temp_video.name
        
        # Get fps from the original video file
        cap = cv2.VideoCapture(original_video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) if cap.isOpened() else 24
        # Get frame count from the original video file
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.isOpened() else 0
        print(f"Extracted frame count from original video: {frame_count}")
        
        # Check if the frame count is valid
        if frame_count <= 0:
            logger.warning(f"Invalid frame count ({frame_count}) detected in video {video_id}")
        cap.release()
        print(f"Extracted fps from original video: {fps}")
        
        if not video:
            raise ValueError(f"Video {video_id} not found")
        if not task:
            raise ValueError(f"Task {task_id} not found")

        # Check if we have a JPG sequence
        if video.jpg_dir_key:
            logger.info(f"Using JPG sequence from {video.jpg_dir_key}")
            process_video_path = None  # We don't need a local video path
            jpg_dir_key = video.jpg_dir_key
        else:
            # We need to download the video file
            logger.info("No JPG sequence available, using video file")
            process_video_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            temp_files.append(process_video_path.name)
            
            if super and video.forward_reverse_key:
                logger.info("Using forward-reverse video")
                s3_client.download_file(BUCKET_NAME, video.forward_reverse_key, process_video_path.name)
            else:
                logger.info("Using original video")
                s3_client.download_file(BUCKET_NAME, video.s3_key, process_video_path.name)
                
            jpg_dir_key = None

        # Update task status
        task.status = "processing"
        db.commit()

        # Create temporary file for mask output
        temp_mask = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        temp_files.append(temp_mask.name)

        # Generate masks using inference manager
        masks_or_chunk_paths = await inference_manager.generate_full_video_masks(
            video_path=process_video_path.name if process_video_path else None,
            points=points,
            bbox=bbox,
            super_mode=super,
            method=method,
            start_frame=start_frame,
            progress_callback=lambda current, total: update_task_progress(db, task_id, current, total),
            jpg_dir_key=jpg_dir_key
        )
        
        # Debug the received masks
        print(f"[Tasks.py] Received masks of type {type(masks_or_chunk_paths)}")
        if isinstance(masks_or_chunk_paths, np.ndarray):
            print(f"[Tasks.py] Mask array shape: {masks_or_chunk_paths.shape}")
        elif isinstance(masks_or_chunk_paths, list):
            print(f"[Tasks.py] Received {len(masks_or_chunk_paths)} chunk paths")

        # Check if we got chunk paths or actual mask array
        if isinstance(masks_or_chunk_paths, list) and all(isinstance(p, str) for p in masks_or_chunk_paths):
            # We got chunk paths, which means masks were too large and saved to disk
            logger.info(f"Received {len(masks_or_chunk_paths)} mask chunks")
            
            # Create temporary file for combined masks
            temp_combined = tempfile.NamedTemporaryFile(delete=False, suffix='.npy')
            temp_files.append(temp_combined.name)
            
            # Load and combine chunks
            mask_chunks = []
            for chunk_path in masks_or_chunk_paths:
                chunk = np.load(chunk_path)
                mask_chunks.append(chunk)
                os.remove(chunk_path)  # Clean up the chunk file
            
            # Combine chunks
            masks = np.concatenate(mask_chunks, axis=0)
            np.save(temp_combined.name, masks)
            
            # Load the saved combined masks
            masks = np.load(temp_combined.name)
            logger.info(f"Combined masks shape: {masks.shape}")

            # After loading/combining masks:
            debug_dir = f"/home/ec2-user/megaton-roto-dev/backend/tmp/sam2_debug/tasks_masks_{str(uuid.uuid4())[:8]}"
            os.makedirs(debug_dir, exist_ok=True)
            print(f"[Tasks.py] Saving ALL masks before video encoding to {debug_dir}")
            # Save ALL masks before encoding
            for i in range(len(masks)):
                sample_mask = masks[i]
                debug_path = f"{debug_dir}/before_encoding_{i:08d}.jpg"
                cv2.imwrite(debug_path, sample_mask)
            print(f"[Tasks.py] Saved all {len(masks)} masks before encoding")
        else:
            # We got the masks directly
            masks = masks_or_chunk_paths
            logger.info(f"Received masks directly, shape: {getattr(masks, 'shape', 'unknown')}")

        # Memory optimization: Process masks in chunks to avoid OOM
        logger.info(f"Saving masks to video file, shape: {masks.shape}")
        height, width = masks[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_mask.name, fourcc, fps, (width, height))
        
        # Write ALL masks directly - no chunking
        masks_written = 0
        for i, mask in enumerate(masks):
            if mask is not None and mask.size > 0:
                out.write(mask)
                masks_written += 1
                # Log progress
                if i % 50 == 0 or i == len(masks) - 1:
                    print(f"[Tasks.py] Written {i+1}/{len(masks)} masks to video")

        # Release the writer
        out.release()
        print(f"[Tasks.py] Completed writing {masks_written}/{len(masks)} masks to video")

        # Verify the output video
        mask_cap = cv2.VideoCapture(temp_mask.name)
        mask_frames = int(mask_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        mask_cap.release()
        print(f"[Tasks.py] Mask video contains {mask_frames} frames according to OpenCV")

        # Ensure we wrote all frames
        if mask_frames != len(masks):
            print(f"[Tasks.py] WARNING: Mask video frame count ({mask_frames}) doesn't match expected ({len(masks)})")
            print(f"[Tasks.py] This will cause issues in the greenscreen process")

        # Save the mask video to debug for inspection
        debug_dir = "/home/ec2-user/megaton-roto-dev/backend/tmp/sam2_debug"
        os.makedirs(debug_dir, exist_ok=True)
        mask_debug_path = f"{debug_dir}/final_mask_video_{str(uuid.uuid4())[:8]}.mp4"
        shutil.copy(temp_mask.name, mask_debug_path)
        print(f"[Tasks.py] Saved copy of mask video to {mask_debug_path}")


        
        # Convert to H.264 for normal videos
        temp_h264 = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        temp_files.append(temp_h264.name)
        convert_to_h264(temp_mask.name, temp_h264.name)
        upload_path = temp_h264.name

        # Upload results to S3
        logger.info("Starting S3 upload...")
        mask_key = f"users/{video.user_id}/masks/{video_id}_mask.mp4"
        
        # Check if object exists
        try:
            s3_client.head_object(Bucket=BUCKET_NAME, Key=mask_key)
            logger.info(f"Object {mask_key} exists in S3, will overwrite")
        except:
            logger.info(f"Object {mask_key} does not exist in S3")
            
        s3_client.upload_file(
            upload_path,
            BUCKET_NAME,
            mask_key,
            ExtraArgs={'ContentType': 'video/mp4'}
        )
        logger.info(f"S3 upload completed for {mask_key}")

        # Update video record with mask
        video.mask_key = mask_key
        
        # Update task status
        task.status = "processing greenscreen"
        task.completed_at = datetime.utcnow()
        formatted_points = format_points(points)
        video.bbox = bbox
        video.points = formatted_points
        db.commit()

        # Split the greenscreen creation into a separate task to reduce memory pressure
        # First, make sure we have a valid video path for the greenscreen process

        # Now call greenscreen with the valid path
        await create_greenscreen_async(video_id, original_video_path, upload_path, task_id)

        # NOW we can safely clear masks from memory
        del masks
        gc.collect()
        torch.cuda.empty_cache()
        logger.info("Masks saved to video file, cleared from memory")
        
    except Exception as e:
        logger.error(f"Error occurred: {str(e)}")
        if task and db:
            task.status = "failed"
            task.error_message = str(e)
            db.commit()
        raise
    finally:
        inference_manager.cleanup()  # Clean up GPU resources
        if db:
            db.close()
        # Clean up temporary files
        for temp_file in temp_files:
            try:
                if os.path.exists(temp_file):
                    print(f"Removing temporary file/directory: {temp_file}")
                    if os.path.isdir(temp_file):
                        shutil.rmtree(temp_file)
                    else:
                        os.unlink(temp_file)
            except Exception as e:
                logger.error(f"Error removing temporary file {temp_file}: {str(e)}")


    async def generate_full_video_masks(
        self,
        video_path: str,
        points: Optional[Dict[str, List[List[float]]]],
        bbox: List[float],
        super_mode: bool = False,
        method: str = "default",
        start_frame: int = 0,
        progress_callback = None,
        jpg_dir_key: Optional[str] = None  # New parameter
    ) -> np.ndarray:
        """Generate masks for all frames in a video or JPG sequence"""
        temp_dirs = []  # Track temporary directories to clean up
        
        try:
            # Clear memory before starting
            self.cleanup()
            torch.cuda.empty_cache()
            gc.collect()
            
            # Use regular model for full masks
            predictor = await self.initialize(is_preview=False)
            self.log_memory_usage("After initialize in generate_full_video_masks")
            
            # Get video info (either from video file or first JPG)
            if jpg_dir_key:
                print(f"[Manager.py] Processing from JPG sequence: {jpg_dir_key}")
                # Download the JPG sequence from S3
                jpg_dir = download_jpg_sequence(jpg_dir_key)
                temp_dirs.append(jpg_dir)
                
                # Get the frame count and dimensions from the JPG sequence
                jpg_files = sorted([f for f in os.listdir(jpg_dir) if f.endswith('.jpg')])
                total_frames = len(jpg_files)
                
                if total_frames == 0:
                    raise ValueError(f"No JPG frames found in directory: {jpg_dir}")
                    
                # Get dimensions from the first frame
                first_frame = cv2.imread(os.path.join(jpg_dir, jpg_files[0]))
                height, width = first_frame.shape[:2]
                
                print(f"[Manager.py] JPG sequence info: {total_frames} frames, {width}x{height}")
                
                # Create a reordered sequence for SAM2
                sam2_dir, sam2_to_original = prepare_sam2_jpg_sequence(
                    jpg_dir, 
                    total_frames, 
                    start_frame
                )
                temp_dirs.append(sam2_dir)
                
                # SAM2 will use this directory path instead of a video
                process_path = sam2_dir
                
            else:
                # Fall back to using the video file
                print(f"[Manager.py] Processing from video file: {video_path}")
                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    print("[Manager.py] Error: Unable to open video file")
                else:
                    print("[Manager.py] Video file opened successfully")
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                cap.release()
                
                print(f"[Manager.py] Video info: {total_frames} frames, {width}x{height}")
                process_path = video_path
                
                # Create frame mapping for regular video processing
                _, sam2_to_original = create_sam2_frame_mapping(total_frames, start_frame)
            
            # Add debug print to verify callback is passed correctly
            print(f"[Manager.py] Progress callback provided: {progress_callback is not None}")
            
            # Determine optimal batch size based on resolution
            if width * height > 1280 * 720:
                batch_size = 10
            else:
                batch_size = 25
                
            print(f"[Manager.py] Using batch size: {batch_size} for resolution {width}x{height}")

            # Run the heavy computation in a thread pool
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, self._generate_masks,
                predictor, process_path, points, bbox, super_mode, method,
                total_frames, height, width, batch_size, 0,  # Use 0 as start_frame for SAM2
                progress_callback, sam2_to_original)  # Pass the mapping
            
            print("[Manager.py] Masks generation completed")
            self.log_memory_usage("After masks generation")
            
            # The result will be correctly ordered based on the original frames
            if progress_callback:
                try:
                    progress_callback(total_frames, total_frames)
                    print(f"[Manager.py] Final progress update: {total_frames}/{total_frames}")
                except Exception as e:
                    print(f"[Manager.py] Error in progress callback: {e}")
            
            return result

        except Exception as e:
            print(f"[Manager.py] Error in generate_full_video_masks: {e}")
            raise
        finally:
            # Clean up temporary directories
            for temp_dir in temp_dirs:
                try:
                    if os.path.exists(temp_dir):
                        shutil.rmtree(temp_dir)
                        print(f"[Manager.py] Removed temporary directory: {temp_dir}")
                except Exception as e:
                    print(f"[Manager.py] Error removing temporary directory {temp_dir}: {e}")
            
            self.cleanup()
            torch.cuda.empty_cache()
            gc.collect()

def create_sam2_frame_mapping(
    total_frames: int, 
    start_frame: int,
    super_mode: bool
) -> Tuple[Dict[int, int], Dict[int, int], List[Tuple[int, int]]]:
    """
    Create mapping plans for reordering frames for SAM2 processing, returning both
    dictionary and tuple-based mappings.
    
    Args:
        total_frames: Total number of frames in the sequence
        start_frame: The user-specified starting frame for mask propagation
    
    Returns:
        Tuple containing:
            - original_to_sam2: Mapping from original frame indices to SAM2 frame indices (with offset key for duplicates)
            - sam2_to_original: Mapping from SAM2 frame indices to original frame indices
            - original_to_sam2_tuples: List of (orig_idx, sam2_idx) pairs for handling duplicates cleanly
    """
    # Basic validation
    if start_frame < 0 or start_frame >= total_frames:
        raise ValueError(f"start_frame {start_frame} out of bounds (total frames: {total_frames})")
    
    original_to_sam2 = {}  # Dictionary mapping
    sam2_to_original = {}
    original_to_sam2_tuples = []  # Tuple-based mapping
    
    if super_mode:
        sam2_idx = 0
        
        # First pass: Forward from 0 to start_frame
        for i in range(start_frame + 1):  # +1 to include start_frame
            orig_idx = i
            original_to_sam2[orig_idx] = sam2_idx
            original_to_sam2_tuples.append((orig_idx, sam2_idx))
            sam2_to_original[sam2_idx] = orig_idx
            sam2_idx += 1
        
        # Second pass: Backward from start_frame to 0
        for i in range(start_frame, -1, -1):  # Start from start_frame, go to 0
            orig_idx = i
            original_to_sam2_tuples.append((orig_idx, sam2_idx))
            # For the first frame in backward pass (which is start_frame again)
            # we need a special case for dictionary mapping to avoid overwriting
            if i == start_frame:
                original_to_sam2[orig_idx + total_frames] = sam2_idx  # Use offset for dict
            else:
                original_to_sam2[orig_idx] = sam2_idx  # Regular mapping
            sam2_to_original[sam2_idx] = orig_idx
            sam2_idx += 1
    else:
        # Regular mode: simple 1:1 mapping
        for i in range(total_frames):
            original_to_sam2[i] = i
            original_to_sam2_tuples.append((i, i))
            sam2_to_original[i] = i
    
    # Save mapping info for debugging
    debug_dir = "/home/ec2-user/megaton-roto-dev/backend/tmp/sam2_debug"
    os.makedirs(debug_dir, exist_ok=True)
    debug_id = str(uuid.uuid4())[:8]
    debug_path = f"{debug_dir}/mapping_{debug_id}.json"
    
    with open(debug_path, 'w') as f:
        json.dump({
            "original_to_sam2": {str(k): v for k, v in original_to_sam2.items()},
            "sam2_to_original": {str(k): v for k, v in sam2_to_original.items()},
            "original_to_sam2_tuples": original_to_sam2_tuples,
            "total_frames": total_frames,
            "start_frame": start_frame
        }, f, indent=2)
    print(f"[JPG Sequence] Saved mapping debug info to {debug_path}")
    
    return original_to_sam2, sam2_to_original, original_to_sam2_tuples

def prepare_sam2_jpg_sequence(
    src_dir: str, 
    total_frames: int, 
    start_frame: int,
    super_mode: bool
) -> Tuple[str, Dict[int, int]]:
    """
    Create a new JPG sequence in the order SAM2 should process it.
    
    Args:
        src_dir: Source directory containing the original JPG sequence
        total_frames: Total number of frames 
        start_frame: The user-specified starting frame for mask propagation
        super_mode: Whether to use super mode (duplicate frames)
    
    Returns:
        Tuple containing:
            - Path to the directory with the reordered sequence
            - Mapping from SAM2 frame indices to original frame indices
    """
    # Create mapping plans
    original_to_sam2, sam2_to_original, original_to_sam2_tuples = create_sam2_frame_mapping(
        total_frames, start_frame, super_mode
    )
    
    # Create a temporary directory for the reordered sequence
    dest_dir = tempfile.mkdtemp()
    
    # Get list of all jpg files in the source directory
    jpg_files = sorted([f for f in os.listdir(src_dir) if f.endswith('.jpg')])
    
    if len(jpg_files) != total_frames:
        print(f"Warning: Found {len(jpg_files)} files but expected {total_frames} frames")
    
    # Create a mapping from frame number to filename
    frame_to_filename = {}
    
    # Try to extract frame numbers using regex pattern
    frame_pattern = re.compile(r'frame_(\d+)\.jpg')
    
    for filename in jpg_files:
        match = frame_pattern.match(filename)
        if match:
            frame_num = int(match.group(1))
            frame_to_filename[frame_num] = filename
    
    # If regex didn't work, assume files are already numbered sequentially
    if not frame_to_filename:
        print("[JPG Sequence] Couldn't extract frame numbers from filenames, assuming sequential ordering")
        for i, filename in enumerate(jpg_files):
            frame_to_filename[i] = filename
    
    # Total number of SAM2 frames (double for super_mode)
    total_sam2_frames = len(sam2_to_original)
    print(f"[JPG Sequence] Creating sequence with {total_sam2_frames} frames")
    
    # Create the reordered sequence with sequential numeric filenames (0.jpg, 1.jpg, etc.)
    for sam2_idx in range(total_sam2_frames):
        orig_idx = sam2_to_original[sam2_idx]
        
        if orig_idx in frame_to_filename:
            src_file = os.path.join(src_dir, frame_to_filename[orig_idx])
            # Create a sequential filename starting from 0.jpg
            dest_file = os.path.join(dest_dir, f"{sam2_idx}.jpg")
            shutil.copy(src_file, dest_file)
            print(f"[JPG Sequence] Copied original frame {orig_idx} to SAM2 frame {sam2_idx}")
        else:
            print(f"[JPG Sequence] Warning: Couldn't find original frame {orig_idx}")
    
    # Save all frames for debugging
    debug_dir = f"/home/ec2-user/megaton-roto-dev/backend/tmp/sam2_debug/frames_{str(uuid.uuid4())[:8]}"
    os.makedirs(debug_dir, exist_ok=True)
    
    for sam2_idx in range(total_sam2_frames):
        src_file = os.path.join(dest_dir, f"{sam2_idx}.jpg")
        if os.path.exists(src_file):
            shutil.copy(src_file, f"{debug_dir}/sam2_frame_{sam2_idx:08d}.jpg")
    
    print(f"[JPG Sequence] Saved all {total_sam2_frames} frames to {debug_dir}")
    print(f"[JPG Sequence] Created reordered sequence with {total_sam2_frames} frames in {dest_dir}")
    
    # Save sequence details for debugging
    with open(os.path.join(debug_dir, "sequence_info.json"), 'w') as f:
        json.dump({
            "total_original_frames": total_frames,
            "total_sam2_frames": total_sam2_frames,
            "start_frame": start_frame,
            "super_mode": super_mode,
            "sam2_to_original": {str(k): v for k, v in sam2_to_original.items()}
        }, f, indent=2)
    
    return dest_dir, sam2_to_original
