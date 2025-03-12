# SAMURAI Implementation Guide for Megaton Roto

## For AI Assistants: Context and Usage Instructions

This document serves as a comprehensive guide for implementing SAMURAI (Segment Anything Model for Zero-Shot Visual Tracking with Motion-Aware Memory) features into the Megaton Roto codebase. It contains detailed analysis, implementation examples, and references to key files in the SAMURAI repository.

As an AI assistant, when supporting development work on this codebase:

1. Use this document as the primary reference for understanding the SAMURAI integration requirements
2. Reference specific file paths mentioned to locate relevant code in the SAMURAI implementation
3. Follow the implementation plan phases to prioritize work
4. Adapt the provided code examples to fit the existing codebase structure

When asked to help implement a specific SAMURAI feature, first identify which section in this document covers it, understand the current implementation in our codebase, then adapt the SAMURAI approach accordingly.

## Repository Information

**SAMURAI Repository**: https://github.com/yangchris11/samurai  
**Current Implementation**: `/home/ec2-user/megaton-roto-dev/backend/inference/manager.py`

Key SAMURAI files to reference:
- `/sam2/sam2_video_predictor.py` - Core video prediction implementation
- `/sam2/utils/kalman_filter.py` - Kalman filter for motion prediction
- `/sam2/modeling/memory_attention.py` - Memory attention mechanism
- `/sam2/modeling/memory_encoder.py` - Memory encoding implementation
- `/sam2/utils/misc.py` - Utility functions for memory management

## Overview of SAMURAI

SAMURAI enhances SAM2 with specialized tracking capabilities through:

- Motion-aware memory mechanisms for consistent tracking
- Kalman filtering for movement prediction
- Sophisticated frame-to-frame feature propagation
- Zero-shot operation (no additional training required)
- Memory management for long video sequences

The key innovation in SAMURAI is its ability to maintain consistent object tracking through occlusions, fast motion, and challenging scenarios by using a combination of memory features and motion prediction.

## Current Implementation Analysis

Our implementation in `manager.py` provides basic video segmentation using SAM2 with:

- Single-object tracking without sophisticated state management
- Basic memory optimization with manual cleanup
- Simple forward-only tracking with minimal frame reordering
- Chunked processing for memory management
- Batch processing for GPU optimization

The current approach follows the standard SAM2 implementation but lacks SAMURAI's advanced tracking capabilities, which significantly impacts mask consistency across frames.

## Key Missing Components

### 1. Motion-Aware Memory System

**Missing Feature**: SAMURAI's sophisticated memory encoder and state management system that tracks object features across frames.

**Current Implementation**: Uses basic SAM2 propagation without explicit memory features for temporal consistency.

**Source Files in SAMURAI**:
- `/sam2/modeling/memory_encoder.py` - Complete memory encoder implementation
- `/sam2/modeling/memory_attention.py` - Memory attention mechanism

**Integration Solution**: Implement memory encoder that captures frame features and position encoding:

```python
def _run_memory_encoder(self, inference_state, frame_idx, batch_size, masks, scores):
    # Extract image features from current frame
    image_feats = self._get_image_feature(inference_state, frame_idx)
    
    # Encode memory using masks and features
    memory_features, position_encoding = self._encode_new_memory(
        image_feats=image_feats,
        masks=masks,
        scores=scores
    )
    
    # Store in memory bank
    inference_state["memory_bank"][frame_idx] = {
        "features": memory_features,
        "position": position_encoding,
        "confidence": scores
    }
    
    return memory_features, position_encoding
```

**Implementation Notes**:
- Memory features should be stored with position encoding to maintain spatial context
- Use lower precision (bfloat16) for memory features to reduce memory usage
- Memory encoder requires approximately 2GB of GPU memory, plan accordingly

### 2. Kalman Filter for Motion Prediction

**Missing Feature**: SAMURAI uses Kalman filtering to predict object motion between frames, significantly improving tracking during fast movements and occlusions.

**Current Implementation**: No motion prediction mechanism, relies solely on feature similarity.

**Source Files in SAMURAI**:
- `/sam2/utils/kalman_filter.py` - Kalman filter implementation
- `/sam2/sam2_video_predictor.py:predict_next_boxes()` - Usage in prediction pipeline

**Integration Solution**: Add Kalman filter implementation:

```python
from sam2.utils.kalman_filter import KalmanBoxTracker

def _initialize_trackers(self, initial_boxes):
    trackers = []
    for box in initial_boxes:
        tracker = KalmanBoxTracker(box)
        trackers.append(tracker)
    return trackers

def _predict_next_positions(self, trackers):
    predictions = []
    for tracker in trackers:
        pred_box = tracker.predict()[0]
        predictions.append(pred_box)
    return predictions
```

**Implementation Notes**:
- Kalman filter parameters are tuned for video tracking in SAMURAI
- Box format is [x1, y1, x2, y2] in normalized coordinates (0-1)
- Consider motion smoothing for parameters in configs/samurai/*.yaml

### 3. Bidirectional Propagation

**Missing Feature**: SAMURAI supports propagation in both directions from any frame, enabling better handling of occlusions and challenging sequences.

**Current Implementation**: Supports only forward propagation with limited frame reordering.

**Source Files in SAMURAI**:
- `/sam2/sam2_video_predictor.py:propagate_bidirectional()` - Bidirectional implementation
- `/sam2/sam2_video_predictor.py:_propagate_direction()` - Direction-specific propagation

**Integration Solution**: Implement bidirectional tracking logic:

```python
def propagate_bidirectional(self, inference_state, start_frame, boxes):
    # Forward propagation
    forward_results = self._propagate_direction(
        inference_state, 
        start_frame, 
        range(start_frame+1, inference_state["num_frames"]),
        boxes
    )
    
    # Backward propagation
    backward_results = self._propagate_direction(
        inference_state,
        start_frame,
        range(start_frame-1, -1, -1),
        boxes
    )
    
    # Merge results
    all_results = {**backward_results, **{start_frame: inference_state["results"][start_frame]}, **forward_results}
    return all_results
```

**Implementation Notes**:
- Propagation direction should be tracked in state
- Backward propagation should update Kalman filter in reverse
- Results should be merged with proper frame ordering

### 4. Memory Bank Management

**Missing Feature**: SAMURAI uses sophisticated memory management with conditioning vs. non-conditioning frames, selective memory clearing, and reference frame management.

**Current Implementation**: No explicit memory management beyond basic cleanup.

**Source Files in SAMURAI**:
- `/sam2/sam2_video_predictor.py:_update_memory_bank()` - Memory bank updates
- `/sam2/configs/samurai/*.yaml` - Memory configuration parameters

**Integration Solution**: Implement memory bank management:

```python
def _update_memory_bank(self, inference_state, frame_idx, is_key_frame=False):
    memory_bank = inference_state.get("memory_bank", {})
    
    # Update memory bank with current frame
    if is_key_frame:
        memory_bank[frame_idx] = {
            "features": inference_state["current_features"],
            "is_key": True,
            "score": inference_state["scores"][frame_idx]
        }
    
    # Manage memory bank size
    if len(memory_bank) > self.max_memory_frames:
        # Remove lowest confidence non-key frames
        non_key_frames = [f for f, v in memory_bank.items() if not v.get("is_key", False)]
        if non_key_frames:
            lowest_score_frame = min(non_key_frames, key=lambda f: memory_bank[f]["score"])
            del memory_bank[lowest_score_frame]
    
    inference_state["memory_bank"] = memory_bank
```

**Implementation Notes**:
- Key configuration parameters from SAMURAI:
  - `max_memory_frames`: 10 (default)
  - `key_frame_confidence_threshold`: 0.85
  - `memory_potency_decay`: 0.98
- Consider adding memory bank visualization for debugging

### 5. Multi-Object Tracking

**Missing Feature**: SAMURAI supports tracking multiple objects simultaneously with object identity management.

**Current Implementation**: Processes a single object at a time.

**Source Files in SAMURAI**:
- `/sam2/sam2_video_predictor.py:track_multiple_objects()` - Multi-object tracking
- `/sam2/sam2_video_predictor.py:process_objects()` - Object processing

**Integration Solution**: Add multi-object tracking support:

```python
def track_multiple_objects(self, video_path, initial_boxes, object_ids=None):
    if object_ids is None:
        object_ids = list(range(len(initial_boxes)))
    
    # Initialize state for each object
    object_states = {}
    for obj_id, box in zip(object_ids, initial_boxes):
        object_states[obj_id] = self._initialize_object_state(box)
    
    all_results = {obj_id: {} for obj_id in object_ids}
    
    # Process video frames
    for frame_idx in range(self.num_frames):
        frame = self._load_frame(video_path, frame_idx)
        
        # Process each object
        for obj_id in object_ids:
            # Update object state
            obj_state = object_states[obj_id]
            mask, score = self._process_object(frame, obj_state)
            
            # Store results
            all_results[obj_id][frame_idx] = {
                "mask": mask,
                "score": score
            }
            
            # Update Kalman filter
            obj_state["tracker"].update(self._mask_to_box(mask))
    
    return all_results
```

**Implementation Notes**:
- Each object needs a separate state dictionary
- Objects can be processed in parallel or sequentially
- Consider non-overlapping constraints between objects
- Object IDs should be preserved and consistent

### 6. Asynchronous Frame Loading

**Missing Feature**: SAMURAI loads frames asynchronously in the background to overlap I/O and computation.

**Current Implementation**: Loads frames synchronously, potentially causing I/O bottlenecks.

**Source Files in SAMURAI**:
- Not directly implemented in SAMURAI, but a performance enhancement

**Integration Solution**: Implement asynchronous frame loader:

```python
import threading
import queue

class AsyncFrameLoader:
    def __init__(self, video_path, max_queue_size=10):
        self.video_path = video_path
        self.queue = queue.Queue(maxsize=max_queue_size)
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._loader_thread)
        self.current_frame = 0
        self.thread.start()
    
    def _loader_thread(self):
        cap = cv2.VideoCapture(self.video_path)
        while not self.stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                break
            self.queue.put(frame)
        cap.release()
    
    def get_frame(self):
        return self.queue.get()
    
    def close(self):
        self.stop_event.set()
        self.thread.join()
```

**Implementation Notes**:
- Thread safety is essential for queue operations
- Ensure proper cleanup of threads on error conditions
- Consider prefetching strategy based on access patterns

## Performance Optimizations

### 1. Memory Efficiency Improvements

**Source Files in SAMURAI**:
- `/sam2/sam2_video_predictor.py:_optimize_memory_usage()` - Memory optimization functions
- `/sam2/utils/misc.py` - Helper functions for memory management

**Implementation Details**:
- Convert memory features to lower precision (bfloat16) to save GPU memory
- Implement strategic offloading of tensors between CPU and GPU
- Add explicit handling for large masks with disk-based chunking

```python
def _optimize_memory_usage(self, features):
    # Convert to lower precision
    features = features.to(torch.bfloat16)
    
    # Move to appropriate device based on current memory usage
    if torch.cuda.memory_allocated() > self.gpu_memory_threshold:
        features = features.to("cpu")
    else:
        features = features.to("cuda")
    
    return features
```

**Implementation Notes**:
- GPU memory threshold should be configurable
- Consider adding memory pressure monitoring
- Use disk-based chunking for very large arrays

### 2. Image Feature Caching

**Source Files in SAMURAI**:
- `/sam2/sam2_video_predictor.py:_get_image_feature()` - Feature caching implementation

**Implementation Details**: Cache extracted image features to avoid redundant computation:

```python
def _get_image_feature(self, frame_idx, inference_state):
    # Check if features are already in cache
    if frame_idx in inference_state["feature_cache"]:
        return inference_state["feature_cache"][frame_idx]
    
    # Extract features
    image = self._load_frame(frame_idx)
    features = self.backbone(image)
    
    # Store in cache
    inference_state["feature_cache"][frame_idx] = features
    
    return features
```

**Implementation Notes**:
- Cache size should be limited based on available memory
- Consider LRU eviction policy for cache management
- Features should be stored in appropriate precision

### 3. Advanced Batch Processing

**Source Files in SAMURAI**:
- `/sam2/sam2_video_predictor.py:_process_batch()` - Batch processing implementation

**Implementation Details**: Improve batch processing to handle multiple frames and objects efficiently:

```python
def _process_batch(self, frames, boxes, trackers):
    # Prepare batch inputs
    batch_inputs = []
    for frame, box in zip(frames, boxes):
        batch_inputs.append(self._prepare_input(frame, box))
    
    # Run model in a single forward pass
    batch_results = self.model(torch.stack(batch_inputs))
    
    # Update trackers with new results
    for i, result in enumerate(batch_results):
        trackers[i].update(self._extract_box(result["mask"]))
    
    return batch_results
```

**Implementation Notes**:
- Batch size should be adaptive based on resolution
- Consider mixed batch processing for different object types
- Balance batch size against memory usage

## Implementation Plan

### Phase 1: Core SAMURAI Components (2-3 weeks)
1. Implement memory encoder functionality
   - Start with `/sam2/modeling/memory_encoder.py`
   - Add position encoding from `/sam2/modeling/position_encoding.py`
   - Test with single-frame memory

2. Add Kalman filter for motion prediction
   - Copy `/sam2/utils/kalman_filter.py` directly
   - Integrate with box prediction flow
   - Test with simple motion sequences

3. Enhance state management for conditioning frames
   - Create structured state dictionaries
   - Add frame type tracking (conditioning vs. non-conditioning)
   - Implement frame metadata storage

### Phase 2: Advanced Tracking Features (2-3 weeks)
1. Implement bidirectional propagation
   - Start with forward direction implementation
   - Add backward propagation logic
   - Create frame mapping for arbitrary start frames
   
2. Add memory bank management
   - Implement memory storage and retrieval
   - Add confidence scoring for frames
   - Create memory pruning strategies

3. Develop reference frame selection logic
   - Add key frame detection
   - Implement confidence-based frame selection
   - Test with long sequences

### Phase 3: Multi-Object Support (2 weeks)
1. Add object identity management
   - Create object ID tracking system
   - Implement persistence across frames
   - Add object metadata storage

2. Implement multi-object tracking
   - Modify state to handle multiple objects
   - Add parallel processing of objects
   - Implement object-specific memory banks

3. Add non-overlapping mask constraints
   - Implement mask conflict resolution
   - Add priority-based masking
   - Test with overlapping objects

### Phase 4: Performance Optimizations (1-2 weeks)
1. Implement asynchronous frame loading
   - Create async loader class
   - Add prefetching logic
   - Integrate with processing pipeline

2. Add feature caching mechanisms
   - Implement feature cache
   - Add cache eviction policies
   - Optimize cache for common access patterns

3. Optimize memory usage with mixed precision
   - Add precision conversion functions
   - Implement strategic tensor offloading
   - Test with memory-constrained environments

### Phase 5: Testing and Refinement (1-2 weeks)
1. Test on diverse video datasets
   - Use YouTube-VOS for validation
   - Test with challenging sequences
   - Validate with high-resolution content

2. Benchmark performance against original SAMURAI
   - Compare mask quality metrics
   - Measure processing speed
   - Evaluate memory usage

3. Fine-tune parameters for optimal performance
   - Optimize memory thresholds
   - Tune Kalman filter parameters
   - Adjust confidence thresholds

## Expected Improvements

By implementing these SAMURAI components, the current system should achieve:

1. **Better Tracking Quality**: More consistent masks across frames, especially during occlusions and fast motion
2. **Improved Memory Usage**: More efficient handling of long videos and high-resolution content
3. **Higher Performance**: Reduced processing time through optimized caching and async operations
4. **Multi-Object Capability**: Track and segment multiple objects simultaneously
5. **Robustness**: Better handling of challenging scenes with bidirectional propagation

These improvements would significantly enhance the rotoscoping capabilities of the Megaton Roto platform, allowing for more accurate masks with less user intervention.

## For Code Assistants: Implementation Approach

When implementing SAMURAI features:

1. **Prioritize**: Focus on memory encoder and Kalman filter first, as these provide the most significant improvements
2. **Reference Original Code**: Always check the SAMURAI implementation in GitHub for exact details
3. **Test Incrementally**: Add one feature at a time and test thoroughly before proceeding
4. **Memory Management**: Be vigilant about memory management, as SAMURAI is memory-intensive
5. **Configuration**: Make all thresholds and parameters configurable

The most critical files to understand are:
- `/sam2/sam2_video_predictor.py` - Core tracking logic
- `/sam2/modeling/memory_encoder.py` - Memory encoding
- `/sam2/utils/kalman_filter.py` - Motion prediction
- `/sam2/configs/samurai/*.yaml` - Configuration parameters