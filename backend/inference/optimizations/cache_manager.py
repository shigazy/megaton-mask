from functools import lru_cache
import hashlib
import json

class MaskCache:
    def __init__(self, max_size=1000):
        self.cache = {}
        self.max_size = max_size
        
    def _generate_key(self, video_id, points, bbox):
        # Create a deterministic key from the input parameters
        data = {
            'video_id': video_id,
            'points': points,
            'bbox': bbox
        }
        return hashlib.md5(json.dumps(data, sort_keys=True).encode()).hexdigest()
        
    def get(self, video_id, points, bbox):
        key = self._generate_key(video_id, points, bbox)
        return self.cache.get(key)
        
    def set(self, video_id, points, bbox, mask):
        key = self._generate_key(video_id, points, bbox)
        
        # Implement LRU eviction if cache is full
        if len(self.cache) >= self.max_size:
            # Remove oldest item
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
            
        self.cache[key] = mask 