def scheduled_chunk_cleanup(max_age_hours=24):
    """Scheduled job to clean up old chunk files"""
    import os
    import time
    import glob
    
    # Define where chunks are stored
    chunk_dir = "/path/to/your/chunks/directory"
    
    # Find all chunk files
    chunk_files = glob.glob(os.path.join(chunk_dir, "*.npy"))
    
    # Get current time
    current_time = time.time()
    
    # Check each file
    for chunk_file in chunk_files:
        # Get file modification time
        file_mod_time = os.path.getmtime(chunk_file)
        
        # If file is older than max_age_hours
        if (current_time - file_mod_time) > (max_age_hours * 3600):
            try:
                os.remove(chunk_file)
                print(f"Removed old chunk file: {chunk_file}")
            except Exception as e:
                print(f"Failed to remove old chunk file {chunk_file}: {e}")
                
    print(f"Scheduled cleanup complete, checked {len(chunk_files)} files") 


def check_disk_space(min_free_gb=10):
    """Check if there's enough disk space available"""
    import shutil
    
    # Get the disk where chunks are stored
    chunk_dir = "/path/to/your/chunks/directory"
    
    # Get disk usage statistics
    disk_usage = shutil.disk_usage(chunk_dir)
    
    # Convert to GB
    free_gb = disk_usage.free / (1024**3)
    
    if free_gb < min_free_gb:
        print(f"WARNING: Low disk space! Only {free_gb:.1f}GB free")
        # Trigger emergency cleanup
        emergency_cleanup()
        
def emergency_cleanup():
    """Delete old chunks to free space in low-disk situations"""
    # Similar to scheduled_chunk_cleanup but more aggressive
    # Delete oldest files first until enough space is freed