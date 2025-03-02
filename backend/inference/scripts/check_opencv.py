import cv2

def check_opencv_info():
    # Print OpenCV version
    print(f"OpenCV Version: {cv2.__version__}")
    
    # Check available codecs
    codecs = [
        'mp4v',
        'avc1',
        'h264',
        'x264',
        'xvid',
        'mjpg'
    ]
    
    print("\nTesting available codecs:")
    for codec in codecs:
        try:
            fourcc = cv2.VideoWriter_fourcc(*codec)
            print(f"Codec {codec}: Available (fourcc code: {fourcc})")
        except Exception as e:
            print(f"Codec {codec}: Not available ({str(e)})")

if __name__ == "__main__":
    check_opencv_info() 