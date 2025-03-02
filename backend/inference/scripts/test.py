import cv2
print(cv2.videoio_registry.getBackends())  # This will show available backends
print(cv2.videoio_registry.getCameraBackends())  # This will show available camera backends
