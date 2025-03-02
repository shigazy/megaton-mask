export const UPLOAD_LIMITS = {
    MAX_FILE_SIZE: 2 * 1024 * 1024 * 1024, // 2GB
    ALLOWED_TYPES: ['video/*'], // Accept all video types
} as const; 