import React, { useEffect, useRef } from 'react';

interface MaskData {
    shape: [number, number];  // [height, width]
    data: string;  // hex encoded binary data
}

interface MaskOverlayProps {
    maskData: MaskData;
    videoWidth: number;
    videoHeight: number;
}

export const MaskOverlay: React.FC<MaskOverlayProps> = ({ maskData, videoWidth, videoHeight }) => {
    const canvasRef = useRef<HTMLCanvasElement>(null);

    useEffect(() => {
        const canvas = canvasRef.current;
        if (!canvas || !maskData) return;

        const ctx = canvas.getContext('2d');
        if (!ctx) return;

        // Set canvas size to match video dimensions
        canvas.width = videoWidth;
        canvas.height = videoHeight;

        try {
            // Convert hex string to binary data
            const bytes = new Uint8Array(maskData.data.match(/.{1,2}/g)!.map(byte => parseInt(byte, 16)));

            // Create ImageData matching mask dimensions
            const imageData = ctx.createImageData(maskData.shape[1], maskData.shape[0]);

            // Fill image data
            for (let y = 0; y < maskData.shape[0]; y++) {
                for (let x = 0; x < maskData.shape[1]; x++) {
                    const maskIndex = y * maskData.shape[1] + x;
                    const imageIndex = maskIndex * 4;

                    const maskValue = bytes[maskIndex] || 0;

                    imageData.data[imageIndex] = 255;     // R
                    imageData.data[imageIndex + 1] = 0;   // G
                    imageData.data[imageIndex + 2] = 0;   // B
                    imageData.data[imageIndex + 3] = maskValue * 128; // A (semi-transparent)
                }
            }

            // Clear canvas
            ctx.clearRect(0, 0, canvas.width, canvas.height);

            // Create temporary canvas for scaling
            const tempCanvas = document.createElement('canvas');
            tempCanvas.width = maskData.shape[1];
            tempCanvas.height = maskData.shape[0];
            const tempCtx = tempCanvas.getContext('2d');
            if (!tempCtx) return;

            // Draw mask to temp canvas
            tempCtx.putImageData(imageData, 0, 0);

            // Scale and draw to main canvas
            ctx.drawImage(
                tempCanvas,
                0, 0, maskData.shape[1], maskData.shape[0],
                0, 0, videoWidth, videoHeight
            );

            console.log('Mask rendered:', {
                maskDimensions: maskData.shape,
                videoDimensions: { width: videoWidth, height: videoHeight },
                canvasDimensions: { width: canvas.width, height: canvas.height }
            });

        } catch (error) {
            console.error('Error rendering mask:', error);
        }

    }, [maskData, videoWidth, videoHeight]);

    return (
        <canvas
            ref={canvasRef}
            className="absolute top-0 left-0 w-full h-full"
            style={{
                pointerEvents: 'none',
                objectFit: 'contain'
            }}
        />
    );
}; 