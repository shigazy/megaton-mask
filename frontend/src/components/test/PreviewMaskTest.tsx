import { useState } from 'react';
import { Point, BBox } from '../types';

export const PreviewMaskTest = () => {
    const [testStatus, setTestStatus] = useState<string>('');
    const [testResults, setTestResults] = useState<any[]>([]);

    // Test data
    const samplePoints: Point[] = [
        { x: 100, y: 100, type: 'positive' },
        { x: 200, y: 200, type: 'negative' }
    ];

    const sampleBbox: BBox = {
        x: 50,
        y: 50,
        w: 200,
        h: 200
    };

    const testPreviewMask = async () => {
        try {
            setTestStatus('Running preview mask test...');
            const results = [];

            // Test 1: Basic preview mask generation
            const result1 = await testBasicPreviewMask();
            results.push({ name: 'Basic Preview Mask', ...result1 });

            // Test 2: Preview mask with empty points
            const result2 = await testEmptyPointsPreviewMask();
            results.push({ name: 'Empty Points Preview Mask', ...result2 });

            // Test 3: Preview mask with invalid bbox
            const result3 = await testInvalidBboxPreviewMask();
            results.push({ name: 'Invalid BBox Preview Mask', ...result3 });

            setTestResults(results);
            setTestStatus('Tests completed');
        } catch (error) {
            setTestStatus(`Test failed: ${error}`);
        }
    };

    const testBasicPreviewMask = async () => {
        const requestData = {
            points: {
                positive: samplePoints
                    .filter(p => p.type === 'positive')
                    .map(p => [p.x, p.y]),
                negative: samplePoints
                    .filter(p => p.type === 'negative')
                    .map(p => [p.x, p.y])
            },
            bbox: [sampleBbox.x, sampleBbox.y, sampleBbox.w, sampleBbox.h]
        };

        const response = await fetch(
            `${process.env.NEXT_PUBLIC_API_URL}/api/videos/test-video-id/preview-mask`,
            {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${localStorage.getItem('token')}`
                },
                body: JSON.stringify(requestData)
            }
        );

        const responseData = await response.json();
        return {
            success: response.ok,
            status: response.status,
            data: responseData
        };
    };

    const testEmptyPointsPreviewMask = async () => {
        // Implementation for empty points test
        // Similar to testBasicPreviewMask but with empty points
    };

    const testInvalidBboxPreviewMask = async () => {
        // Implementation for invalid bbox test
        // Similar to testBasicPreviewMask but with invalid bbox
    };

    return (
        <div className="p-4">
            <h2 className="text-xl font-bold mb-4">Preview Mask Tests</h2>
            <button
                onClick={testPreviewMask}
                className="bg-blue-500 text-white px-4 py-2 rounded"
            >
                Run Tests
            </button>

            <div className="mt-4">
                <p>Status: {testStatus}</p>

                {testResults.map((result, index) => (
                    <div key={index} className="mt-2 p-2 border rounded">
                        <h3 className="font-bold">{result.name}</h3>
                        <p>Success: {result.success ? 'Yes' : 'No'}</p>
                        <p>Status: {result.status}</p>
                        <pre className="mt-2 bg-gray-100 p-2 rounded">
                            {JSON.stringify(result.data, null, 2)}
                        </pre>
                    </div>
                ))}
            </div>
        </div>
    );
}; 