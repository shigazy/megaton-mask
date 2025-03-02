interface TooltipProps {
    position: { x: number; y: number; text: string } | null;
}

export const CoordinateTooltip: React.FC<TooltipProps> = ({ position }) => {
    if (!position) return null;

    return (
        <div
            className="absolute z-50 bg-gray-800 text-white px-2 py-1 rounded text-sm"
            style={{
                left: position.x,
                top: position.y,
                transform: 'translate(-50%, -100%)',
                pointerEvents: 'none'
            }}
        >
            {position.text}
        </div>
    );
}; 