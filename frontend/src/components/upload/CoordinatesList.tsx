interface CoordinatesListProps {
    points: Point[];
    onDeletePoint: (index: number) => void;
}

export const CoordinatesList: React.FC<CoordinatesListProps> = ({ points, onDeletePoint }) => {
    return (
        <div className="absolute right-0 top-0 bg-white p-4 shadow-lg rounded-l-lg max-h-full overflow-y-auto">
            <h3 className="text-lg font-semibold mb-4">Coordinates</h3>
            <ul className="space-y-2">
                {points.map((point, index) => (
                    <li key={index} className="flex items-center justify-between">
                        <span className="mr-4">
                            ({Math.round(point.x)}, {Math.round(point.y)})
                        </span>
                        <button
                            onClick={() => onDeletePoint(index)}
                            className="text-red-500 hover:text-red-700"
                        >
                            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                            </svg>
                        </button>
                    </li>
                ))}
            </ul>
        </div>
    );
}; 