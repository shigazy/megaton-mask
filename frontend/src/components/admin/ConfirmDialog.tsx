interface ConfirmDialogProps {
    isOpen: boolean;
    title: string;
    message: string;
    confirmText?: string;
    cancelText?: string;
    onConfirm: () => void;
    onCancel: () => void;
    isDestructive?: boolean;
}

export const ConfirmDialog = ({
    isOpen,
    title,
    message,
    confirmText = 'Confirm',
    cancelText = 'Cancel',
    onConfirm,
    onCancel,
    isDestructive = false
}: ConfirmDialogProps) => {
    if (!isOpen) return null;

    return (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
            <div className="bg-[var(--card-background)] rounded-lg p-6 max-w-md w-full">
                <h3 className="text-lg font-semibold mb-2">{title}</h3>
                <p className="text-[var(--text-secondary)] mb-6">{message}</p>
                <div className="flex justify-end gap-3">
                    <button
                        onClick={onCancel}
                        className="px-4 py-2 border border-[var(--border-color)] rounded-lg 
                                 hover:border-[var(--accent-purple)] transition-colors"
                    >
                        {cancelText}
                    </button>
                    <button
                        onClick={onConfirm}
                        className={`px-4 py-2 rounded-lg text-white transition-colors
                                ${isDestructive
                                ? 'bg-red-500 hover:bg-red-600'
                                : 'bg-[var(--accent-purple)] hover:bg-[var(--accent-purple-hover)]'}`}
                    >
                        {confirmText}
                    </button>
                </div>
            </div>
        </div>
    );
}; 