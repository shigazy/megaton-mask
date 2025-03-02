import { create } from 'zustand';

interface VideoInstance {
    isPlaying: boolean;
    isMuted: boolean;
    progress: number;
    currentTime: number;
    duration: number;
    showFrames: boolean;
}

interface VideoStore {
    instances: Record<string, VideoInstance>;

    // Actions for managing instances
    registerInstance: (id: string) => void;
    removeInstance: (id: string) => void;

    // Actions for updating instance state
    setIsPlaying: (id: string, playing: boolean) => void;
    setIsMuted: (id: string, muted: boolean) => void;
    setProgress: (id: string, progress: number) => void;
    setCurrentTime: (id: string, time: number) => void;
    setDuration: (id: string, duration: number) => void;
    setShowFrames: (id: string, show: boolean) => void;

    // Utility to get instance state
    getInstance: (id: string) => VideoInstance | undefined;
}

const DEFAULT_INSTANCE: VideoInstance = {
    isPlaying: false,
    isMuted: false,
    progress: 0,
    currentTime: 0,
    duration: 0,
    showFrames: false,
};

export const useVideoStore = create<VideoStore>()((set, get) => ({
    instances: {},

    registerInstance: (id) => {
        set((state) => ({
            instances: {
                ...state.instances,
                [id]: DEFAULT_INSTANCE,
            },
        }));
    },

    removeInstance: (id) => {
        set((state) => {
            const { [id]: _, ...rest } = state.instances;
            return { instances: rest };
        });
    },

    setIsPlaying: (id, playing) => {
        set((state) => ({
            instances: {
                ...state.instances,
                [id]: { ...state.instances[id], isPlaying: playing },
            },
        }));
    },

    setIsMuted: (id, muted) => {
        set((state) => ({
            instances: {
                ...state.instances,
                [id]: { ...state.instances[id], isMuted: muted },
            },
        }));
    },

    setProgress: (id, progress) => {
        set((state) => ({
            instances: {
                ...state.instances,
                [id]: { ...state.instances[id], progress },
            },
        }));
    },

    setCurrentTime: (id, time) => {
        set((state) => ({
            instances: {
                ...state.instances,
                [id]: { ...state.instances[id], currentTime: time },
            },
        }));
    },

    setDuration: (id, duration) => {
        set((state) => ({
            instances: {
                ...state.instances,
                [id]: { ...state.instances[id], duration },
            },
        }));
    },

    setShowFrames: (id, show) => {
        set((state) => ({
            instances: {
                ...state.instances,
                [id]: { ...state.instances[id], showFrames: show },
            },
        }));
    },

    getInstance: (id) => {
        return get().instances[id];
    },
})); 