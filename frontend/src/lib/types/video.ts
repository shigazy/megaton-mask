export interface Video {
    id: string;
    title: string;
    videoUrl: string;
    thumbnailUrl: string;
    createdAt: string;
    bbox: Array<number>;
    points: Array<number>;
    mask_data: Array<number>;
}