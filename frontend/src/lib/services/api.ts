import axios from 'axios';

const api = axios.create({
  baseURL: process.env.NEXT_PUBLIC_API_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

export const uploadService = {
  getUploadUrl: async (fileName: string, fileType: string) => {
    return api.post('/upload/presign', { fileName, fileType });
  },

  confirmUpload: async (videoKey: string) => {
    return api.post('/upload/confirm', { videoKey });
  },
};