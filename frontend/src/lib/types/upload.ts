export interface UploadResponse {
    uploadUrl: string;
    key: string;
  }
  
  export interface ConfirmUploadResponse {
    success: boolean;
    videoUrl: string;
  }