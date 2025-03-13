# Megaton Roto: Current Implementation Analysis

## 0. Project Structure

### 0.1 Repository Organization
- **Production Repository**: `/home/ec2-user/megaton-roto` (main production codebase)
- **Development Repository**: `/home/ec2-user/megaton-roto-dev` (development branch, ahead of production)

### 0.2 Key Directories and Files
- **Backend Core**: `/backend/app/` - FastAPI application with routes and models
  - Main application: `/backend/app/main.py`
  - Auth implementation: `/backend/app/core/auth.py`
  - Auth routes: `/backend/app/api/auth/routes.py`
  - Database models: `/backend/app/models/`
- **AI Engine**: 
  - `/backend/inference/` - Main inference management system
  - `/backend/inference/sam2/` - SAM2 model implementation with SAMURAI integration
  - `/backend/inference/manager.py` - Core inference orchestration
- **Frontend**: `/frontend/src/` - Next.js frontend application
  - Authentication context: `/frontend/src/contexts/AuthContext.tsx`
  - Auth modal: `/frontend/src/components/auth/AuthModal.tsx`
  - Admin route protection: `/frontend/src/components/auth/AdminRoute.tsx`
  - Main page: `/frontend/src/app/page.tsx`
- **Configuration**: 
  - `/backend/inference/sam2/configs/samurai/` - SAMURAI-specific configuration files
  - `/backend/inference/sam2/sam2/modeling/sam2_base.py` - SAM2 model with SAMURAI integration
- **Services**: `/_services/` - System service definitions for deployment

### 0.3 SAMURAI Implementation
- **Configuration Files**: `/backend/inference/sam2/configs/samurai/sam2.1_hiera_*.yaml`
- **Core Implementation**: `/backend/inference/sam2/sam2/modeling/sam2_base.py` (lines 212-218 initialize SAMURAI)
- **Memory Module**: `/backend/inference/sam2/sam2/modeling/memory_encoder.py`
- **Kalman Filter**: `/backend/inference/sam2/sam2/utils/kalman_filter.py`
- **Video Processing**: `/backend/inference/sam2/sam2/sam2_video_predictor.py`

## 1. Current Architecture

### 1.1 Backend Architecture
- **Framework**: FastAPI (Python) with SQLAlchemy ORM (`/backend/app/main.py` and `/backend/app/db/session.py`)
- **Database**: PostgreSQL with Alembic migrations (`/backend/migrations/`)
- **Authentication**: JWT-based auth with refresh tokens
  - Core auth functions: `/backend/app/core/auth.py`
  - Auth routes: `/backend/app/api/auth/routes.py`
  - JWT token creation: `/backend/app/core/auth.py:create_access_token()`
  - Refresh token management: `/backend/app/core/auth.py:create_refresh_token()` and `validate_refresh_token()`
- **Storage**: AWS S3 for videos, thumbnails, masks, and JPG sequences
  - S3 implementation: `/backend/app/core/storage.py`
- **Processing**: Custom video processing pipeline with SAM2 for rotoscoping
  - Video processor: `/backend/app/core/video_processor.py`
- **Object Tracking**: SAMURAI integration for advanced video object tracking
  - Integration in `/backend/inference/sam2/sam2/modeling/sam2_base.py`
- **Compute**: GPU-accelerated inference for mask generation using PyTorch
- **Payment**: Stripe integration for subscription and credits
  - Stripe routes: `/backend/app/api/stripe/routes.py`
- **Deployment**: Microservice architecture with systemd services (`/_services/`)

### 1.2 Frontend Architecture
- **Framework**: Next.js with React (`/frontend/src/`)
- **State Management**: 
  - Zustand for global state
  - Context API for auth: `/frontend/src/contexts/AuthContext.tsx`
  - Credits context: `/frontend/src/contexts/CreditsContext.tsx`
- **Styling**: Tailwind CSS (`/frontend/tailwind.config.js`)
- **Media Handling**: Custom video player with frame-by-frame controls
  - Video player: `/frontend/src/components/video/VideoPlayer.tsx`
- **UI Components**: 
  - Authentication modal: `/frontend/src/components/auth/AuthModal.tsx`
  - Admin route protection: `/frontend/src/components/auth/AdminRoute.tsx`
  - Video gallery: `/frontend/src/components/video/VideoGallery.tsx`
  - Video upload: `/frontend/src/components/upload/Video.tsx`

### 1.3 Core Workflow
1. Users upload videos (`/frontend/src/components/upload/Video.tsx`), which are transcoded and stored in S3
2. Users annotate videos with bounding boxes and positive/negative points (`/frontend/src/components/annotation/AnnotationTool.tsx`)
3. The system processes videos using SAM2 to generate masks (`/backend/inference/manager.py`)
4. Processed masks and greenscreen videos are made available for download (`/frontend/src/components/video/VideoDownload.tsx`)
5. Users are charged credits based on duration and processing options (`/backend/app/api/videos/routes.py` for credit deduction)

### 1.4 Business Model
- Credit-based system for processing videos
- Tiered membership plans (Free, Basic, Pro, Enterprise) (`/backend/app/config/membership_config.py`)
- Stripe integration for subscriptions and one-time credit purchases (`/backend/app/api/stripe/routes.py`)
- Admin controls for managing users and credits (`/frontend/src/app/admin/page.tsx`)

## 2. Authentication Implementation

### 2.1 Backend Authentication System
- **JWT Generation and Validation**: (`/backend/app/core/auth.py`)
  - Token creation: `create_access_token()` - Creates JWT with configurable expiration
  - Current user middleware: `get_current_user()` - Validates tokens on protected routes
  - Password handling: `verify_password()` and `get_password_hash()` - Secure password storage

- **Refresh Token System**: (`/backend/app/core/auth.py` and `/backend/app/api/auth/routes.py`)
  - Token creation: `create_refresh_token()` - Creates DB-stored refresh tokens
  - Token validation: `validate_refresh_token()` - Checks token validity and expiration
  - Token invalidation: `invalidate_refresh_token()` - Marks tokens as used after refresh

- **Auth Endpoints**: (`/backend/app/api/auth/routes.py`)
  - Login: `/api/auth/login` - Authenticates users and issues tokens
  - Register: `/api/auth/register` - Creates new user accounts
  - Refresh: `/api/auth/refresh` - Handles token refresh requests
  - Email confirmation: `/api/auth/confirm/{token}` - Verifies email addresses

### 2.2 Frontend Authentication System
- **Auth Context Provider**: (`/frontend/src/contexts/AuthContext.tsx`)
  - User state management
  - Token storage and retrieval
  - API client configuration
  - Token refresh logic with race condition prevention
  - HTTP interceptors for handling 401 errors

- **Authentication UI**: (`/frontend/src/components/auth/AuthModal.tsx`)
  - Login form
  - Registration form
  - Email confirmation handling
  - Error display

- **Protected Routes**: (`/frontend/src/components/auth/AdminRoute.tsx`)
  - Admin-only route protection
  - Authentication state check
  - Redirection for unauthorized users

- **API Client**: (`/frontend/src/contexts/AuthContext.tsx`)
  - Centralized axios instance
  - Authorization header management
  - Response interceptors for token refresh

## 3. Recent Authentication Improvements

### 3.1 Race Condition Prevention
- **Mutex Pattern**: (`/frontend/src/contexts/AuthContext.tsx`)
  - Implementation of refresh promise reference to prevent multiple simultaneous refresh attempts
  - Clean state management for refreshing token status

### 3.2 Centralized API Client
- **Unified HTTP Client**: (`/frontend/src/contexts/AuthContext.tsx`)
  - Consolidated API requests using axios
  - Exported apiClient for use throughout the application
  - Consistent header management

### 3.3 Error Handling
- **Enhanced Recovery**: (`/frontend/src/contexts/AuthContext.tsx`)
  - Better error handling for failed refresh attempts
  - Proper cleanup of refresh state after completion or failure

## 4. Key Bottlenecks and Scalability Issues

### 4.1 Authentication Issues
- **Token Storage Security**: Using localStorage instead of HttpOnly cookies (`/frontend/src/contexts/AuthContext.tsx`)
- **Token Expiration Management**: No proactive refresh before expiration
- **API Request Handling**: Multiple authentication approaches across components

### 4.2 Inference Processing
- **GPU Memory**: Current design loads entire videos into memory (`/backend/inference/manager.py`)
- **Sequential Processing**: Tasks are processed in-sequence on a single machine
- **Lack of Queue Management**: No robust job queue system for distributed processing
- **Batch Size Limitations**: Hard-coded batch sizes based on resolution (`/backend/inference/sam2/configs/`)

### 4.3 Storage Management
- **S3 Organization**: Simple object structure without optimization for access patterns (`/backend/app/core/storage.py`)
- **Temporary Files**: Heavy reliance on local temporary storage for processing
- **URL Expiration**: Presigned URLs expire after 1 hour, requiring refresh (`/frontend/src/components/video/VideoGallery.tsx`)

### 4.4 Database Load
- **Polling**: Task status polling creates unnecessary database load (`/frontend/src/components/video/VideoProcessingStatus.tsx`)
- **Session Management**: Database session cleanup may be inconsistent (`/backend/app/api/` routes)
- **Transaction Isolation**: Lack of explicit transaction boundaries

### 4.5 Frontend Performance
- **Large Component State**: Complex state management in annotation components (`/frontend/src/components/annotation/`)
- **Canvas Redrawing**: Inefficient redrawing of canvas elements
- **Video Processing**: All video processing happens client-side before upload (`/frontend/src/components/upload/Video.tsx`)

## 5. Technical Debt and Maintenance Challenges

### 5.1 Code Organization
- **Monolithic Structure**: Backend combines API, processing, and business logic
- **Long Functions**: Several functions exceed reasonable complexity (e.g., `/backend/app/api/videos/routes.py`)
- **Debug Code**: Numerous debug print statements throughout the codebase
- **Commented Code**: Old, commented-out code sections in several files
- **Duplicate Logic**: Repeated code for S3 interactions and video processing

### 5.2 Error Handling
- **Inconsistent Patterns**: Mix of try/except with return values and exception propagation
- **Missing Recovery**: Limited automatic recovery from processing failures
- **Error Messages**: Inconsistent user-facing error messages

### 5.3 Configuration Management
- **Hard-coded Values**: Several hard-coded paths and values throughout the code
- **Environment Variables**: Inconsistent use of environment variables vs. settings
- **Secrets Management**: Potential security issues with AWS credentials handling

### 5.4 Testing
- **Limited Testing**: Minimal automated tests for core functionality
- **Manual Testing**: Heavy reliance on debug code suggests manual testing
- **No CI/CD**: No evidence of continuous integration or deployment

## 6. Recommendations for Future Improvements

### 6.1 Authentication Enhancements
- **Token Expiration Tracking**: Add token expiration checks to refresh preemptively
- **HttpOnly Cookies**: Replace localStorage token storage with secure cookies
- **API Client Standardization**: Ensure all components use the centralized API client
- **JWT Secret Rotation**: Implement proper secret rotation for production security

### 6.2 Short-term Infrastructure Improvements
- Implement proper queue management with Redis or RabbitMQ
- Add detailed logging and monitoring with Prometheus and Grafana
- Refactor for consistent error handling with structured errors
- Clean up debug code and implement proper logging levels

### 6.3 Medium-term Refactoring
- Split monolithic application into microservices (API, Worker, Auth)
- Containerize services with Docker for easier deployment
- Implement Kubernetes for orchestration and scaling
- Add comprehensive unit and integration tests
- Refactor frontend for better code organization and performance

### 6.4 Scalability Enhancements
- Implement worker pool architecture for distributed processing
- Add database read replicas and connection pooling
- Optimize S3 structure with lifecycle policies and access patterns
- Implement auto-scaling based on processing queue depth
- Utilize CDN for faster content delivery

### 6.5 Business Continuity
- Implement backup and disaster recovery procedures
- Add monitoring and alerting for system health
- Create documentation for operations and maintenance
- Implement API versioning for backward compatibility