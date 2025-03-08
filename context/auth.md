# Authentication in Megaton Roto

## Overview

Megaton Roto uses a JWT-based authentication system for securing API endpoints and managing user sessions. The system supports user registration, login, email confirmation, and different authorization levels (regular users vs. admin users).

## Key Components

### User Authentication Flow

1. **Registration**: Users register with email and password
   - Password is hashed before storage
   - A confirmation token is generated and stored
   - Confirmation email is sent to the user

2. **Email Confirmation**: 
   - User clicks link with confirmation token
   - Account is marked as confirmed upon successful verification

3. **Login**: Users authenticate with email and password
   - System verifies credentials and issues a JWT token
   - Token contains user ID and permission level

4. **Session Management**: 
   - Frontend stores JWT token in secure HTTP-only cookies
   - Token is included in Authorization header for API requests
   - Tokens have expiration times requiring periodic re-authentication

## Implementation Details

### Database Schema

The User model contains authentication-related fields:
- `hashed_password`: Securely stored password hash
- `is_confirmed`: Boolean indicating email verification status
- `confirmation_token`: Token for email verification
- `super_user`: Boolean indicating admin privileges

### API Endpoints

Authentication is handled through these endpoints:
- `POST /api/auth/register`: Register a new user
- `POST /api/auth/login`: Authenticate and receive JWT token
- `POST /api/auth/confirm/{token}`: Confirm email registration
- `POST /api/auth/resend-confirmation`: Request new confirmation email

### Authorization Middleware

The backend uses middleware to:
- Validate JWT tokens on protected routes
- Check user permissions for admin-only actions
- Handle token expiration and renewal

### Security Measures

- Passwords are hashed using secure algorithms
- JWT tokens are signed with a secret key
- Email verification reduces risk of account abuse
- Admin actions are logged for audit purposes

## Admin Authentication

Administrators (super_users) have additional privileges:
- Access to admin-only endpoints
- Ability to manage other users' accounts
- Actions are logged in the AdminAction table for accountability

## Session Lifecycle

1. User logs in and receives token
2. Token is used for authorized API requests
3. Token expires after a set time period
4. User must re-authenticate to receive a new token