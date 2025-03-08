# Google Authentication Implementation: Next Steps

## Overview

This document outlines the steps needed to complete the Google Authentication implementation for Megaton Roto after the code changes have been made.

## 1. Create a Google OAuth Client

1. Go to the [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select an existing one
3. Navigate to "APIs & Services" > "Credentials"
4. Click "Create Credentials" > "OAuth client ID"
5. Select "Web application" as the application type
6. Add a name for your OAuth client (e.g., "Megaton Roto")
7. Add authorized JavaScript origins:
   - `https://app.megaton.ai` (production)
   - `http://localhost:3000` (development)
8. Add authorized redirect URIs:
   - `https://app.megaton.ai/auth/callback` (production)
   - `http://localhost:3000/auth/callback` (development)
9. Click "Create" to generate your client ID and client secret
10. Save these credentials securely

## 2. Configure Environment Variables

Update your environment variables in both development and production environments:

### Backend Environment Variables

Add these to your .env file and AWS Parameter Store:

```
GOOGLE_CLIENT_ID=your_google_client_id
GOOGLE_CLIENT_SECRET=your_google_client_secret
GOOGLE_REDIRECT_URI=https://app.megaton.ai/auth/callback (or http://localhost:3000/auth/callback for dev)
```

### Frontend Environment Variables

Add this to your Next.js .env.local file:

```
NEXT_PUBLIC_GOOGLE_CLIENT_ID=your_google_client_id
```

## 3. Run the Database Migration

Run the migration script to update the database schema:

```bash
cd /path/to/megaton-roto/backend
python migrations/add_google_auth.py
```

This will:
- Make the `hashed_password` field nullable (required for Google-only accounts)
- Add the `google_id` column to store Google user IDs
- Add the `auth_providers` JSON column to track authentication methods
- Update existing users with default values

## 4. Test the Implementation

Test the following scenarios:

1. **New User Registration**:
   - Test signing up with Google for a new user
   - Verify the user is properly created in the database
   - Verify the user can access protected resources

2. **Existing User Login**:
   - Test logging in with Google for an existing user
   - Verify account linking when the email matches

3. **Account Linking**:
   - Test linking a Google account to an existing account
   - Test unlinking a Google account

4. **Edge Cases**:
   - Test what happens when a user tries to link a Google account that's already linked to another user
   - Test account recovery flows

## 5. Security Review

Conduct a security review:

1. Verify that token validation is properly implemented
2. Ensure proper error handling for authentication failures
3. Check that sensitive data is not exposed in logs
4. Review account linking/unlinking flows for security issues

## 6. Monitoring

Set up monitoring for the new authentication flow:

1. Add logging for Google authentication attempts (success/failure)
2. Set up alerts for unusual authentication patterns
3. Monitor the ratio of Google vs. password authentications

## 7. Documentation

Update your documentation:

1. Add user documentation explaining how to use Google Sign-In
2. Update developer documentation explaining the authentication flow
3. Document any known limitations or edge cases

## Support Contact

If you encounter any issues during implementation, contact:
- Google Cloud Support: https://cloud.google.com/support
- OAuth 2.0 documentation: https://developers.google.com/identity/protocols/oauth2