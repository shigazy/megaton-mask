## Setup Instructions

1. Activate the conda environment :

conda activate python310

2. Install the dependencies:

pip install -r requirements.txt

/node_modules

## CLI Commands:

### Log Viewing Commands:

- `frontdevlog` - View frontend development service logs
- `backdevlog` - View backend development service logs 
- `frontlog` - View production frontend logs
- `backlog` - View production backend logs
- `nginxlog` - View Nginx error logs

### Kill Commands:

# Find processes using port 3000
sudo fuser 3000/tcp

# Kill all processes using port 3000
sudo fuser -k 3000/tcp

sudo fuser -k 3001/tcp && sudo systemctl restart megaton-roto-frontend-dev.service

# Find processes using port 8000
sudo fuser 8000/tcp

# Kill all processes using port 8000
sudo fuser -k 8000/tcp

sudo fuser -k 8001/tcp && sudo systemctl restart megaton-roto-backend-dev.service

## Available Scripts

- `npm run dev` - Start development server
- `npm run build` - Build for production
- `npm run start` - Start production server
- `npm run lint` - Run ESLint

## Features

- Video upload to S3
- Upload progress tracking
- Video playback
- [Future features to be added]


## Port Already in Use

If you encounter a "port already in use" error when starting the backend server, run:
sudo pkill -f node && cd ~/megaton-roto/frontend && npm run dev
sudo pkill -f uvicorn && cd ~/megaton-roto/backend && uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
## Backend Integration

This frontend connects to a FastAPI backend. Make sure the backend server is running at `http://localhost:8000` before starting the frontend.

cd backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

## Contributing

1. Create a feature branch
2. Make your changes
3. Submit a pull request

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| NEXT_PUBLIC_API_URL | Backend API URL | http://localhost:8000 |

## Troubleshooting

Common issues and solutions:

1. **CORS Issues**
   - Ensure backend CORS settings match frontend URL
   - Check environment variables are set correctly

2. **Upload Failures**
   - Verify AWS credentials
   - Check S3 bucket permissions

## License

[Your license here]


# For the backend FastAPI service
sudo systemctl start megaton-roto-frontend.service
sudo systemctl stop megaton-roto-frontend.service
sudo systemctl enable megaton-roto-frontend.service
sudo systemctl disable megaton-roto-frontend.service
sudo systemctl restart megaton-roto-frontend.service

sudo journalctl -u megaton-roto-frontend.service -f

# For the Roto service
sudo systemctl start megaton-roto.service
sudo systemctl stop megaton-roto.service
sudo systemctl enable megaton-roto.service
sudo systemctl disable megaton-roto.service
sudo systemctl restart megaton-roto.service

sudo journalctl -u megaton-roto.service -f

sudo systemctl kill -s 9 megaton-roto.service


# IMPORTANT: When adding new columns to any table, remember to update the add_columns() function
# in backend/app/db/session.py to include the new column definitions.
# Example:
# ADD COLUMN IF NOT EXISTS new_column_name VARCHAR/JSONB/etc;
# TODO: Automate this process.