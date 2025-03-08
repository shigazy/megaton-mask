# CLAUDE.md - Guidelines for the Megaton Roto Project

## Build & Run Commands
- **Frontend**: `cd frontend && npm run dev` (development) / `npm run build && npm run start` (production)
- **Backend**: `cd backend && ./start.sh` or `cd backend && uvicorn app.main:app --reload --port 8000`
- **Lint**: `cd frontend && npm run lint` (frontend) / `black .` and `flake8` (backend, if installed)
- **Migrations**: `cd backend && alembic revision --autogenerate -m "message"` and `alembic upgrade head`

## Code Style
- **TypeScript**: Strong typing with interfaces, strict mode enabled, path aliases with @/* for src/
- **Python**: Organized imports (stdlib → third-party → local), comprehensive error handling with try/except
- **Components**: React functional components with TypeScript interfaces, organized by feature
- **Naming**: PascalCase for components/classes, camelCase for functions/variables, snake_case in Python
- **Error Handling**: Try/catch with specific error types, logging at appropriate levels

## Architecture
- **Frontend**: React/Next.js with TypeScript, Zustand for state, Tailwind CSS for styling
- **Backend**: FastAPI with SQLAlchemy ORM, JWT auth, background tasks for processing
- **AI**: Custom inference manager with SAM2 model, processing queue for video/masks