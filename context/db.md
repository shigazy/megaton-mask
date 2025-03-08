# Database Configuration

## ORM and Database
- SQLAlchemy with PostgreSQL is used as the ORM
- Models defined in models.py file

## Schema Management
- No traditional migrations framework (like Alembic)
- Schema changes handled via `add_columns()` function in session.py
- Tables created automatically with `Base.metadata.create_all(bind=engine)` on application startup

## Models
- User
- Video
- Task
- AdminAction
- GlobalConfig

## Database Credentials
- Stored in AWS Secrets Manager (Secret ID: "rds!db-b6e814ea-dcfb-4201-aa97-0c7f0dad8b81")
- Development credentials:
  - Host: megaton-roto-db.c9e2eoe0qp8t.us-east-1.rds.amazonaws.com
  - Port: 5432
  - Database: postgres
  - Username: postgres
  - Password: KW?qBfh7BshzgxX#~e63!mQWlHuy

## Notes
- Schema changes require manual updates to the `add_columns()` function
- Comment in models.py: "TODO: Automate this process."