#!/bin/bash
ENV=${ENV:-development}
echo "Starting server in $ENV environment"
export ENV=$ENV
uvicorn app.main:app --reload --port 8000