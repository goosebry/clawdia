#!/bin/bash
echo "Stopping all Clawdia proxy and worker processes..."

# Kill main brain instances (handles macOS framework binary paths)
pkill -f "ella.main" || true
echo "Terminated main brain instances."

# Force kill all Celery workers
pkill -9 -f "celery -A ella.tasks.celery_app"
echo "Terminated background Celery workers."

echo "All zombie processes cleared. You may now start a single clean instance."
