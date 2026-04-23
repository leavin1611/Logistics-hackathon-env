FROM python:3.11-slim

# HuggingFace Spaces uses port 7860
ENV PORT=7860
ENV HOST=0.0.0.0
ENV WORKERS=2

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*

# Install OpenEnv from source (includes core libs)
RUN pip install --no-cache-dir \
    "git+https://github.com/meta-pytorch/OpenEnv.git" \
    fastmcp fastapi uvicorn pydantic

# Copy environment source
COPY . /app

# Add src to path so openenv.core imports resolve
ENV PYTHONPATH=/app/src:/app

EXPOSE 7860

CMD uvicorn server.app:app \
    --host $HOST \
    --port $PORT \
    --workers $WORKERS
