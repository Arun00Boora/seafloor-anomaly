# Backend image: FastAPI + the packaged deploy_data (no PyTorch needed).
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1
WORKDIR /app

# Install dependencies first, so Docker can cache this layer between builds.
COPY backend/requirements.txt backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt

# Copy the code and the packaged data. The folder layout matches the project,
# so the backend finds deploy_data the same way it does on your laptop.
COPY backend backend
COPY deploy_data deploy_data

WORKDIR /app/backend
EXPOSE 8000

# Hosts like Render pass the port in $PORT; locally it falls back to 8000.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]