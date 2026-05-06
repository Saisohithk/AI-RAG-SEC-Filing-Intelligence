# Dockerfile — Containerizes the FastAPI backend for deployment
#
# Build:   docker build -t sec-rag-api .
# Run:     docker run -p 8000:8000 --env-file .env sec-rag-api
#
# Why Docker?
# - Ensures your code runs the same everywhere (dev, staging, production)
# - Required for Hugging Face Spaces deployment
# - Makes environment setup reproducible

# Use official Python 3.11 slim image (smaller than full Python image)
FROM python:3.11-slim

# Set working directory inside the container
WORKDIR /app

# Install system dependencies needed by PyMuPDF and sentence-transformers
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (Docker caches this layer)
# If requirements.txt doesn't change, pip install won't re-run on rebuild
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all application code
COPY . .

# Expose the FastAPI port
EXPOSE 8000

# Start the FastAPI server
# --host 0.0.0.0 makes it accessible from outside the container
# --port 8000 matches the EXPOSE above
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
