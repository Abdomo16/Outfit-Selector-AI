# Build stage
FROM python:3.10-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


# Production stage
FROM python:3.10-slim

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder and application code
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY . .

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Expose port
EXPOSE 8000

# Run FastAPI using Uvicorn
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
