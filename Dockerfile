# Base image
FROM python:3.11-slim

# Avoid interactive prompts during apt installs
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies needed by OpenCV/TensorFlow at runtime
# libgl1, libglib2.0-0 for cv2; libgomp1 often required by TensorFlow
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       libgl1 \
       libglib2.0-0 \
       libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Set workdir
WORKDIR /app

# Install Python deps
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY . .

# Environment
ENV PORT=8000

# Start server
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:${PORT}"]
