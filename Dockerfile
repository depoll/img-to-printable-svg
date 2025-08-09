FROM python:3.11-slim

# Install system dependencies
# - potrace: Required for bitmap to vector conversion
# - libcairo2-dev: Required for SVG rasterization features (--rasterize option)
# - gcc/g++/pkg-config: Build dependencies for Python packages that need compilation
# Note: Inkscape can be added for additional SVG rasterization fallback support
RUN apt-get update && apt-get install -y \
    potrace \
    gcc \
    g++ \
    libcairo2-dev \
    pkg-config \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY img2svg.py .
COPY app.py .
COPY original_scripts/ ./original_scripts/
COPY templates/ ./templates/

# Create necessary directories
RUN mkdir -p /app/output /app/input /app/temp

# Expose port for web server (using 5000 as configured in app.py)
EXPOSE 5000

# Set environment variables for production
ENV FLASK_APP=app.py
ENV PYTHONUNBUFFERED=1
ENV FLASK_ENV=production
ENV WERKZEUG_RUN_MAIN=true
ENV PYTHONDONTWRITEBYTECODE=1

# Default command runs the web server
CMD ["python", "app.py"]