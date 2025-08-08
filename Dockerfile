FROM python:3.11-slim

# Install system dependencies including potrace
RUN apt-get update && apt-get install -y \
    potrace \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY img2svg.py .
COPY original_scripts/ ./original_scripts/

# Create output directory
RUN mkdir -p /app/output

# Set the default command
ENTRYPOINT ["python", "img2svg.py"]
CMD ["--help"]