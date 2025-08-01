FROM python:3.12-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install uv for faster package management
RUN pip install uv

# Set work directory
WORKDIR /app

# Copy dependency files first (for better caching)
COPY pyproject.toml ./
COPY README.md ./

# Install Python dependencies (this layer will be cached if deps don't change)
RUN uv pip install --system -e "."

# Install Playwright browsers for crawl4ai (before copying app code for better caching)
RUN playwright install chromium
RUN playwright install-deps

# Copy application code (after dependencies and browsers are installed)
COPY . .

# Create logs directory
RUN mkdir -p logs

# Expose port
EXPOSE 8000

# Default command (can be overridden in docker-compose)
CMD ["uvicorn", "context_server.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
