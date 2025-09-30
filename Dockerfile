FROM python:3.11-slim

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Set working directory
WORKDIR /app

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install dependencies
RUN uv sync --frozen --no-dev

# Copy application code
COPY . .

# Expose port (Heroku assigns this dynamically)
EXPOSE 8000

# Use JSON array syntax (exec form) for proper signal handling
# Configuration is handled in gunicorn.conf.py
CMD ["uv", "run", "gunicorn", "-c", "gunicorn.conf.py", "main:app"]