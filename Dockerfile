# Start from a minimal Python 3.11 image (~150 MB instead of ~900 MB for the full image).
# "slim" strips out compilers and docs we don't need, keeping the container small and fast.
FROM python:3.11-slim

# Set the working directory inside the container. All subsequent commands
# (COPY, RUN, CMD) execute relative to /app.
WORKDIR /app

# Copy ONLY requirements.txt first, then install dependencies.
# Docker caches each step ("layer"). If requirements.txt hasn't changed,
# Docker skips the slow pip install on rebuilds. This is why we copy
# requirements.txt separately from the rest of the code.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Now copy the rest of the source code. This layer rebuilds on every
# code change, but the pip install layer above stays cached.
COPY . .

# Document that the container listens on port 8501 (Streamlit's default).
# This doesn't actually open the port — that's done by docker-compose.
EXPOSE 8501

# A health check lets Docker (and cloud platforms) know if the app is alive.
# If this curl fails, the container is marked "unhealthy" and can be restarted.
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# The default command to run when the container starts.
# These flags configure Streamlit for container environments:
#   --server.address=0.0.0.0    Listen on all network interfaces (not just localhost)
#   --server.enableCORS=false   Disable cross-origin restrictions for local dev
CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.enableCORS=false", \
     "--server.enableXsrfProtection=false"]