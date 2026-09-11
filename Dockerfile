FROM python:3.11-slim

WORKDIR /app

# LightGBM requires libgomp (OpenMP runtime)
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements-server.txt .
RUN pip install --no-cache-dir -r requirements-server.txt

# Copy source code
COPY src/ src/

# Copy trained models
COPY models/ models/

# Copy historical data required by GAGNE TEMPS
COPY data/ data/

# Expose API port
EXPOSE 8001

# Start FastAPI / Uvicorn
CMD ["uvicorn", "src.server:app", "--host", "0.0.0.0", "--port", "8001"]
