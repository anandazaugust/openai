FROM python:3.11-slim

WORKDIR /app

# Python optimization
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install required system packages
# - curl: needed for Azure Managed Identity IMDS calls
# - ca-certificates: required for TLS handshake to Azure Redis + OpenAI
# - build-essential, gcc, libffi-dev: needed by libs "azure-identity", "redis-entraid", "cryptography"
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    build-essential \
    gcc \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose FastAPI port
EXPOSE 8080

# Run Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
