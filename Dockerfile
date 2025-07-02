FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies including CA certificates and curl
RUN apt-get update && apt-get install -y \
    ca-certificates \
    curl \
    && update-ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Set environment variable to help Python locate certificates
ENV SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt

# Copy project files
COPY . /app

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port
EXPOSE 8080

# Start the app
CMD ["python", "app.py"]
