FROM python:3.11-slim

# Set working directory
WORKDIR /app


# Copy project files
COPY . /app

# Install Python dependencies
RUN pip3 install -r requirements.txt

# Expose port
EXPOSE 8080

# Start the app
CMD ["python", "app.py"]
