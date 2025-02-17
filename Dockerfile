# Use Python 3.12.2 as base image
FROM python:3.12.2-slim

# Set working directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install dependencies
RUN pip install -r requirements.txt

# Copy application code
COPY . .

# Expose port if necessary
EXPOSE 8000

# Run command
CMD ["python", "app.py"]
