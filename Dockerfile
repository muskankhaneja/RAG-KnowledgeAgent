FROM python:3.10-slim

WORKDIR /app

# Install system dependencies (git needed for GitHub ingest feature)
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app source
COPY . .

# HuggingFace Spaces requires port 7860
EXPOSE 7860

# Start the FastAPI server
CMD ["python", "-m", "uvicorn", "src.agent.app:app", "--host", "0.0.0.0", "--port", "7860"]
