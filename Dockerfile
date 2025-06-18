# Use an official Python runtime as a parent image
FROM python:3.10.6-buster

# Install ffmpeg (needed by Whisper) and clean up apt caches
RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg \
 && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Copy and install Python dependencies
COPY requirements_prod.txt requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# DEBUG: show which torch got installed
RUN python -c "import torch; print('torch version:', torch.__version__)" \
 && pip show torch

# Copy application code
COPY api        ./api
COPY pocketcoach ./pocketcoach
COPY models     ./models
COPY tokenizer.pkl ./tokenizer.pkl

# (Optional) expose the port and set a default
#ENV PORT=8000
#EXPOSE $PORT

# Start the app via Uvicorn
CMD ["sh","-c","uvicorn api.fast:app --host 0.0.0.0 --port ${PORT}"]
