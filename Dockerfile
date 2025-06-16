# Use an official Python runtime as a parent image
FROM python:3.10.6-buster

# Set the working directory in the container
WORKDIR /app

COPY requirements.txt ./

RUN pip install --no-cache-dir --upgrade pip
RUN pip install -r requirements.txt

COPY api ./api
COPY pocketcoach ./pocketcoach
COPY models ./models
COPY tokenizer.pkl ./tokenizer.pkl

CMD ["sh", "-c", "uvicorn api.fast:app --host 0.0.0.0 --port $PORT"]
