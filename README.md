# Pocket Coach

This is the backend part of the Pocket Coach. It trains a deep learning model on the [emotion-dataset](https://github.com/dair-ai/emotion_dataset) to map the emotions of a users dialog with a chat-therapist to help guide the therapist to ask relevant questions and to make the users are aware of what they are truly feeling.

## How to run locally

1. Clone this repo
2. Download the [data](https://www.kaggle.com/datasets/parulpandey/emotion-dataset)
3. Copy the data into the raw_data with its file names intact (training.csv, test.csv, validation.csv)
4. Copy the env.sample to .env and populate the necessary env vars
5. make run_preprocess
6. make run_server_locally
7. Server is now running on http://localhost:8000/
