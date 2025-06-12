from pocketcoach.dl_logic.data import get_data, clean_data_set, pad, emotion_of
from pocketcoach.params import *
from pathlib import Path
from pocketcoach.dl_logic.model import train_base_model
import tensorflow as tf
from pocketcoach.dl_logic.tokenizer import save
from pocketcoach.dl_logic.model import load_model

def preprocess():
    """
    - Reads the train, test, and validation data sets.
    - Performs basic cleaning of the data-sets
    - Fits a tokenizer on the training set
    - Stores the tokenizer
    - Train and evaluate model
    - Store model and return it
    """

    print("Read data")
    train_df = get_data(Path(LOCAL_DATA_PATH).joinpath("training.csv"))
    test_df = get_data(Path(LOCAL_DATA_PATH).joinpath("test.csv"))
    validation_df = get_data(Path(LOCAL_DATA_PATH).joinpath("validation.csv"))

    print("Clean data")
    train_cleaned_df = clean_data_set(train_df)
    test_cleaned_df = clean_data_set(test_df)
    validation_cleaned_df = clean_data_set(validation_df)

    print("Create Tokenizer")
    X = train_cleaned_df['text']
    tokenizer = tf.keras.preprocessing.text.Tokenizer()
    tokenizer.fit_on_texts(X)

    save(tokenizer)

    vocab_size = len(tokenizer.word_index)

    X_train = pad(X, tokenizer)
    y_train = train_cleaned_df['label']

    X_val = pad(validation_cleaned_df['text'], tokenizer)
    y_val = validation_cleaned_df['label']

    X_test = pad(test_cleaned_df['text'], tokenizer)
    y_test = test_cleaned_df['label']

    model = train_base_model(X_train, y_train, X_val, y_val, vocab_size)
    evaluation = model.evaluate(X_test, y_test, return_dict=True)
    print(f"Model evalauated {evaluation}")

    model.save(BASE_MODEL_NAME)

    print(f"✅ Model has been trained and stored as {BASE_MODEL_NAME}")


def classify(text):
    print(f"Predicting text {text}")
    model = load_model()
    prediction = model.predict(text)

    mapped_predictions = [
        {
            'score': item['score'],
            'category': emotion_of(item['label'])
        }
        for item in prediction
    ]

    print(f"Result of the prediction is {mapped_predictions}")
    return mapped_predictions

def category(id):
    dict = {
        0: 'sadness',
        1: 'joy',
        2: 'love',
        3: 'anger',
        4: 'fear',
        5: 'surprised'
    }

    return dict.get(id, f'Invalid id {id}')
