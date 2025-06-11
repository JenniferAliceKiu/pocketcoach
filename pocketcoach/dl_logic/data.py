import pandas as pd
from pathlib import Path
import string
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer

def get_data(
        cache_path:Path,
        data_has_header=True
    ) -> pd.DataFrame:
    """
    Retrieve data from `cache_path` if the file exists
    """
    if cache_path.is_file():
        print("\nLoad data from local CSV...")
        df = pd.read_csv(cache_path, header='infer' if data_has_header else None)
    else:
        print(f"\nFile could not be found at {cache_path} ...")

    print(f"✅ Data loaded from file {cache_path.name}, with shape {df.shape}")

    return df


def clean_data_set(df: pd.DataFrame):
    """
    Adds a column with cleaned texts to the data frame.
    """

    df['cleaned_text'] = df['text'].apply(clean)
    print("✅ data cleaned")

    return df

def clean(text: str):
    """
    Cleans the raw texts by lower casing, strip for leading or trailing white-
    spaces, remove digits and punctiations and perform lemmatization on the
    verbs and nouns.
    """

    text = text.strip()
    text = text.lower()
    text = ''.join(word for word in text if not word.isdigit())
    for punctuation in string.punctuation:
        text = text.replace(punctuation, '')
    tokenized = word_tokenize(text)
    tokenized_lemmatized = lemmatize(tokenized, "v")
    tokenized_lemmatized = lemmatize(tokenized_lemmatized, "n")
    cleaned_text = " ".join(tokenized_lemmatized)

    return cleaned_text


def lemmatize(word_tokens, pos):
    """
    Lemmatizes word tokens based on pos
    """
    return [
        WordNetLemmatizer().lemmatize(word, pos=pos)
        for word in word_tokens
    ]
