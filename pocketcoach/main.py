from pocketcoach.dl_logic.data import get_data, clean_data_set
from pocketcoach.params import *
from pathlib import Path

def preprocess():
    """
    - Reads the train, test, and validation data sets.
    - Performs basic cleaning of the data-sets
    """

    train_df = get_data(Path(LOCAL_DATA_PATH).joinpath("training.csv"))
    test_df = get_data(Path(LOCAL_DATA_PATH).joinpath("test.csv"))
    validation_df = get_data(Path(LOCAL_DATA_PATH).joinpath("validation.csv"))

    train_cleaned_df = clean_data_set(train_df)
    test_cleaned_df = clean_data_set(test_df)
    validation_cleaned_df = clean_data_set(validation_df)
