from pocketcoach.ml_logic.data import get_data
from pocketcoach.params import *
from pathlib import Path

def preprocess():
    """
    Reads the train, test, and validation data sets.
    """

    train_df = get_data(Path(LOCAL_DATA_PATH).joinpath("training.csv"))
    test_df = get_data(Path(LOCAL_DATA_PATH).joinpath("test.csv"))
    validation_df = get_data(Path(LOCAL_DATA_PATH).joinpath("validation.csv"))
