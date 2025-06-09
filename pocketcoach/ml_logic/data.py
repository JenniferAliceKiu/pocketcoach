import pandas as pd
from pathlib import Path

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
