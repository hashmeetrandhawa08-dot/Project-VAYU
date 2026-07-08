import pandas as pd

def load_data(file_path):
    df = pd.read_csv(file_path)

    print("=" * 50)
    print("Dataset Loaded Successfully")
    print("=" * 50)
    print(f"Shape: {df.shape}")

    return df


if __name__ == "__main__":
    dataset = load_data("data/raw/air_quality.csv")
    print(dataset.head())