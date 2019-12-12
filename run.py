from pathlib import Path
import pandas as pd
from dataset import create_df

dataset_path = Path("./resources")
image_root_path = dataset_path / "common-mobile-web-app-icons"


def main():
    df_path = dataset_path / "data.csv"
    if df_path.exists():
        df = pd.read_csv(df_path)
    else:
        df = create_df(root_path=image_root_path)
        df.to_csv(df_path, index=False)


if __name__ == '__main__':
    main()
