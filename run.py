from pathlib import Path

from dataset import create_df

dataset_path = Path("./resources")
image_root_path = dataset_path / "common-mobile-web-app-icons"


def main():
    df_path = dataset_path / "data.csv"
    df = create_df(root_path=image_root_path)
    df.to_csv(df_path, index=False)


if __name__ == '__main__':
    main()
