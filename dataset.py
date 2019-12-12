from pathlib import Path

import pandas as pd

dataset_path = Path("../resources/common-mobile-web-app-icons/")


def create_df():
    image_paths = [path for path in dataset_path.glob("*/*.jpg")]
    x_col_name = "image_path"
    y_col_name = "class"
    df = pd.DataFrame({x_col_name: image_paths})
    df[y_col_name] = df[x_col_name].map(lambda x: x.parent.stem)
    df[x_col_name] = df[x_col_name].map(lambda x: str(x))
    return df
