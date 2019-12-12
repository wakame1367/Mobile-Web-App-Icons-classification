import pandas as pd


def create_df(root_path):
    image_paths = [path for path in root_path.glob("*/*.jpg")]
    x_col_name = "image_path"
    y_col_name = "class"
    df = pd.DataFrame({x_col_name: image_paths})
    df[y_col_name] = df[x_col_name].map(lambda x: x.parent.stem)
    df[x_col_name] = df[x_col_name].map(lambda x: str(x))
    return df
