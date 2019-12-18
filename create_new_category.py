import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from dataset import create_df

dataset_path = Path("./resources")
# Download from Kaggle Dataset and place it in a suitable directory.
# https://www.kaggle.com/testdotai/common-mobile-web-app-icons
image_root_path = dataset_path / "common-mobile-web-app-icons"

# common using mobile app UI labels
USE_LABELS = ['arrow_left', 'notifications', 'play', 'info', 'mail',
              'globe', 'upload', 'music', 'close', 'user', 'settings', 'home',
              'fast_forward', 'trash', 'question', 'map', 'eye', 'check_mark',
              'sort', 'overflow_menu', 'minimize', 'save', 'delete',
              'maximize', 'download', 'share', 'external_link', 'thumbs_up',
              'search', 'arrow_right', 'crop', 'camera', 'refresh', 'add',
              'volume', 'favorite', 'menu', 'edit', 'fab', 'link', 'arrow_up',
              'arrow_down', 'tag', 'warning', 'bookmark', 'cart', 'cloud',
              'filter', '_negative']

other_dir_name = "other"
other_cat_path = image_root_path / other_dir_name
if not other_cat_path.exists():
    other_cat_path.mkdir()

MAX_NUM_IMAGES = 3000


def main():
    df_path = dataset_path / "data.csv"
    if df_path.exists():
        df = pd.read_csv(df_path)
    else:
        df = create_df(root_path=image_root_path)
        df.to_csv(df_path, index=False)
    x_col_name = "image_path"
    y_col_name = "class"
    drop_indexes = pd.Index([])
    for label in USE_LABELS:
        drop_index = df[df[y_col_name] == label].index
        drop_indexes = drop_indexes.union(drop_index)
    df.drop(index=drop_indexes, inplace=True)
    other_image_indexes = np.random.choice(df.index, MAX_NUM_IMAGES,
                                           replace=False)
    print(df.loc[other_image_indexes][y_col_name].unique())
    for other_image_path in df.loc[other_image_indexes][x_col_name]:
        print(other_image_path)

        dst_path = other_cat_path / Path(other_image_path).name
        shutil.copy(src=str(other_image_path),
                    dst=str(dst_path))


if __name__ == '__main__':
    main()
