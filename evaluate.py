from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import mobilenet, resnet_v2
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from dataset import create_df
from model import build_model

dataset_path = Path("./resources")
# Download from Kaggle Dataset and place it in a suitable directory.
# https://www.kaggle.com/testdotai/common-mobile-web-app-icons
image_root_path = dataset_path / "common-mobile-web-app-icons"
log_path = dataset_path / "logs"
# common using mobile app UI labels
USE_LABELS = ['arrow_left', 'notifications', 'play', 'info', 'mail',
              'globe', 'upload', 'music', 'close', 'user', 'settings', 'home',
              'fast_forward', 'trash', 'question', 'map', 'eye', 'check_mark',
              'sort', 'overflow_menu', 'minimize', 'save', 'delete',
              'maximize', 'download', 'share', 'external_link', 'thumbs_up',
              'search', 'arrow_right', 'crop', 'camera', 'refresh', 'add',
              'volume', 'favorite', 'menu', 'edit', 'fab', 'link', 'arrow_up',
              'arrow_down', 'tag', 'warning', 'bookmark', 'cart', 'cloud',
              'filter', 'other']


def main():
    df_path = dataset_path / "data.csv"
    if df_path.exists():
        df = pd.read_csv(df_path)
    else:
        df = create_df(root_path=image_root_path)
        df.to_csv(df_path, index=False)
    x_col_name = "image_path"
    y_col_name = "class"
    labels = set(df[y_col_name].unique()).difference(set(USE_LABELS))
    drop_indexes = pd.Index([])
    for label in labels:
        drop_index = df[df[y_col_name] == label].index
        drop_indexes = drop_indexes.union(drop_index)
    df.drop(index=drop_indexes, inplace=True)
    num_classes = len(USE_LABELS)
    width, height = 224, 224
    num_channels = 3
    target_size = (height, width)
    input_shapes = (height, width, num_channels)
    random_state = 1224
    batch_size = 32
    base_model = resnet_v2.ResNet101V2(include_top=False,
                                       weights='imagenet',
                                       input_shape=input_shapes)
    model = build_model(base_model, n_classes=num_classes)

    model.load_weights(str(log_path / "weights.16-0.38.hdf5"))

    x_train, x_val, y_train, y_val = train_test_split(df[x_col_name],
                                                      df[y_col_name],
                                                      test_size=0.2,
                                                      shuffle=True,
                                                      random_state=random_state,
                                                      stratify=df[y_col_name])
    valid_gen = ImageDataGenerator(rescale=1. / 255)
    valid_generator = valid_gen.flow_from_dataframe(pd.concat([x_val, y_val],
                                                              axis=1),
                                                    x_col=x_col_name,
                                                    y_col=y_col_name,
                                                    target_size=target_size,
                                                    batch_size=batch_size,
                                                    class_mode='categorical',
                                                    subset='training')

    y_pred = model.predict_generator(valid_generator)
    y_pred = np.argmax(y_pred, axis=1)
    print('Confusion Matrix')
    print(confusion_matrix(valid_generator.classes, y_pred))
    print('Classification Report')
    print(classification_report(valid_generator.classes, y_pred))


if __name__ == '__main__':
    main()
