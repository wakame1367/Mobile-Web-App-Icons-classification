import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras import optimizers
from tensorflow.keras.applications import resnet_v2
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from model import build_model
from random_eraser import get_random_eraser

dataset_path = Path("./resources")
# Download from Kaggle Dataset and place it in a suitable directory.
# https://www.kaggle.com/testdotai/common-mobile-web-app-icons
image_root_path = dataset_path / "common-mobile-web-app-icons"
log_path = dataset_path / "logs"

if not log_path.exists():
    log_path.mkdir()

SEED = 1234

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


def plot_history(history):
    # Plot training & validation accuracy values
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig(str(log_path / "Plot_accuracy_values.jpg"))
    plt.close()

    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig(str(log_path / "Plot_loss_values.jpg"))
    plt.close()


def prepare_generator(df, x_col, y_col, width, height, batch_size, test_size,
                      classes):
    x_train, x_val, y_train, y_val = train_test_split(df[x_col],
                                                      df[y_col],
                                                      test_size=test_size,
                                                      shuffle=True,
                                                      random_state=SEED,
                                                      stratify=df[y_col])
    cutout = get_random_eraser(v_l=0, v_h=1, pixel_level=True)
    train_gen = ImageDataGenerator(rotation_range=45,
                                   width_shift_range=.15,
                                   height_shift_range=.15,
                                   horizontal_flip=True,
                                   vertical_flip=True,
                                   zoom_range=0.5,
                                   preprocessing_function=cutout,
                                   rescale=1. / 255)
    train_generator = train_gen.flow_from_dataframe(
        pd.concat([x_train, y_train],
                  axis=1),
        x_col=x_col,
        y_col=y_col,
        target_size=(width, height),
        batch_size=batch_size,
        class_mode='categorical',
        subset='training',
        classes=classes)
    valid_gen = ImageDataGenerator(rescale=1. / 255)
    valid_generator = valid_gen.flow_from_dataframe(
        pd.concat([x_val, y_val],
                  axis=1),
        x_col=x_col,
        y_col=y_col,
        target_size=(width, height),
        batch_size=batch_size,
        class_mode='categorical',
        subset='training',
        classes=classes)
    return train_generator, valid_generator


def train(train_generator, valid_generator, model, lr, epochs):
    opt = optimizers.Adam(lr=lr)
    model.compile(optimizer=opt,
                  loss='categorical_crossentropy',
                  metrics=[Precision(), Recall()])
    step_size_train = train_generator.n // train_generator.batch_size
    step_size_valid = valid_generator.n // valid_generator.batch_size
    chk_filename = "weights.{epoch:02d}-{val_loss:.2f}.hdf5"
    chk_path = log_path / chk_filename
    model_chk = ModelCheckpoint(filepath=str(chk_path),
                                save_best_only=True,
                                save_weights_only=True)
    early_stopping = EarlyStopping(patience=10)
    callbacks = [model_chk, early_stopping]
    history = model.fit_generator(generator=train_generator,
                                  steps_per_epoch=step_size_train,
                                  validation_data=valid_generator,
                                  validation_steps=step_size_valid,
                                  epochs=epochs,
                                  callbacks=callbacks
                                  )

    plot_history(history)
    # model.evaluate_generator(valid_generator)


class LabelParser:
    def __init__(self, labels):
        self.labels = labels
        self.icon_name2label = {}
        self.model_use = "use"
        self.model_label = "label"
        self.set_labels()
        self.label2icon_name = {label: icon_name for icon_name, label in
                                self.icon_name2label.items()}

    def set_labels(self):
        for icon_name, values in self.labels.items():
            model_use = values.get(self.model_use)
            if model_use:
                label = values.get(self.model_label)
                self.icon_name2label[icon_name] = label


def get_classes(label_path):
    with open(label_path) as f:
        labels = json.load(f)
    lp = LabelParser(labels)
    classes = list(lp.icon_name2label.keys())
    return classes


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", type=str)
    parser.add_argument("label_path", type=str)
    parser.add_argument("--x_col_name", type=str, default="image_path")
    parser.add_argument("--y_col_name", type=str, default="class")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--width", type=int, default=224)
    parser.add_argument("--height", type=int, default=224)
    parser.add_argument("--channels", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=0.0001)
    parser.add_argument("--test_size", type=float, default=0.2)
    _args = parser.parse_args()
    return _args


def main():
    args = get_arguments()
    df_path = Path(args.csv_path)
    if not df_path.exists():
        raise FileExistsError("{}".format(df_path))
    else:
        df = pd.read_csv(df_path)

    labels_path = Path(args.label_path)
    if not labels_path.exists():
        raise FileExistsError("{}".format(labels_path))
    else:
        classes = get_classes(labels_path)

    x_col_name = args.x_col_name
    y_col_name = args.y_col_name
    labels = set(df[y_col_name].unique())
    num_classes = len(USE_LABELS)
    width, height = args.width, args.height
    num_channels = args.channels
    input_shapes = (height, width, num_channels)
    batch_size = args.batch_size
    test_size = args.test_size
    epochs = args.epochs
    learning_rate = args.learning_rate
    base_model = resnet_v2.ResNet101V2(include_top=False,
                                       weights='imagenet',
                                       input_shape=input_shapes)
    model = build_model(base_model, n_classes=num_classes)

    labels = set(df[y_col_name].unique()).difference(set(USE_LABELS))
    drop_indexes = pd.Index([])
    for label in labels:
        drop_index = df[df[y_col_name] == label].index
        drop_indexes = drop_indexes.union(drop_index)
    df.drop(index=drop_indexes, inplace=True)
    train_generator, valid_generator = prepare_generator(df=df,
                                                         x_col=x_col_name,
                                                         y_col=y_col_name,
                                                         width=width,
                                                         height=height,
                                                         batch_size=batch_size,
                                                         test_size=test_size,
                                                         classes=classes)
    train(train_generator=train_generator, valid_generator=valid_generator,
          model=model, epochs=epochs, lr=learning_rate)


if __name__ == '__main__':
    main()
