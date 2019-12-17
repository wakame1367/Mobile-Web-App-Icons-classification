from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras import optimizers
from tensorflow.keras.applications import mobilenet
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from dataset import create_df
from model import build_model

dataset_path = Path("./resources")
# Download from Kaggle Dataset and place it in a suitable directory.
# https://www.kaggle.com/testdotai/common-mobile-web-app-icons
image_root_path = dataset_path / "common-mobile-web-app-icons"
log_path = dataset_path / "logs"

if not log_path.exists():
    log_path.mkdir()

# common using mobile app UI labels
USE_LABELS = ['arrow_left', 'notifications', 'play', 'info', 'mail',
              'globe', 'upload', 'music', 'close', 'user', 'settings', 'home',
              'fast_forward', 'trash', 'question', 'map', 'eye', 'check_mark',
              'sort', 'overflow_menu', 'minimize', 'save', 'delete',
              'maximize', 'download', 'share', 'external_link', 'thumbs_up',
              'search', 'arrow_right', 'crop', 'camera', 'refresh', 'add',
              'volume', 'favorite', 'menu', 'edit', 'fab', 'link', 'arrow_up',
              'arrow_down', 'tag', 'warning', 'bookmark', 'cart', 'cloud',
              'filter']


def plot_history(history):
    # Plot training & validation accuracy values
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig(str(log_path / "Plot_accuracy_values.jpg"))

    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig(str(log_path / "Plot_loss_values.jpg"))


def train(df, x_col, y_col, is_test=True):
    # n_splits = 4
    random_state = 1224
    # multiple class and
    # skf = StratifiedKFold(n_splits=n_splits, shuffle=True,
    #                       random_state=random_state)

    width, height = 224, 224
    target_size = (height, width)
    num_channels = 3
    epochs = 100
    lr = 0.001
    batch_size = 128
    test_size = 0.2
    num_classes = len(df[y_col].unique())
    opt = optimizers.SGD(lr=lr)
    x_train, x_val, y_train, y_val = train_test_split(df[x_col], df[y_col],
                                                      test_size=test_size,
                                                      shuffle=True,
                                                      random_state=random_state,
                                                      stratify=df[y_col])
    # for fold_idx, (train_idx, val_idx) in enumerate(skf.split(df[x_col],
    #                                                           df[y_col])):
    #     print("Fold: {}".format(fold_idx))
    #     print(df.loc[train_idx].shape)
    #     print(df.loc[val_idx].shape)
    if is_test:
        # May not be included in all class
        # train_idx = np.random.choice(x_train, batch_size * 200)
        # val_idx = np.random.choice(val_idx, batch_size * 100)
        pass
    model = build_model(n_classes=num_classes,
                        input_shapes=(height, width, num_channels))
    train_gen = ImageDataGenerator(
        preprocessing_function=mobilenet.preprocess_input)
    train_generator = train_gen.flow_from_dataframe(
        pd.concat([x_train, y_train],
                  axis=1),
        x_col=x_col,
        y_col=y_col,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training')
    valid_gen = ImageDataGenerator(
        preprocessing_function=mobilenet.preprocess_input)
    valid_generator = valid_gen.flow_from_dataframe(pd.concat([x_val, y_val],
                                                              axis=1),
                                                    x_col=x_col,
                                                    y_col=y_col,
                                                    target_size=target_size,
                                                    batch_size=batch_size,
                                                    class_mode='categorical',
                                                    subset='validation')
    model.compile(optimizer=opt,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    step_size_train = train_generator.n // train_generator.batch_size
    step_size_valid = valid_generator.n // valid_generator.batch_size
    chk_filename = "weights.{epoch:02d}-{val_loss:.2f}.hdf5"
    chk_path = log_path / chk_filename
    model_chk = ModelCheckpoint(filepath=str(chk_path),
                                save_best_only=True,
                                save_weights_only=True)
    ealry_stopping = EarlyStopping()
    callbacks = [model_chk, ealry_stopping]
    history = model.fit_generator(generator=train_generator,
                                  steps_per_epoch=step_size_train,
                                  validation_data=valid_generator,
                                  validation_steps=step_size_valid,
                                  epochs=epochs,
                                  callbacks=callbacks
                                  )

    plot_history(history)
    # model.evaluate_generator(valid_generator)


def main():
    df_path = dataset_path / "data.csv"
    if df_path.exists():
        df = pd.read_csv(df_path)
    else:
        df = create_df(root_path=image_root_path)
        df.to_csv(df_path, index=False)
    x_col_name = "image_path"
    y_col_name = "class"

    # https://stackoverflow.com/questions/26577516/how-to-test-if-a-string-contains-one-of-the-substrings-in-a-list-in-pandas
    drop_indexes = df[~df[y_col_name].str.contains("|".join(USE_LABELS))].index
    df.drop(index=drop_indexes, inplace=True)
    train(df, x_col=x_col_name, y_col=y_col_name, is_test=False)


if __name__ == '__main__':
    main()
