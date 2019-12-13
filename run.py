from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from tensorflow.keras import optimizers
from tensorflow.keras.applications import mobilenet
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from dataset import create_df
from model import build_model

dataset_path = Path("./resources")
image_root_path = dataset_path / "common-mobile-web-app-icons"


def train(df, log_path, x_col, y_col, is_test=True):
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
    batch_size = 64
    num_classes = len(df[y_col].unique())
    opt = optimizers.Adam(lr=lr)
    x_train, x_val, y_train, y_val = train_test_split(df[x_col], df[y_col],
                                                      test_size=0.3,
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
    train_generator = train_gen.flow_from_dataframe(pd.concat([x_train, y_train],
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
                                                    subset='training')
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

    # Plot training & validation accuracy values
    # plt.plot(history.history['acc'])
    # plt.plot(history.history['val_acc'])
    # plt.title('Model accuracy')
    # plt.ylabel('Accuracy')
    # plt.xlabel('Epoch')
    # plt.legend(['Train', 'Test'], loc='upper left')
    # plt.show()
    #
    # # Plot training & validation loss values
    # plt.plot(history.history['loss'])
    # plt.plot(history.history['val_loss'])
    # plt.title('Model loss')
    # plt.ylabel('Loss')
    # plt.xlabel('Epoch')
    # plt.legend(['Train', 'Test'], loc='upper left')
    # plt.show()
    # model.evaluate_generator(valid_generator)


def main():
    log_path = dataset_path / "logs"
    if not log_path.exists():
        log_path.mkdir()
    df_path = dataset_path / "data.csv"
    if df_path.exists():
        df = pd.read_csv(df_path)
    else:
        df = create_df(root_path=image_root_path)
        df.to_csv(df_path, index=False)
    x_col_name = "image_path"
    y_col_name = "class"
    thresh_count = 100
    d = df[y_col_name].value_counts() < thresh_count
    drop_classes = list(d[d == True].index)
    # https://stackoverflow.com/questions/26577516/how-to-test-if-a-string-contains-one-of-the-substrings-in-a-list-in-pandas
    drop_indexes = df[df["class"].str.contains("|".join(drop_classes))].index
    df.drop(index=drop_indexes, inplace=True)

    train(df, log_path=log_path, x_col=x_col_name, y_col=y_col_name,
          is_test=False)


if __name__ == '__main__':
    main()
