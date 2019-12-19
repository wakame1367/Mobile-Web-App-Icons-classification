from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras import optimizers
from tensorflow.keras.applications import resnet_v2
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, \
    ReduceLROnPlateau
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
USE_LABELS = ['arrow_left', 'notifications', 'info', 'upload', 'close',
              'settings', 'home', 'trash', 'question', 'eye', 'check_mark',
              'sort', 'overflow_menu', 'delete', 'download', 'share',
              'external_link', 'search', 'arrow_right', 'crop', 'refresh',
              'add', 'favorite', 'menu', 'edit', 'link', 'tag', 'warning',
              'bookmark', 'filter']


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


def train(model, preprocess_func, df, x_col, y_col, is_test=True):
    # n_splits = 4
    random_state = 1224
    # multiple class and
    # skf = StratifiedKFold(n_splits=n_splits, shuffle=True,
    #                       random_state=random_state)

    width, height = 224, 224
    target_size = (height, width)
    num_channels = 3
    epochs = 100
    lr = 0.002
    batch_size = 32
    test_size = 0.2
    opt = optimizers.RMSprop(learning_rate=lr,
                             rho=0.9,
                             momentum=0.1,
                             epsilon=1e-07,
                             centered=True,
                             name='RMSprop')
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
    train_gen = ImageDataGenerator(rotation_range=45,
                                   width_shift_range=.15,
                                   height_shift_range=.15,
                                   horizontal_flip=True,
                                   zoom_range=0.5,
                                   preprocessing_function=preprocess_func)
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
        preprocessing_function=preprocess_func)
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
    early_stopping = EarlyStopping(patience=10)
    learning_rate_reduction = ReduceLROnPlateau(
        monitor='loss',  # Quantity to be monitored.
        factor=0.25,
        # Factor by which the learning rate will be reduced. new_lr = lr * factor
        patience=2,
        # The number of epochs with no improvement after which learning rate will be reduced.
        verbose=1,  # 0: quiet - 1: update messages.
        mode="auto",
        # {auto, min, max}. In min mode, lr will be reduced when the quantity monitored has stopped decreasing;
        # in the max mode it will be reduced when the quantity monitored has stopped increasing;
        # in auto mode, the direction is automatically inferred from the name of the monitored quantity.
        min_delta=0.0001,
        # threshold for measuring the new optimum, to only focus on significant changes.
        cooldown=0,
        # number of epochs to wait before resuming normal operation after learning rate (lr) has been reduced.
        min_lr=0.00001  # lower bound on the learning rate.
    )
    callbacks = [learning_rate_reduction, model_chk, early_stopping]
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
    num_classes = len(USE_LABELS)
    width, height = 224, 224
    num_channels = 3
    input_shapes = (height, width, num_channels)
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
    train(preprocess_func=resnet_v2.preprocess_input, df=df, model=model,
          x_col=x_col_name, y_col=y_col_name, is_test=False)


if __name__ == '__main__':
    main()
