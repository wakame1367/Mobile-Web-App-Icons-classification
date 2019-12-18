from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model


def build_model(base_model, n_classes):
    base_model.trainable = False
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(1024, activation="relu")(x)
    x = Dropout(0.25)(x)
    x = Dense(512, activation="relu")(x)
    x = Dropout(0.25)(x)
    y = Dense(n_classes, activation="softmax")(x)

    model = Model(inputs=base_model.input,
                  outputs=y)
    return model
