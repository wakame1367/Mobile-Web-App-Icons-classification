from tensorflow.keras.applications import mobilenet
from tensorflow.keras.applications import resnet_v2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model


def build_model(n_classes, input_shapes=(224, 224, 3)):
    base_model = mobilenet.MobileNet(include_top=False,
                                     weights='imagenet',
                                     input_shape=input_shapes)
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(1024, activation="relu")(x)
    x = Dropout(0.25)(x)
    x = Dense(512, activation="relu")(x)
    x = Dropout(0.25)(x)
    y = Dense(n_classes, activation="softmax")(x)

    model = Model(inputs=base_model.input,
                  outputs=y)
    for layer in model.layers[:87]:
        layer.trainable = False
    return model


def build_resnet_model(n_classes, input_shapes=(224, 224, 3)):
    base_model = resnet_v2.ResNet101V2(include_top=False,
                                       weights='imagenet',
                                       input_shape=input_shapes)
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
