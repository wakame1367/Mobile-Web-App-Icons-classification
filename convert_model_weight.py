"""
Reference:
https://www.tensorflow.org/guide/keras/save_and_serialize
"""

import argparse
from pathlib import Path

import tensorflow as tf
from tensorflow.keras.applications import resnet_v2

from evaluate import USE_LABELS
from model import build_model


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str, help="path to model_weight(.h5)")
    parser.add_argument("output_path", type=str, help="")
    _args = parser.parse_args()
    return _args


def main():
    args = get_arguments()
    model_weight_path = Path(args.path)
    if not model_weight_path.exists():
        raise FileExistsError(model_weight_path)
    output_path = Path(args.output_path)
    if not output_path.exists():
        raise FileExistsError(output_path)
    width, height = 224, 224
    num_channels = 3
    num_classes = len(USE_LABELS)
    input_shapes = (height, width, num_channels)
    base_model = resnet_v2.ResNet101V2(include_top=False,
                                       weights='imagenet',
                                       input_shape=input_shapes)
    model = build_model(base_model, n_classes=num_classes)
    model.load_weights(str(model_weight_path))

    tf.keras.backend.clear_session()
    model.save(str(output_path), save_format="tf")


if __name__ == '__main__':
    main()
