import warnings
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from albumentations import (
    Compose, HueSaturationValue, RandomBrightness, RandomContrast, RandomGamma,
    ToFloat, Cutout, HorizontalFlip, PadIfNeeded, RandomCrop, RandomScale,
    Resize, Rotate
)
from albumentations import ImageOnlyTransform
from sklearn.preprocessing import LabelEncoder
from tensorflow.python.keras.utils import to_categorical
from tensorflow.python.keras.utils.data_utils import Sequence

warnings.filterwarnings('ignore')

Height, Width = 224, 224
dataset_path = Path("./resources")
image_root_path = dataset_path / "common-mobile-web-app-icons"


class GrayToRGB(ImageOnlyTransform):
    """
    Targets:
        image

    Image types:
        uint8, float32
    """

    def __init__(self, p=1.0):
        super(GrayToRGB, self).__init__(p)

    def apply(self, img, **params):
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)


AUGMENTATIONS_TRAIN = Compose([
    # https://github.com/albumentations-team/albumentations/issues/67
    GrayToRGB(),
    HorizontalFlip(p=0.5),
    RandomContrast(limit=0.2, p=0.5),
    RandomGamma(gamma_limit=(80, 120), p=0.5),
    RandomBrightness(limit=0.2, p=0.5),
    HueSaturationValue(hue_shift_limit=5, sat_shift_limit=20,
                       val_shift_limit=10, p=.9),
    # CLAHE(p=1.0, clip_limit=2.0),
    # ShiftScaleRotate(
    #     shift_limit=0.0625, scale_limit=0.1,
    #     rotate_limit=15, border_mode=cv2.BORDER_REFLECT_101, p=0.8),
    ToFloat(max_value=255)
])

# Reference
# https://github.com/diceroll/kmnist/blob/master/dataloader.py
AUGMENTATIONS_KMNIST = Compose([
    Rotate(p=0.8, limit=5),
    PadIfNeeded(p=0.5, min_height=Height + 2, min_width=Width),
    PadIfNeeded(p=0.5, min_height=Height, min_width=Width + 2),
    Resize(p=1.0, height=Height, width=Width),
    RandomScale(p=1.0, scale_limit=0.1),
    PadIfNeeded(p=1.0, min_height=Height + 4, min_width=Width + 4),
    RandomCrop(p=1.0, height=Height, width=Width),
    Cutout(p=0.5, num_holes=4, max_h_size=4, max_w_size=4),
    ToFloat(max_value=255)
])

AUGMENTATIONS_VALID = Compose([
    # CLAHE(p=1.0, clip_limit=2.0),
    ToFloat(max_value=255)
])


class MobileAppImageSequence(Sequence):
    def __init__(self, image_path, y, batch_size, target_size, augmentations):
        self.image_paths = image_path
        self.y = y
        self.batch_size = batch_size
        self.target_size = target_size
        self.augment = augmentations

    def __len__(self):
        return int(np.ceil(len(self.image_paths) / float(self.batch_size)))

    def get_image(self, batch_paths):
        for img_path in batch_paths:
            yield np.array(Image.open(img_path))

    def __getitem__(self, idx):
        start_idx = idx * self.batch_size
        end_idx = (idx + 1) * self.batch_size
        paths = self.image_paths[start_idx:end_idx]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

        return np.stack([
            self.augment(image=x)["image"] for x in self.get_image(paths)],
            axis=0), np.array(batch_y)


def main():
    df = pd.read_csv(dataset_path / "data.csv")
    x_col = "image_path"
    y_col = "class"
    batch_size = 10
    num_classes = df[y_col].nunique()
    target_size = (Height, Width)
    label_encoder = LabelEncoder()
    label_encoder.fit(df[y_col])
    label = to_categorical(label_encoder.transform(df[y_col]),
                           num_classes=num_classes)
    gen = MobileAppImageSequence(image_path=df[x_col], y=label,
                                 target_size=target_size, batch_size=batch_size,
                                 augmentations=AUGMENTATIONS_KMNIST)
    imgs, label = gen.__getitem__(0)
    print(imgs.shape, label.shape)


if __name__ == '__main__':
    main()
