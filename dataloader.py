import warnings

import cv2
import numpy as np
from albumentations import (
    Compose, HueSaturationValue, RandomBrightness, RandomContrast, RandomGamma,
    ToFloat, Cutout, HorizontalFlip, PadIfNeeded, RandomCrop, RandomScale,
    Resize, Rotate
)
from albumentations import ImageOnlyTransform
from tensorflow.python.keras.utils.data_utils import Sequence

warnings.filterwarnings('ignore')

Height, Width = 224, 224


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


class MNISTSequence(Sequence):
    def __init__(self, x_set, y_set, batch_size, augmentations):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size
        self.augment = augmentations

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

        return np.stack([
            self.augment(image=x)["image"] for x in batch_x
        ], axis=0), np.array(batch_y)


def main():
    pass


if __name__ == '__main__':
    main()
