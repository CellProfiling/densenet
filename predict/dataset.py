import os

import cv2
import numpy as np
import torch
from torch.utils.data.dataset import Dataset

opj = os.path.join
ope = os.path.exists


class ProteinDataset(Dataset):
    def __init__(
        self, image_dir, image_size=512, crop_size=0, in_channels=4, suffix="png"
    ):
        self.image_dir = image_dir

        image_names = os.listdir(self.image_dir)
        # eg. ffd91122-bad0-11e8-b2b8-ac1f6b6435d0_red.png -> ffd91122-bad0-11e8-b2b8-ac1f6b6435d0
        self.image_ids = np.sort(
            np.unique(
                [image_name[: image_name.rfind("_")] for image_name in image_names]
            )
        )
        self.suffix = suffix

        self.transform = None

        self.image_size = image_size
        self.crop_size = crop_size
        self.in_channels = in_channels
        if in_channels == 3:
            self.colors = ["red", "green", "blue"]
        elif in_channels == 4:
            self.colors = ["red", "green", "blue", "yellow"]
        else:
            raise ValueError(in_channels)
        self.random_crop = False

        self.num = len(self.image_ids)

    def set_transform(self, transform=None):
        self.transform = transform

    def set_random_crop(self, random_crop=False):
        self.random_crop = random_crop

    def crop_image(self, image):
        random_crop_size = int(np.random.uniform(self.crop_size, self.image_size))
        x = int(np.random.uniform(0, self.image_size - random_crop_size))
        y = int(np.random.uniform(0, self.image_size - random_crop_size))
        crop_image = image[x : x + random_crop_size, y : y + random_crop_size]
        return crop_image

    def read_rgby(self, image_id):
        image = [
            cv2.imread(
                opj(self.image_dir, "%s_%s.%s" % (image_id, color, self.suffix)),
                cv2.IMREAD_GRAYSCALE,
            )
            for color in self.colors
        ]

        if image[0] is None:
            print(self.image_dir, image_id)

        image = np.stack(image, axis=-1)

        h, w = image.shape[:2]
        if self.image_size != h or self.image_size != w:
            image = cv2.resize(
                image,
                (self.image_size, self.image_size),
                interpolation=cv2.INTER_LINEAR,
            )

        if self.random_crop and self.crop_size > 0:
            image = self.crop_image(image)
        if self.crop_size > 0:
            h, w = image.shape[:2]
            if self.crop_size != h or self.crop_size != w:
                image = cv2.resize(
                    image,
                    (self.crop_size, self.crop_size),
                    interpolation=cv2.INTER_LINEAR,
                )

        return image

    def image_to_tensor(self, image, mean=0, std=1.0):
        image = image.astype(np.float32)
        image = (image - mean) / std
        image = image.transpose((2, 0, 1))
        tensor = torch.from_numpy(image)
        return tensor

    def __getitem__(self, index):
        image_id = self.image_ids[index]

        image = self.read_rgby(image_id)

        if self.transform is not None:
            image = self.transform(image)

        image = image / 255.0
        image = self.image_to_tensor(image)

        return image, index

    def __len__(self):
        return self.num


def augment_default(image):
    return image


def augment_flipud(image):
    image = np.flipud(image)
    return image


def augment_fliplr(image):
    image = np.fliplr(image)
    return image


def augment_transpose(image):
    image = np.transpose(image, (1, 0, 2))
    return image


def augment_flipud_lr(image):
    image = np.flipud(image)
    image = np.fliplr(image)
    return image


def augment_flipud_transpose(image):
    image = augment_flipud(image)
    image = augment_transpose(image)
    return image


def augment_fliplr_transpose(image):
    image = augment_fliplr(image)
    image = augment_transpose(image)
    return image


def augment_flipud_lr_transpose(image):
    image = augment_flipud(image)
    image = augment_fliplr(image)
    image = augment_transpose(image)
    return image
