from typing import Tuple

import cv2
import numpy as np

from torchvision.datasets import VOCSegmentation
from torchvision.transforms import transforms
from data.base import BaseDataset

from data.transform_utils import ratio_preserved_resize, to_float, img2roi


class PascalVOCDataset(BaseDataset):
    # TODO: resize image, mask at getitem to fixed shapes
    def __init__(self, data_dir: str, _type: str = "train", **kwargs):
        super().__init__(data_dir, _type)
        self.data = self.load_data(**kwargs)
        self.transform = {
            "image": transforms.Compose([ratio_preserved_resize((256, 256)), to_float(), transforms.ToTensor()]),
            "mask": transforms.Compose([ratio_preserved_resize((256, 256)), to_float(), transforms.ToTensor()])
        }

    def __getitem__(self, index):
        image, mask = self.data[index]
        image = np.array(image.getdata()).reshape(image.size[0], image.size[1], 3)
        mask = np.array(mask.getdata()).reshape(mask.size[0], mask.size[1])
        if self.transform:
            image = self.transform["image"](image)
            mask = self.transform["mask"](mask)
        return image, mask

    def __len__(self):
        return len(self.data)

    def load_data(self, **kwargs):
        _type = self._type if self._type == "train" else "val"
        return VOCSegmentation(root=self.data_dir, image_set=_type, download=True, year=kwargs.get("year", "2012"))

    def get_alias(self):
        return "voc_segmentation"
