
from torchvision import transforms
import torch
from torch import Tensor
import math
import numbers
import warnings
import numpy as np
from typing import List, Tuple
from collections.abc import Sequence

import torch
from torch import Tensor
import torchvision.transforms as transforms

from torchvision import transforms
from torchvision.transforms import functional as F, InterpolationMode
from torchvision.transforms.functional import _interpolation_modes_from_int

from asym_patch_sampling import get_random_patch_sequence_index_pair


class ComposeWithBox:

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):

        img, bbox = self.transforms[0](img)

        for t in self.transforms[1:]:
            img = t(img)

        return img, bbox

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += f"    {t}"
        format_string += "\n)"
        return format_string


class TwoCropsTransform:

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return [q, k]


class TwoCropsTransformBox:
    """Take two random crops of one image"""

    def __init__(self, base_transform1, base_transform2, time ,sampling_ratio,power):
        self.base_transform1 = base_transform1
        self.base_transform2 = base_transform2
        self.time=time
        self.sampling_ratio=sampling_ratio
        self.power=power
    def __call__(self, x):
        im1, box1 = self.base_transform1(x)
        im2, box2 = self.base_transform2(x)
        box1 = torch.tensor(box1)
        box2 = torch.tensor(box2)
        idxs1, idxs2 = get_random_patch_sequence_index_pair(box1.tolist(), box2.tolist(), ngrid=16,  sampling_ratio=self.sampling_ratio, time=self.time,power=self.power)     
        return (im1, np.array(idxs1)), (im2, np.array(idxs2))


def _setup_size(size, error_msg):
    if isinstance(size, numbers.Number):
        return int(size), int(size)

    if isinstance(size, Sequence) and len(size) == 1:
        return size[0], size[0]

    if len(size) != 2:
        raise ValueError(error_msg)

    return size


class RandomResizedCropBox(torch.nn.Module):

    def __init__(self, size, scale=(0.08, 1.0), ratio=(3.0 / 4.0, 4.0 / 3.0), interpolation=InterpolationMode.BILINEAR):
        super().__init__()
        self.size = _setup_size(size, error_msg="Please provide only two dimensions (h, w) for size.")

        if not isinstance(scale, Sequence):
            raise TypeError("Scale should be a sequence")
        if not isinstance(ratio, Sequence):
            raise TypeError("Ratio should be a sequence")
        if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
            warnings.warn("Scale and ratio should be of kind (min, max)")

        # Backward compatibility with integer value
        if isinstance(interpolation, int):
            warnings.warn(
                "Argument interpolation should be of type InterpolationMode instead of int. "
                "Please, use InterpolationMode enum."
            )
            interpolation = _interpolation_modes_from_int(interpolation)

        self.interpolation = interpolation
        self.scale = scale
        self.ratio = ratio

    @staticmethod
    def get_params(img: Tensor, scale: List[float], ratio: List[float]) -> Tuple[int, int, int, int]:
        width, height = F.get_image_size(img)
        # width, height = F._get_image_size(img)
        area = height * width

        log_ratio = torch.log(torch.tensor(ratio))
        for _ in range(10):
            target_area = area * torch.empty(1).uniform_(scale[0], scale[1]).item()
            aspect_ratio = torch.exp(torch.empty(1).uniform_(log_ratio[0], log_ratio[1])).item()

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if 0 < w <= width and 0 < h <= height:
                i = torch.randint(0, height - h + 1, size=(1,)).item()
                j = torch.randint(0, width - w + 1, size=(1,)).item()
                return i, j, h, w

        # Fallback to central crop
        in_ratio = float(width) / float(height)
        if in_ratio < min(ratio):
            w = width
            h = int(round(w / min(ratio)))
        elif in_ratio > max(ratio):
            h = height
            w = int(round(h * max(ratio)))
        else:  # whole image
            w = width
            h = height
        i = (height - h) // 2
        j = (width - w) // 2
        return i, j, h, w

    def forward(self, img):
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        img = F.resized_crop(img, i, j, h, w, self.size, self.interpolation)
        return img, (i, j, h, w)

    def __repr__(self) -> str:
        interpolate_str = self.interpolation.value
        format_string = self.__class__.__name__ + f"(size={self.size}"
        format_string += f", scale={tuple(round(s, 4) for s in self.scale)}"
        format_string += f", ratio={tuple(round(r, 4) for r in self.ratio)}"
        format_string += f", interpolation={interpolate_str})"
        return format_string


train_transforms = ComposeWithBox([
    RandomResizedCropBox(32, scale=(0.15, 1.)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomApply([
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
    ], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])


eval_transform_train = transforms.Compose([
    transforms.RandomResizedCrop(32, scale=(0.2, 1.)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

eval_transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])
