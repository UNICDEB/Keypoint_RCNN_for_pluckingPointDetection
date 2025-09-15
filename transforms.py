# transforms.py
import torchvision
from PIL import Image
import random
import torch
import torchvision.transforms.functional as F

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms
    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

class ToTensor:
    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target

class RandomHorizontalFlip:
    def __init__(self, prob=0.5):
        self.prob = prob
    def __call__(self, image, target):
        if random.random() < self.prob:
            image = F.hflip(image)
            w, h = image.shape[2], image.shape[1]
            # flip boxes
            boxes = target['boxes']
            boxes = boxes.clone()
            boxes[:, [0,2]] = w - boxes[:, [2,0]]
            target['boxes'] = boxes
            # flip keypoints: x -> w - x if visible
            kps = target['keypoints']
            if kps is not None:
                kps = kps.clone()
                kps[:, :, 0] = w - kps[:, :, 0]
                target['keypoints'] = kps
        return image, target

def get_train_transforms():
    return Compose([
        ToTensor(),
        RandomHorizontalFlip(0.5),
    ])

def get_test_transforms():
    return Compose([
        ToTensor(),
    ])
