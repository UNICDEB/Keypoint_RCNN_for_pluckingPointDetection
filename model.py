# model.py
import torchvision
from torchvision.models.detection.keypoint_rcnn import KeypointRCNN
from torchvision.models.detection import keypointrcnn_resnet50_fpn

def get_model(num_classes=2, num_keypoints=1, pretrained_backbone=True):
    # load model with pre-trained weights on COCO backbone
    model = keypointrcnn_resnet50_fpn(pretrained=False, progress=True,
                                      num_classes=num_classes,
                                      pretrained_backbone=pretrained_backbone,
                                      num_keypoints=num_keypoints)
    return model
