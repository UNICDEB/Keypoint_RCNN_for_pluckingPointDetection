# # model.py
# import torchvision
# from torchvision.models.detection.keypoint_rcnn import KeypointRCNN
# from torchvision.models.detection import keypointrcnn_resnet50_fpn

# def get_model(num_classes=2, num_keypoints=1, pretrained_backbone=True):
#     # load model with pre-trained weights on COCO backbone
#     model = keypointrcnn_resnet50_fpn(pretrained=False, progress=True,
#                                       num_classes=num_classes,
#                                       pretrained_backbone=pretrained_backbone,
#                                       num_keypoints=num_keypoints)
#     return model

# model.py
import torchvision
from torchvision.models.detection import KeypointRCNN
from torchvision.models.detection.rpn import AnchorGenerator


def get_model(num_classes=2, num_keypoints=1, pretrained=True):
    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.keypointrcnn_resnet50_fpn(
        weights="DEFAULT" if pretrained else None
    )

    # Replace the box predictor (for your dataset)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
        in_features, num_classes
    )

    # Replace the keypoint predictor (for your dataset)
    in_features_keypoint = model.roi_heads.keypoint_predictor.kps_score_lowres.in_channels
    model.roi_heads.keypoint_predictor = torchvision.models.detection.keypoint_rcnn.KeypointRCNNPredictor(
        in_features_keypoint, num_keypoints
    )

    return model
