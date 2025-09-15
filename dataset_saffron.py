# dataset_saffron.py
import torch
from torch.utils.data import Dataset
from PIL import Image
import json
import os

class SaffronKeypointDataset(Dataset):
    def __init__(self, images_dir, annotation_json, transforms=None):
        """
        annotation_json: path to the coco_like list saved earlier
        """
        with open(annotation_json, "r") as f:
            self.records = json.load(f)
        self.images_dir = images_dir
        self.transforms = transforms

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        img_path = os.path.join(self.images_dir, rec['file_name'])
        img = Image.open(img_path).convert("RGB")
        boxes = rec['boxes']
        # Filter out invalid boxes
        valid_boxes = []
        valid_labels = []
        valid_keypoints = []
        for i, b in enumerate(boxes):
            if b[2] > b[0] and b[3] > b[1]:
                valid_boxes.append(b)
                valid_labels.append(rec['labels'][i])
                if 'keypoints' in rec and len(rec['keypoints']) > i:
                    valid_keypoints.append(rec['keypoints'][i])
        if not valid_boxes:
            raise ValueError(f"No valid boxes for image {rec['file_name']}")
        boxes = torch.as_tensor(valid_boxes, dtype=torch.float32)
        labels = torch.as_tensor(valid_labels, dtype=torch.int64)

        # keypoints expected shape (N, K, 3) -> K=1
        kps = rec.get('keypoints', [])
        if len(kps) == 0:
            keypoints = torch.zeros((boxes.shape[0], 1, 3), dtype=torch.float32)
        else:
            # convert list [x,y,v] to (N,1,3)
            kps_tensor = torch.tensor(kps, dtype=torch.float32)
            keypoints = kps_tensor.view(-1, 1, 3)

        image_id = torch.tensor([rec['image_id']])
        area = torch.as_tensor(rec.get('area', [ (b[2]-b[0])*(b[3]-b[1]) for b in rec['boxes'] ]), dtype=torch.float32)
        iscrowd = torch.as_tensor(rec.get('iscrowd', [0]*len(rec['boxes'])), dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "keypoints": keypoints,
            "image_id": image_id,
            "area": area,
            "iscrowd": iscrowd
        }

        if self.transforms:
            img, target = self.transforms(img, target)

        return img, target
