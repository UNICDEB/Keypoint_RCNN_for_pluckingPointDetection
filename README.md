# Saffron Flower Plucking Point Detection (Keypoint R-CNN)

This project detects **saffron flowers** and their **plucking point** using a **Keypoint R-CNN** model in PyTorch.  
It is designed for automating saffron harvesting by identifying the flower side view and the correct plucking location.

---

## 📂 Dataset
- Images: `Dataset_txt/images`
- Annotations: JSON / TXT format with:
  - **1 detection class** → `saffron_flower_side`
  - **1 keypoint** → `saffron_flower_plucking_point`

### Convert Annotations
- **From JSON** → `convert_annotations.py`
- **From TXT** → `convert_annotations_txt.py`  
  - Handles YOLO-like format:  
    ```
    class x_center y_center width height keypoint_x keypoint_y visibility
    ```
  - Converts to COCO-like JSON with absolute pixel coordinates.
  - Optionally saves visualized images in `corrected_images/`.

---

## 🏗️ Model
- Backbone: **ResNet50-FPN** (pretrained on COCO)
- Detection Head: **2 classes** (`background`, `saffron_flower_side`)
- Keypoints: **1 plucking point per flower**

Defined in [`model.py`](model.py):

```python
model = torchvision.models.detection.keypointrcnn_resnet50_fpn(weights="DEFAULT")
```

---

## ⚙️ Training
Run:

```bash
python train.py
```

Key features:
- Train/Val split = 80/20
- Data augmentations: flip, rotation, color jitter
- Optimizer: SGD (lr=0.0025, momentum=0.9, weight_decay=0.0005)
- LR scheduler: Step decay (every 50 epochs)
- Epochs: default `200`
- Checkpoints saved every 25 epochs → `Sample_Weight/`
- Final model saved → `Sample_Weight/keypointrcnn_saffron_final.pth`

Training loop logs per-iteration and per-epoch losses:
```
[Epoch 0 Iter 0] Total Loss 9.0513 | loss_classifier: 0.6746 | loss_box_reg: 0.2340 | loss_keypoint: 8.0783 ...
Epoch 1/200 --> Train Loss: 7.2345 | Val Loss: 7.1021
```

---

## 🔍 Inference
Run inference on a single image:

```bash
python inference.py
```

- Loads trained model
- Predicts bounding boxes + plucking point
- Filters detections by score (default ≥0.6)
- Saves visualization (`out_example.jpg`)
- Prints prediction details:
```
Det 0: score=0.74, box=[120, 250, 300, 480], keypoint=[[210, 460, 1]]
```

---

## 📊 Improvements Implemented
- Fixed YOLO TXT → COCO conversion (normalized → absolute pixels).
- Safe validation loop (handles variable keypoints per image).
- Pretrained COCO backbone initialization.
- Loss logging for each component.
- Visualization of corrected annotations for debugging.

---

## 🚀 Next Steps
- Add **TensorBoard logging** for live monitoring.
- Implement evaluation metrics (mAP, keypoint error).
- Deploy trained model in a real-time robotic plucking system.

---

## 📝 Citation
If you use this work, please cite:
```
Saffron Flower Plucking Point Detection using Keypoint R-CNN, 2025.
```
