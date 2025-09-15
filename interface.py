# # Inference & visualization
# # Load model, run on an image, filter by score, and draw box + keypoint.
# # inference.py

# import torch
# from PIL import Image, ImageDraw
# import torchvision.transforms.functional as F
# from model import get_model

# device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# model = get_model(num_classes=2, num_keypoints=1)
# model.load_state_dict(torch.load("keypointrcnn_saffron_final.pth", map_location=device))
# model.to(device)
# model.eval()

# def predict_and_visualize(image_path, score_thresh=0.6, save_path=None):
#     img = Image.open(image_path).convert("RGB")
#     img_tensor = F.to_tensor(img).to(device)
#     with torch.no_grad():
#         outputs = model([img_tensor])
#     out = outputs[0]
#     boxes = out['boxes'].cpu()
#     scores = out['scores'].cpu()
#     keypoints = out['keypoints'].cpu()  # shape (num_dets, num_kpts, 3)
#     draw = ImageDraw.Draw(img)

#     for i, s in enumerate(scores):
#         if s < score_thresh:
#             continue
#         box = boxes[i].numpy().tolist()
#         draw.rectangle(box, outline="red", width=2)
#         kpt = keypoints[i][0]  # only one keypoint
#         x, y, v = float(kpt[0]), float(kpt[1]), float(kpt[2])
#         if v > 0:
#             r = 3
#             draw.ellipse((x-r, y-r, x+r, y+r), fill="blue")

#     if save_path:
#         img.save(save_path)
#     return img

# if __name__ == "__main__":
#     img = predict_and_visualize("image_8.jpeg", save_path="out_example.jpg")
#     img.show()

####################
# inference.py
import torch
from PIL import Image, ImageDraw
import torchvision.transforms.functional as F
from model import get_model

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model = get_model(num_classes=2, num_keypoints=1)
model.load_state_dict(torch.load("keypointrcnn_saffron_final.pth", map_location=device))
model.to(device)
model.eval()

def predict_and_visualize(image_path, score_thresh=0.2, save_path=None):
    img = Image.open(image_path).convert("RGB")
    img_tensor = F.to_tensor(img).to(device)

    with torch.no_grad():
        outputs = model([img_tensor])
    out = outputs[0]

    boxes = out['boxes'].cpu().numpy()
    scores = out['scores'].cpu().numpy()
    keypoints = out['keypoints'].cpu().numpy()  # shape (num_dets, num_kpts, 3)

    draw = ImageDraw.Draw(img)

    print(f"\n🔎 Predictions for {image_path}:")
    if len(scores) == 0:
        print("⚠️ No detections found")
    else:
        for i, (box, score, kpts) in enumerate(zip(boxes, scores, keypoints)):
            print(f" Det {i}: score={score:.4f}, box={box}, keypoint={kpts.tolist()}")

            if score < score_thresh:
                continue

            # Draw bounding box
            draw.rectangle(box.tolist(), outline="red", width=2)

            # Draw keypoint (only one in your case)
            x, y, v = kpts[0]
            if v > 0:
                r = 3
                draw.ellipse((x-r, y-r, x+r, y+r), fill="blue")

    if save_path:
        img.save(save_path)
        print(f"✅ Saved visualization → {save_path}")

    return img

if __name__ == "__main__":
    img = predict_and_visualize("image_10.jpeg", save_path="out_example.jpg")
    img.show()
