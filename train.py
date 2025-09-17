# # train.py
# import torch
# from torch.utils.data import DataLoader, random_split
# import torch.optim as optim
# from dataset_saffron import SaffronKeypointDataset
# from transforms import get_train_transforms, get_test_transforms
# from model import get_model


# def collate_fn(batch):
#     return tuple(zip(*batch))

# def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=50):
#     model.train()
#     for i, (images, targets) in enumerate(data_loader):
#         images = list(img.to(device) for img in images)
#         targets = [{k: v.to(device) for k,v in t.items()} for t in targets]
#         loss_dict = model(images, targets)
#         losses = sum(loss for loss in loss_dict.values())

#         optimizer.zero_grad()
#         losses.backward()
#         optimizer.step()

#         if i % print_freq == 0:
#             print(f"Epoch {epoch} Iter {i} Loss {losses.item():.4f}")

# def main():
#     device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
#     images_dir = "Dataset_txt/images"
#     ann_json = "annotations_coco_like.json"

#     dataset_full = SaffronKeypointDataset(images_dir, ann_json, transforms=get_train_transforms())
#     # split
#     n = len(dataset_full)
#     val_size = int(n*0.2)
#     train_size = n - val_size
#     train_dataset, val_dataset = random_split(dataset_full, [train_size, val_size])

#     train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4, collate_fn=collate_fn)
#     val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=2, collate_fn=collate_fn)

#     model = get_model(num_classes=2, num_keypoints=1)
#     model.to(device)

#     params = [p for p in model.parameters() if p.requires_grad]
#     optimizer = optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
#     lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

#     num_epochs = 500
#     for epoch in range(num_epochs):
#         train_one_epoch(model, optimizer, train_loader, device, epoch, print_freq=20)
#         lr_scheduler.step()

#         # save checkpoint everty 25 epochs
#         if (epoch + 1) % 25 == 0:
#             torch.save(model.state_dict(), f"Sample_Weight/checkpoint_epoch_{epoch+1}.pth")

#     # Save final
#     torch.save(model.state_dict(), "Sample_Weight/keypointrcnn_saffron_final.pth")

# if __name__ == "__main__":
#     main()

##############
# train.py
import torch
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
from dataset_saffron import SaffronKeypointDataset
from transforms import get_train_transforms, get_test_transforms
from model import get_model
import os


def collate_fn(batch):
    return tuple(zip(*batch))


def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=50):
    model.train()
    running_loss = 0.0

    for i, (images, targets) in enumerate(data_loader):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        running_loss += losses.item()

        if i % print_freq == 0:
            loss_details = " | ".join([f"{k}: {v.item():.4f}" for k, v in loss_dict.items()])
            print(f"[Epoch {epoch} Iter {i}] Total Loss {losses.item():.4f} | {loss_details}")

    return running_loss / len(data_loader)


def evaluate_one_epoch(model, data_loader, device):
    model.train()  # <-- set to train mode to get losses!
    val_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for images, targets in data_loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            batch_loss = 0.0
            for img, tgt in zip(images, targets):
                loss_dict = model([img], [tgt])  # single image
                if isinstance(loss_dict, dict):
                    loss_val = sum(loss for loss in loss_dict.values()).item()
                else:
                    # If not dict, skip or set to zero
                    loss_val = 0.0
                batch_loss += loss_val

            batch_loss /= len(images)
            val_loss += batch_loss
            num_batches += 1

    return val_loss / max(1, num_batches)




def main():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Using device: {device}")

    # dataset paths
    images_dir = "Dataset_txt/images"
    ann_json = "annotations_coco_like.json"

    dataset_full = SaffronKeypointDataset(images_dir, ann_json, transforms=get_train_transforms())

    # split train/val
    n = len(dataset_full)
    val_size = int(n * 0.2)
    train_size = n - val_size
    train_dataset, val_dataset = random_split(dataset_full, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=2, collate_fn=collate_fn)

    # 🔹 load model with pretrained COCO weights
    model = get_model(num_classes=2, num_keypoints=1, pretrained=True)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(params, lr=0.0025, momentum=0.9, weight_decay=0.0005)

    # 🔹 slower LR schedule
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

    os.makedirs("Sample_Weight", exist_ok=True)

    num_epochs = 500
    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, optimizer, train_loader, device, epoch, print_freq=20)
        val_loss = evaluate_one_epoch(model, val_loader, device)
        lr_scheduler.step()

        print(f"\nEpoch {epoch+1}/{num_epochs} --> Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        # save checkpoint every 25 epochs
        if (epoch + 1) % 25 == 0:
            ckpt_path = f"Sample_Weight/checkpoint_epoch_{epoch+1}.pth"
            torch.save(model.state_dict(), ckpt_path)
            print(f"✅ Saved checkpoint {ckpt_path}")

    # Save final model
    torch.save(model.state_dict(), "Sample_Weight/keypointrcnn_saffron_final.pth")
    print("✅ Training finished, final model saved.")


if __name__ == "__main__":
    main()
