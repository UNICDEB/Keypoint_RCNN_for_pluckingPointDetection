# train.py
import torch
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
from dataset_saffron import SaffronKeypointDataset
from transforms import get_train_transforms, get_test_transforms
from model import get_model


def collate_fn(batch):
    return tuple(zip(*batch))

def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=50):
    model.train()
    for i, (images, targets) in enumerate(data_loader):
        images = list(img.to(device) for img in images)
        targets = [{k: v.to(device) for k,v in t.items()} for t in targets]
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if i % print_freq == 0:
            print(f"Epoch {epoch} Iter {i} Loss {losses.item():.4f}")

def main():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    images_dir = "Dataset_txt/images"
    ann_json = "annotations_coco_like.json"

    dataset_full = SaffronKeypointDataset(images_dir, ann_json, transforms=get_train_transforms())
    # split
    n = len(dataset_full)
    val_size = int(n*0.2)
    train_size = n - val_size
    train_dataset, val_dataset = random_split(dataset_full, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=2, collate_fn=collate_fn)

    model = get_model(num_classes=2, num_keypoints=1)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    num_epochs = 500
    for epoch in range(num_epochs):
        train_one_epoch(model, optimizer, train_loader, device, epoch, print_freq=20)
        lr_scheduler.step()

        # save checkpoint everty 25 epochs
        if (epoch + 1) % 25 == 0:
            torch.save(model.state_dict(), f"Sample_Weight/checkpoint_epoch_{epoch+1}.pth")

    # Save final
    torch.save(model.state_dict(), "Sample_Weight/keypointrcnn_saffron_final.pth")

if __name__ == "__main__":
    main()
