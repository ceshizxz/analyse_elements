import os
import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from PIL import Image
import xml.etree.ElementTree as ET
from tqdm import tqdm

# Custom dataset for loading images and bounding boxes
class ShapeDataset(Dataset):
    def __init__(self, image_dir, xml_dir, transforms=None):
        self.image_dir = image_dir
        self.xml_dir = xml_dir
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]
        self.transforms = transforms

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        xml_path = os.path.join(self.xml_dir, img_name.replace(".jpg", ".xml"))

        img = Image.open(img_path).convert("RGB")
        boxes = []
        labels = []

        # Parse bounding boxes from Pascal VOC style XML
        tree = ET.parse(xml_path)
        root = tree.getroot()
        for obj in root.findall("object"):
            bndbox = obj.find("bndbox")
            xmin = int(float(bndbox.find("xmin").text))
            ymin = int(float(bndbox.find("ymin").text))
            xmax = int(float(bndbox.find("xmax").text))
            ymax = int(float(bndbox.find("ymax").text))
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(1)  # only one class: shape

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx])
        }

        if self.transforms:
            img = self.transforms(img)

        return img, target

# Allow DataLoader to handle different-size targets
def collate_fn(batch):
    return tuple(zip(*batch))

def train_model(
    image_dir,
    xml_dir,
    num_epochs,
    output_path,
    lr=0.005,
    step_size=5,
    gamma=0.1,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create dataset and dataloader
    dataset = ShapeDataset(image_dir, xml_dir, transforms=F.to_tensor)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)

    # Load Faster R-CNN model with ResNet50 backbone + FPN
    model = fasterrcnn_resnet50_fpn(
        weights=torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    )
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
        in_features, num_classes=2  # background + shape
    )

    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=0.0005)

    # Learning rate scheduler (StepLR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    best_loss = float("inf")
    best_epoch = -1

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        batch_count = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for images, targets in pbar:
            # Skip empty targets (no boxes)
            filtered = [(img, tgt) for img, tgt in zip(images, targets) if tgt["boxes"].numel() > 0]
            if not filtered:
                continue
            images, targets = zip(*filtered)

            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # Compute loss
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            loss_value = losses.item()
            total_loss += loss_value
            batch_count += 1

            pbar.set_postfix({
                "loss": f"{loss_value:.4f}",
                "lr": f"{scheduler.get_last_lr()[0]:.5f}"
            })

        epoch_loss = total_loss / max(batch_count, 1)
        print(f"[Epoch {epoch+1}] avg_loss: {epoch_loss:.4f} | lr: {scheduler.get_last_lr()[0]:.6f}")

        scheduler.step()

        # Save best model
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_epoch = epoch + 1
            torch.save(model.state_dict(), output_path)
            print(f"Best model updated at epoch {best_epoch} (avg_loss={best_loss:.4f}) -> {output_path}")

    print(f"Training done. Best epoch: {best_epoch} with avg_loss={best_loss:.4f}")
    torch.save(model.state_dict(), os.path.splitext(output_path)[0] + "_last.pth")
    print(f"Last epoch model saved at: {os.path.splitext(output_path)[0] + '_last.pth'}")

if __name__ == "__main__":
    train_model(
        image_dir=r"E:\zxz\project\images\generated_regions\images",
        xml_dir=r"E:\zxz\project\images\generated_regions\xmls",
        num_epochs=20,
        output_path=r"E:\zxz\project\models\fasterrcnn_shapes.pth",
        lr=0.005,
        step_size=5,
        gamma=0.1
    )
