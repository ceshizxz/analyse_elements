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

        tree = ET.parse(xml_path)
        root = tree.getroot()
        for obj in root.findall("object"):
            bndbox = obj.find("bndbox")
            xmin = int(float(bndbox.find("xmin").text))
            ymin = int(float(bndbox.find("ymin").text))
            xmax = int(float(bndbox.find("xmax").text))
            ymax = int(float(bndbox.find("ymax").text))
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(1) 

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

def collate_fn(batch):
    return tuple(zip(*batch))

def train_model(image_dir, xml_dir, num_epochs=10, output_path="fasterrcnn_model.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = ShapeDataset(image_dir, xml_dir, transforms=F.to_tensor)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)

    model = fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes=2)  # 1类 + 背景

    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for images, targets in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            images = list(img.to(device) for img in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            total_loss += losses.item()

        print(f"Epoch {epoch+1} Loss: {total_loss:.4f}")

    torch.save(model.state_dict(), output_path)
    print(f"model saved at: {output_path}")

if __name__ == "__main__":
    train_model(
        image_dir=r"E:\zxz\project\images\generated_regions\images",
        xml_dir=r"E:\zxz\project\images\generated_regions\xmls",
        num_epochs=10,
        output_path=r"C:\Users\ce-sh\Desktop\UCL\PHAS0077\analyse_elements\save\fasterrcnn_shapes.pth"
    )
