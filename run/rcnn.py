import os
import torch
import torchvision
from torchvision.transforms import functional as F
from PIL import Image, ImageDraw
import xml.etree.ElementTree as ET

def load_model(model_path, device):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None, weights_backbone=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes=2)
    
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

def create_pascal_voc_xml(filename, image_size, boxes, output_path):
    annotation = ET.Element('annotation')
    ET.SubElement(annotation, 'filename').text = filename

    size = ET.SubElement(annotation, 'size')
    ET.SubElement(size, 'width').text = str(image_size[0])
    ET.SubElement(size, 'height').text = str(image_size[1])
    ET.SubElement(size, 'depth').text = '3'

    for box in boxes:
        xmin, ymin, xmax, ymax = map(int, box)
        obj = ET.SubElement(annotation, 'object')
        ET.SubElement(obj, 'name').text = 'object'
        bndbox = ET.SubElement(obj, 'bndbox')
        ET.SubElement(bndbox, 'xmin').text = str(xmin)
        ET.SubElement(bndbox, 'ymin').text = str(ymin)
        ET.SubElement(bndbox, 'xmax').text = str(xmax)
        ET.SubElement(bndbox, 'ymax').text = str(ymax)

    tree = ET.ElementTree(annotation)
    tree.write(output_path, encoding='utf-8', xml_declaration=True)

def predict_and_process(model, image_path, device, pred_dir, xml_dir, crop_dir, confidence_threshold=0.5):
    original_image = Image.open(image_path).convert("RGB") 
    image_for_drawing = original_image.copy()

    image_tensor = F.to_tensor(original_image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image_tensor)

    boxes = outputs[0]['boxes'].cpu()
    scores = outputs[0]['scores'].cpu()

    draw = ImageDraw.Draw(image_for_drawing)
    valid_boxes = []

    base_filename = os.path.splitext(os.path.basename(image_path))[0]  # page_1
    pred_image_name = f"pred_{base_filename}.png"

    for i, (box, score) in enumerate(zip(boxes, scores), start=1):
        if score >= confidence_threshold:
            box = list(map(int, box.tolist()))
            valid_boxes.append(box)

            draw.rectangle(box, outline="red", width=3)
            draw.text((box[0], box[1]), f"{score:.2f}", fill="red")

            cropped = original_image.crop(box)
            crop_name = f"{base_filename}_{i}.png"
            cropped.save(os.path.join(crop_dir, crop_name))

    image_for_drawing.save(os.path.join(pred_dir, pred_image_name))

    xml_path = os.path.join(xml_dir, f"{base_filename}.xml")
    create_pascal_voc_xml(pred_image_name, original_image.size, valid_boxes, xml_path)

def batch_process(model_path, input_dir, pred_dir, xml_dir, crop_dir, confidence_threshold=0.5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(model_path, device)

    os.makedirs(pred_dir, exist_ok=True)
    os.makedirs(xml_dir, exist_ok=True)
    os.makedirs(crop_dir, exist_ok=True)

    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    for filename in image_files:
        input_path = os.path.join(input_dir, filename)
        predict_and_process(model, input_path, device, pred_dir, xml_dir, crop_dir, confidence_threshold)

    print("Done!")

if __name__ == "__main__":
    batch_process(
        model_path=r"E:\zxz\project\models\fasterrcnn_shapes_m_3.pth",
        input_dir=r"E:\zxz\project\pages_b",
        pred_dir=r"E:\zxz\project\extracted\b2\pages",
        xml_dir=r"E:\zxz\project\extracted\b2\xml",
        crop_dir=r"E:\zxz\project\extracted\b2\cropped",
        confidence_threshold=0.5
    )
