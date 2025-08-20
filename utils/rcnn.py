import os
import torch
import torchvision
from torchvision.transforms import functional as F
from PIL import Image, ImageDraw
import xml.etree.ElementTree as ET

# Allowed image extensions
ALLOWED_EXTS = ('.jpg', '.jpeg', '.png')

def load_model(model_path, device):
    """
    Load a Faster R-CNN model from checkpoint.
    """
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None, weights_backbone=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # Replace the predictor for 2 classes: background + object
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes=2)

    # Load checkpoint (supporting torch >=2.0 with weights_only argument)
    try:
        state_dict = torch.load(model_path, map_location=device, weights_only=True)
    except TypeError:
        state_dict = torch.load(model_path, map_location=device)

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

def create_pascal_voc_xml(filename, image_size, boxes, output_path):
    """
    Create a Pascal VOC style XML annotation file.
    """
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

def find_corresponding_image(folder, base_name):
    # First check for exact matches with allowed extensions
    for ext in ALLOWED_EXTS:
        candidate = os.path.join(folder, base_name + ext)
        if os.path.isfile(candidate):
            return candidate
    # Otherwise, search through folder
    for f in os.listdir(folder):
        name, ext = os.path.splitext(f)
        if name == base_name and ext.lower() in ALLOWED_EXTS:
            return os.path.join(folder, f)
    return None

def predict_and_process(
    model,
    pred_image_path,
    crop_source_path,
    device,
    pred_dir,
    xml_dir,
    crop_dir,
    confidence_threshold=0.5
):
    pred_image = Image.open(pred_image_path).convert("RGB")
    image_tensor = F.to_tensor(pred_image).unsqueeze(0).to(device)

    # Run inference
    with torch.no_grad():
        outputs = model(image_tensor)

    boxes = outputs[0]['boxes'].cpu()
    scores = outputs[0]['scores'].cpu()

    # Draw predictions on image
    vis_image = pred_image.copy()
    draw = ImageDraw.Draw(vis_image)
    valid_boxes = []

    base_filename = os.path.splitext(os.path.basename(pred_image_path))[0]
    pred_image_name = f"pred_{base_filename}.png"

    for i, (box, score) in enumerate(zip(boxes, scores), start=1):
        if score >= confidence_threshold:
            box = list(map(int, box.tolist()))
            valid_boxes.append(box)
            # Draw bounding box and score
            draw.rectangle(box, outline="red", width=3)
            draw.text((box[0], box[1]), f"{score:.2f}", fill="red")

    # Save visualization image
    vis_save_path = os.path.join(pred_dir, pred_image_name)
    vis_image.save(vis_save_path)

    # Crop objects from source image
    crop_source_img = Image.open(crop_source_path).convert("RGB")
    crop_base = os.path.splitext(os.path.basename(crop_source_path))[0]
    for i, box in enumerate(valid_boxes, start=1):
        cropped = crop_source_img.crop(box)
        crop_name = f"{crop_base}_{i}.png"
        cropped.save(os.path.join(crop_dir, crop_name))

    # Save XML annotations (reference the source image)
    xml_filename_ref = os.path.basename(crop_source_path)
    xml_path = os.path.join(xml_dir, f"{base_filename}.xml")
    create_pascal_voc_xml(xml_filename_ref, crop_source_img.size, valid_boxes, xml_path)

def batch_process(
    model_path,
    input_dir,
    crop_source_dir,
    pred_dir,
    xml_dir,
    crop_dir,
    confidence_threshold=0.5
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(model_path, device)

    # Ensure output directories exist
    os.makedirs(pred_dir, exist_ok=True)
    os.makedirs(xml_dir, exist_ok=True)
    os.makedirs(crop_dir, exist_ok=True)

    # Collect images from input_dir
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(ALLOWED_EXTS)]

    for filename in image_files:
        pred_path = os.path.join(input_dir, filename)
        base = os.path.splitext(filename)[0]

        crop_source_path = find_corresponding_image(crop_source_dir, base)

        predict_and_process(
            model=model,
            pred_image_path=pred_path,
            crop_source_path=crop_source_path,
            device=device,
            pred_dir=pred_dir,
            xml_dir=xml_dir,
            crop_dir=crop_dir,
            confidence_threshold=confidence_threshold
        )

    print("Done!")

if __name__ == "__main__":
    batch_process(
        model_path=r"E:\zxz\project\models\fasterrcnn_shapes.pth",
        input_dir=r"E:\zxz\project\pages_b",
        crop_source_dir=r"E:\zxz\project\pages_a",
        pred_dir=r"E:\zxz\project\extracted\b\pages",
        xml_dir=r"E:\zxz\project\extracted\b\xml",
        crop_dir=r"E:\zxz\project\extracted\b\cropped",
        confidence_threshold=0.5
    )
