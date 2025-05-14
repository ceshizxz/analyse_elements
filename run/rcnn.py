import os
import torch
import torchvision
from torchvision.transforms import functional as F
from PIL import Image, ImageDraw

def load_model(model_path, device):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None, weights_backbone=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes=2)
    
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

def predict_and_draw(model, image_path, device, confidence_threshold=0.5):
    image = Image.open(image_path).convert("RGB")
    image_tensor = F.to_tensor(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image_tensor)

    boxes = outputs[0]['boxes'].cpu()
    scores = outputs[0]['scores'].cpu()

    draw = ImageDraw.Draw(image)
    for box, score in zip(boxes, scores):
        if score >= confidence_threshold:
            draw.rectangle(box.tolist(), outline="red", width=3)
            draw.text((box[0], box[1]), f"{score:.2f}", fill="red")

    return image

def batch_inference(model_path, input_dir, output_dir, confidence_threshold=0.5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(model_path, device)

    os.makedirs(output_dir, exist_ok=True)
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    for i, filename in enumerate(image_files):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, f"pred_{filename}")

        result_image = predict_and_draw(model, input_path, device, confidence_threshold=confidence_threshold)
        result_image.save(output_path)

    print("Done!")

if __name__ == "__main__":
    batch_inference(
        model_path=r"C:\Users\ce-sh\Desktop\UCL\PHAS0077\analyse_elements\save\fasterrcnn_shapes.pth",
        input_dir=r"E:\zxz\project\pages", 
        output_dir=r"E:\zxz\project\extracted",
        confidence_threshold=0.5
    )
