import os
import random
import xml.etree.ElementTree as ET
from PIL import Image

def bboxes_overlap(bbox1, bbox2):
    x1_min, y1_min, x1_max, y1_max = bbox1
    x2_min, y2_min, x2_max, y2_max = bbox2
    return not (x1_max <= x2_min or x2_max <= x1_min or y1_max <= y2_min or y2_max <= y1_min)

def paste_shape_no_overlap(background, shape_path, side, existing_bboxes, scale_factor=10, max_attempts=50):
    from PIL import ImageOps
    import numpy as np

    bg_width, bg_height = background.size
    shape_orig = Image.open(shape_path).convert("RGBA")

    # === 1. 缩放 ===
    max_scale_w = (bg_width * 0.3) / shape_orig.width
    max_scale_h = (bg_height * 0.8) / shape_orig.height
    safe_scale = min(scale_factor, max_scale_w, max_scale_h, 1.0)
    shape = shape_orig.resize(
        (int(shape_orig.width * safe_scale), int(shape_orig.height * safe_scale)),
        Image.LANCZOS
    )

    # === 2. 40% 概率随机镜像 ===
    if random.random() < 0.4:
        flip_mode = random.choice(["horizontal", "vertical"])
        if flip_mode == "horizontal":
            shape = shape.transpose(Image.FLIP_LEFT_RIGHT)
        else:
            shape = shape.transpose(Image.FLIP_TOP_BOTTOM)

    # === 3. 40% 概率平行四边形拉伸 ===
    if random.random() < 0.4:
        dx = random.uniform(-0.2, 0.2)
        dy = random.uniform(-0.2, 0.2)
        w, h = shape.size
        shape = shape.transform(
            (w, h),
            Image.AFFINE,
            (1, dx, 0, dy, 1, 0),
            resample=Image.BICUBIC,
            fillcolor=(0, 0, 0, 0)
        )

    # === 4. 40% 概率旋转 ===
    if random.random() < 0.4:
        angle = random.uniform(-30, 30)
        shape = shape.rotate(angle, expand=True)

    shape_width, shape_height = shape.size

    # === 5. 区域限制 ===
    if side == "right":
        x_min = int(bg_width * 0.67)
        x_max = bg_width - shape_width
    else:
        x_min = 0
        x_max = int(bg_width * 0.33) - shape_width

    # === 6. 查找合适位置（不重叠）===
    for _ in range(max_attempts):
        x = random.randint(x_min, max(x_min + 1, x_max))
        y = random.randint(0, bg_height - shape_height)
        new_bbox = (x, y, x + shape_width, y + shape_height)
        if all(not bboxes_overlap(new_bbox, bbox) for bbox in existing_bboxes):
            background.paste(shape, (x, y), shape)
            return new_bbox

    return None

def generate_image_with_shapes(background_path, shape_paths, output_image_path, output_xml_path, side="right"):
    background = Image.open(background_path).convert("RGB")
    bboxes = []

    num_shapes = random.randint(1, 2)
    chosen_shapes = random.choices(shape_paths, k=num_shapes)

    for shape_path in chosen_shapes:
        bbox = paste_shape_no_overlap(background, shape_path, side, bboxes)
        if bbox:
            bboxes.append(bbox)

    background.save(output_image_path)

    bg_width, bg_height = background.size
    annotation = ET.Element('annotation')
    ET.SubElement(annotation, 'folder').text = os.path.basename(os.path.dirname(output_image_path))
    ET.SubElement(annotation, 'filename').text = os.path.basename(output_image_path)
    size = ET.SubElement(annotation, 'size')
    ET.SubElement(size, 'width').text = str(bg_width)
    ET.SubElement(size, 'height').text = str(bg_height)
    ET.SubElement(size, 'depth').text = '3'

    for bbox in bboxes:
        obj = ET.SubElement(annotation, 'object')
        ET.SubElement(obj, 'name').text = 'Shape'
        ET.SubElement(obj, 'pose').text = 'Unspecified'
        ET.SubElement(obj, 'truncated').text = '0'
        ET.SubElement(obj, 'difficult').text = '0'
        bndbox = ET.SubElement(obj, 'bndbox')
        ET.SubElement(bndbox, 'xmin').text = str(bbox[0])
        ET.SubElement(bndbox, 'ymin').text = str(bbox[1])
        ET.SubElement(bndbox, 'xmax').text = str(bbox[2])
        ET.SubElement(bndbox, 'ymax').text = str(bbox[3])

    tree = ET.ElementTree(annotation)
    tree.write(output_xml_path)

def generate_combined(
    background_paths,
    shape_dir,
    output_image_dir,
    output_xml_dir,
    total_count=20
):
    os.makedirs(output_image_dir, exist_ok=True)
    os.makedirs(output_xml_dir, exist_ok=True)

    shape_paths = [
        os.path.join(shape_dir, f)
        for f in os.listdir(shape_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]
    if not shape_paths:
        raise ValueError(f"No shape images found in {shape_dir}")

    for i in range(total_count):
        background_path = random.choice(background_paths)
        side = "right" if "right" in os.path.basename(background_path) else "left"

        output_img_path = os.path.join(output_image_dir, f"generated_{i}.jpg")
        output_xml_path = os.path.join(output_xml_dir, f"generated_{i}.xml")

        generate_image_with_shapes(
            background_path=background_path,
            shape_paths=shape_paths,
            output_image_path=output_img_path,
            output_xml_path=output_xml_path,
            side=side
        )

if __name__ == "__main__":
    generate_combined(
        background_paths=[
            r"E:\zxz\project\region_right.jpg",
            r"E:\zxz\project\region_left.jpg"
        ],
        shape_dir=r"E:\zxz\project\shapes\e",
        output_image_dir=r"E:\zxz\project\images\generated_regions_e\images",
        output_xml_dir=r"E:\zxz\project\images\generated_regions_e\xmls",
        total_count=50
    )
