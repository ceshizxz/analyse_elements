import os
import random
import xml.etree.ElementTree as ET
from PIL import Image

# Shapes not to annotate in XML
CONFUSING_NAMES = {"no1.png", "no2.png", "no3.png"}

def apply_transformations(shape, shrink=1.0):
    """Apply random flip, stretch, and rotation to the shape image."""
    from PIL import ImageOps
    import numpy as np

    shape_width, shape_height = shape.size
    shape = shape.resize((int(shape_width * shrink), int(shape_height * shrink)), Image.LANCZOS)

    if random.random() < 0.4:
        flip_mode = random.choice(["horizontal", "vertical"])
        shape = shape.transpose(Image.FLIP_LEFT_RIGHT if flip_mode == "horizontal" else Image.FLIP_TOP_BOTTOM)

    if random.random() < 0.4:
        dx, dy = random.uniform(-0.2, 0.2), random.uniform(-0.2, 0.2)
        shape = shape.transform(
            shape.size, Image.AFFINE, (1, dx, 0, dy, 1, 0),
            resample=Image.BICUBIC, fillcolor=(0, 0, 0, 0)
        )

    if random.random() < 0.4:
        angle = random.uniform(-30, 30)
        shape = shape.rotate(angle, expand=True)

    return shape

def paste_shape_absolute(background, shape, x, y):
    """Paste shape at absolute coordinates and return its bounding box."""
    background.paste(shape, (x, y), shape)
    return (x, y, x + shape.width, y + shape.height)

def bboxes_overlap(b1, b2):
    """Check if two bounding boxes overlap."""
    return not (b1[2] <= b2[0] or b2[2] <= b1[0] or b1[3] <= b2[1] or b2[3] <= b1[1])

def paste_random_nonoverlapping(background, shape, existing_bboxes, side, max_attempts=50):
    """Paste shape non-overlapping in left or right area with y-axis limits (5%~95%)."""
    bg_w, bg_h = background.size
    shape_w, shape_h = shape.size

    if side == "right":
        x_min = int(bg_w * 0.67)
        x_max = bg_w - shape_w
    else:
        x_min = 0
        x_max = int(bg_w * 0.33) - shape_w

    y_min = int(bg_h * 0.05)
    y_max = int(bg_h * 0.95) - shape_h

    if x_max < x_min or y_max < y_min:
        return None

    for _ in range(max_attempts):
        x = random.randint(x_min, x_max)
        y = random.randint(y_min, y_max)
        new_bbox = (x, y, x + shape_w, y + shape_h)
        if all(not bboxes_overlap(new_bbox, b) for b in existing_bboxes):
            background.paste(shape, (x, y), shape)
            return new_bbox
    return None

def generate_image_with_shapes(background_path, shape_paths, output_img_path, output_xml_path, side="right"):
    """Generate one image with 0–5 shapes, and save as image + XML."""
    background = Image.open(background_path).convert("RGB")
    bg_w, bg_h = background.size
    bboxes = []
    names = []

    count = random.choices([0, 1, 2, 3, 4, 5], weights=[5, 25, 25, 20, 15, 10])[0]

    if count == 0:
        background.save(output_img_path)
        with open(output_xml_path, 'w') as f:
            f.write('<annotation></annotation>')
        return

    # Load and transform shape images
    shape_imgs = []
    for _ in range(count):
        sp = random.choice(shape_paths)
        shape = Image.open(sp).convert("RGBA")
        shape = apply_transformations(shape)
        shape_imgs.append((shape, os.path.basename(sp)))

    # First try adjacent placement if count ≥ 4
    if count >= 4:
        s1, n1 = shape_imgs.pop(0)
        s2, n2 = shape_imgs.pop(0)
        s1 = apply_transformations(s1, shrink=0.5)
        s2 = apply_transformations(s2, shrink=0.5)
        w1, h1 = s1.size
        w2, h2 = s2.size

        y_min = int(bg_h * 0.05)
        y_max = int(bg_h * 0.95) - max(h1, h2)

        if side == "right":
            x_min = int(bg_w * 0.67)
            x_max = bg_w - (w1 + w2)
        else:
            x_min = 0
            x_max = int(bg_w * 0.33) - (w1 + w2)

        if x_max >= x_min and y_max >= y_min:
            for _ in range(50):
                x = random.randint(x_min, x_max)
                y = random.randint(y_min, y_max)
                bbox1 = paste_shape_absolute(background, s1, x, y)
                bbox2 = paste_shape_absolute(background, s2, x + w1, y)
                if all(not bboxes_overlap(bbox1, b) and not bboxes_overlap(bbox2, b) for b in bboxes):
                    bboxes.extend([bbox1, bbox2])
                    names.extend([n1, n2])
                    break

    # Paste remaining shapes
    for shape, name in shape_imgs:
        bbox = paste_random_nonoverlapping(background, shape, bboxes, side)
        if bbox:
            bboxes.append(bbox)
            names.append(name)

    background.save(output_img_path)

    # Write XML
    annotation = ET.Element('annotation')
    ET.SubElement(annotation, 'folder').text = os.path.basename(os.path.dirname(output_img_path))
    ET.SubElement(annotation, 'filename').text = os.path.basename(output_img_path)
    size = ET.SubElement(annotation, 'size')
    ET.SubElement(size, 'width').text = str(bg_w)
    ET.SubElement(size, 'height').text = str(bg_h)
    ET.SubElement(size, 'depth').text = '3'

    for bbox, name in zip(bboxes, names):
        if name not in CONFUSING_NAMES:
            obj = ET.SubElement(annotation, 'object')
            ET.SubElement(obj, 'name').text = 'Shape'
            ET.SubElement(obj, 'pose').text = 'Unspecified'
            ET.SubElement(obj, 'truncated').text = '0'
            ET.SubElement(obj, 'difficult').text = '0'
            bndbox = ET.SubElement(obj, 'bndbox')
            ET.SubElement(bndbox, 'xmin').text = str(int(bbox[0]))
            ET.SubElement(bndbox, 'ymin').text = str(int(bbox[1]))
            ET.SubElement(bndbox, 'xmax').text = str(int(bbox[2]))
            ET.SubElement(bndbox, 'ymax').text = str(int(bbox[3]))

    tree = ET.ElementTree(annotation)
    tree.write(output_xml_path)

def generate_combined(background_paths, shape_dir, output_img_dir, output_xml_dir, total_count=20):
    """Main loop to generate images and annotations."""
    os.makedirs(output_img_dir, exist_ok=True)
    os.makedirs(output_xml_dir, exist_ok=True)

    shape_paths = [
        os.path.join(shape_dir, f)
        for f in os.listdir(shape_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]
    if not shape_paths:
        raise ValueError(f"No shape images found in {shape_dir}")

    for i in range(total_count):
        bg_path = random.choice(background_paths)
        side = "right" if "right" in os.path.basename(bg_path).lower() else "left"
        img_out = os.path.join(output_img_dir, f"generated_{i}.jpg")
        xml_out = os.path.join(output_xml_dir, f"generated_{i}.xml")
        generate_image_with_shapes(bg_path, shape_paths, img_out, xml_out, side)

    print("Done!")

if __name__ == "__main__":
    generate_combined(
        background_paths=[
            r"E:\zxz\project\region_right.png",
            r"E:\zxz\project\region_left.png"
        ],
        shape_dir=r"E:\zxz\project\shapes\m",
        output_img_dir=r"E:\zxz\project\images\generated_regions_b\images",
        output_xml_dir=r"E:\zxz\project\images\generated_regions_b\xmls",
        total_count=600
    )