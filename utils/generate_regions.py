import os
import random
import xml.etree.ElementTree as ET
from PIL import Image

# Images to paste but NOT annotate in XML
CONFUSING_NAMES = {"no1.png", "no2.png", "no3.png"}

def apply_transformations(shape, shrink=1.0):
    # Apply random flip, stretch, and rotation to the shape image.
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
    # Paste shape at absolute coordinates and return its bounding box.
    background.paste(shape, (x, y), shape)
    return (x, y, x + shape.width, y + shape.height)

def paste_random_nonoverlapping(background, shape, existing_bboxes, side, max_attempts=50):
    # Paste shape in a non-overlapping location on the left or right side.
    bg_w, bg_h = background.size
    shape_w, shape_h = shape.size

    if side == "right":
        x_min = int(bg_w * 0.67)
        x_max = bg_w - shape_w
    else:
        x_min = 0
        x_max = int(bg_w * 0.33) - shape_w

    y_min = int(bg_h * 0.05)  # Avoid top 5%
    y_max = int(bg_h * 0.95) - shape_h  # Avoid bottom 5%

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

def bboxes_overlap(b1, b2):
    # Check if two bounding boxes overlap.
    return not (b1[2] <= b2[0] or b2[2] <= b1[0] or b1[3] <= b2[1] or b2[3] <= b1[1])

def generate_image_with_shapes(background_path, shape_paths, output_image_path, output_xml_path, side="right"):
    """Main image generation function with 3 layout modes: normal, vertical and horizontal overlap."""
    background = Image.open(background_path).convert("RGB")
    bg_w, bg_h = background.size
    bboxes = []
    xml_boxes = []
    names = [] 
    mode = random.choices(["normal", "vertical", "horizontal"], weights=[0.8, 0.1, 0.1])[0]
    shapes = random.choices(shape_paths, k=2)
    shape_imgs = []
    for sp in shapes:
        shape = Image.open(sp).convert("RGBA")
        shrink = 0.5 if mode == "horizontal" else 1.0
        shape = apply_transformations(shape, shrink=shrink)
        shape_imgs.append((shape, os.path.basename(sp)))

    if mode == "vertical":
        # Stack shapes top and bottom
        shape1, name1 = shape_imgs[0]
        shape2, name2 = shape_imgs[1]
        w1, h1 = shape1.size
        w2, h2 = shape2.size

        if side == "right":
            x_min = int(bg_w * 0.67)
            x_max = bg_w - w1
        else:
            x_min = 0
            x_max = int(bg_w * 0.33) - w1

        y_min = int(bg_h * 0.05)
        y_max = int(bg_h * 0.95) - (h1 + h2)

        for _ in range(50):
            if x_max < x_min or y_max < 0:
                break
            x = random.randint(x_min, x_max)
            y = random.randint(y_min, y_max)
            bbox1 = paste_shape_absolute(background, shape1, x, y)
            bbox2 = paste_shape_absolute(background, shape2, x, y + h1)
            bboxes.extend([bbox1, bbox2])
            names = [name1, name2]
            break

    elif mode == "horizontal":
        # Stack shapes left and right, with 50% scaling
        shape1, name1 = shape_imgs[0]
        shape2, name2 = shape_imgs[1]
        w1, h1 = shape1.size
        w2, h2 = shape2.size

        if side == "right":
            x_min = int(bg_w * 0.67)
            x_max = bg_w - w1 - w2
        else:
            x_min = 0
            x_max = int(bg_w * 0.33) - w1 - w2

        y_min = int(bg_h * 0.05)
        y_max = int(bg_h * 0.95) - max(h1, h2)

        for _ in range(50):
            if x_max < x_min or y_max < 0:
                break
            x = random.randint(x_min, x_max)
            y = random.randint(y_min, y_max)
            bbox1 = paste_shape_absolute(background, shape1, x, y)
            bbox2 = paste_shape_absolute(background, shape2, x + w1, y)
            bboxes.extend([bbox1, bbox2])
            names = [name1, name2]
            break

    else:
        # Default: randomly place 1–2 shapes with no overlap
        for shape, name in shape_imgs:
            bbox = paste_random_nonoverlapping(background, shape, bboxes, side)
            if bbox:
                bboxes.append(bbox)
                names.append(name)

    background.save(output_image_path)

    # Write XML annotation (exclude confusing shapes)
    annotation = ET.Element('annotation')
    ET.SubElement(annotation, 'folder').text = os.path.basename(os.path.dirname(output_image_path))
    ET.SubElement(annotation, 'filename').text = os.path.basename(output_image_path)
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

def generate_combined(background_paths, shape_dir, output_image_dir, output_xml_dir, total_count=20):
    # Main loop: generate multiple training images with bounding box labels.
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
        generate_image_with_shapes(background_path, shape_paths, output_img_path, output_xml_path, side)

    print("Done!")

if __name__ == "__main__":
    generate_combined(
        background_paths=[
            r"E:\zxz\project\region_right.jpg",
            r"E:\zxz\project\region_left.jpg"
        ],
        shape_dir=r"E:\zxz\project\shapes\c",
        output_image_dir=r"E:\zxz\project\images\generated_regions_c\images",
        output_xml_dir=r"E:\zxz\project\images\generated_regions_c\xmls",
        total_count=500
    )

    generate_combined(
        background_paths=[
            r"E:\zxz\project\region_right.jpg",
            r"E:\zxz\project\region_left.jpg"
        ],
        shape_dir=r"E:\zxz\project\shapes\e",
        output_image_dir=r"E:\zxz\project\images\generated_regions_e\images",
        output_xml_dir=r"E:\zxz\project\images\generated_regions_e\xmls",
        total_count=500
    )

    generate_combined(
        background_paths=[
            r"E:\zxz\project\region_right.jpg",
            r"E:\zxz\project\region_left.jpg"
        ],
        shape_dir=r"E:\zxz\project\shapes\m",
        output_image_dir=r"E:\zxz\project\images\generated_regions_m\images",
        output_xml_dir=r"E:\zxz\project\images\generated_regions_m\xmls",
        total_count=500
    )
