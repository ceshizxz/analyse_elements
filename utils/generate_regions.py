import os
import random
import xml.etree.ElementTree as ET
from PIL import Image
import numpy as np

def apply_transformations(shape, shrink=1.0):
    w, h = shape.size
    shape = shape.resize((int(w * shrink), int(h * shrink)), Image.LANCZOS)

    if random.random() < 0.4:  # flip
        if random.random() < 0.5:
            shape = shape.transpose(Image.FLIP_LEFT_RIGHT)
        else:
            shape = shape.transpose(Image.FLIP_TOP_BOTTOM)

    if random.random() < 0.4:  # shear
        dx, dy = random.uniform(-0.2, 0.2), random.uniform(-0.2, 0.2)
        shape = shape.transform(
            shape.size, Image.AFFINE, (1, dx, 0, dy, 1, 0),
            resample=Image.BICUBIC, fillcolor=(0, 0, 0, 0)
        )

    if random.random() < 0.4:  # rotate
        angle = random.uniform(-30, 30)
        shape = shape.rotate(angle, expand=True)

    return shape

def random_transparency(shape):
    alpha = np.array(shape.getchannel("A"), dtype=np.uint8)
    mask = np.ones_like(alpha, dtype=np.float32)

    for _ in range(random.randint(3, 6)):
        cx = random.randint(0, shape.width - 1)
        cy = random.randint(0, shape.height - 1)
        r = random.randint(10, max(10, min(shape.width, shape.height) // 3))
        y, x = np.ogrid[:shape.height, :shape.width]
        dist = (x - cx) ** 2 + (y - cy) ** 2
        mask[dist <= r ** 2] *= random.uniform(0.4, 0.8)

    noise_strength = random.uniform(0.9, 1.0)
    noise = np.random.uniform(noise_strength, 1.0, size=alpha.shape)
    mask *= noise

    alpha = (alpha.astype(np.float32) * mask).clip(0, 255).astype(np.uint8)
    shape.putalpha(Image.fromarray(alpha))
    return shape

def paste_shape_absolute(background, shape, x, y):
    background.paste(shape, (x, y), shape)
    return (x, y, x + shape.width, y + shape.height)

def bboxes_overlap(b1, b2):
    return not (b1[2] <= b2[0] or b2[2] <= b1[0] or b1[3] <= b2[1] or b2[3] <= b1[1])

def paste_random_nonoverlapping(background, shape, existing_bboxes, side, max_attempts=50, allow_overlap=False):
    bg_w, bg_h = background.size
    w, h = shape.size

    if side == "right":
        x_min = int(bg_w * 0.67); x_max = bg_w - w
    else:
        x_min = 0; x_max = int(bg_w * 0.33) - w
    y_min = int(bg_h * 0.05); y_max = int(bg_h * 0.95) - h

    if x_max < x_min or y_max < y_min:
        return None

    for _ in range(max_attempts):
        x = random.randint(x_min, x_max)
        y = random.randint(y_min, y_max)
        new_bbox = (x, y, x + w, y + h)
        if allow_overlap or all(not bboxes_overlap(new_bbox, b) for b in existing_bboxes):
            background.paste(shape, (x, y), shape)
            return new_bbox

    # Retry with shrink
    shape = apply_transformations(shape, shrink=0.8)
    w, h = shape.size
    y_max = int(bg_h * 0.95) - h
    x_max = x_max if side == "right" else int(bg_w * 0.33) - w

    for _ in range(max_attempts):
        x = random.randint(x_min, x_max)
        y = random.randint(y_min, y_max)
        new_bbox = (x, y, x + w, y + h)
        if allow_overlap or all(not bboxes_overlap(new_bbox, b) for b in existing_bboxes):
            background.paste(shape, (x, y), shape)
            return new_bbox
    return None

def generate_image_with_shapes(background_path, shape_imgs, output_img_path, output_xml_path, side="right"):
    bg = Image.open(background_path).convert("RGB")
    bg_w, bg_h = bg.size
    bboxes, names = [], []

    count = random.choices([0, 1, 2, 3, 4, 5], weights=[5, 25, 25, 20, 15, 10])[0]

    if count == 0:
        bg.save(output_img_path)
        with open(output_xml_path, 'w', encoding='utf-8') as f:
            f.write('<?xml version="1.0" encoding="UTF-8"?>\n<annotation></annotation>')
        return

    selected_shapes = []
    for _ in range(count):
        shape, name = random.choice(shape_imgs)
        s = shape.copy().convert("RGBA")
        s = apply_transformations(s)
        if random.random() < 0.7:  # transparency
            s = random_transparency(s)
        selected_shapes.append((s, name))

    if count >= 4:
        s1, n1 = selected_shapes.pop(0); s2, n2 = selected_shapes.pop(0)
        mode = random.choice(["adjacent", "overlap"])

        if mode == "adjacent":
            s1 = apply_transformations(s1, shrink=0.5)
            s2 = apply_transformations(s2, shrink=0.5)
            w1, h1 = s1.size
            w2, h2 = s2.size
            y_min = int(bg_h * 0.05)
            y_max = int(bg_h * 0.95) - max(h1, h2)
            if side == "right":
                x_min = int(bg_w * 0.67)
                x_max = bg_w - w1 - w2
            else:
                x_min = 0
                x_max = int(bg_w * 0.33) - w1 - w2

            if x_max >= x_min and y_max >= y_min:
                for _ in range(50):
                    x = random.randint(x_min, x_max)
                    y = random.randint(y_min, y_max)
                    bbox1 = paste_shape_absolute(bg, s1, x, y)
                    bbox2 = paste_shape_absolute(bg, s2, x + w1, y)
                    if all(not bboxes_overlap(bbox1, b) and not bboxes_overlap(bbox2, b) for b in bboxes):
                        bboxes.extend([bbox1, bbox2])
                        names.extend([n1, n2])
                        break

        elif mode == "overlap":
            s1 = apply_transformations(s1)
            s2 = apply_transformations(s2)
            w1, h1 = s1.size
            w2, h2 = s2.size
            W = max(w1, w2)
            H = max(h1, h2)
            if side == "right":
                x_min = int(bg_w * 0.67)
                x_max = bg_w - W
            else:
                x_min = 0
                x_max = int(bg_w * 0.33) - W
            y_min = int(bg_h * 0.05)
            y_max = int(bg_h * 0.95) - H
            if x_max >= x_min and y_max >= y_min:
                for _ in range(50):
                    x = random.randint(x_min, x_max)
                    y = random.randint(y_min, y_max)
                    bg.paste(s1, (x,y), s1)
                    bg.paste(s2, (x,y), s2)
                    union_bbox = (x, y, x + W, y + H)
                    if all(not bboxes_overlap(union_bbox, b) for b in bboxes):
                        bboxes.append(union_bbox)
                        names.append("OVERLAPPED_PAIR")
                        break

    for shape, name in selected_shapes:
        bbox = paste_random_nonoverlapping(bg, shape, bboxes, side)
        if bbox:
            bboxes.append(bbox)
            names.append(name)

    bg.save(output_img_path)

    ann = ET.Element('annotation')
    ET.SubElement(ann, 'folder').text = os.path.basename(os.path.dirname(output_img_path))
    ET.SubElement(ann, 'filename').text = os.path.basename(output_img_path)
    size = ET.SubElement(ann, 'size')
    ET.SubElement(size, 'width').text = str(bg_w)
    ET.SubElement(size, 'height').text = str(bg_h)
    ET.SubElement(size, 'depth').text = '3'

    for bbox in bboxes:
        obj = ET.SubElement(ann, 'object')
        ET.SubElement(obj, 'name').text = 'Shape'
        bndbox = ET.SubElement(obj, 'bndbox')
        ET.SubElement(bndbox, 'xmin').text = str(int(bbox[0]))
        ET.SubElement(bndbox, 'ymin').text = str(int(bbox[1]))
        ET.SubElement(bndbox, 'xmax').text = str(int(bbox[2]))
        ET.SubElement(bndbox, 'ymax').text = str(int(bbox[3]))

    ET.ElementTree(ann).write(output_xml_path, encoding='utf-8', xml_declaration=True)

def generate_combined(background_paths, shape_dir, output_img_dir, output_xml_dir, total_count=20):
    """Main loop to generate images and annotations."""
    os.makedirs(output_img_dir, exist_ok=True)
    os.makedirs(output_xml_dir, exist_ok=True)

    shape_imgs = []
    for f in os.listdir(shape_dir):
        if f.lower().endswith((".png", ".jpg", ".jpeg")):
            img = Image.open(os.path.join(shape_dir, f)).convert("RGBA")
            shape_imgs.append((img,f))
    if not shape_imgs:
        raise ValueError(f"No shape images found in {shape_dir}")

    for i in range(total_count):
        bg_path = random.choice(background_paths)
        side = "right" if "right" in os.path.basename(bg_path).lower() else "left"
        img_out = os.path.join(output_img_dir, f"generated_{i}.jpg")
        xml_out = os.path.join(output_xml_dir, f"generated_{i}.xml")
        generate_image_with_shapes(bg_path, shape_imgs, img_out, xml_out, side)

    print("Done!")

if __name__ == "__main__":
    generate_combined(
        background_paths=[
            r"E:\zxz\project\region_right.png",
            r"E:\zxz\project\region_left.png"
        ],
        shape_dir=r"E:\zxz\project\shapes\m0",
        output_img_dir=r"E:\zxz\project\images\generated_regions\images",
        output_xml_dir=r"E:\zxz\project\images\generated_regions\xmls",
        total_count=600
    )
