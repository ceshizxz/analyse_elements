from PIL import Image
import numpy as np
import os

def crop_to_black_content(input_path, output_path):
    img = Image.open(input_path).convert("L")
    img_np = np.array(img)

    black_pixels = np.where(img_np < 10)

    if black_pixels[0].size == 0:
        print(f"No black pixels found in {input_path}")
        return

    top = np.min(black_pixels[0])
    bottom = np.max(black_pixels[0])
    left = np.min(black_pixels[1])
    right = np.max(black_pixels[1])

    cropped_img = img.crop((left, top, right + 1, bottom + 1))
    cropped_img.save(output_path)

def batch_crop(folder_in, folder_out):
    os.makedirs(folder_out, exist_ok=True)
    for filename in os.listdir(folder_in):
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            input_path = os.path.join(folder_in, filename)
            output_path = os.path.join(folder_out, filename)
            crop_to_black_content(input_path, output_path)
    print("Done")

batch_crop(r"E:\zxz\project\shapes\e_b", r"E:\zxz\project\shapes\e_c")