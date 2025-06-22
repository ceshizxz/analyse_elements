from PIL import Image, ImageEnhance, ImageFilter
import os

def convert_to_black_white(input_path, output_path, threshold=150):
    img = Image.open(input_path).convert("RGBA")

    img = img.filter(ImageFilter.SHARPEN)
    
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(2.0)
    
    datas = img.getdata()
    new_data = []
    for item in datas:
        r, g, b, a = item
        gray = (r + g + b) / 3
        if gray < threshold:
            new_data.append((0, 0, 0))
        else:
            new_data.append((255, 255, 255))

    bw_img = Image.new("RGB", img.size)
    bw_img.putdata(new_data)
    bw_img.save(output_path)

def batch_convert(folder_in, folder_out, threshold=150):
    os.makedirs(folder_out, exist_ok=True)
    for filename in os.listdir(folder_in):
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            input_path = os.path.join(folder_in, filename)
            output_path = os.path.join(folder_out, filename)
            convert_to_black_white(input_path, output_path, threshold)

batch_convert(r"E:\zxz\project\shapes\a", r"E:\zxz\project\shapes\b", threshold=150)