import os
from PIL import Image

def keep_dark_lines_only(image_path, output_path, threshold=160):
    img = Image.open(image_path).convert("RGBA")
    datas = img.getdata()

    newData = []
    for item in datas:
        r, g, b, a = item
        gray = (r + g + b) / 3
        if gray < threshold:
            newData.append((0, 0, 0, 255))
        else:
            newData.append((255, 255, 255, 0))
    img.putdata(newData)
    img.save(output_path, "PNG")

folder = r"E:\zxz\project\shapes\a"
output_folder = r"E:\zxz\project\shapes\c"


for file in os.listdir(folder):
    if file.lower().endswith(".png"):
        input_path = os.path.join(folder, file)
        output_path = os.path.join(output_folder, file)
        keep_dark_lines_only(input_path, output_path, threshold=160)
