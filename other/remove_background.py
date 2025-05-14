import os
from PIL import Image

def remove_white_background(image_path, output_path):
    img = Image.open(image_path).convert("RGBA")
    datas = img.getdata()

    newData = []
    for item in datas:
        if item[0] > 240 and item[1] > 240 and item[2] > 240:
            newData.append((255, 255, 255, 0))
        else:
            newData.append(item)

    img.putdata(newData)
    img.save(output_path, "PNG")

folder = r"E:\zxz\project\shapes\a"
for file in os.listdir(folder):
    if file.lower().endswith(".png"):
        input_path = os.path.join(folder, file)
        output_path = os.path.join(r"E:\zxz\project\shapes\b",file)
        remove_white_background(input_path, output_path)
