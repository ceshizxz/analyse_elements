import os
import shutil
import random

source_dir = r"C:\Users\ce-sh\Desktop\UCL\PHAS0077\analyse_elements\images\generated_images"

train_dir = r"C:\Users\ce-sh\Desktop\UCL\PHAS0077\analyse_elements\images\Train"
test_dir = r"C:\Users\ce-sh\Desktop\UCL\PHAS0077\analyse_elements\images\Test"

os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

split_ratio = 0.8  

for letter in os.listdir(source_dir):
    letter_path = os.path.join(source_dir, letter)
    
    if os.path.isdir(letter_path): 
        images = os.listdir(letter_path)
        random.shuffle(images)

        split_idx = int(len(images) * split_ratio)
        train_images = images[:split_idx]
        test_images = images[split_idx:]

        os.makedirs(os.path.join(train_dir, letter), exist_ok=True)
        os.makedirs(os.path.join(test_dir, letter), exist_ok=True)

        for img in train_images:
            shutil.copy(os.path.join(letter_path, img), os.path.join(train_dir, letter, img))

        for img in test_images:
            shutil.copy(os.path.join(letter_path, img), os.path.join(test_dir, letter, img))

print("Done!")