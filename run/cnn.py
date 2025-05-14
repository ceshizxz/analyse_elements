import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models')))

from classify import ResNetClassifier

train_dir = r"D:\images\Train"
test_dir = r"D:\images\Test"
save_dir = r"C:\Users\ce-sh\Desktop\UCL\PHAS0077\analyse_elements\save"

cnn_trainer = ResNetClassifier(train_dir, test_dir, save_dir)
cnn_trainer.train_and_save()

model_path = os.path.join(save_dir, "model_full_dataset.h5")
cnn_trainer.evaluate_model(model_path)