import numpy as np
import os
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Flatten, Dropout, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix

class ResNetClassifier:

    def __init__(self, train_dir, test_dir, save_dir="model"):
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

    def _create_data_generators(self):
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.3,
            brightness_range=[0.7, 1.3],
            horizontal_flip=True,
            fill_mode='nearest'
        )

        test_datagen = ImageDataGenerator(rescale=1./255)

        train_generator = train_datagen.flow_from_directory(
            self.train_dir,
            target_size=(128, 128),
            batch_size=32,
            class_mode='categorical'
        )

        test_generator = test_datagen.flow_from_directory(
            self.test_dir,
            target_size=(128, 128),
            batch_size=64,
            class_mode='categorical',
            shuffle=False
        )

        return train_generator, test_generator

    def _build_model(self, num_classes):
        base_model = ResNet50(input_shape=(128, 128, 3), include_top=False, weights="imagenet")
        base_model.trainable = False

        model = Sequential([
            base_model,
            GlobalAveragePooling2D(),
            Dense(128, activation='relu'),
            BatchNormalization(),
            Dropout(0.4),
            Dense(64, activation='relu'),
            Dropout(0.3),
            Dense(num_classes, activation='softmax') 
        ])

        model.compile(
            loss='categorical_crossentropy',
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
            metrics=['accuracy']
        )

        return model

    def train_and_save(self):
        train_generator, test_generator = self._create_data_generators()
        num_classes = len(train_generator.class_indices)

        model = self._build_model(num_classes)

        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=5, restore_best_weights=True
        )
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=3, verbose=1
        )

        model.fit(
            train_generator,
            epochs=30,
            validation_data=test_generator,
            callbacks=[early_stopping, reduce_lr]
        )

        model_save_path = os.path.join(self.save_dir, "resnet_classifier.h5")
        model.save(model_save_path)
        print(f"Model saved at {model_save_path}")

    def evaluate_model(self):
        model_path = os.path.join(self.save_dir, "resnet_classifier.h5")
        model = load_model(model_path)
        _, test_generator = self._create_data_generators()

        predictions = model.predict(test_generator)
        predicted_classes = np.argmax(predictions, axis=1)
        true_classes = test_generator.classes
        class_labels = list(test_generator.class_indices.keys())

        cm = confusion_matrix(true_classes, predicted_classes)
        print("Confusion Matrix:\n", cm)

        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")
        plt.show()

        classification_report_result = classification_report(
            true_classes, predicted_classes, target_names=class_labels, output_dict=True
        )
        print("Classification Report:\n", classification_report_result)

        overall_accuracy = np.mean(true_classes == predicted_classes) * 100
        print(f"Overall Accuracy: {overall_accuracy:.2f}%")
