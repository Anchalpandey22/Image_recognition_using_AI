"""
train_model.py

Trains a custom CNN on images placed inside the dataset/ folder, using the
following structure (one sub-folder per class):

    dataset/
        cats/
        dogs/
        cars/
        flowers/

Run with:
    python train_model.py

After training completes, image_model.h5 and class_names.pkl are saved into
model/. The Flask app (predict.py) automatically detects and switches to using
this custom model the next time it starts.

NOTE: This repository ships with empty dataset/ sub-folders as placeholders.
Add your own labelled images before running this script. The app works out of
the box without training (using a pretrained MobileNetV2 model) -- this script
is for when you want predictions specific to your own classes.
"""

import os
import pickle
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    BatchNormalization,
    Dropout,
    Flatten,
    Dense,
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from config import Config

IMAGE_SIZE = Config.IMAGE_SIZE
BATCH_SIZE = 32
EPOCHS = 30
DATASET_DIR = Config.DATASET_FOLDER
MODEL_DIR = Config.MODEL_FOLDER


def count_images(dataset_dir):
    total = 0
    classes_found = []
    for entry in sorted(os.listdir(dataset_dir)):
        class_path = os.path.join(dataset_dir, entry)
        if os.path.isdir(class_path):
            files = [
                f for f in os.listdir(class_path)
                if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
            ]
            if files:
                classes_found.append(entry)
                total += len(files)
    return total, classes_found


def build_model(num_classes, input_shape=(224, 224, 3)):
    """Builds the CNN architecture as specified in the project requirements."""
    model = Sequential([
        Conv2D(32, (3, 3), activation="relu", padding="same", input_shape=input_shape),
        MaxPooling2D(pool_size=(2, 2)),

        Conv2D(64, (3, 3), activation="relu", padding="same"),
        MaxPooling2D(pool_size=(2, 2)),

        Conv2D(128, (3, 3), activation="relu", padding="same"),
        MaxPooling2D(pool_size=(2, 2)),

        BatchNormalization(),
        Dropout(0.3),

        Flatten(),
        Dense(512, activation="relu"),
        Dropout(0.3),
        Dense(256, activation="relu"),
        Dense(num_classes, activation="softmax"),
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def plot_history(history, output_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(history.history["accuracy"], label="Train Accuracy")
    axes[0].plot(history.history["val_accuracy"], label="Val Accuracy")
    axes[0].set_title("Model Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()

    axes[1].plot(history.history["loss"], label="Train Loss")
    axes[1].plot(history.history["val_loss"], label="Val Loss")
    axes[1].set_title("Model Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Training graphs saved to {output_path}")


def main():
    total_images, classes_found = count_images(DATASET_DIR)

    if total_images == 0 or len(classes_found) < 2:
        print("=" * 70)
        print("No training data found (or fewer than 2 classes populated).")
        print(f"Please add images into sub-folders under: {DATASET_DIR}")
        print("Example: dataset/cats/*.jpg, dataset/dogs/*.jpg, etc.")
        print("The Flask app will continue to use the pretrained MobileNetV2")
        print("model until you train a custom model with real data.")
        print("=" * 70)
        return

    print(f"Found {total_images} images across classes: {classes_found}")

    os.makedirs(MODEL_DIR, exist_ok=True)

    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode="nearest",
        validation_split=0.2,
    )

    train_generator = train_datagen.flow_from_directory(
        DATASET_DIR,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="training",
    )

    val_generator = train_datagen.flow_from_directory(
        DATASET_DIR,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="validation",
    )

    class_names = sorted(train_generator.class_indices, key=train_generator.class_indices.get)
    num_classes = len(class_names)

    model = build_model(num_classes, input_shape=(*IMAGE_SIZE, 3))
    model.summary()

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        ModelCheckpoint(
            os.path.join(MODEL_DIR, "image_model.h5"),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6),
    ]

    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=EPOCHS,
        callbacks=callbacks,
    )

    # Final save (in case ModelCheckpoint did not trigger e.g. very short training)
    model.save(os.path.join(MODEL_DIR, "image_model.h5"))

    with open(os.path.join(MODEL_DIR, "class_names.pkl"), "wb") as f:
        pickle.dump(class_names, f)

    plot_history(history, os.path.join(MODEL_DIR, "training_history.png"))

    print("\nTraining complete!")
    print(f"Model saved to: {os.path.join(MODEL_DIR, 'image_model.h5')}")
    print(f"Class names saved to: {os.path.join(MODEL_DIR, 'class_names.pkl')}")
    print("Restart the Flask app to start using your custom-trained model.")


if __name__ == "__main__":
    main()
