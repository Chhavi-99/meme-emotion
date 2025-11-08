#!/usr/bin/env python3
import os
import sys
import argparse
import numpy as np
import tensorflow as tf
from PIL import ImageFile
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout

ImageFile.LOAD_TRUNCATED_IMAGES = True

CLASS_NAMES = ["neutral", "positive", "very_positive", "negative", "very_negative"]

def build_model(num_classes: int = 5) -> tf.keras.Model:
    """
    Build the same architecture you trained before:
    VGG16 (imagenet, no top, global pooling via flatten) + small dense head.
    """
    model = Sequential(name="sentiment_vgg16")
    model.add(VGG16(include_top=False, weights="imagenet", input_shape=(224, 224, 3)))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(num_classes, activation='softmax'))

    # freeze the VGG16 convolutional base (as in your original code)
    for layer in model.layers[0].layers:
        layer.trainable = False

    # compile so it's ready if someone wants to .evaluate quickly
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["acc"])
    return model

def load_weights_if_available(model: tf.keras.Model, weights_path: str) -> bool:
    if weights_path and os.path.isfile(weights_path):
        model.load_weights(weights_path)
        return True
    return False

def load_and_preprocess(img_path: str) -> np.ndarray:
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x

def predict_sentiment(model: tf.keras.Model, img_batch: np.ndarray) -> dict:
    preds = model.predict(img_batch, verbose=0)
    probs = preds[0].astype(float)  # shape (5,)
    idx = int(np.argmax(probs))
    return {
        "class_index": idx,
        "class_name": CLASS_NAMES[idx],
        "probabilities": {name: float(probs[i]) for i, name in enumerate(CLASS_NAMES)},
    }

def main():
    parser = argparse.ArgumentParser(description="Predict sentiment from an image (and optional text).")
    parser.add_argument("--image_path", required=True, help="Path to input image.")
    parser.add_argument("--text", required=True, help='Associated caption/text (not used by the model yet).')
    parser.add_argument("--weights", default="weights/sentiment_vgg16.h5",
                        help="Path to trained model weights (.h5). Default: weights/sentiment_vgg16.h5")
    args = parser.parse_args()

    if not os.path.isfile(args.image_path):
        print(f"ERROR: image not found at '{args.image_path}'", file=sys.stderr)
        sys.exit(1)

    # Optional: make TF a bit quieter
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

    model = build_model(num_classes=5)

    had_weights = load_weights_if_available(model, args.weights)
    if not had_weights:
        print(
            f"WARNING: Could not find weights at '{args.weights}'. "
            "Proceeding with randomly initialized classifier (predictions will be meaningless).",
            file=sys.stderr
        )

    # Log the text (for traceability). You can later add a text encoder + fusion if desired.
    print(f"[INFO] Received text: {args.text}")

    x = load_and_preprocess(args.image_path)
    result = predict_sentiment(model, x)

    # Pretty print
    print("\n=== Sentiment Prediction ===")
    print(f"Image: {args.image_path}")
    print(f"Predicted: {result['class_name']} (class {result['class_index']})")
    print("Probabilities:")
    for name, p in result["probabilities"].items():
        print(f"  - {name:>14}: {p:.4f}")

if __name__ == "__main__":
    main()
