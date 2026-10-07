# Image Classification: Happy vs Sad

A CNN-based binary image classifier built with TensorFlow/Keras that predicts whether a face shows a **happy** or **sad** expression.

## Overview

- **Task:** Binary image classification (happy vs. sad)
- **Framework:** TensorFlow / Keras
- **Environment:** Google Colab (GPU)
- **Dataset:** ~21,000 grayscale facial images (48×48), FER2013-style
- **Final test accuracy:** ~81.4% on 12,045 unseen test images

## Project Structure

```
image_classification/
├── data/
│   ├── train/
│   │   ├── happy/
│   │   └── sad/
│   └── test/
│       ├── happy/
│       └── sad/
├── Imagetonumbers/      # exploration of image-to-array fundamentals
├── model/
│   └── happysadmodel.keras
├── deep_learning.ipynb  # main notebook: data prep, training, evaluation
├── requirements.txt
└── README.md
```

## Dataset

- 8,977 training images (happy / sad), 12,045 test images
- All images are **48×48 grayscale**, confirmed via per-image size check and R/G/B channel comparison
- Cleaned prior to training: removed invalid files (mislabeled `.txt`/`.svg` files, broken images) using OpenCV + PIL validation

## Preprocessing Pipeline

1. Load images with `tf.keras.utils.image_dataset_from_directory` (`image_size=(48,48)`, `color_mode='grayscale'`)
2. Normalize pixel values to `0–1` range (`img / 255.0`)
3. Split: 80% train / 20% validation (from the training folder); separate held-out test folder used for final evaluation

## Model Architecture

```
Input (48, 48, 1)
→ RandomFlip (horizontal)
→ RandomRotation (0.05)
→ Conv2D(16, 3x3, relu) → MaxPooling2D
→ Conv2D(32, 3x3, relu) → MaxPooling2D
→ Conv2D(64, 3x3, relu) → MaxPooling2D
→ GlobalAveragePooling2D
→ Dropout(0.4)
→ Dense(64, relu)
→ Dense(1, sigmoid)
```

- **Optimizer:** Adam
- **Loss:** Binary Crossentropy
- **Callbacks:** EarlyStopping (`monitor='val_loss'`, `patience=3`, `restore_best_weights=True`)
- **Total parameters:** ~27,500

## Results

| Metric | Validation | Test (unseen) |
|---|---|---|
| Accuracy | 82.1% | 81.4% |
| Loss | 0.404 | — |
| Precision (test) | — | 72.2% |
| Recall (test) | — | 87.4% |

Training stopped early at epoch 25/50 via `EarlyStopping`, restoring the best checkpoint (epoch 22).

## Usage

### Load the trained model

```python
from tensorflow.keras.models import load_model

model = load_model('model/happysadmodel.keras')
```

### Predict on a new image

```python
import cv2
import numpy as np
import tensorflow as tf

def predict_image(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resize = tf.image.resize(img[..., np.newaxis], (48, 48))
    img_normalized = resize.numpy() / 255.0
    img_input = np.expand_dims(img_normalized, 0)

    prediction = model.predict(img_input, verbose=0)
    label = "sad" if prediction[0][0] > 0.5 else "happy"
    print(f"Predicted: {label} (raw: {prediction[0][0]:.4f})")

predict_image('path/to/image.jpg')
```

## Requirements

See `requirements.txt`. Core dependencies: `tensorflow`, `opencv-python`, `pillow`, `numpy`, `matplotlib`.

## Notes

- Dataset images are low-resolution (48×48) and grayscale, so some genuinely ambiguous expressions (neutral faces, partially occluded faces) are harder to classify correctly — this accounts for most of the ~19% error rate.
- This project was built as a learning exercise covering the full pipeline: image-to-array fundamentals, dataset cleaning, CNN architecture design, training diagnostics, and evaluation.
