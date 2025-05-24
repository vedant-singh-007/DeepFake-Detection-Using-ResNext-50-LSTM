# DeepFake Detection Using ResNext50-LSTM

This project implements a deepfake video classification system using a combination of ResNeXt-50 (for spatial feature extraction) and LSTM (for temporal sequence modeling). The model is trained to distinguish between real and fake videos from the Celeb-DF dataset.

## Dataset

- **Source:** Celeb-DF (v1)
- **Composition:**
  - 432 fake videos
  - 428 real videos

## Preprocessing Pipeline

1. **Video Trimming:** Each video is trimmed to 150 frames using OpenCV.
2. **Face Extraction:** Faces are cropped from these 150 frames. From these, 10 frames with valid face detections are selected per video.
3. **Debugging Support:** The frame extraction function displays videos with no face detections to aid debugging.
4. **Transformations:** Selected face frames are resized and transformed into the appropriate shape for ResNeXt-50 input.

## Model Architecture

- **CNN Backbone:** Pretrained ResNeXt-50 used to extract spatial features from each frame.
- **Temporal Modeling:** Features from 10 frames are passed through an LSTM to capture sequential dependencies.
- **Classification Head:** A final linear layer outputs class logits.
- **Loss Function:** CrossEntropyLoss with class weights to handle class imbalance.

## Label Mapping

- Real: `0`
- Fake: `1`

## Frameworks Used

- PyTorch
- OpenCV

## Training Results

- **Training Accuracy:** 89.7%
- **Training Loss:** 0.17

## Testing Results

- **Testing Accuracy:** 98.26%
- **Average Test Loss:** 0.0703

### Confusion Matrix

|               | Predicted: Real (0) | Predicted: Fake (1) |
|---------------|---------------------|----------------------|
| **Actual: Real (0)** | 86                  | 2                    |
| **Actual: Fake (1)** | 1                   | 83                   |

![Confusion Matrix](f5bdd776-3185-4c37-8ef5-89b6bbc0f386.png)
