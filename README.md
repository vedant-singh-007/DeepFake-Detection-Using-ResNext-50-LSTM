# DeepFake-Detection-Using-ResNext-50-LSTM

This project implements a deepfake video classification system using a combination of ResNeXt-50 (for spatial feature extraction) and LSTM (for temporal sequence modeling). The model is trained to distinguish between real and fake videos from the Celeb-DF dataset.

Dataset
Source: Celeb-DF (v1)

Number of videos:

432 fake videos

428 real videos

Labeling:

Fake: 1

Real: 0

A dictionary is used to map each video path to its corresponding label.

Preprocessing Pipeline
Frame Extraction
Videos are trimmed to a maximum of 150 frames using OpenCV.

Face Detection and Cropping
Faces are detected in each frame. If no face is detected, the issue is logged for debugging. From the 150 frames, 10 frames with detected faces are selected for further processing.

Frame Transformation
The selected frames are resized and normalized to fit the input requirements of the pretrained ResNeXt-50 model.

Model Architecture
Spatial Feature Extraction
A pretrained ResNeXt-50 model extracts features from each of the 10 selected frames per video.

Temporal Modeling
An LSTM layer captures temporal dependencies across the 10-frame sequence.

Classification
The output of the LSTM is passed through a linear layer for binary classification (real or fake).
CrossEntropyLoss with class weighting is used to handle class imbalance.

Training and Testing
Training accuracy: 89.7%

Training loss: 0.17

Testing accuracy: 98.26%

A confusion matrix is generated after testing to confirm that the model is not biased toward a single class.

Highlights
Full video preprocessing pipeline including trimming, face detection, and frame selection

Real-time debug messages for missing face detections

Efficient spatial-temporal modeling using ResNeXt and LSTM

Clear performance metrics and confusion matrix visualization

Dependencies
Python 3.x

PyTorch

OpenCV

torchvision

matplotlib

scikit-learn

seaborn

How to Run
Install dependencies:

nginx
Copy
Edit
pip install -r requirements.txt
Train the model:

nginx
Copy
Edit
python train.py
Test the model and view the confusion matrix:

nginx
Copy
Edit
python test.py
