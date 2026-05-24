# Face Mask Detection Using Deep Learning
## Overview
This project implements a deep learning-based face mask detection system using a U-Net architecture for semantic segmentation. The model is designed to identify and segment face masks in images, providing pixel-level predictions of mask regions.

## Features
U-Net Architecture: Encoder-decoder structure with skip connections for precise segmentation

Image Preprocessing: Resizes input images to 128x128 pixels with normalization

Real-time Prediction: Generates mask predictions for input images

Evaluation Metrics: Tracks loss and accuracy during training

## Dataset
The project uses the Face Mask Detection Dataset from Kaggle, which contains images of people with and without face masks.

## Model Architecture
The U-Net model consists of:

### Encoder (Downsampling Path)
Conv2D + Dropout + Conv2D blocks

MaxPooling layers for downsampling

Filter sizes: 16 → 32 → 64

### Bottleneck
Two Conv2D layers with 64 filters

### Decoder (Upsampling Path)
Conv2DTranspose layers for upsampling

Skip connections concatenating encoder features

Filter sizes: 32 → 16

### Output Layer
1x1 Conv2D with sigmoid activation for binary segmentation

## Requirements
text
numpy
tensorflow
os
imageio
matplotlib
kagglehub
scikit-learn
Setup and Usage
1. Install Dependencies
bash
pip install numpy tensorflow imageio matplotlib kagglehub scikit-learn
2. Run the Notebook
Execute the Jupyter notebook cells sequentially to:

Download the dataset using kagglehub

Preprocess images (resize to 128x128, normalize)

Build and compile the U-Net model

Train the model

Evaluate performance

Visualize predictions

3. Model Training Parameters
Optimizer: Adam

Loss Function: Binary Crossentropy

Metrics: Accuracy

Batch Size: 8

Epochs: 10

Train/Test Split: 80/20

## Results
The model outputs:

Segmentation masks indicating mask presence

Training/validation loss curves

Prediction visualizations comparing input, ground truth, and predicted masks

Usage Example
python
# Load and preprocess image
img = load_img(image_path, target_size=(128, 128))
img = img_to_array(img) / 255.0
img = np.expand_dims(img, axis=0)

# Generate prediction
prediction = model.predict(img)
Future Improvements
Increase dataset size for better generalization

Implement data augmentation

Fine-tune hyperparameters

Add post-processing for cleaner mask outputs

Deploy as web service or mobile application
