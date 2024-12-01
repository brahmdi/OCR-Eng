# English OCR Using Deep Learning

This repository contains an Optical Character Recognition (OCR) system designed to recognize English handwritten characters. The project uses deep learning techniques and is trained on a combination of the Kaggle A-Z Handwritten Alphabets dataset and the MNIST Handwritten Digits dataset.

## Table of Contents
- [About the Project](#about-the-project)
- [Datasets](#datasets)
- [Model Architecture](#model-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)
- [Future Work](#future-work)
- [License](#license)

## About the Project
The goal of this project is to create a robust OCR system capable of recognizing both alphabetic characters (A-Z) and numeric digits (0-9) in handwritten text. This system can be integrated into various applications such as automated form digitization, educational tools, and more.

### Key Objectives
- Recognize individual characters (A-Z, 0-9) in handwritten text.
- Achieve high accuracy using a combination of CNN and LSTM architectures.

## Datasets
This project uses the following datasets:
1. **Kaggle A-Z Handwritten Alphabets Dataset**  
   - Contains 372,450 grayscale images of size 28x28 representing English alphabets.
2. **MNIST Handwritten Digits Dataset**  
   - Contains 70,000 grayscale images of size 28x28 representing digits 0-9.

These datasets are preprocessed to normalize pixel values and merge them into a unified dataset for training.

## Model Architecture
The OCR system uses a hybrid deep learning model:
1. **Convolutional Neural Network (CNN)**:  
   Extracts spatial features from input images.
2. **Fully Connected Layers**:  
   Processes features into vector representations.
3. **Softmax Output Layer**:  
   Classifies each image into one of 36 classes (A-Z and 0-9).

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/brahmdi/OCR-Eng.git
   cd OCR-Eng
   ```
2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```


### Streamlit Interface
Run the Streamlit app to test the OCR system interactively:
```bash
streamlit run OCREng.py
```

## Features
- **36-Class Character Recognition**: Supports both uppercase English alphabets (A-Z) and digits (0-9).
- **Interactive Demo**: Streamlit interface for real-time predictions.
- **Modular Codebase**: Easy to adapt and extend for other datasets or languages.

## Future Work
- **Sequence Recognition**: Extend the model to recognize full words and sentences.
- **Mobile Integration**: Develop a mobile app for on-the-go handwriting recognition.
- **Performance Optimization**: Experiment with advanced architectures like transformers.



## Contact
**Author**: Brahim El Hamdi  
**Email**: hmdibrahim09@gmail.com  
