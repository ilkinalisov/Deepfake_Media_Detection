# Multi-Modal Deepfake Detection System

## Overview

This project provides a comprehensive **multi-modal deepfake detection system** that can analyze three types of media:

1. **🎵 Audio Detection** - Detects AI-generated/deepfake audio using MFCC features
2. **📝 Text Detection** - Identifies fake news using TF-IDF and machine learning
3. **🎬 Video Detection** - Detects deepfake videos using MobileNetV2 frame analysis

### Audio Detection Credits
Audio detection model credits: https://github.com/noorchauhan/DeepFake-Audio-Detection-MFCC
Research Paper: [IEEE Access](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9996362)

The audio model uses MFCC features to detect real or fake audio. Originally developed using the "Fake-or-Real" dataset of 195,000 combined audio samples, we've expanded it with additional datasets to handle modern AI-generated voices.


## Citation
A. Hamza et al., "Deepfake Audio Detection via MFCC Features Using Machine Learning," in IEEE Access, vol. 10, pp. 134018-134028, 2022, doi: 10.1109/ACCESS.2022.3231480.
Abstract: Deepfake content is created or altered synthetically using artificial intelligence (AI) approaches to appear real. It can include synthesizing audio, video, images, and text. Deepfakes may now produce natural-looking content, making them harder to identify. Much progress has been achieved in identifying video deepfakes in recent years; nevertheless, most investigations in detecting audio deepfakes have employed the ASVSpoof or AVSpoof dataset and various machine learning, deep learning, and deep learning algorithms. This research uses machine and deep learning-based approaches to identify deepfake audio. Mel-frequency cepstral coefficients (MFCCs) technique is used to acquire the most useful information from the audio. We choose the Fake-or-Real dataset, which is the most recent benchmark dataset. The dataset was created with a text-to-speech model and is divided into four sub-datasets: for-rece, for-2-sec, for-norm and for-original. These datasets are classified into sub-datasets mentioned above according to audio length and bit rate. The experimental results show that the support vector machine (SVM) outperformed the other machine learning (ML) models in terms of accuracy on for-rece and for-2-sec datasets, while the gradient boosting model performed very well using for-norm dataset. The VGG-16 model produced highly encouraging results when applied to the for-original dataset. The VGG-16 model outperforms other state-of-the-art approaches.
keywords: {Deepfakes;Deep learning;Speech synthesis;Training data;Feature extraction;Machine learning algorithms;Data models;Acoustics;Deepfakes;deepfake audio;synthetic audio;machine learning;acoustic data},
URL: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9996362&isnumber=9668973

## Features

- **Multi-Modal Detection**: Analyze audio, text, and video files
- **Web Interface**: Beautiful, modern web UI with tabbed interface
- **RESTful API**: Separate endpoints for each media type
- **High Accuracy**: Uses state-of-the-art ML models (SVM, MLP)
- **Easy to Use**: Drag-and-drop file upload or paste text directly

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/deepfake-media-detection.git
   cd deepfake-media-detection
   ```

2. Create a virtual environment (recommended):
   ```bash
   # For Windows
   python -m venv venv
   venv\Scripts\activate

   # For Linux/macOS
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Training Models

### 1. Train Audio Detection Model

```bash
python main.py
```

This trains an SVM model using MFCC features from audio files in `data/azvoices/`.

### 2. Train Text Detection Model

```bash
python train_text_model.py
```

This creates a text fake news detection model using TF-IDF features. By default, it uses sample data. To use your own dataset, place a CSV file with 'text' and 'label' columns and update the script.

### 3. Train Video Detection Model (Optional)

```bash
python train_video_model.py
```

**Note**: Video detection requires:
- TensorFlow installed (`pip install tensorflow`)
- Video files organized in `data/video/real/` and `data/video/fake/`
- Sufficient GPU/CPU resources (frame extraction is computationally intensive)

## Running the Web Application

Start the Flask server:

```bash
python app.py
```

The server will start on `http://localhost:5000`. Open this URL in your browser to access the web interface.

### Available Endpoints

- **GET** `/` - Main web interface with tabs for all three modes
- **POST** `/predict/audio` - Analyze audio files
- **POST** `/predict/text` - Analyze text content
- **POST** `/predict/video` - Analyze video files
- **GET** `/status` - Check which models are available

## Usage

### Via Web Interface

1. Open `http://localhost:5000` in your browser
2. Select the appropriate tab (Audio, Text, or Video)
3. Upload your file or paste text
4. Click "Analyze" and view results with confidence scores

### Via API

#### Audio Detection
```bash
curl -X POST -F "file=@audio.mp3" http://localhost:5000/predict/audio
```

#### Text Detection
```bash
curl -X POST -F "text=Your text here" http://localhost:5000/predict/text
```

#### Video Detection
```bash
curl -X POST -F "file=@video.mp4" http://localhost:5000/predict/video
```

## Supported File Formats

- **Audio**: MP3, WAV, OGG, FLAC, M4A
- **Text**: TXT files or direct text input
- **Video**: MP4, AVI, MOV, MKV, FLV, WMV

## Model Details

### Audio Detection
- **Features**: MFCC (Mel-Frequency Cepstral Coefficients)
- **Model**: Support Vector Machine (SVM) with linear kernel
- **Input**: Audio files up to 100MB

### Text Detection
- **Features**: TF-IDF (Term Frequency-Inverse Document Frequency)
- **Model**: SVM with linear kernel & Multi-Layer Perceptron (MLP)
- **Input**: Text content (any length)

### Video Detection
- **Features**: MobileNetV2 pre-trained on ImageNet
- **Model**: SVM with RBF kernel
- **Process**: Extracts 30 frames per video, averages features
- **Input**: Video files up to 100MB


