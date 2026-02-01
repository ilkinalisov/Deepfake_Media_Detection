# Deepfake Audio Detection Project

## Overview
ALL CREDITS GO TO https://github.com/noorchauhan/DeepFake-Audio-Detection-MFCC
Find the Paper [here](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9996362)

We trained an open source model from Github that uses MFCC features to detect real or fake audio. This model was developed 2 years ago and uses "Fake-or-Real" dataset of 195,000 combined audio samples. The model is highly accurate but the issue is that they use a dataset that is consisting of only TTS generated audio, ever since then deepfake or AI generated audios have advanced to mimic real human voice, and this model was not previously trained on those. (https://github.com/noorchauhan/DeepFake-Audio-Detection-MFCC)
For that reason, more datasets were used.


## Citation
A. Hamza et al., "Deepfake Audio Detection via MFCC Features Using Machine Learning," in IEEE Access, vol. 10, pp. 134018-134028, 2022, doi: 10.1109/ACCESS.2022.3231480.
Abstract: Deepfake content is created or altered synthetically using artificial intelligence (AI) approaches to appear real. It can include synthesizing audio, video, images, and text. Deepfakes may now produce natural-looking content, making them harder to identify. Much progress has been achieved in identifying video deepfakes in recent years; nevertheless, most investigations in detecting audio deepfakes have employed the ASVSpoof or AVSpoof dataset and various machine learning, deep learning, and deep learning algorithms. This research uses machine and deep learning-based approaches to identify deepfake audio. Mel-frequency cepstral coefficients (MFCCs) technique is used to acquire the most useful information from the audio. We choose the Fake-or-Real dataset, which is the most recent benchmark dataset. The dataset was created with a text-to-speech model and is divided into four sub-datasets: for-rece, for-2-sec, for-norm and for-original. These datasets are classified into sub-datasets mentioned above according to audio length and bit rate. The experimental results show that the support vector machine (SVM) outperformed the other machine learning (ML) models in terms of accuracy on for-rece and for-2-sec datasets, while the gradient boosting model performed very well using for-norm dataset. The VGG-16 model produced highly encouraging results when applied to the for-original dataset. The VGG-16 model outperforms other state-of-the-art approaches.
keywords: {Deepfakes;Deep learning;Speech synthesis;Training data;Feature extraction;Machine learning algorithms;Data models;Acoustics;Deepfakes;deepfake audio;synthetic audio;machine learning;acoustic data},
URL: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9996362&isnumber=9668973

## Installation
To initialize the project, follow these steps:

1. Clone the repository to your local machine:
   ```
   git clone https://github.com/your-username/deepfake-audio-detection.git
   cd deepfake-audio-detection
   ```

2. Set up a virtual environment (optional but recommended):
   ```
   # For Windows
   python -m venv venv
   venv\Scripts\activate

   # For Linux/macOS
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install the required dependencies

## How to Use Training the Model
To train the SVM model with the provided data, follow these steps:

1. Run the training script:
   ```
   python main.py
   ```
   After sucessfully running the main script, it will initially ask you to provide the path of the voice to analyze, provide it with the path and the
2. Run the web app by:
   ```
   python app.py
   ```

   The training script will extract MFCC features from the audio files, split the data into training and testing sets, scale the features, train the SVM model, and save the trained model and scaler for future use.

### Analyzing Audio
To classify an audio file as genuine or deepfake, follow these steps:

1. Ensure the trained model and scaler are available (already saved during training).

2. Run the analysis script:
   ```
   python analyze_audio.py path/to/your/audio/file.wav
   ```

   Replace `path/to/your/audio/file.wav` with the path to the audio file you want to analyze. The script will extract MFCC features from the audio, scale the features using the saved scaler, pass the features to the trained SVM model, and display the classification result.


