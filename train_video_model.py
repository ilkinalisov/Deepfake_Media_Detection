"""
Video Deepfake Detection Model Training Script

This script trains a video-based deepfake detection model using MobileNetV2 for feature extraction.
Based on the S_colab_deepfake_detection.ipynb notebook.
"""

import os
import random
import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict
import joblib
from tqdm import tqdm

# Try to import TensorFlow/Keras
try:
    from tensorflow.keras.applications import MobileNetV2
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import GlobalAveragePooling2D
    TF_AVAILABLE = True
except ImportError:
    print("Warning: TensorFlow not available. Install with: pip install tensorflow")
    TF_AVAILABLE = False

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix


class VideoFeatureExtractor:
    """Extract features from video files using MobileNetV2"""

    def __init__(self):
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow is required for video feature extraction")

        print("Building MobileNetV2 feature extractor...")
        base_model = MobileNetV2(
            weights="imagenet",
            include_top=False,
            input_shape=(224, 224, 3)
        )
        base_model.trainable = False

        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        self.model = Model(inputs=base_model.input, outputs=x)
        print(f"✓ Feature extractor ready (output dimension: {self.model.output_shape[1]})")

    def extract_frames(self, video_path, max_frames=30):
        """Extract frames from a video file"""
        cap = cv2.VideoCapture(str(video_path))
        frames = []

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0:
            return frames

        # Sample frames evenly or take max_frames
        frame_indices = np.linspace(0, total_frames - 1, min(max_frames, total_frames), dtype=int)

        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                # Preprocess frame
                frame = cv2.resize(frame, (224, 224))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = frame.astype('float32') / 255.0
                frames.append(frame)

        cap.release()
        return frames

    def extract_features(self, frames, batch_size=32):
        """Extract features from frames"""
        if not frames:
            return None

        frames_array = np.array(frames)
        features = self.model.predict(frames_array, verbose=0, batch_size=batch_size)
        # Average features across all frames to get video-level feature
        video_feature = np.mean(features, axis=0)
        return video_feature


def find_video_files(directory, extensions=['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']):
    """Find all video files in a directory"""
    video_files = []
    directory = Path(directory)

    for ext in extensions:
        video_files.extend(list(directory.rglob(f"*{ext}")))
        video_files.extend(list(directory.rglob(f"*{ext.upper()}")))

    return video_files


def train_video_model(data_dir='data/video', max_frames_per_video=30):
    """
    Train video deepfake detection model

    Args:
        data_dir: Directory containing 'real' and 'fake' subdirectories with video files
        max_frames_per_video: Maximum number of frames to extract per video
    """
    if not TF_AVAILABLE:
        print("ERROR: TensorFlow is not installed.")
        print("Install with: pip install tensorflow")
        return None

    print("="*60)
    print("VIDEO DEEPFAKE DETECTION - MODEL TRAINING")
    print("="*60)

    data_path = Path(data_dir)
    real_path = data_path / 'real'
    fake_path = data_path / 'fake'

    # Check if data directories exist
    if not real_path.exists() or not fake_path.exists():
        print(f"\nERROR: Data directories not found!")
        print(f"Expected structure:")
        print(f"  {data_dir}/")
        print(f"    ├── real/  (real videos)")
        print(f"    └── fake/  (fake videos)")
        print(f"\nPlease organize your video files in this structure.")
        return None

    # Find video files
    print(f"\nSearching for video files in {data_dir}...")
    real_videos = find_video_files(real_path)
    fake_videos = find_video_files(fake_path)

    print(f"Found {len(real_videos)} real videos")
    print(f"Found {len(fake_videos)} fake videos")

    if len(real_videos) == 0 or len(fake_videos) == 0:
        print("\nERROR: No video files found!")
        print("Supported formats: .mp4, .avi, .mov, .mkv, .flv, .wmv")
        return None

    # Initialize feature extractor
    extractor = VideoFeatureExtractor()

    # Extract features from all videos
    X = []
    y = []

    print("\nExtracting features from REAL videos...")
    for video_path in tqdm(real_videos):
        frames = extractor.extract_frames(video_path, max_frames=max_frames_per_video)
        if frames:
            feature = extractor.extract_features(frames)
            if feature is not None:
                X.append(feature)
                y.append(0)  # 0 = real

    print(f"✓ Processed {sum(np.array(y) == 0)} real videos")

    print("\nExtracting features from FAKE videos...")
    for video_path in tqdm(fake_videos):
        frames = extractor.extract_frames(video_path, max_frames=max_frames_per_video)
        if frames:
            feature = extractor.extract_features(frames)
            if feature is not None:
                X.append(feature)
                y.append(1)  # 1 = fake

    print(f"✓ Processed {sum(np.array(y) == 1)} fake videos")

    # Convert to numpy arrays
    X = np.array(X)
    y = np.array(y)

    print(f"\nFeature matrix shape: {X.shape}")
    print(f"Labels shape: {y.shape}")
    print(f"Real videos: {sum(y == 0)}, Fake videos: {sum(y == 1)}")

    # Save raw features
    os.makedirs('models', exist_ok=True)
    np.save('models/video_X_features.npy', X)
    np.save('models/video_y_labels.npy', y)
    print("\n✓ Raw features saved")

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\nTrain set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")

    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Save scaler
    joblib.dump(scaler, 'models/video_scaler.pkl')
    print("✓ Scaler saved")

    # Train SVM Model
    print("\n" + "="*60)
    print("Training SVM Model...")
    print("="*60)
    svm = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42)
    svm.fit(X_train_scaled, y_train)

    y_pred_svm = svm.predict(X_test_scaled)
    svm_acc = accuracy_score(y_test, y_pred_svm)
    svm_f1 = f1_score(y_test, y_pred_svm)

    print(f"\nSVM Accuracy: {svm_acc:.4f}")
    print(f"SVM F1 Score: {svm_f1:.4f}")
    print("\nSVM Classification Report:")
    print(classification_report(y_test, y_pred_svm, target_names=['Real', 'Fake']))
    print("\nSVM Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred_svm))

    joblib.dump(svm, 'models/video_svm_model.pkl')
    print("\n✓ SVM model saved to models/video_svm_model.pkl")

    # Train Logistic Regression
    print("\n" + "="*60)
    print("Training Logistic Regression...")
    print("="*60)
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train_scaled, y_train)

    y_pred_lr = lr.predict(X_test_scaled)
    lr_acc = accuracy_score(y_test, y_pred_lr)
    lr_f1 = f1_score(y_test, y_pred_lr)

    print(f"\nLR Accuracy: {lr_acc:.4f}")
    print(f"LR F1 Score: {lr_f1:.4f}")

    joblib.dump(lr, 'models/video_lr_model.pkl')
    print("✓ LR model saved")

    # Train MLP Model
    print("\n" + "="*60)
    print("Training MLP Model...")
    print("="*60)
    mlp = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        max_iter=300,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1
    )
    mlp.fit(X_train_scaled, y_train)

    y_pred_mlp = mlp.predict(X_test_scaled)
    mlp_acc = accuracy_score(y_test, y_pred_mlp)
    mlp_f1 = f1_score(y_test, y_pred_mlp)

    print(f"\nMLP Accuracy: {mlp_acc:.4f}")
    print(f"MLP F1 Score: {mlp_f1:.4f}")

    joblib.dump(mlp, 'models/video_mlp_model.pkl')
    print("✓ MLP model saved")

    # Summary
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"SVM - Accuracy: {svm_acc:.4f}, F1: {svm_f1:.4f}")
    print(f"LR  - Accuracy: {lr_acc:.4f}, F1: {lr_f1:.4f}")
    print(f"MLP - Accuracy: {mlp_acc:.4f}, F1: {mlp_f1:.4f}")
    print("\nModels saved to 'models/' directory:")
    print("  - video_svm_model.pkl (best performance)")
    print("  - video_lr_model.pkl")
    print("  - video_mlp_model.pkl")
    print("  - video_scaler.pkl")

    return svm, lr, mlp, scaler


if __name__ == "__main__":
    train_video_model()
