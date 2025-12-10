import os
import glob
import librosa
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib


def extract_mfcc_features(audio_path, n_mfcc=30, n_fft=2048, hop_length=512):
    try:
        audio_data, sr = librosa.load(audio_path, sr=None)
        audio_data = librosa.util.normalize(audio_data)
    except Exception as e:
        print(f"Error loading audio file {audio_path}: {e}")
        return None

    mfcc = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)

    features = np.concatenate((mfcc, delta, delta2), axis=0)
    return np.mean(features.T, axis=0)


# for datasets that seperates the audio files into two folders (real, fake)
def create_dataset(directory, label):
    X, y = [], []
    audio_files = []
    audio_files.extend(glob.glob(os.path.join(directory, "*.wav")))
    audio_files.extend(glob.glob(os.path.join(directory, "*.flac")))
    audio_files.extend(glob.glob(os.path.join(directory, "*.mp3")))
                  
    for audio_path in audio_files:
        mfcc_features = extract_mfcc_features(audio_path)
        if mfcc_features is not None:
            X.append(mfcc_features)
            y.append(label)
        else:
            print(f"Skipping audio file {audio_path}")

    print("Loaded", len(X), "samples from", directory)
    return X, y


# for release_in_the_wild or any dataset that uses meta.csv files for classification
def create_dataset_from_metadata(metadata_path, base_audio_dir):
    df = pd.read_csv(metadata_path)

    X, y = [], []
    for _, row in df.iterrows():
        # Adapt these column names to match your real metadata file
        filename = row["file"]          # e.g. 'obama_001.wav'
        label_str = row["label"]        # e.g. 'bonafide' or 'spoof'

        # Build full path to audio
        audio_path = os.path.join(base_audio_dir, filename)

        # Map string label to numeric (0 = real, 1 = fake)
        if label_str.lower() in ["bonafide", "bona-fide", "real"]:
            label = 0
        else:  # spoof, fake, deepfake, etc.
            label = 1

        mfcc_features = extract_mfcc_features(audio_path)
        if mfcc_features is not None:
            X.append(mfcc_features)
            y.append(label)
        else:
            print(f"Skipping audio file {audio_path}")

    X = np.array(X)
    y = np.array(y)
    print("Total samples:", len(X))
    return X, y


def train_model(X, y):
    unique_classes = np.unique(y)
    print("Unique classes in y_train:", unique_classes)

    if len(unique_classes) < 2:
        raise ValueError("Atleast 2 set is required to train")

    print("Size of X:", X.shape)
    print("Size of y:", y.shape)

    class_counts = np.bincount(y)
    if np.min(class_counts) < 2:
        print("Combining both classes into one for training")
        X_train, y_train = X, y
        X_test, y_test = None, None
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        print("Size of X_train:", X_train.shape)
        print("Size of X_test:", X_test.shape)
        print("Size of y_train:", y_train.shape)
        print("Size of y_test:", y_test.shape)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    if X_test is not None:
        X_test_scaled = scaler.transform(X_test)

        svm_classifier = SVC(kernel='linear', class_weight='balanced', random_state=42)
        svm_classifier.fit(X_train_scaled, y_train)

        y_pred = svm_classifier.predict(X_test_scaled)

        accuracy = accuracy_score(y_test, y_pred)
        confusion_mtx = confusion_matrix(y_test, y_pred)

        print("Accuracy:", accuracy)
        print("Confusion Matrix:")
        print(confusion_mtx)
    else:
        print("Insufficient samples for stratified splitting. Combine both classes into one for training.")
        print("Train on all available data.")

        svm_classifier = SVC(kernel='linear', random_state=42)
        svm_classifier.fit(X_train_scaled, y_train)

    # Save the trained SVM model and scaler
    model_filename = "svm_model.pkl"
    scaler_filename = "scaler.pkl"
    joblib.dump(svm_classifier, model_filename)
    joblib.dump(scaler, scaler_filename)

def analyze_audio(input_audio_path):
    model_filename = "svm_model.pkl"
    scaler_filename = "scaler.pkl"
    svm_classifier = joblib.load(model_filename)
    scaler = joblib.load(scaler_filename)

    if not os.path.exists(input_audio_path):
        print("Error: The specified file does not exist.")
        return
    elif not (input_audio_path.lower().endswith(".wav") or input_audio_path.lower().endswith(".flac") or input_audio_path.lower().endswith(".mp3")):
        print("Error: The specified file is not a .wav, .mp3 or a .flac file.")
        return

    mfcc_features = extract_mfcc_features(input_audio_path)

    if mfcc_features is not None:
        mfcc_features_scaled = scaler.transform(mfcc_features.reshape(1, -1))
        prediction = svm_classifier.predict(mfcc_features_scaled)
        if prediction[0] == 0:
            print("The input audio is classified as genuine.")
        else:
            print("The input audio is classified as deepfake.")
    else:
        print("Error: Unable to process the input audio.")

def main():

    # for datasets that seperates the audio files into two folders (real, fake)
    genuine_dir = r"data/azvoices/real"
    deepfake_dir = r"data/azvoices/fake"
 
    X_real, y_real = create_dataset(genuine_dir, label=0)
    X_fake, y_fake = create_dataset(deepfake_dir, label=1)

    X = np.array(X_real + X_fake)
    y = np.array(y_real + y_fake)

    # for release_in_the_wild or any dataset that uses meta.csv files for classification
    # metadata_path = r"release_in_the_wild/metadata/meta.csv"
    # base_audio_dir = r"release_in_the_wild/"
    # X, y = create_dataset_from_metadata(metadata_path, base_audio_dir)
    train_model(X, y)


if __name__ == "__main__":
    main()

    user_input_file = input("Enter the path of the .wav file to analyze: ")
    analyze_audio(user_input_file)

