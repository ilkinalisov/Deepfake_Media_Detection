#app.py

import os
from flask import Flask, request, render_template, jsonify
import librosa
import numpy as np
import joblib

# Lazy-loaded models for text and video detection
text_model = None
text_tokenizer = None
video_model = None
device = None

app = Flask(__name__)
ALLOWED_AUDIO_EXTENSIONS = {"wav", "flac", "mp3"}
ALLOWED_VIDEO_EXTENSIONS = {"mp4", "avi", "mov", "mkv"}


def get_device():
    """Check for Apple Silicon MPS, CUDA, or fallback to CPU"""
    import torch
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


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


def analyze_audio(input_audio_path):
    """Analyze audio file for deepfake detection using SVM model."""
    model_filename = "models/audio_svm.pkl"
    scaler_filename = "models/audio_scaler.pkl"

    if not os.path.exists(input_audio_path):
        return {"error": "The specified file does not exist."}

    ext = input_audio_path.lower().rsplit(".", 1)[-1] if "." in input_audio_path else ""
    if ext not in ALLOWED_AUDIO_EXTENSIONS:
        return {"error": "Unsupported audio format. Use .wav, .flac, or .mp3"}

    mfcc_features = extract_mfcc_features(input_audio_path)
    if mfcc_features is None:
        return {"error": "Unable to process the input audio."}

    try:
        scaler = joblib.load(scaler_filename)
        svm_classifier = joblib.load(model_filename)
    except FileNotFoundError:
        return {"error": "Model files not found. Please train the model first."}

    mfcc_features_scaled = scaler.transform(mfcc_features.reshape(1, -1))
    prediction = svm_classifier.predict(mfcc_features_scaled)

    # Get decision function for confidence
    decision = svm_classifier.decision_function(mfcc_features_scaled)[0]
    confidence = 1 / (1 + np.exp(-abs(decision)))  # Sigmoid to normalize

    if prediction[0] == 0:
        return {"result": "genuine", "confidence": float(confidence)}
    else:
        return {"result": "deepfake", "confidence": float(confidence)}


def load_text_model():
    """Lazy load the text detection model."""
    global text_model, text_tokenizer, device
    if text_model is None:
        import torch
        from transformers import RobertaForSequenceClassification, RobertaTokenizer

        device = get_device()
        print(f"Text model loading on device: {device}")

        model_name = "roberta-base-openai-detector"
        text_tokenizer = RobertaTokenizer.from_pretrained(model_name)
        text_model = RobertaForSequenceClassification.from_pretrained(model_name)
        text_model.to(device)
        text_model.eval()
    return text_model, text_tokenizer


def analyze_text(text):
    """Analyze text for AI-generated content detection."""
    import torch

    if not text or len(text.strip()) < 10:
        return {"error": "Text must be at least 10 characters long."}

    try:
        model, tokenizer = load_text_model()
    except Exception as e:
        return {"error": f"Failed to load text model: {str(e)}"}

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.softmax(outputs.logits, dim=-1)

    # Model outputs: index 0 = fake (AI), index 1 = real (human)
    fake_prob = probabilities[0][0].item()
    real_prob = probabilities[0][1].item()

    if real_prob > fake_prob:
        return {"result": "human", "confidence": float(real_prob)}
    else:
        return {"result": "ai_generated", "confidence": float(fake_prob)}


def load_video_model():
    """Lazy load the video detection model (MobileNetV2 for feature extraction)."""
    global video_model, device
    if video_model is None:
        import torch
        from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

        device = get_device()
        print(f"Video model loading on device: {device}")

        video_model = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
        video_model.to(device)
        video_model.eval()
    return video_model


def analyze_video(input_video_path, num_frames=16):
    """
    Analyze video for deepfake detection.
    Note: This is a placeholder implementation - the model is not trained for deepfake detection.
    """
    import torch
    import cv2
    from torchvision import transforms

    if not os.path.exists(input_video_path):
        return {"error": "The specified file does not exist."}

    ext = input_video_path.lower().rsplit(".", 1)[-1] if "." in input_video_path else ""
    if ext not in ALLOWED_VIDEO_EXTENSIONS:
        return {"error": "Unsupported video format. Use .mp4, .avi, .mov, or .mkv"}

    try:
        model = load_video_model()
    except Exception as e:
        return {"error": f"Failed to load video model: {str(e)}"}

    # Open video and extract frames
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        return {"error": "Unable to open video file."}

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        cap.release()
        return {"error": "Video has no frames."}

    # Sample frames evenly
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    frames = []

    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
    cap.release()

    if len(frames) == 0:
        return {"error": "Could not extract frames from video."}

    # Preprocess frames for MobileNetV2
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    processed_frames = torch.stack([preprocess(f) for f in frames]).to(device)

    # Get features from model (placeholder: using ImageNet predictions)
    with torch.no_grad():
        outputs = model(processed_frames)
        avg_output = torch.mean(outputs, dim=0)
        probabilities = torch.softmax(avg_output, dim=0)
        max_prob = torch.max(probabilities).item()

    # Placeholder logic: This doesn't actually detect deepfakes
    # In a real implementation, you would train a classifier on deepfake video features
    # For now, return a mock result with a note
    mock_confidence = 0.5 + (np.random.random() * 0.3)  # Random between 0.5-0.8
    mock_result = "genuine" if np.random.random() > 0.5 else "deepfake"

    return {
        "result": mock_result,
        "confidence": float(mock_confidence),
        "note": "Placeholder: Video model not trained for deepfake detection"
    }


def allowed_audio_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_AUDIO_EXTENSIONS


def allowed_video_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_VIDEO_EXTENSIONS


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/predict/audio", methods=["POST"])
def predict_audio():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    audio_file = request.files["file"]
    if audio_file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    if not allowed_audio_file(audio_file.filename):
        return jsonify({"error": "Invalid format. Use .wav, .flac, or .mp3"}), 400

    if not os.path.exists("uploads"):
        os.makedirs("uploads")

    audio_path = os.path.join("uploads", audio_file.filename)
    audio_file.save(audio_path)

    try:
        result = analyze_audio(audio_path)
    finally:
        if os.path.exists(audio_path):
            os.remove(audio_path)

    if "error" in result:
        return jsonify(result), 400

    return jsonify(result)


@app.route("/predict/text", methods=["POST"])
def predict_text():
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "No text provided"}), 400

    text = data["text"]
    result = analyze_text(text)

    if "error" in result:
        return jsonify(result), 400

    return jsonify(result)


@app.route("/predict/video", methods=["POST"])
def predict_video():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    video_file = request.files["file"]
    if video_file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    if not allowed_video_file(video_file.filename):
        return jsonify({"error": "Invalid format. Use .mp4, .avi, .mov, or .mkv"}), 400

    if not os.path.exists("uploads"):
        os.makedirs("uploads")

    video_path = os.path.join("uploads", video_file.filename)
    video_file.save(video_path)

    try:
        result = analyze_video(video_path)
    finally:
        if os.path.exists(video_path):
            os.remove(video_path)

    if "error" in result:
        return jsonify(result), 400

    return jsonify(result)


if __name__ == "__main__":
    print(f"Starting Multi-Modal Deepfake Detector...")
    print(f"Device: {get_device()}")
    app.run(debug=True)
