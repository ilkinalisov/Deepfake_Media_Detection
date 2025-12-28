"""
Expanded Flask Backend for Multi-Modal Deepfake Detection
Supports: Audio, Text, and Video deepfake detection
"""

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import numpy as np
import librosa
import joblib
import os
import cv2
import tempfile
from werkzeug.utils import secure_filename

# Try to import TensorFlow for video processing
try:
    from tensorflow.keras.applications import MobileNetV2
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import GlobalAveragePooling2D
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("⚠️  TensorFlow not available. Video detection will be disabled.")

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB (larger for video files)

# File type extensions
AUDIO_EXTENSIONS = {'wav', 'mp3', 'ogg', 'flac', 'm4a'}
TEXT_EXTENSIONS = {'txt'}
VIDEO_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'flv', 'wmv'}

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# =============================================================================
# MODEL LOADING
# =============================================================================

# Audio Models
audio_model = None
audio_scaler = None

# Text Models
text_model = None
text_vectorizer = None

# Video Models
video_model = None
video_scaler = None
video_feature_extractor = None

def load_models():
    """Load all trained models"""
    global audio_model, audio_scaler
    global text_model, text_vectorizer
    global video_model, video_scaler, video_feature_extractor

    print("\n" + "="*60)
    print("LOADING MODELS")
    print("="*60)

    # Load Audio Models
    try:
        audio_model = joblib.load('svm_model.pkl')
        audio_scaler = joblib.load('scaler.pkl')
        print("✅ Audio models loaded successfully")
    except Exception as e:
        print(f"⚠️  Audio models not found: {e}")

    # Load Text Models
    try:
        text_model = joblib.load('models/text_svm_model.pkl')
        text_vectorizer = joblib.load('models/text_tfidf_vectorizer.pkl')
        print("✅ Text models loaded successfully")
    except Exception as e:
        print(f"⚠️  Text models not found: {e}")

    # Load Video Models
    if TF_AVAILABLE:
        try:
            video_model = joblib.load('models/video_svm_model.pkl')
            video_scaler = joblib.load('models/video_scaler.pkl')

            # Build feature extractor
            base_model = MobileNetV2(
                weights="imagenet",
                include_top=False,
                input_shape=(224, 224, 3)
            )
            base_model.trainable = False
            x = base_model.output
            x = GlobalAveragePooling2D()(x)
            video_feature_extractor = Model(inputs=base_model.input, outputs=x)

            print("✅ Video models loaded successfully")
        except Exception as e:
            print(f"⚠️  Video models not found: {e}")
    else:
        print("⚠️  Video models disabled (TensorFlow not available)")

    print("="*60 + "\n")

load_models()

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def allowed_file(filename, file_type):
    """Check if file extension is allowed for the given file type"""
    if '.' not in filename:
        return False

    ext = filename.rsplit('.', 1)[1].lower()

    if file_type == 'audio':
        return ext in AUDIO_EXTENSIONS
    elif file_type == 'text':
        return ext in TEXT_EXTENSIONS
    elif file_type == 'video':
        return ext in VIDEO_EXTENSIONS

    return False


def extract_audio_features(audio_path):
    """Extract MFCC features from audio file"""
    try:
        n_mfcc = 30
        n_fft = 2048
        hop_length = 512

        y, sr = librosa.load(audio_path, sr=None)
        y = librosa.util.normalize(y)

        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
        delta = librosa.feature.delta(mfcc)
        delta2 = librosa.feature.delta(mfcc, order=2)

        features = np.concatenate((mfcc, delta, delta2), axis=0)
        return np.mean(features.T, axis=0)

    except Exception as e:
        print(f"Error extracting audio features: {e}")
        return None


def extract_text_features(text_content):
    """Extract TF-IDF features from text"""
    try:
        # Transform text using the loaded vectorizer
        features = text_vectorizer.transform([text_content])
        return features
    except Exception as e:
        print(f"Error extracting text features: {e}")
        return None


def extract_video_features(video_path, max_frames=30):
    """Extract features from video using MobileNetV2"""
    try:
        cap = cv2.VideoCapture(str(video_path))
        frames = []

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0:
            return None

        # Sample frames evenly
        frame_indices = np.linspace(0, total_frames - 1, min(max_frames, total_frames), dtype=int)

        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame = cv2.resize(frame, (224, 224))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = frame.astype('float32') / 255.0
                frames.append(frame)

        cap.release()

        if not frames:
            return None

        # Extract features using MobileNetV2
        frames_array = np.array(frames)
        frame_features = video_feature_extractor.predict(frames_array, verbose=0, batch_size=32)

        # Average features across frames
        video_feature = np.mean(frame_features, axis=0)
        return video_feature

    except Exception as e:
        print(f"Error extracting video features: {e}")
        return None


# =============================================================================
# API ENDPOINTS
# =============================================================================

@app.route('/')
def index():
    """Serve the main page with all detection options"""
    return render_template('index_multimodal.html')


@app.route('/simple')
def simple():
    """Serve the simple audio-only page"""
    return render_template('index_simple.html')


@app.route('/status', methods=['GET'])
def status():
    """Check which models are available"""
    return jsonify({
        'audio': audio_model is not None,
        'text': text_model is not None,
        'video': video_model is not None and TF_AVAILABLE
    })


@app.route('/predict/audio', methods=['POST'])
def predict_audio():
    """Predict if audio is AI-generated or real"""
    if audio_model is None or audio_scaler is None:
        return jsonify({'error': 'Audio model not loaded'}), 500

    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if not allowed_file(file.filename, 'audio'):
        return jsonify({'error': f'Invalid file type. Allowed: {", ".join(AUDIO_EXTENSIONS)}'}), 400

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))

    try:
        file.save(filepath)

        # Extract features
        features = extract_audio_features(filepath)
        if features is None:
            return jsonify({'error': 'Failed to extract audio features'}), 500

        # Scale and predict
        features_scaled = audio_scaler.transform(features.reshape(1, -1))
        prediction_val = audio_model.predict(features_scaled)[0]

        # Calculate confidence
        try:
            probs = audio_model.predict_proba(features_scaled)[0]
            confidence = round(max(probs) * 100, 2)
        except:
            decision = audio_model.decision_function(features_scaled)[0]
            confidence = round(min(100, max(50, 50 + abs(decision) * 10)), 2)

        result_label = 'fake' if prediction_val == 1 else 'real'

        return jsonify({
            'prediction': result_label,
            'confidence': confidence,
            'type': 'audio'
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)


@app.route('/predict/text', methods=['POST'])
def predict_text():
    """Predict if text is fake news or real"""
    if text_model is None or text_vectorizer is None:
        return jsonify({'error': 'Text model not loaded'}), 500

    # Handle both file upload and direct text input
    text_content = None

    if 'file' in request.files:
        file = request.files['file']
        if file.filename != '' and allowed_file(file.filename, 'text'):
            text_content = file.read().decode('utf-8')

    if text_content is None and 'text' in request.form:
        text_content = request.form['text']

    if text_content is None or text_content.strip() == '':
        return jsonify({'error': 'No text provided'}), 400

    try:
        # Extract features
        features = extract_text_features(text_content)
        if features is None:
            return jsonify({'error': 'Failed to extract text features'}), 500

        # Predict
        prediction_val = text_model.predict(features)[0]

        # Calculate confidence
        try:
            probs = text_model.predict_proba(features)[0]
            confidence = round(max(probs) * 100, 2)
        except:
            decision = text_model.decision_function(features)[0]
            confidence = round(min(100, max(50, 50 + abs(decision) * 10)), 2)

        result_label = 'fake' if prediction_val == 1 else 'real'

        return jsonify({
            'prediction': result_label,
            'confidence': confidence,
            'type': 'text'
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/predict/video', methods=['POST'])
def predict_video():
    """Predict if video is deepfake or real"""
    if not TF_AVAILABLE:
        return jsonify({'error': 'Video detection not available (TensorFlow not installed)'}), 500

    if video_model is None or video_scaler is None or video_feature_extractor is None:
        return jsonify({'error': 'Video model not loaded'}), 500

    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if not allowed_file(file.filename, 'video'):
        return jsonify({'error': f'Invalid file type. Allowed: {", ".join(VIDEO_EXTENSIONS)}'}), 400

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))

    try:
        file.save(filepath)

        # Extract features
        features = extract_video_features(filepath)
        if features is None:
            return jsonify({'error': 'Failed to extract video features'}), 500

        # Scale and predict
        features_scaled = video_scaler.transform(features.reshape(1, -1))
        prediction_val = video_model.predict(features_scaled)[0]

        # Calculate confidence
        try:
            probs = video_model.predict_proba(features_scaled)[0]
            confidence = round(max(probs) * 100, 2)
        except:
            decision = video_model.decision_function(features_scaled)[0]
            confidence = round(min(100, max(50, 50 + abs(decision) * 10)), 2)

        result_label = 'fake' if prediction_val == 1 else 'real'

        return jsonify({
            'prediction': result_label,
            'confidence': confidence,
            'type': 'video'
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)


# Legacy endpoint for backwards compatibility
@app.route('/predict', methods=['POST'])
def predict():
    """Legacy audio prediction endpoint"""
    return predict_audio()


if __name__ == '__main__':
    print("\n" + "="*60)
    print("🌐 MULTI-MODAL DEEPFAKE DETECTION SERVER")
    print("="*60)
    print("Server running on: http://localhost:5000")
    print("\nAvailable endpoints:")
    print("  • POST /predict/audio - Audio deepfake detection")
    print("  • POST /predict/text  - Text fake news detection")
    print("  • POST /predict/video - Video deepfake detection")
    print("  • GET  /status        - Check model availability")
    print("="*60 + "\n")

    app.run(debug=True, port=5000, host='0.0.0.0')
