from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import numpy as np
import librosa
import pickle
import joblib
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
# Enable CORS for all routes so your HTML can talk to port 5000
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'ogg', 'flac', 'm4a'}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Load the trained model and scaler
try:
    # Using joblib as per your main.py training script
    model = joblib.load('svm_model.pkl')
    scaler = joblib.load('scaler.pkl')
    print("✅ Models loaded successfully!")
except Exception as e:
    print(f"❌ Error loading models: {e}")
    print("Make sure svm_model.pkl and scaler.pkl are in the same directory and you have run main.py first.")
    model = None
    scaler = None


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def extract_features(audio_path):
    """
    Extract MFCC features exactly matching main.py logic.
    Do NOT change parameters here unless you retrain the model.
    """
    try:
        # Parameters from main.py
        n_mfcc = 30
        n_fft = 2048
        hop_length = 512

        # Load audio
        y, sr = librosa.load(audio_path, sr=None)
        y = librosa.util.normalize(y)
        
        # Extract features
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
        delta = librosa.feature.delta(mfcc)
        delta2 = librosa.feature.delta(mfcc, order=2)

        # Concatenate features
        features = np.concatenate((mfcc, delta, delta2), axis=0)
        
        # Return mean of transpose
        return np.mean(features.T, axis=0)
    
    except Exception as e:
        print(f"Error extracting features: {e}")
        return None


@app.route('/predict', methods=['POST'])
def predict():
    if model is None or scaler is None:
        return jsonify({'error': 'Model not loaded.'}), 500
    
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400
    
    file = request.files['audio']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400
    
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
    
    try:
        file.save(filepath)
        
        # 1. Extract Features
        features = extract_features(filepath)
        if features is None:
            return jsonify({'error': 'Failed to extract audio features'}), 500
        
        # 2. Scale Features
        features_scaled = scaler.transform(features.reshape(1, -1))
        
        # 3. Predict
        # The model returns [0] for Real, [1] for Fake (based on your main.py)
        prediction_val = model.predict(features_scaled)[0]
        
        # 4. Calculate Confidence
        try:
            # Try to get probability if the SVM was trained with probability=True
            # Note: Your main.py didn't explicitly set probability=True, so this might fail back to decision_function
            probs = model.predict_proba(features_scaled)[0]
            confidence = round(max(probs) * 100, 2)
        except:
            # Fallback using decision distance
            decision = model.decision_function(features_scaled)[0]
            confidence = round(min(100, max(50, 50 + abs(decision) * 10)), 2)
        
        # 5. Result
        result_label = 'fake' if prediction_val == 1 else 'real'
        
        return jsonify({
            'prediction': result_label,
            'confidence': confidence
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        # Clean up uploaded file
        if os.path.exists(filepath):
            os.remove(filepath)

@app.route('/simple')
def simple():
    return render_template('index_simple.html')

if __name__ == '__main__':
    print("🌐 Server starting on http://localhost:5000")
    app.run(debug=True, port=5000)