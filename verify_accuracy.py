import os
from app import analyze_audio, analyze_text

def test_audio_accuracy(test_dir_real, test_dir_fake):
    results = []
    # Test Real
    for f in os.listdir(test_dir_real):
        res = analyze_audio(os.path.join(test_dir_real, f))
        results.append(res['result'] == 'genuine')
    
    # Test Fake
    for f in os.listdir(test_dir_fake):
        res = analyze_audio(os.path.join(test_dir_fake, f))
        results.append(res['result'] == 'deepfake')
        
    accuracy = sum(results) / len(results)
    print(f"✅ Verified Audio Accuracy: {accuracy * 100:.2f}%")

# Run this to get your official project metrics