"""
Text Fake News Detection Model Training Script

This script trains a text-based fake news detection model using TF-IDF and SVM/MLP.
Based on the S_train_model.ipynb notebook.
"""

import pandas as pd
import joblib
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.neural_network import MLPClassifier
import warnings
warnings.filterwarnings('ignore')


def create_sample_dataset():
    """Create a sample dataset for demonstration purposes"""
    print("Creating sample dataset...")

    # Sample fake and real news texts
    fake_news = [
        "shocking discovery scientists found aliens living among us",
        "breaking celebrity caught in massive scandal cover up exposed",
        "you won't believe what this politician did next",
        "miracle cure discovered that doctors don't want you to know",
        "conspiracy revealed government hiding truth from citizens",
    ] * 20  # Repeat to create more samples

    real_news = [
        "stock market closes higher amid positive economic indicators",
        "new scientific study published in peer reviewed journal",
        "government announces policy changes following consultation",
        "international summit addresses climate change concerns",
        "researchers develop new medical treatment after clinical trials",
    ] * 20  # Repeat to create more samples

    texts = fake_news + real_news
    labels = [1] * len(fake_news) + [0] * len(real_news)  # 1 = fake, 0 = real

    df = pd.DataFrame({
        'text': texts,
        'label': labels
    })

    return df


def train_text_model(data_source='sample'):
    """
    Train text fake news detection model

    Args:
        data_source: 'sample' for demo data, or path to CSV file with 'text' and 'label' columns
    """
    print("="*60)
    print("TEXT FAKE NEWS DETECTION - MODEL TRAINING")
    print("="*60)

    # Load or create dataset
    if data_source == 'sample':
        df = create_sample_dataset()
        print(f"Using sample dataset with {len(df)} examples")
    else:
        print(f"Loading dataset from: {data_source}")
        df = pd.read_csv(data_source)

        # Handle different column names
        if 'clean_text' in df.columns:
            df = df.rename(columns={'clean_text': 'text'})

    # Clean data
    df['text'] = df['text'].fillna("")
    df = df[df['text'] != ""].reset_index(drop=True)

    print(f"\nDataset size: {len(df)} samples")
    print(f"Real news (0): {sum(df['label'] == 0)}")
    print(f"Fake news (1): {sum(df['label'] == 1)}")

    # Extract features and labels
    X = df['text']
    y = df['label']

    # TF-IDF Vectorization
    print("\nCreating TF-IDF vectorizer...")
    tfidf = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        stop_words='english'
    )

    X_tfidf = tfidf.fit_transform(X)
    print(f"Feature matrix shape: {X_tfidf.shape}")
    print(f"Average non-zero features per document: {(X_tfidf != 0).sum(axis=1).mean():.2f}")

    # Save TF-IDF vectorizer
    os.makedirs('models', exist_ok=True)
    joblib.dump(tfidf, 'models/text_tfidf_vectorizer.pkl')
    print("✓ TF-IDF vectorizer saved to models/text_tfidf_vectorizer.pkl")

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_tfidf, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\nTrain set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")

    # Train SVM Model
    print("\n" + "="*60)
    print("Training SVM Model...")
    print("="*60)
    svm_model = SVC(kernel='linear', probability=True, random_state=42)
    svm_model.fit(X_train, y_train)

    # Evaluate SVM
    y_pred_svm = svm_model.predict(X_test)
    svm_acc = accuracy_score(y_test, y_pred_svm)
    svm_f1 = f1_score(y_test, y_pred_svm)

    print(f"\nSVM Accuracy: {svm_acc:.4f}")
    print(f"SVM F1 Score: {svm_f1:.4f}")
    print("\nSVM Classification Report:")
    print(classification_report(y_test, y_pred_svm, target_names=['Real', 'Fake']))
    print("\nSVM Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred_svm))

    # Save SVM model
    joblib.dump(svm_model, 'models/text_svm_model.pkl')
    print("\n✓ SVM model saved to models/text_svm_model.pkl")

    # Train MLP Model
    print("\n" + "="*60)
    print("Training MLP Model...")
    print("="*60)
    mlp_model = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        max_iter=300,
        random_state=42
    )
    mlp_model.fit(X_train, y_train)

    # Evaluate MLP
    y_pred_mlp = mlp_model.predict(X_test)
    mlp_acc = accuracy_score(y_test, y_pred_mlp)
    mlp_f1 = f1_score(y_test, y_pred_mlp)

    print(f"\nMLP Accuracy: {mlp_acc:.4f}")
    print(f"MLP F1 Score: {mlp_f1:.4f}")
    print("\nMLP Classification Report:")
    print(classification_report(y_test, y_pred_mlp, target_names=['Real', 'Fake']))
    print("\nMLP Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred_mlp))

    # Save MLP model
    joblib.dump(mlp_model, 'models/text_mlp_model.pkl')
    print("\n✓ MLP model saved to models/text_mlp_model.pkl")

    # Summary
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"SVM - Accuracy: {svm_acc:.4f}, F1: {svm_f1:.4f}")
    print(f"MLP - Accuracy: {mlp_acc:.4f}, F1: {mlp_f1:.4f}")
    print("\nModels saved to 'models/' directory")
    print("  - text_tfidf_vectorizer.pkl")
    print("  - text_svm_model.pkl")
    print("  - text_mlp_model.pkl")

    return svm_model, mlp_model, tfidf


if __name__ == "__main__":
    # You can specify a CSV file path or use 'sample' for demo data
    # train_text_model('data/processed/clean_fake_news_dataset.csv')
    train_text_model('sample')
