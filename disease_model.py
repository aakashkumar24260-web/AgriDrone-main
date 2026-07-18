"""Train and save the Random Forest disease classifier."""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib
from pathlib import Path
import pandas as pd
import config

def generate_synthetic_data(n_samples=5000):
    """Generate synthetic training data for disease classification.
    
    Returns:
        X: Feature matrix (n_samples, 5)
        y: Target labels (0=Healthy, 1=Early, 2=Severe)
    """
    np.random.seed(42)
    
    # Initialize arrays
    X = np.zeros((n_samples, 5))
    y = np.zeros(n_samples, dtype=int)
    
    # Class distributions: 60% healthy, 25% early, 15% severe
    n_healthy = int(n_samples * 0.60)
    n_early = int(n_samples * 0.25)
    n_severe = n_samples - n_healthy - n_early
    
    # Healthy samples (class 0)
    # NDVI: high (0.55-0.85), Red: low, Green: high, Texture: low, Moisture: medium-high
    idx = 0
    for i in range(n_healthy):
        X[idx, 0] = np.random.uniform(0.55, 0.85)  # NDVI
        X[idx, 1] = np.random.uniform(0.10, 0.30)  # Red intensity
        X[idx, 2] = np.random.uniform(0.60, 0.90)  # Green intensity
        X[idx, 3] = np.random.uniform(0.05, 0.20)  # Texture variance
        X[idx, 4] = np.random.uniform(0.50, 0.80)  # Moisture index
        y[idx] = 0
        idx += 1
    
    # Early disease samples (class 1)
    for i in range(n_early):
        X[idx, 0] = np.random.uniform(0.30, 0.55)  # NDVI
        X[idx, 1] = np.random.uniform(0.30, 0.50)  # Red intensity (increased)
        X[idx, 2] = np.random.uniform(0.35, 0.60)  # Green intensity (decreased)
        X[idx, 3] = np.random.uniform(0.20, 0.45)  # Texture variance (increased)
        X[idx, 4] = np.random.uniform(0.30, 0.60)  # Moisture index (decreased)
        y[idx] = 1
        idx += 1
    
    # Severe disease samples (class 2)
    for i in range(n_severe):
        X[idx, 0] = np.random.uniform(0.05, 0.30)  # NDVI (very low)
        X[idx, 1] = np.random.uniform(0.50, 0.80)  # Red intensity (very high)
        X[idx, 2] = np.random.uniform(0.10, 0.35)  # Green intensity (very low)
        X[idx, 3] = np.random.uniform(0.40, 0.70)  # Texture variance (very high)
        X[idx, 4] = np.random.uniform(0.10, 0.40)  # Moisture index (very low)
        y[idx] = 2
        idx += 1
    
    return X, y

def train_model():
    """Train and save the Random Forest classifier."""
    print("Generating synthetic training data...")
    X, y = generate_synthetic_data(5000)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train Random Forest
    print("Training Random Forest classifier...")
    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        class_weight='balanced'
    )
    clf.fit(X_train, y_train)
    
    # Evaluate
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {accuracy:.3f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, 
          target_names=['Healthy', 'Early Disease', 'Severe Disease']))
    
    # Feature importance
    importance = clf.feature_importances_
    for name, imp in zip(config.FEATURE_NAMES, importance):
        print(f"  {name}: {imp:.3f}")
    
    # Save model
    config.MODELS_DIR.mkdir(exist_ok=True)
    model_path = config.MODELS_DIR / "disease_clf.pkl"
    joblib.dump(clf, model_path)
    print(f"\nModel saved to {model_path}")
    
    return clf

if __name__ == "__main__":
    train_model()