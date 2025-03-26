import os
import numpy as np
import csv
from tqdm import tqdm
from sklearn.svm import OneClassSVM

def load_training_data(filename):
    """Load training data and return as a NumPy array, excluding headers and category column."""
    data = []
    with open(filename, 'r') as f:
        lines = f.readlines()[1:]  # Skip header
        for line in lines:
            parts = line.strip().split(',')[1:]  # Ignore the first column (category)
            data.append([int(x) for x in parts])
    return np.array(data, dtype=np.float32)

def load_test_data(filename):
    """Load test data and return as a NumPy array, excluding the header."""
    data = []
    with open(filename, 'r') as f:
        lines = f.readlines()[1:]  # Skip header
        for line in lines:
            parts = line.strip().split(',')
            data.append([int(x) for x in parts])
    return np.array(data, dtype=np.float32)

def train_oneclass_svm(data, nu=0.01, kernel='rbf', gamma='scale'):
    """Train a One-Class SVM for anomaly detection."""
    model = OneClassSVM(nu=nu, kernel=kernel, gamma=gamma)
    model.fit(data)
    return model

def anomaly_score(model, test_samples):
    """Compute anomaly scores for test samples using One-Class SVM."""
    scores = -model.decision_function(test_samples)  # Negate to align with MSE convention
    return scores

def save_anomaly_scores(test_data, scores, output_file):
    """Save anomaly scores to a CSV file."""
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["id", "outliers"])
        for idx, score in enumerate(scores):
            writer.writerow([idx, score])


if __name__ == '__main__':
    training_data = load_training_data("training.csv")
    test_data = load_test_data("test_X.csv")
    svm_model = train_oneclass_svm(training_data)
    scores = anomaly_score(svm_model, test_data)
    save_anomaly_scores(test_data, scores, "svm_anomaly_scores.csv")
    print("Anomaly scores saved to svm_anomaly_scores.csv")
