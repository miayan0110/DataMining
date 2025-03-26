import numpy as np
import csv
from tqdm import tqdm
from sklearn.neighbors import LocalOutlierFactor

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

def train_knn(data, n_neighbors=5, contamination=0.1):
    """Train a k-Nearest Neighbors (kNN) model for anomaly detection using Local Outlier Factor."""
    model = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination, novelty=True)
    model.fit(data)
    return model

def anomaly_score(model, test_samples):
    """Compute anomaly scores for test samples using kNN-based Local Outlier Factor."""
    scores = -model.decision_function(test_samples)  # Negate to align with anomaly scoring convention
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
    knn_model = train_knn(training_data)
    scores = anomaly_score(knn_model, test_data)
    save_anomaly_scores(test_data, scores, "knn_anomaly_scores.csv")
    print("Anomaly scores saved to knn_anomaly_scores.csv")
