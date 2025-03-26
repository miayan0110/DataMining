import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import csv
from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter

logdir = f'./records'
os.makedirs(logdir, exist_ok=True)
writer = SummaryWriter(logdir)

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

class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim=16):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, latent_dim),
            nn.Tanh()
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.Tanh(),
            nn.Linear(32, 64),
            nn.Tanh(),
            nn.Linear(64, input_dim)
        )
    
    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

def train_autoencoder(data, epochs=1000, lr=0.001):
    """Train an autoencoder with MSE loss."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = data.shape[1]
    model = Autoencoder(input_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    train_data = torch.tensor(data, dtype=torch.float32).to(device)
    pbar = tqdm(range(epochs), desc=f"Training Progress: ")
    
    for epoch in pbar:
        optimizer.zero_grad()
        output = model(train_data)
        loss = criterion(output, train_data)
        loss.backward()
        optimizer.step()
        
        writer.add_scalar('AutoEncoder Loss', loss.item(), epoch+1)
        pbar.set_postfix(loss=loss.item())
    
    torch.save(model.state_dict(), "autoencoder_model.pth")
    print("Model saved to autoencoder_model.pth")
    return model

def anomaly_score(model, test_samples):
    """Compute the MSE anomaly scores for given test samples."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_samples = torch.tensor(test_samples, dtype=torch.float32).to(device)
    with torch.no_grad():
        reconstructed = model(test_samples)
    mse = ((reconstructed - test_samples) ** 2).mean(dim=1).cpu().numpy()
    return mse

def save_anomaly_scores(test_data, scores, output_file):
    """Save anomaly scores to a CSV file."""
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["id", "outliers"])
        for idx, score in enumerate(scores):
            writer.writerow([idx, score])



if __name__ == '__main__':
    training_data = load_training_data("training.csv")  # (4200, 16)
    test_data = load_test_data("test_X.csv")    # (1000, 16)
    ae_model = train_autoencoder(training_data, epochs=5000)
    scores = anomaly_score(ae_model, test_data)
    save_anomaly_scores(test_data, scores, "ae_anomaly_scores.csv")
    print("Anomaly scores saved to ae_anomaly_scores.csv")
