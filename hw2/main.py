import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

def load_training_data(filename):
    """Load training data and return as a NumPy array."""
    data = []
    with open(filename, 'r') as f:
        for line in f:
            parts = line.strip().split(',')[1:]  # Ignore the first column (category)
            data.append([int(x) for x in parts])
    return np.array(data, dtype=np.float32)

class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim=8):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, latent_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 16),
            nn.ReLU(),
            nn.Linear(16, input_dim)
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
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(train_data)
        loss = criterion(output, train_data)
        loss.backward()
        optimizer.step()
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.6f}")
    
    return model

def anomaly_score(model, test_sample):
    """Compute the MSE anomaly score for a given test sample."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_sample = torch.tensor(test_sample, dtype=torch.float32).to(device).unsqueeze(0)
    with torch.no_grad():
        reconstructed = model(test_sample)
    mse = ((reconstructed - test_sample) ** 2).mean().item()
    return mse

# Example usage:
# training_data = load_training_data("training_data.csv")
# ae_model = train_autoencoder(training_data)
# test_sample = np.array([3,11,4,8,2,1,13,5,4,12,10,7,0,8,3,6], dtype=np.float32)
# score = anomaly_score(ae_model, test_sample)
# print("Anomaly Score (MSE):", score)
