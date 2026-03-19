import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

DATA_PATH = "Train"
BATCH_SIZE = 32
EPOCHS = 5
LR = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PaintingsDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.image_paths = []
        self.transform = transform
        for file in os.listdir(root_dir):
            if file.lower().endswith((".jpg", ".png", ".jpeg")):
                self.image_paths.append(os.path.join(root_dir, file))
        if len(self.image_paths) == 0:
            raise Exception("No images found")
        print(f"Found {len(self.image_paths)} images")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image

class ConvAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, 2, 1, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, 2, 1, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 3, 2, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

def show_images(original, reconstructed):
    original = original.cpu().numpy().transpose(1, 2, 0)
    reconstructed = reconstructed.cpu().detach().numpy().transpose(1, 2, 0)
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(original)
    plt.title("Original")
    plt.subplot(1, 2, 2)
    plt.imshow(reconstructed)
    plt.title("Reconstructed")
    plt.show()

def compute_anomaly_map(original, reconstructed):
    diff = torch.abs(original - reconstructed)
    return diff.mean(dim=0)

def show_heatmap(anomaly_map):
    anomaly_map = anomaly_map.cpu().detach().numpy()
    plt.imshow(anomaly_map, cmap="hot")
    plt.colorbar()
    plt.show()

def threshold_map(anomaly_map, threshold=0.1):
    return (anomaly_map > threshold).float()

def train():
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])
    dataset = PaintingsDataset(DATA_PATH, transform=transform)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    model = ConvAutoencoder().to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    for epoch in range(EPOCHS):
        total_loss = 0
        for images in tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            images = images.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, images)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch [{epoch+1}] Loss: {total_loss/len(loader):.4f}")
    torch.save(model.state_dict(), "model.pth")

def detect(image_path):
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])
    model = ConvAutoencoder().to(DEVICE)
    model.load_state_dict(torch.load("model.pth", map_location=DEVICE))
    model.eval()
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        reconstructed = model(image)
    original = image.squeeze(0)
    reconstructed = reconstructed.squeeze(0)
    show_images(original, reconstructed)
    anomaly_map = compute_anomaly_map(original, reconstructed)
    show_heatmap(anomaly_map)
    thresholded = threshold_map(anomaly_map)
    show_heatmap(thresholded)

if __name__ == "__main__":
    print("1. Train Model")
    print("2. Detect Hidden Layer")
    choice = input("Enter choice: ")
    if choice == "1":
        train()
    elif choice == "2":
        path = input("Enter image path: ")
        detect(path)