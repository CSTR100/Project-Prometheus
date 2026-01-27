"""
Technosignature Detection: Variational Anomaly Detection Pipeline (Mk5)
====================================================================

This script implements a Variational Autoencoder (VAE) based anomaly detection system.
Upgrades from Mk4:
1. Variational Autoencoder (VAE) architecture with reparameterization.
2. Robust data augmentation (flips, rotations).
3. Reconstruction heatmaps for visualization.
4. KLD + MSE combined loss.

Usage:
    python xenarch_mk5_script.py
"""

import os
import sys
from pathlib import Path
from typing import List, Tuple, Dict, Union
import json
from datetime import datetime

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from loguru import logger

# Configure logger
logger.remove()
logger.add(sys.stderr, level="INFO")
logger.add("logs/xenarch_mk5_{time}.log", rotation="10 MB")


# ============================================
# 1. MODELS
# ============================================

class ConvolutionalVAE(nn.Module):
    """Variational Autoencoder for learning normal geological features"""
    
    def __init__(self, latent_dim=128):
        super(ConvolutionalVAE, self).__init__()
        self.latent_dim = latent_dim
        
        # Encoder
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),  # 64 -> 32
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # 32 -> 16
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), # 16 -> 8
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1), # 8 -> 4
            nn.ReLU(),
        )
        
        self.fc_mu = nn.Linear(256 * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(256 * 4 * 4, latent_dim)
        
        # Decoder
        self.decoder_input = nn.Linear(latent_dim, 256 * 4 * 4)
        
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1), # 4 -> 8
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),  # 8 -> 16
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),   # 16 -> 32
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 3, stride=2, padding=1, output_padding=1),    # 32 -> 64
            nn.Sigmoid()
        )
    
    def encode(self, x):
        h = self.encoder_conv(x)
        h = torch.flatten(h, 1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        h = self.decoder_input(z)
        h = h.view(-1, 256, 4, 4)
        return self.decoder_conv(h)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def get_reconstruction_loss_map(self, x):
        """Calculate per-pixel reconstruction error"""
        reconstructed, _, _ = self.forward(x)
        return (x - reconstructed) ** 2


# ============================================
# 2. DATASET & PROCESSING
# ============================================

class ChipExtractor:
    def __init__(self, chip_size=64, overlap=0.0):
        self.chip_size = chip_size
        self.overlap = overlap
        
    def extract_grid(self, image_path: str, output_dir: str, max_size_mb: float = None) -> List[Dict]:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        chips_metadata = []
        
        with rasterio.open(image_path) as src:
            width, height = src.width, src.height
            stride = int(self.chip_size * (1 - self.overlap))
            
            chip_id = 0
            total_size_bytes = 0
            max_size_bytes = max_size_mb * 1024 * 1024 if max_size_mb else float('inf')
            
            for y in range(0, height - self.chip_size, stride):
                for x in range(0, width - self.chip_size, stride):
                    if total_size_bytes >= max_size_bytes:
                        break
                        
                    window = Window(x, y, self.chip_size, self.chip_size)
                    chip = src.read(1, window=window)
                    
                    if chip.std() < 5:
                        continue
                    
                    source_stem = Path(image_path).stem
                    chip_filename = f"{source_stem}_chip_{chip_id:04d}.tif"
                    chip_path = output_path / chip_filename
                    
                    window_transform = src.window_transform(window)
                    chip_bounds = rasterio.transform.array_bounds(
                        self.chip_size, self.chip_size, window_transform
                    )
                    
                    with rasterio.open(
                        chip_path, 'w',
                        driver='GTiff',
                        height=self.chip_size,
                        width=self.chip_size,
                        count=1,
                        dtype=chip.dtype,
                        crs=src.crs,
                        transform=window_transform
                    ) as dst:
                        dst.write(chip, 1)
                    
                    total_size_bytes += chip.nbytes
                    
                    chips_metadata.append({
                        'chip_id': chip_id,
                        'chip_path': str(chip_path),
                        'bbox': chip_bounds
                    })
                    chip_id += 1
                if total_size_bytes >= max_size_bytes:
                    break
        return chips_metadata

class LunarDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        path = self.df.iloc[idx]['chip_path']
        with rasterio.open(path) as src:
            img = src.read(1).astype(np.float32)
        
        # Normalize
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        img = img[np.newaxis, :, :] # Add channel
        
        image_tensor = torch.from_numpy(img)
        if self.transform:
            image_tensor = self.transform(image_tensor)
            
        return image_tensor, path

# ============================================
# 3. TRAINING & EVALUATION
# ============================================

def vae_loss_function(recon_x, x, mu, logvar):
    MSE = F.mse_loss(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return MSE + KLD, MSE, KLD

class VAETrainer:
    def __init__(self, model, device='cpu', lr=1e-3):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
    def train_epoch(self, loader):
        self.model.train()
        total_loss = 0
        total_mse = 0
        total_kld = 0
        
        for images, _ in tqdm(loader, desc="Training VAE"):
            images = images.to(self.device)
            self.optimizer.zero_grad()
            
            recon, mu, logvar = self.model(images)
            loss, mse, kld = vae_loss_function(recon, images, mu, logvar)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            total_mse += mse.item()
            total_kld += kld.item()
            
        return total_loss / len(loader.dataset), total_mse / len(loader.dataset), total_kld / len(loader.dataset)

    def evaluate(self, loader):
        self.model.eval()
        results = []
        
        with torch.no_grad():
            for images, paths in tqdm(loader, desc="Evaluating"):
                images = images.to(self.device)
                recon, mu, logvar = self.model(images)
                
                # Per-sample MSE
                mse = torch.mean((images - recon)**2, dim=[1, 2, 3])
                
                for i in range(len(paths)):
                    results.append({
                        'chip_path': paths[i],
                        'mse_score': mse[i].item()
                    })
        return pd.DataFrame(results)

# ============================================
# 4. VISUALIZATION
# ============================================

def save_reconstruction_examples(model, dataset, output_path, device='cpu', n=8):
    model.eval()
    indices = np.random.choice(len(dataset), n, replace=False)
    
    fig, axes = plt.subplots(3, n, figsize=(n*2, 6))
    
    with torch.no_grad():
        for i, idx in enumerate(indices):
            img, _ = dataset[idx]
            img_batch = img.unsqueeze(0).to(device)
            recon, _, _ = model(img_batch)
            
            img_np = img.squeeze().numpy()
            recon_np = recon.cpu().squeeze().numpy()
            diff_np = (img_np - recon_np)**2
            
            axes[0, i].imshow(img_np, cmap='gray')
            axes[0, i].axis('off')
            if i == 0: axes[0, i].set_ylabel('Original')
            
            axes[1, i].imshow(recon_np, cmap='gray')
            axes[1, i].axis('off')
            if i == 0: axes[1, i].set_ylabel('Recon')
            
            axes[2, i].imshow(diff_np, cmap='hot')
            axes[2, i].axis('off')
            if i == 0: axes[2, i].set_ylabel('Diff')
            
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

# ============================================
# 5. MAIN
# ============================================

def main():
    logger.info("Starting Xenarch Mk5 Anomaly Detection Pipeline")
    
    # Dirs
    data_root = Path("data")
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    # 1. Extraction
    extractor = ChipExtractor(chip_size=64)
    
    logger.info("Extracting training chips...")
    train_imgs = list(Path("training data").glob("*"))
    all_train_chips = []
    for img in train_imgs:
        if img.suffix.lower() in ['.png', '.jpg', '.tif', '.tiff']:
            chips = extractor.extract_grid(str(img), str(data_root / "processed" / "train_chips"))
            all_train_chips.extend(chips)
    
    logger.info("Extracting test chips...")
    test_imgs = list(Path("Test data").glob("*"))
    all_test_chips = []
    for img in test_imgs:
         if img.suffix.lower() in ['.png', '.jpg', '.tif', '.tiff']:
            chips = extractor.extract_grid(str(img), str(data_root / "processed" / "test_chips"), max_size_mb=10.0)
            all_test_chips.extend(chips)
            break # Just one for now
            
    # Datasets
    train_df = pd.DataFrame(all_train_chips)
    test_df = pd.DataFrame(all_test_chips)
    
    # Augmentation for training
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        # transforms.RandomRotation(90) # Requires PIL or specific torch version/wrap
    ])
    
    train_ds = LunarDataset(train_df, transform=train_transform)
    test_ds = LunarDataset(test_df)
    
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)
    
    # Model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ConvolutionalVAE(latent_dim=128)
    trainer = VAETrainer(model, device=device)
    
    # Train
    num_epochs = 5
    for epoch in range(num_epochs):
        loss, mse, kld = trainer.train_epoch(train_loader)
        logger.info(f"Epoch {epoch+1}/{num_epochs} - Loss: {loss:.4f} (MSE: {mse:.4f}, KLD: {kld:.4f})")
        
    # Save Model (NEW)
    model_path = data_root / "models" / "vae_mk5.pth"
    model_path.parent.mkdir(exist_ok=True)
    torch.save(model.state_dict(), model_path)
    logger.info(f"Model weight saved to {model_path}")

    # Evaluate
    results_df = trainer.evaluate(test_loader)
    
    # Threshold (95th percentile)
    threshold = results_df['mse_score'].quantile(0.95)
    results_df['is_anomaly'] = results_df['mse_score'] > threshold
    
    # Save
    results_df.to_csv(results_dir / "xenarch_mk5_results.csv", index=False)
    logger.info(f"Results saved to {results_dir / 'xenarch_mk5_results.csv'}")
    
    # Visualize
    save_reconstruction_examples(model, test_ds, results_dir / "mk5_reconstructions.png", device=device)
    logger.info(f"Visualization saved to {results_dir / 'mk5_reconstructions.png'}")
    
    logger.info("Pipeline Complete")

if __name__ == "__main__":
    main()