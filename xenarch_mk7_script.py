"""
Technosignature Detection: Integrated High-Resolution VAE Pipeline (Mk6)
=======================================================================

This script combines the Variational Autoencoder (VAE) architecture from Mk5
with the High-Resolution Visualization module from Mk6.

Improvements:
1. High-DPI outputs (300 DPI)
2. Upsampled visualization for detailed inspection
3. Error heatmaps for pinpointing anomalies
4. Integrated VAE training and evaluation

Usage:
    python xenarch_mk6_script.py
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
from scipy.ndimage import zoom

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
logger.add("logs/xenarch_mk6_{time}.log", rotation="10 MB")


# ============================================
# 1. MODELS (from Mk5)
# ============================================

class ConvolutionalVAE(nn.Module):
    """Variational Autoencoder for learning normal geological features"""
    
    def __init__(self, latent_dim=128):
        super(ConvolutionalVAE, self).__init__()
        self.latent_dim = latent_dim
        
        # Encoder
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),  # 128 -> 64
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # 64 -> 32
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), # 32 -> 16
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1), # 16 -> 8
            nn.ReLU(),
        )
        
        self.fc_mu = nn.Linear(256 * 8 * 8, latent_dim)
        self.fc_logvar = nn.Linear(256 * 8 * 8, latent_dim)
        
        # Decoder
        self.decoder_input = nn.Linear(latent_dim, 256 * 8 * 8)
        
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1), # 8 -> 16
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),  # 16 -> 32
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),   # 32 -> 64
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 3, stride=2, padding=1, output_padding=1),    # 64 -> 128
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
        h = h.view(-1, 256, 8, 8)
        return self.decoder_conv(h)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


# ============================================
# 2. DATASET & PROCESSING (from Mk5)
# ============================================

class ChipExtractor:
    def __init__(self, chip_size=128, overlap=0.0):
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
# 3. HIGH-RES VISUALIZATION (from user's Mk6)
# ============================================

class HighResVisualizer:
    """Enhanced visualizer with high-resolution outputs"""
    
    def __init__(self, output_dir='./results'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set matplotlib defaults for high quality
        plt.rcParams['figure.dpi'] = 300
        plt.rcParams['savefig.dpi'] = 300
        plt.rcParams['figure.figsize'] = (20, 10)
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.titlesize'] = 12
        plt.rcParams['axes.labelsize'] = 11
    
    def plot_reconstruction_examples(
        self, 
        model, 
        dataset, 
        n_samples=6, 
        device='cpu',
        upsample_factor=2,
        interpolation='bilinear'
    ):
        model.eval()
        fig, axes = plt.subplots(
            3, n_samples, 
            figsize=(n_samples * 4, 12),
            gridspec_kw={'hspace': 0.3, 'wspace': 0.2}
        )
        
        indices = np.random.choice(len(dataset), n_samples, replace=False)
        
        with torch.no_grad():
            for i, idx in enumerate(indices):
                image, path = dataset[idx]
                image_batch = image.unsqueeze(0).to(device)
                recon, _, _ = model(image_batch)
                
                original_np = image.cpu().squeeze().numpy()
                recon_np = recon.cpu().squeeze().numpy()
                error_map = np.abs(original_np - recon_np)
                
                if upsample_factor > 1:
                    original_np = zoom(original_np, upsample_factor, order=1)
                    recon_np = zoom(recon_np, upsample_factor, order=1)
                    error_map = zoom(error_map, upsample_factor, order=1)
                
                axes[0, i].imshow(original_np, cmap='gray', interpolation=interpolation, vmin=0, vmax=1)
                axes[0, i].axis('off')
                if i == 0: axes[0, i].set_ylabel('Original', fontsize=14, fontweight='bold')
                axes[0, i].set_title(f'Sample {i+1}', fontsize=11)
                
                axes[1, i].imshow(recon_np, cmap='gray', interpolation=interpolation, vmin=0, vmax=1)
                axes[1, i].axis('off')
                if i == 0: axes[1, i].set_ylabel('Reconstructed', fontsize=14, fontweight='bold')
                
                im = axes[2, i].imshow(error_map, cmap='hot', interpolation=interpolation)
                axes[2, i].axis('off')
                if i == 0: axes[2, i].set_ylabel('Recon Error', fontsize=14, fontweight='bold')
                
                if i == n_samples - 1:
                    cbar = plt.colorbar(im, ax=axes[2, i], fraction=0.046, pad=0.04)
                    cbar.set_label('Error Magnitude', rotation=270, labelpad=15)
        
        plt.suptitle('High-Resolution Reconstruction Examples', fontsize=16, fontweight='bold', y=0.98)
        output_path = self.output_dir / 'reconstruction_examples_hires.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        logger.info(f"Saved high-res reconstructions: {output_path}")
        plt.close()

    def plot_top_anomalies_hires(self, labels_df, n_samples=12):
        top_anomalies = labels_df.nlargest(n_samples, 'mse_score')
        fig, axes = plt.subplots(3, 4, figsize=(20, 15), gridspec_kw={'hspace': 0.3, 'wspace': 0.2})
        axes = axes.flatten()
        
        for idx, (_, row) in enumerate(top_anomalies.iterrows()):
            if idx >= len(axes): break
            with rasterio.open(row['chip_path']) as src:
                chip = src.read(1)
            chip_upsampled = zoom(chip, 2, order=1)
            axes[idx].imshow(chip_upsampled, cmap='gray', interpolation='bilinear')
            
            is_anomaly = row.get('is_anomaly', False)
            color = '#f44336' if is_anomaly else '#4CAF50'
            axes[idx].set_title(f"Score: {row['mse_score']:.4f}\n{Path(row['chip_path']).name}", fontsize=10, color=color, fontweight='bold')
            axes[idx].axis('off')
            
        plt.suptitle('Top Anomalies - High Resolution', fontsize=18, fontweight='bold', y=0.98)
        output_path = self.output_dir / 'top_anomalies_hires.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        logger.info(f"Saved high-res top anomalies: {output_path}")
        plt.close()


# ============================================
# 4. TRAINING & EVALUATION (from Mk5)
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
        total_loss, total_mse, total_kld = 0, 0, 0
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
                mse = torch.mean((images - recon)**2, dim=[1, 2, 3])
                for i in range(len(paths)):
                    results.append({'chip_path': paths[i], 'mse_score': mse[i].item()})
        return pd.DataFrame(results)


# ============================================
# 5. MAIN
# ============================================

def main():
    logger.info("Starting Xenarch Mk6 High-Resolution Pipeline")
    data_root = Path("data")
    results_dir = Path("results") / "high_res"
    results_dir.mkdir(exist_ok=True, parents=True)
    
    # 1. Extraction (reusing directories)
    extractor = ChipExtractor(chip_size=128)
    train_imgs = list(Path("training data").glob("*"))
    all_train_chips = []
    for img in train_imgs:
        if img.suffix.lower() in ['.png', '.jpg', '.tif', '.tiff']:
            chips = extractor.extract_grid(str(img), str(data_root / "processed" / "train_chips"), max_size_mb=10.0)
            all_train_chips.extend(chips)
    
    test_imgs = list(Path("Test data").glob("*"))
    all_test_chips = []
    for img in test_imgs:
         if img.suffix.lower() in ['.png', '.jpg', '.tif', '.tiff']:
            chips = extractor.extract_grid(str(img), str(data_root / "processed" / "test_chips"), max_size_mb=10.0)
            all_test_chips.extend(chips)
            break
            
    train_df = pd.DataFrame(all_train_chips)
    test_df = pd.DataFrame(all_test_chips)
    
    train_transform = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip()])
    train_ds = LunarDataset(train_df, transform=train_transform)
    test_ds = LunarDataset(test_df)
    
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)
    
    # 2. Model & Training
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ConvolutionalVAE(latent_dim=128)
    trainer = VAETrainer(model, device=device)
    
    num_epochs = 5  # Quicker run for demonstration
    for epoch in range(num_epochs):
        loss, mse, kld = trainer.train_epoch(train_loader)
        logger.info(f"Epoch {epoch+1}/{num_epochs} - Loss: {loss:.4f}")
        
    # 3. Evaluation & High-Res Viz
    results_df = trainer.evaluate(test_loader)
    threshold = results_df['mse_score'].quantile(0.95)
    results_df['is_anomaly'] = results_df['mse_score'] > threshold
    
    viz = HighResVisualizer(output_dir=results_dir)
    viz.plot_reconstruction_examples(model, test_ds, n_samples=6, device=device, upsample_factor=2)
    viz.plot_top_anomalies_hires(results_df, n_samples=12)
    
    results_df.to_csv(results_dir / "xenarch_mk6_results.csv", index=False)
    logger.info("Mk6 Pipeline Complete. High-res results saved to results/high_res/")

if __name__ == "__main__":
    main()
