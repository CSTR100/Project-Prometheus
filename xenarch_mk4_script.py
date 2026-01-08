"""
Technosignature Detection: Unsupervised Anomaly Detection Pipeline
====================================================================

This script implements an autoencoder-based anomaly detection system that:
1. Trains ONLY on natural geological features
2. Detects anomalies via reconstruction error
3. Validates using synthetic "artificial" features as test cases

This matches the research proposal approach: learn what natural looks like,
flag anything that doesn't fit the pattern.

Usage:
    python anomaly_detection_pipeline.py
"""

import os
import sys
from pathlib import Path
from typing import List, Tuple, Dict
import json
from datetime import datetime

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from loguru import logger

# Configure logger
logger.add("logs/anomaly_pipeline_{time}.log", rotation="10 MB")


# ============================================
# 1. DATA INGESTION
# ============================================

class LROAdapter:
    """Download and manage LRO imagery"""
    
    def __init__(self, data_dir='./data/raw/lro'):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
    def download_apollo_landing_site(self) -> str:
        """
        Download high-resolution imagery of Apollo 11 landing site.
        
        This provides real validation data: the model should flag the Lunar Module,
        rover tracks, and experiments as anomalies when trained on natural terrain.
        """
        image_path = self.data_dir / "apollo11_landing_site.tif"
        
        # Check if already downloaded
        if image_path.exists():
            logger.info(f"Apollo 11 image already exists: {image_path}")
            return str(image_path)
        
        logger.info("Downloading Apollo 11 landing site imagery from NASA...")
        
        import urllib.request
        from PIL import Image
        import io
        
        # Apollo 11 landing site from LRO NAC
        # This is a publicly accessible tile showing Tranquility Base
        # Coordinates: 0.67408°N, 23.47297°E
        # Using NASA TREK WMS service for Apollo 11 site
        
        # NASA TREK tile covering Apollo 11 landing site
        # This shows the descent stage, EASEP, and disturbed regolith
        apollo_url = "https://trek.nasa.gov/tiles/Moon/EQ/LRO_NAC_Apollo11_M1129476670_80cm/7/67/42.png"
        
        try:
            logger.info(f"Downloading from NASA TREK: {apollo_url}")
            logger.info("This image shows Apollo 11 Lunar Module descent stage and experiments")
            
            with urllib.request.urlopen(apollo_url, timeout=60) as response:
                image_data = response.read()
            
            # Convert to grayscale numpy array
            img = Image.open(io.BytesIO(image_data)).convert('L')
            data = np.array(img, dtype=np.uint8)
            
            logger.success(f"Downloaded Apollo 11 image: {data.shape}")
            
            # Save as GeoTIFF with proper geolocation
            # Apollo 11 landing site: 0.67408°N, 23.47297°E
            height, width = data.shape
            
            # Calculate approximate bounds for this tile
            # Each tile at zoom level 7 covers ~0.703° at equator
            center_lon = 23.47297
            center_lat = 0.67408
            half_size = 0.35  # degrees
            
            transform = rasterio.transform.from_bounds(
                center_lon - half_size,
                center_lat - half_size,
                center_lon + half_size,
                center_lat + half_size,
                width, height
            )
            
            with rasterio.open(
                image_path, 'w',
                driver='GTiff',
                height=height,
                width=width,
                count=1,
                dtype=np.uint8,
                crs='EPSG:4326',
                transform=transform
            ) as dst:
                dst.write(data, 1)
                dst.update_tags(
                    mission='LRO',
                    instrument='NAC',
                    source='NASA TREK',
                    target='Apollo 11 Landing Site',
                    location='Tranquility Base',
                    coordinates='0.67408°N, 23.47297°E',
                    resolution_meters=0.5,
                    features='Lunar Module descent stage, EASEP experiments, disturbed regolith'
                )
            
            logger.success(f"Saved Apollo 11 landing site image: {image_path}")
            logger.info("Image contains actual human artifacts for validation testing")
            return str(image_path)
            
        except Exception as e:
            logger.error(f"Failed to download Apollo 11 data: {e}")
            logger.info("Falling back to generic lunar imagery...")
            return self.download_generic_lunar_imagery()
    
    def download_generic_lunar_imagery(self) -> str:
        """
        Download generic lunar imagery (no human artifacts).
        This is used for training the model on natural geology only.
        """
        image_path = self.data_dir / "lunar_natural.tif"
        
        if image_path.exists():
            logger.info(f"Natural lunar image already exists: {image_path}")
            return str(image_path)
        
        logger.info("Downloading natural lunar terrain from NASA...")
        
        import urllib.request
        from PIL import Image
        import io
        
        # Use LRO WAC global mosaic tile from lunar farside
        # (guaranteed no human artifacts)
        wac_url = "https://trek.nasa.gov/tiles/Moon/EQ/LRO_WAC_Mosaic_global_100m_June2013/4/8/5.jpg"
        
        try:
            logger.info(f"Downloading from: {wac_url}")
            with urllib.request.urlopen(wac_url, timeout=30) as response:
                image_data = response.read()
            
            img = Image.open(io.BytesIO(image_data)).convert('L')
            data = np.array(img, dtype=np.uint8)
            
            logger.success(f"Downloaded natural lunar image: {data.shape}")
            
            height, width = data.shape
            transform = rasterio.transform.from_bounds(
                -180, -45, -90, 0,
                width, height
            )
            
            with rasterio.open(
                image_path, 'w',
                driver='GTiff',
                height=height,
                width=width,
                count=1,
                dtype=np.uint8,
                crs='EPSG:4326',
                transform=transform
            ) as dst:
                dst.write(data, 1)
                dst.update_tags(
                    mission='LRO',
                    instrument='WAC',
                    source='NASA TREK',
                    resolution_meters=100.0,
                    features='Natural lunar terrain - no human artifacts'
                )
            
            logger.success(f"Saved natural lunar image: {image_path}")
            return str(image_path)
            
        except Exception as e:
            logger.error(f"Failed to download lunar data: {e}")
            logger.error("Please check internet connection or provide local image file")
            raise
    
    def download_example_nac(self) -> str:
        """
        Main download function - gets Apollo landing site for validation.
        """
        return self.download_apollo_landing_site()
    
    def extract_metadata(self, image_path: str) -> Dict:
        """Extract metadata"""
        with rasterio.open(image_path) as src:
            bounds = src.bounds
            metadata = {
                'image_id': Path(image_path).stem,
                'width': src.width,
                'height': src.height,
                'bbox_coords': (bounds.left, bounds.bottom, bounds.right, bounds.top)
            }
        return metadata


# ============================================
# 2. CHIP EXTRACTION
# ============================================

class ChipExtractor:
    """Extract training chips from images"""
    
    def __init__(self, chip_size=64, overlap=0.0):
        self.chip_size = chip_size
        self.overlap = overlap
        
    def extract_grid(self, image_path: str, output_dir: str, max_size_mb: float = None) -> List[Dict]:
        """Extract regular grid of chips"""
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
                        logger.warning(f"Reached 10MB limit. Stopping chip extraction.")
                        break
                        
                    window = Window(x, y, self.chip_size, self.chip_size)
                    chip = src.read(1, window=window)
                    
                    # Skip empty chips
                    if chip.std() < 5:
                        continue
                    
                    # Save chip
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
                    
                    # Update total size (approximate size of chip data)
                    total_size_bytes += chip.nbytes
                    
                    center_x = (chip_bounds[0] + chip_bounds[2]) / 2
                    center_y = (chip_bounds[1] + chip_bounds[3]) / 2
                    
                    chips_metadata.append({
                        'chip_id': chip_id,
                        'chip_path': str(chip_path),
                        'center_lon': center_x,
                        'center_lat': center_y,
                        'bbox': chip_bounds
                    })
                    
                    chip_id += 1
                if total_size_bytes >= max_size_bytes:
                    break
            
            logger.info(f"Extracted {chip_id} chips (Total data: {total_size_bytes / 1024 / 1024:.2f} MB)")
            
        return chips_metadata


# ============================================
# 3. SYNTHETIC LABELING (for validation)
# ============================================

class SyntheticAnnotator:
    """Create labels to simulate natural vs. artificial for testing"""
    
    def generate_labels(self, chips_metadata: List[Dict], image_path: str) -> pd.DataFrame:
        """Generate synthetic labels based on image analysis"""
        labels = []
        
        for chip_meta in chips_metadata:
            chip_path = chip_meta['chip_path']
            
            with rasterio.open(chip_path) as chip_src:
                chip_data = chip_src.read(1)
                
                # Heuristics for labeling
                edges = self._detect_edges(chip_data)
                linear_score = self._detect_linear_features(edges)
                circular_score = self._detect_circular_features(chip_data)
                
                # Label assignment
                if linear_score > 0.3 and chip_data.mean() > 150:
                    category = 'artificial'
                    subcategory = 'linear_feature'
                    confidence = min(linear_score, 0.95)
                elif circular_score > 0.4:
                    category = 'natural'
                    subcategory = 'crater'
                    confidence = min(circular_score, 0.98)
                else:
                    category = 'natural'
                    subcategory = 'terrain'
                    confidence = 0.85
                
                labels.append({
                    'chip_id': chip_meta['chip_id'],
                    'chip_path': chip_path,
                    'category': category,
                    'subcategory': subcategory,
                    'confidence': confidence,
                    'center_lon': chip_meta['center_lon'],
                    'center_lat': chip_meta['center_lat']
                })
        
        df = pd.DataFrame(labels)
        logger.info(f"Generated {len(df)} labels: {df['category'].value_counts().to_dict()}")
        return df
    
    def _detect_edges(self, image: np.ndarray) -> np.ndarray:
        """Detect edges using Sobel filter"""
        from scipy import ndimage
        sx = ndimage.sobel(image, axis=0, mode='constant')
        sy = ndimage.sobel(image, axis=1, mode='constant')
        return np.hypot(sx, sy)
    
    def _detect_linear_features(self, edges: np.ndarray) -> float:
        """Score for linear features"""
        if edges.max() == 0:
            return 0.0
        
        threshold = edges.max() * 0.3
        edge_pixels = edges > threshold
        
        if edge_pixels.sum() < 10:
            return 0.0
        
        row_sums = edge_pixels.sum(axis=1)
        col_sums = edge_pixels.sum(axis=0)
        
        max_alignment = max(row_sums.max(), col_sums.max())
        total_edges = edge_pixels.sum()
        
        return min(max_alignment / total_edges, 1.0)
    
    def _detect_circular_features(self, image: np.ndarray) -> float:
        """Score for circular features"""
        center = image.shape[0] // 2
        y, x = np.ogrid[:image.shape[0], :image.shape[1]]
        
        center_region = ((x - center)**2 + (y - center)**2) < (center * 0.4)**2
        edge_region = ((x - center)**2 + (y - center)**2) > (center * 0.6)**2
        edge_region &= ((x - center)**2 + (y - center)**2) < (center * 0.9)**2
        
        if center_region.sum() == 0 or edge_region.sum() == 0:
            return 0.0
        
        center_mean = image[center_region].mean()
        edge_mean = image[edge_region].mean()
        
        contrast = edge_mean - center_mean
        return min(max(contrast / 50, 0), 1.0)


# ============================================
# 4. AUTOENCODER MODEL
# ============================================

class ConvolutionalAutoencoder(nn.Module):
    """Autoencoder for learning normal geological features"""
    
    def __init__(self, latent_dim=128):
        super(ConvolutionalAutoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),  # 64 -> 32
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # 32 -> 16
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), # 16 -> 8
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1), # 8 -> 4
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, latent_dim)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256 * 4 * 4),
            nn.ReLU(),
            nn.Unflatten(1, (256, 4, 4)),
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed
    
    def get_reconstruction_error(self, x):
        """Calculate per-sample reconstruction error"""
        reconstructed = self.forward(x)
        error = torch.mean((x - reconstructed) ** 2, dim=[1, 2, 3])
        return error


# ============================================
# 5. DATASET
# ============================================

class LunarChipDataset(Dataset):
    """Dataset for lunar image chips"""
    
    def __init__(self, labels_df: pd.DataFrame, natural_only=True, transform=None):
        self.transform = transform
        
        # KEY: Filter to only natural samples for training
        if natural_only:
            self.labels_df = labels_df[labels_df['category'] == 'natural'].copy()
            logger.info(f"Filtered to {len(self.labels_df)} natural samples only")
        else:
            self.labels_df = labels_df
    
    def __len__(self):
        return len(self.labels_df)
    
    def __getitem__(self, idx):
        row = self.labels_df.iloc[idx]
        
        with rasterio.open(row['chip_path']) as src:
            image = src.read(1).astype(np.float32)
        
        # Normalize to [0, 1]
        image = (image - image.min()) / (image.max() - image.min() + 1e-8)
        
        # Single channel
        image = image[np.newaxis, :, :]
        
        if self.transform:
            image = self.transform(torch.from_numpy(image))
        else:
            image = torch.from_numpy(image)
        
        return image, row['chip_path']  # Return path for tracking


# ============================================
# 6. TRAINER
# ============================================

class AutoencoderTrainer:
    """Train autoencoder on reconstruction task"""
    
    def __init__(self, model, device='cpu'):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
    
    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        
        pbar = tqdm(train_loader, desc="Training")
        for images, _ in pbar:
            images = images.to(self.device)
            
            self.optimizer.zero_grad()
            reconstructed = self.model(images)
            loss = self.criterion(reconstructed, images)
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            pbar.set_postfix({'loss': running_loss / (pbar.n + 1)})
        
        return running_loss / len(train_loader)
    
    def detect_anomalies(self, test_loader, threshold_percentile=95):
        """Detect anomalies based on reconstruction error"""
        self.model.eval()
        all_errors = []
        all_paths = []
        
        with torch.no_grad():
            for images, paths in tqdm(test_loader, desc="Computing anomaly scores"):
                images = images.to(self.device)
                errors = self.model.get_reconstruction_error(images)
                all_errors.extend(errors.cpu().numpy())
                all_paths.extend(paths)
        
        all_errors = np.array(all_errors)
        threshold = np.percentile(all_errors, threshold_percentile)
        
        logger.info(f"Anomaly threshold ({threshold_percentile}th percentile): {threshold:.6f}")
        
        return all_errors, all_paths, threshold


# ============================================
# 7. VISUALIZATION
# ============================================

class ResultsVisualizer:
    """Visualize results"""
    
    def __init__(self, output_dir='./results'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def plot_reconstruction_examples(self, model, dataset, n_samples=6, device='cpu'):
        """Plot original vs reconstructed images"""
        model.eval()
        
        fig, axes = plt.subplots(2, n_samples, figsize=(15, 5))
        
        indices = np.random.choice(len(dataset), n_samples, replace=False)
        
        with torch.no_grad():
            for i, idx in enumerate(indices):
                image, _ = dataset[idx]
                image = image.unsqueeze(0).to(device)
                
                reconstructed = model(image)
                
                # Original
                axes[0, i].imshow(image.cpu().squeeze(), cmap='gray')
                axes[0, i].axis('off')
                if i == 0:
                    axes[0, i].set_ylabel('Original', fontsize=12)
                
                # Reconstructed
                axes[1, i].imshow(reconstructed.cpu().squeeze(), cmap='gray')
                axes[1, i].axis('off')
                if i == 0:
                    axes[1, i].set_ylabel('Reconstructed', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'reconstruction_examples.png', dpi=150)
        logger.info(f"Saved reconstruction examples")
        plt.close()
    
    def plot_anomaly_distribution(self, labels_df):
        """Plot distribution of anomaly scores by category"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Histogram
        for category in labels_df['category'].unique():
            data = labels_df[labels_df['category'] == category]['anomaly_score']
            ax1.hist(data, bins=30, alpha=0.6, label=category)
        
        ax1.axvline(labels_df['anomaly_threshold'].iloc[0], 
                   color='r', linestyle='--', label='Threshold')
        ax1.set_xlabel('Reconstruction Error')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Anomaly Score Distribution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Box plot
        labels_df.boxplot(column='anomaly_score', by='category', ax=ax2)
        ax2.axhline(labels_df['anomaly_threshold'].iloc[0], 
                   color='r', linestyle='--', label='Threshold')
        ax2.set_xlabel('Category')
        ax2.set_ylabel('Reconstruction Error')
        ax2.set_title('Anomaly Scores by Category')
        plt.suptitle('')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'anomaly_distribution.png', dpi=150)
        logger.info(f"Saved anomaly distribution plot")
        plt.close()
    
    def plot_training_curve(self, train_losses):
        """Plot training loss curve"""
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, marker='o')
        plt.xlabel('Epoch')
        plt.ylabel('Reconstruction Loss (MSE)')
        plt.title('Training Curve: Learning Natural Geology')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'training_curve.png', dpi=150)
        logger.info(f"Saved training curve")
        plt.close()
    
    def plot_top_anomalies(self, labels_df, n_samples=12):
        """Plot the top anomalies detected"""
        top_anomalies = labels_df.nlargest(n_samples, 'anomaly_score')
        
        fig, axes = plt.subplots(3, 4, figsize=(12, 9))
        axes = axes.flatten()
        
        for idx, (_, row) in enumerate(top_anomalies.iterrows()):
            if idx >= len(axes):
                break
            
            with rasterio.open(row['chip_path']) as src:
                chip = src.read(1)
            
            axes[idx].imshow(chip, cmap='gray')
            axes[idx].set_title(
                f"{row['category']}\nScore: {row['anomaly_score']:.4f}",
                fontsize=8
            )
            axes[idx].axis('off')
        
        plt.suptitle('Top 12 Anomalies (Highest Reconstruction Error)', fontsize=14)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'top_anomalies.png', dpi=150)
        logger.info(f"Saved top anomalies visualization")
        plt.close()


# ============================================
# 8. MAIN PIPELINE
# ============================================

def main():
    """Run complete anomaly detection pipeline"""
    
    logger.info("="*70)
    logger.info("UNSUPERVISED ANOMALY DETECTION PIPELINE")
    logger.info("Training on natural geology only, detecting outliers")
    logger.info("="*70)
    
    # Configuration
    config = {
        'data_dir': './data',
        'chip_size': 64,
        'batch_size': 16,
        'num_epochs': 10,
        'latent_dim': 128,
        'anomaly_threshold_percentile': 95,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    logger.info(f"Configuration: {json.dumps(config, indent=2)}")
    
    # Step 1 & 2: Load local images and extract chips
    logger.info("\n[1/6] Extracting training chips from 'training data' folder...")
    training_data_dir = Path("training data")
    training_chips_dir = Path(config['data_dir']) / "processed" / "training_chips"
    training_chips_dir.mkdir(parents=True, exist_ok=True)
    
    extractor = ChipExtractor(chip_size=config['chip_size'], overlap=0.0)
    all_training_chips = []
    
    for img_file in training_data_dir.glob("*"):
        if img_file.suffix.lower() in ['.png', '.jpg', '.jpeg', '.tif', '.tiff']:
            logger.info(f"Processing training image: {img_file.name}")
            chips = extractor.extract_grid(str(img_file), str(training_chips_dir))
            all_training_chips.extend(chips)
    
    chips_metadata = all_training_chips
    
    logger.info("\n[2/6] Extracting test chips from 'Test data' folder (10MB limit)...")
    test_data_dir = Path("Test data")
    test_chips_dir = Path(config['data_dir']) / "processed" / "test_chips"
    test_chips_dir.mkdir(parents=True, exist_ok=True)
    
    test_chips_metadata = []
    for img_file in test_data_dir.glob("*"):
        if img_file.suffix.lower() in ['.png', '.jpg', '.jpeg', '.tif', '.tiff']:
            logger.info(f"Processing test image: {img_file.name}")
            chips = extractor.extract_grid(str(img_file), str(test_chips_dir), max_size_mb=10.0)
            test_chips_metadata.extend(chips)
            break # Only process first image in test data as per request or just one
    
    # Step 3: Labels
    # For training, we assume all training data is "natural"
    training_labels = pd.DataFrame([{
        'chip_id': c['chip_id'],
        'chip_path': c['chip_path'],
        'category': 'natural',
        'subcategory': 'training',
        'confidence': 1.0,
        'center_lon': c['center_lon'],
        'center_lat': c['center_lat']
    } for c in all_training_chips])
    
    # For testing, we don't know, but we'll label them as 'test' to monitor
    test_labels = pd.DataFrame([{
        'chip_id': c['chip_id'],
        'chip_path': c['chip_path'],
        'category': 'test',
        'subcategory': 'test_data',
        'confidence': 1.0,
        'center_lon': c['center_lon'],
        'center_lat': c['center_lat']
    } for c in test_chips_metadata])
    
    training_labels.to_csv(f"{config['data_dir']}/training_labels.csv", index=False)
    test_labels.to_csv(f"{config['data_dir']}/test_labels.csv", index=False)
    
    # Step 4: Train autoencoder ONLY on natural terrain
    logger.info(f"\n[4/6] Training autoencoder on NATURAL terrain only...")
    logger.info("(Artificial samples held out for validation)")
    
    # Create dataset with ONLY natural samples
    train_dataset = LunarChipDataset(training_labels, natural_only=True)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'], 
        shuffle=True,
        num_workers=0
    )
    
    # Initialize model
    model = ConvolutionalAutoencoder(latent_dim=config['latent_dim'])
    trainer = AutoencoderTrainer(model, device=config['device'])
    
    # Training loop
    train_losses = []
    logger.info(f"Training on {config['device']}...")
    
    for epoch in range(config['num_epochs']):
        logger.info(f"\nEpoch {epoch+1}/{config['num_epochs']}")
        loss = trainer.train_epoch(train_loader)
        train_losses.append(loss)
        logger.info(f"Reconstruction Loss: {loss:.6f}")
    
    # Save model
    model_path = Path(config['data_dir']) / 'models' / 'autoencoder.pth'
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_path)
    logger.success(f"Model saved to {model_path}")
    
    # Step 5: Detect anomalies on ALL data
    logger.info("\n[5/6] Running anomaly detection on all chips...")
    
    # Create test dataset with test samples
    test_dataset = LunarChipDataset(test_labels, natural_only=False)
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config['batch_size'], 
        shuffle=False,
        num_workers=0
    )
    
    # Get anomaly scores
    errors, paths, threshold = trainer.detect_anomalies(
        test_loader, 
        threshold_percentile=config['anomaly_threshold_percentile']
    )
    
    # Add scores to dataframe
    test_labels['anomaly_score'] = errors
    test_labels['anomaly_threshold'] = threshold
    test_labels['is_anomaly'] = errors > threshold
    
    # Save results
    test_labels.to_csv(f"{config['data_dir']}/anomaly_results.csv", index=False)
    
    # Generate visualizations
    logger.info("\nGenerating visualizations...")
    viz = ResultsVisualizer(output_dir='./results')
    
    viz.plot_training_curve(train_losses)
    viz.plot_reconstruction_examples(model, test_dataset, device=config['device'])
    viz.plot_anomaly_distribution(test_labels)
    viz.plot_top_anomalies(test_labels)
    
    # Final summary
    logger.info("\n" + "="*70)
    logger.info("PIPELINE COMPLETE!")
    logger.info("="*70)
    logger.info(f"✓ Processed {len(all_training_chips)} training chips")
    logger.info(f"✓ Processed {len(test_chips_metadata)} test chips")
    logger.info(f"✓ Detected {test_labels['is_anomaly'].sum()} anomalies in test data")
    logger.info(f"✓ Results saved to: {config['data_dir']}/anomaly_results.csv")
    logger.info(f"✓ Visualizations saved to: ./results/")
    logger.info("\nNext steps:")
    logger.info("  1. Review top anomalies in results/top_anomalies.png")
    logger.info("  2. Apply to more real lunar imagery")
    logger.info("  3. Refine threshold based on expert review")


if __name__ == "__main__":
    main()
