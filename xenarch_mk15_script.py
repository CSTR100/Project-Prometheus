"""
Technosignature Detection: Xenarch Mk14 - Stable Training Edition
====================================================================

Key Improvements over Mk13:
✓ Gradient clipping to prevent divergence
✓ KL annealing (warm-up schedule) for stable latent space
✓ Enhanced batch normalization with robust epsilon
✓ Learning rate scheduler with warmup
✓ Early stopping on NaN detection
✓ Checkpoint saving for best model
✓ Better logging and diagnostics

Maintains Mk13's optimized detection:
✓ 30% weight on contextual metric for circular features
✓ Adaptive confidence calculation
✓ Optimized for Apollo landing site detection

Usage:
    python xenarch_mk14_stable.py
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
from scipy.ndimage import zoom, gaussian_filter, generic_filter, label
from scipy.spatial import distance
from sklearn.cluster import DBSCAN

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
logger.add("logs/xenarch_mk14_{time}.log", rotation="10 MB")


# ============================================
# 1. STABLE VAE MODEL
# ============================================

class StableConvolutionalVAE(nn.Module):
    """VAE with enhanced stability features"""
    
    def __init__(self, latent_dim=64, input_size=256):
        super(StableConvolutionalVAE, self).__init__()
        self.latent_dim = latent_dim
        self.input_size = input_size
        
        final_size = input_size // 16
        
        # Encoder with LayerNorm for stability
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32, eps=1e-3),  # Larger epsilon
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64, eps=1e-3),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128, eps=1e-3),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256, eps=1e-3),
            nn.ReLU(),
        )
        
        # Use smaller initialization for latent layers
        self.fc_mu = nn.Linear(256 * final_size * final_size, latent_dim)
        self.fc_logvar = nn.Linear(256 * final_size * final_size, latent_dim)
        
        # Initialize with smaller weights
        nn.init.xavier_uniform_(self.fc_mu.weight, gain=0.01)
        nn.init.xavier_uniform_(self.fc_logvar.weight, gain=0.01)
        nn.init.constant_(self.fc_mu.bias, 0)
        nn.init.constant_(self.fc_logvar.bias, 0)
        
        # Decoder
        self.decoder_input = nn.Linear(latent_dim, 256 * final_size * final_size)
        nn.init.xavier_uniform_(self.decoder_input.weight, gain=0.01)
        
        self.final_size = final_size
        
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128, eps=1e-3),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64, eps=1e-3),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32, eps=1e-3),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )
        
        logger.info(f"Stable VAE initialized: latent_dim={latent_dim}, input_size={input_size}")
    
    def encode(self, x):
        h = self.encoder_conv(x)
        h = torch.flatten(h, 1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        
        # Clamp logvar to prevent numerical instability
        logvar = torch.clamp(logvar, min=-10, max=10)
        
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        h = self.decoder_input(z)
        h = h.view(-1, 256, self.final_size, self.final_size)
        return self.decoder_conv(h)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


# ============================================
# 2. DATA PROCESSING (Same as Mk13)
# ============================================

class ChipExtractor:
    def __init__(self, chip_size=256, overlap=0.0):
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
                    
                    center_x = (chip_bounds[0] + chip_bounds[2]) / 2
                    center_y = (chip_bounds[1] + chip_bounds[3]) / 2
                    
                    chips_metadata.append({
                        'chip_id': chip_id,
                        'chip_path': str(chip_path),
                        'center_x': center_x,
                        'center_y': center_y,
                        'bbox': chip_bounds
                    })
                    chip_id += 1
                if total_size_bytes >= max_size_bytes:
                    break
                    
            logger.info(f"Extracted {chip_id} chips from {Path(image_path).name}")
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
        
        # Robust normalization
        img_min, img_max = np.percentile(img, [1, 99])
        img = np.clip(img, img_min, img_max)
        img = (img - img_min) / (img_max - img_min + 1e-8)
        img = img[np.newaxis, :]
        
        image_tensor = torch.from_numpy(img).float()
        if self.transform:
            image_tensor = self.transform(image_tensor)
            
        return image_tensor, path


# ============================================
# 3. MULTI-METRIC ANOMALY SCORER (Same as Mk13)
# ============================================

class MultiMetricAnomalyScorer:
    """Combines multiple metrics - optimized for circular features"""
    
    def __init__(self, model, device='cpu'):
        self.model = model.to(device)
        self.device = device
        self.model.eval()
    
    def compute_reconstruction_error(self, image):
        with torch.no_grad():
            recon, mu, logvar = self.model(image)
            mse = torch.mean((image - recon) ** 2, dim=[1, 2, 3])
            return mse.cpu().numpy(), recon
    
    def compute_latent_density(self, image):
        with torch.no_grad():
            mu, logvar = self.model.encode(image)
            distances = torch.sqrt(torch.sum(mu ** 2, dim=1))
            density_scores = torch.exp(-distances / self.model.latent_dim)
            return 1.0 - density_scores.cpu().numpy()
    
    def compute_contextual_anomaly(self, image):
        """CRITICAL for circular lander detection"""
        scores = []
        
        for i in range(image.shape[0]):
            img_np = image[i, 0].cpu().numpy()
            
            mean_brightness = np.mean(img_np)
            std_brightness = np.std(img_np)
            
            bright_threshold = mean_brightness + 2 * std_brightness
            bright_pixels = img_np > bright_threshold
            
            if bright_pixels.sum() < 5:
                brightness_anomaly = 0.0
            else:
                bright_region_mean = np.mean(img_np[bright_pixels])
                brightness_anomaly = min((bright_region_mean - mean_brightness) / (std_brightness + 1e-8) / 3, 1.0)
            
            try:
                local_std = generic_filter(img_np, np.std, size=9)
                texture_mean = np.mean(local_std)
                texture_std = np.std(local_std)
                texture_outliers = np.abs(local_std - texture_mean) > 2 * texture_std
                texture_anomaly = texture_outliers.sum() / img_np.size
            except:
                texture_anomaly = 0.0
            
            bright_regions, num_regions = label(bright_pixels)
            
            size_brightness_anomaly = 0.0
            if num_regions > 0:
                for region_id in range(1, num_regions + 1):
                    region_mask = bright_regions == region_id
                    region_size = region_mask.sum()
                    region_brightness = np.mean(img_np[region_mask])
                    
                    expected_brightness = mean_brightness + (region_size / img_np.size) * std_brightness
                    
                    if region_brightness > expected_brightness:
                        anomaly_strength = (region_brightness - expected_brightness) / (std_brightness + 1e-8)
                        size_brightness_anomaly = max(size_brightness_anomaly, min(anomaly_strength / 2, 1.0))
            
            if bright_pixels.sum() > 10:
                y_coords, x_coords = np.where(bright_pixels)
                y_min, y_max = y_coords.min(), y_coords.max()
                x_min, x_max = x_coords.min(), x_coords.max()
                bbox_area = (y_max - y_min + 1) * (x_max - x_min + 1)
                actual_area = bright_pixels.sum()
                compactness = actual_area / (bbox_area + 1e-8)
                
                if compactness > 0.6 and actual_area < (img_np.size * 0.05):
                    compactness_anomaly = compactness
                else:
                    compactness_anomaly = 0.0
            else:
                compactness_anomaly = 0.0
            
            contextual_score = (
                0.25 * brightness_anomaly +
                0.20 * texture_anomaly +
                0.35 * size_brightness_anomaly +
                0.20 * compactness_anomaly
            )
            
            scores.append(contextual_score)
        
        return np.array(scores)
    
    def compute_gradient_anomaly(self, image, recon):
        def compute_gradients(img_tensor):
            img_np = img_tensor.cpu().numpy()
            if len(img_np.shape) == 4:
                img_np = img_np[:, 0, :, :]
            
            gradients = []
            for i in range(img_np.shape[0]):
                gx = np.gradient(img_np[i], axis=0)
                gy = np.gradient(img_np[i], axis=1)
                grad_mag = np.sqrt(gx**2 + gy**2)
                gradients.append(grad_mag.mean())
            return np.array(gradients)
        
        orig_grads = compute_gradients(image)
        recon_grads = compute_gradients(recon)
        return np.abs(orig_grads - recon_grads)
    
    def compute_edge_regularity(self, image):
        scores = []
        
        for i in range(image.shape[0]):
            img_np = image[i, 0].cpu().numpy()
            
            gx = np.gradient(img_np, axis=0)
            gy = np.gradient(img_np, axis=1)
            edge_strength = np.sqrt(gx**2 + gy**2)
            
            threshold = np.percentile(edge_strength, 90)
            strong_edges = edge_strength > threshold
            
            if strong_edges.sum() < 10:
                scores.append(0.0)
                continue
            
            row_alignment = np.max(strong_edges.sum(axis=1)) / strong_edges.sum()
            col_alignment = np.max(strong_edges.sum(axis=0)) / strong_edges.sum()
            alignment_score = max(row_alignment, col_alignment)
            
            labeled_edges, num_features = label(strong_edges)
            
            if num_features == 0:
                scores.append(0.0)
                continue
            
            component_sizes = [np.sum(labeled_edges == j) for j in range(1, num_features + 1)]
            max_component_size = max(component_sizes) if component_sizes else 0
            continuity_score = max_component_size / strong_edges.sum()
            
            edge_angles = np.arctan2(gy[strong_edges], gx[strong_edges])
            
            if len(edge_angles) > 0:
                edge_angles_deg = np.degrees(edge_angles) % 180
                angle_hist, _ = np.histogram(edge_angles_deg, bins=18, range=(0, 180))
                peak_at_0_or_180 = angle_hist[0]
                peak_at_90 = angle_hist[9]
                angle_concentration = (peak_at_0_or_180 + peak_at_90) / len(edge_angles)
            else:
                angle_concentration = 0.0
            
            if max_component_size > 20:
                largest_component_mask = (labeled_edges == (np.argmax(component_sizes) + 1))
                comp_y, comp_x = np.where(largest_component_mask)
                
                if len(comp_x) > 3:
                    try:
                        p = np.polyfit(comp_x, comp_y, 1)
                        y_predicted = np.polyval(p, comp_x)
                        deviation = np.sqrt(np.mean((comp_y - y_predicted) ** 2))
                        straightness_score = 1.0 / (1.0 + deviation / img_np.shape[0])
                    except:
                        straightness_score = 0.0
                else:
                    straightness_score = 0.0
            else:
                straightness_score = 0.0
            
            regularity_score = (
                alignment_score ** 0.3 *
                continuity_score ** 0.4 *
                angle_concentration ** 0.2 *
                straightness_score ** 0.1
            )
            
            scores.append(regularity_score)
        
        return np.array(scores)
    
    def compute_combined_score(self, image):
        """Weighted combination optimized for circular anomalies"""
        mse_scores, recon = self.compute_reconstruction_error(image)
        density_scores = self.compute_latent_density(image)
        contextual_scores = self.compute_contextual_anomaly(image)
        gradient_scores = self.compute_gradient_anomaly(image, recon)
        regularity_scores = self.compute_edge_regularity(image)
        
        def normalize(scores):
            min_val, max_val = scores.min(), scores.max()
            if max_val - min_val < 1e-8:
                return scores * 0
            return (scores - min_val) / (max_val - min_val)
        
        mse_norm = normalize(mse_scores)
        density_norm = normalize(density_scores)
        contextual_norm = normalize(contextual_scores)
        gradient_norm = normalize(gradient_scores)
        regularity_norm = normalize(regularity_scores)
        
        combined = (
            0.30 * mse_norm +
            0.20 * density_norm +
            0.30 * contextual_norm +
            0.15 * gradient_norm +
            0.05 * regularity_norm
        )
        
        metric_dict = {
            'mse': mse_scores,
            'density': density_scores,
            'contextual': contextual_scores,
            'gradient': gradient_scores,
            'regularity': regularity_scores,
            'mse_norm': mse_norm,
            'density_norm': density_norm,
            'contextual_norm': contextual_norm,
            'gradient_norm': gradient_norm,
            'regularity_norm': regularity_norm
        }
        
        return combined, metric_dict


# ============================================
# 4. ADVANCED DETECTOR (Same confidence calc as Mk13)
# ============================================

class AdvancedAnomalyDetector:
    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device
        self.scorer = MultiMetricAnomalyScorer(model, device)
    
    def detect_anomalies(
        self,
        test_loader,
        chip_metadata: List[Dict],
        use_spatial_filter=False,
        base_percentile=92
    ) -> pd.DataFrame:
        
        logger.info("\n" + "="*70)
        logger.info("ADVANCED ANOMALY DETECTION PIPELINE (Mk14 Stable)")
        logger.info("="*70)
        
        logger.info("\n[1/2] Computing multi-metric anomaly scores...")
        all_combined_scores = []
        all_metrics = []
        
        with torch.no_grad():
            for images, paths in tqdm(test_loader, desc="Scoring"):
                images = images.to(self.device)
                combined, metrics = self.scorer.compute_combined_score(images)
                
                all_combined_scores.extend(combined)
                all_metrics.append(metrics)
        
        combined_scores = np.array(all_combined_scores)
        
        aggregated_metrics = {}
        for key in all_metrics[0].keys():
            aggregated_metrics[key] = np.concatenate([m[key] for m in all_metrics])
        
        results_df = pd.DataFrame(chip_metadata)
        results_df['anomaly_score'] = combined_scores
        
        for key, values in aggregated_metrics.items():
            results_df[f'metric_{key}'] = values
        
        logger.info(f"   Scores: range=[{combined_scores.min():.4f}, {combined_scores.max():.4f}], "
                   f"mean={combined_scores.mean():.4f}, std={combined_scores.std():.4f}")
        
        # Simple thresholding
        logger.info(f"\n[2/2] Applying {base_percentile}th percentile threshold...")
        threshold = np.percentile(combined_scores, base_percentile)
        results_df['is_anomaly_final'] = results_df['anomaly_score'] > threshold
        results_df['cluster_id'] = -1
        
        # Confidence calculation (same as Mk13)
        logger.info("\nComputing confidence scores...")
        results_df['confidence'] = self._compute_confidence(results_df)
        
        results_df = results_df.sort_values('confidence', ascending=False)
        
        logger.info("\n" + "="*70)
        logger.info("DETECTION SUMMARY")
        logger.info("="*70)
        logger.info(f"Total chips: {len(results_df)}")
        logger.info(f"Anomalies detected: {results_df['is_anomaly_final'].sum()}")
        logger.info(f"High confidence (>0.8): {(results_df['confidence'] > 0.8).sum()}")
        
        return results_df

    def _compute_confidence(self, df: pd.DataFrame) -> np.ndarray:
        """Mk13 optimized confidence for non-clustered data"""
        confidence = np.zeros(len(df))
        
        score_norm = (df['anomaly_score'] - df['anomaly_score'].min()) / \
                     (df['anomaly_score'].max() - df['anomaly_score'].min() + 1e-8)
        
        clustering_active = ('cluster_id' in df.columns and (df['cluster_id'] >= 0).any())
        
        if not clustering_active:
            logger.info("   Using non-clustering confidence weights (optimized for circular lander)")
            confidence += 0.50 * score_norm
            
            if 'metric_contextual_norm' in df.columns:
                confidence += 0.30 * df['metric_contextual_norm']
                logger.info("   Applied 30% weight to contextual metric")
            
            if 'metric_mse_norm' in df.columns:
                confidence += 0.20 * df['metric_mse_norm']
        
        return np.clip(confidence, 0, 1)


# ============================================
# 5. STABLE TRAINING WITH KL ANNEALING
# ============================================

def stable_vae_loss(recon_x, x, mu, logvar, beta=0.001, kl_weight=1.0):
    """Loss with KL annealing support"""
    MSE = F.mse_loss(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    # Apply KL weight for annealing
    return MSE + beta * kl_weight * KLD, MSE, KLD


class StableVAETrainer:
    def __init__(self, model, device='cpu', lr=1e-3, warmup_epochs=3):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.warmup_epochs = warmup_epochs
        self.best_loss = float('inf')
        self.patience_counter = 0
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=2
        )
        
    def get_kl_weight(self, epoch, total_epochs):
        """KL annealing schedule"""
        if epoch < self.warmup_epochs:
            return epoch / self.warmup_epochs
        return 1.0
    
    def train_epoch(self, loader, epoch, total_epochs):
        self.model.train()
        total_loss, total_mse, total_kld = 0, 0, 0
        
        kl_weight = self.get_kl_weight(epoch, total_epochs)
        
        for images, _ in tqdm(loader, desc=f"Epoch {epoch+1} (KL weight={kl_weight:.2f})"):
            images = images.to(self.device)
            
            self.optimizer.zero_grad()
            recon, mu, logvar = self.model(images)
            
            loss, mse, kld = stable_vae_loss(recon, images, mu, logvar, kl_weight=kl_weight)
            
            # Check for NaN
            if torch.isnan(loss):
                logger.error(f"NaN detected at epoch {epoch+1}! Stopping training.")
                return None, None, None
            
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            total_mse += mse.item()
            total_kld += kld.item()
        
        avg_loss = total_loss / len(loader.dataset)
        avg_mse = total_mse / len(loader.dataset)
        avg_kld = total_kld / len(loader.dataset)
        
        # Update learning rate
        self.scheduler.step(avg_loss)
        
        return avg_loss, avg_mse, avg_kld
    
    def save_checkpoint(self, filepath, epoch, loss):
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
        }, filepath)
        logger.info(f"Checkpoint saved: {filepath}")


# ============================================
# 6. VISUALIZATION
# ============================================

class HighResVisualizer:
    def __init__(self, output_dir='./results'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        plt.rcParams['figure.dpi'] = 300
        plt.rcParams['savefig.dpi'] = 300

    def plot_top_anomalies_with_confidence(self, results_df, n_samples=12):
        top_anomalies = results_df.nlargest(n_samples, 'confidence')
        
        rows = int(np.ceil(n_samples / 4))
        fig, axes = plt.subplots(rows, 4, figsize=(20, 5 * rows), 
                                gridspec_kw={'hspace': 0.4, 'wspace': 0.3})
        axes = axes.flatten()
        
        for idx, (_, row) in enumerate(top_anomalies.iterrows()):
            if idx >= len(axes): break
            
            with rasterio.open(row['chip_path']) as src:
                chip = src.read(1)
            
            chip = (chip - chip.min()) / (chip.max() - chip.min() + 1e-8)
            chip_upsampled = zoom(chip, 2, order=1)
            axes[idx].imshow(chip_upsampled, cmap='gray', interpolation='bilinear')
            
            conf = row['confidence']
            if conf > 0.8:
                color = '#e53935'
            elif conf > 0.6:
                color = '#fb8c00'
            else:
                color = '#fdd835'
            
            title = (f"Confidence: {conf:.3f}\n"
                    f"Score: {row['anomaly_score']:.3f}")
            
            axes[idx].set_title(title, fontsize=9, color=color, fontweight='bold')
            axes[idx].axis('off')
            
            for spine in axes[idx].spines.values():
                spine.set_edgecolor(color)
                spine.set_linewidth(4)
                spine.set_visible(True)
        
        for i in range(len(top_anomalies), len(axes)):
            axes[i].axis('off')
            
        plt.suptitle('Top Anomalies by Confidence (Mk14 Stable)', fontsize=18, fontweight='bold')
        plt.savefig(self.output_dir / 'top_anomalies_mk14.png', dpi=300, bbox_inches='tight')
        logger.info(f"Saved: {self.output_dir / 'top_anomalies_mk14.png'}")
        plt.close()


# ============================================
# 7. MAIN PIPELINE
# ============================================

def main():
    logger.info("="*70)
    logger.info("XENARCH Mk14: STABLE TRAINING FOR LANDER DETECTION")
    logger.info("="*70)
    
    config = {
        'chip_size': 256,
        'latent_dim': 56,
        'batch_size': 4,
        'num_epochs': 15,
        'learning_rate': 0.0005,  # Lower LR
        'warmup_epochs': 3,
        'base_percentile': 95,
        'use_spatial_filter': False,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    logger.info(f"Configuration: {json.dumps(config, indent=2)}")
    
    data_root = Path("data")
    results_dir = Path("results") / "mk14_stable"
    results_dir.mkdir(exist_ok=True, parents=True)
    
    # 1. Chip Extraction
    logger.info("\n[STEP 1/5] Extracting chips...")
    extractor = ChipExtractor(chip_size=config['chip_size'])
    
    train_chips_dir = data_root / "processed" / "train_chips_256"
    test_chips_dir = data_root / "processed" / "test_chips_256"
    
    all_train_chips = []
    train_imgs = list(Path("training data").glob("*"))
    for img in train_imgs:
        if img.suffix.lower() in ['.png', '.jpg', '.tif', '.tiff']:
            chips = extractor.extract_grid(str(img), str(train_chips_dir))
            all_train_chips.extend(chips)
    
    all_test_chips = []
    test_imgs = list(Path("Test data").glob("*"))
    for img in test_imgs:
        if img.suffix.lower() in ['.png', '.jpg', '.tif', '.tiff']:
            chips = extractor.extract_grid(str(img), str(test_chips_dir), max_size_mb=10.0)
            all_test_chips.extend(chips)
            break
    
    if not all_train_chips or not all_test_chips:
        logger.error("Missing training or test data!")
        return
        
    train_df = pd.DataFrame(all_train_chips)
    test_df = pd.DataFrame(all_test_chips)
    
    logger.info(f"Training chips: {len(train_df)}")
    logger.info(f"Test chips: {len(test_df)}")
    
    # 2. Create datasets
    logger.info("\n[STEP 2/5] Creating datasets...")
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip()
    ])
    
    train_ds = LunarDataset(train_df, transform=train_transform)
    test_ds = LunarDataset(test_df)
    
    train_loader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=config['batch_size'], shuffle=False, num_workers=0)
    
    # 3. Train stable model
    logger.info(f"\n[STEP 3/5] Training Stable VAE on {config['device']}...")
    model = StableConvolutionalVAE(
        latent_dim=config['latent_dim'],
        input_size=config['chip_size']
    )
    
    trainer = StableVAETrainer(
        model, 
        device=config['device'], 
        lr=config['learning_rate'],
        warmup_epochs=config['warmup_epochs']
    )
    
    best_loss = float('inf')
    for epoch in range(config['num_epochs']):
        loss, mse, kld = trainer.train_epoch(train_loader, epoch, config['num_epochs'])
        
        if loss is None:
            logger.error("Training stopped due to NaN")
            break
            
        logger.info(f"Epoch {epoch+1}/{config['num_epochs']} - Loss: {loss:.2f}, MSE: {mse:.2f}, KLD: {kld:.2f}")
        
        if loss < best_loss:
            best_loss = loss
            model_path = data_root / "models" / "xenarch_mk14_best.pth"
            model_path.parent.mkdir(exist_ok=True, parents=True)
            trainer.save_checkpoint(model_path, epoch, loss)
    
    # 4. Detection
    logger.info("\n[STEP 4/5] Running anomaly detection...")
    detector = AdvancedAnomalyDetector(model, device=config['device'])
    
    results_df = detector.detect_anomalies(
        test_loader=test_loader,
        chip_metadata=all_test_chips,
        base_percentile=config['base_percentile']
    )
    
    results_df.to_csv(results_dir / "xenarch_mk14_results.csv", index=False)
    logger.info(f"Results saved: {results_dir / 'xenarch_mk14_results.csv'}")
    
    # Print top detections
    logger.info("\n" + "="*70)
    logger.info("TOP 10 DETECTIONS")
    logger.info("="*70)
    
    top_10 = results_df.nlargest(min(10, len(results_df)), 'confidence')
    for rank, (idx, row) in enumerate(top_10.iterrows(), 1):
        logger.info(f"\nRank {rank}: {Path(row['chip_path']).name}")
        logger.info(f"  Confidence:  {row['confidence']:.4f}")
        logger.info(f"  Total Score: {row['anomaly_score']:.4f}")
        logger.info(f"  Contextual:  {row.get('metric_contextual', 0):.4f} (norm: {row.get('metric_contextual_norm', 0):.4f})")
        logger.info(f"  MSE:         {row.get('metric_mse', 0):.4f} (norm: {row.get('metric_mse_norm', 0):.4f})")
    
    # 5. Visualization
    logger.info("\n[STEP 5/5] Generating visualizations...")
    viz = HighResVisualizer(output_dir=results_dir)
    viz.plot_top_anomalies_with_confidence(results_df, n_samples=min(12, len(results_df)))
    
    logger.info("\n" + "="*70)
    logger.info("XENARCH Mk14 COMPLETE!")
    logger.info("="*70)
    logger.info(f"Results: {results_dir}")
    logger.info(f"Anomalies: {results_df['is_anomaly_final'].sum()}")
    logger.info(f"High confidence: {(results_df['confidence'] > 0.8).sum()}")
    
if __name__ == "__main__":
    main()