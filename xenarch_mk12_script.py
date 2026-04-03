
"""
Technosignature Detection: Xenarch Mk7 - Integrated Advanced Pipeline
======================================================================

Combines:
- VAE architecture from Mk5
- High-resolution visualization from Mk6
- Advanced false positive reduction
- Multi-metric anomaly scoring (MSE, density, contextual, gradient, regularity)
- Spatial consistency filtering (optional - disable for small datasets)
- Adaptive thresholding

IMPORTANT: Spatial Filtering Configuration
------------------------------------------
The spatial filter requires anomalies to appear in multiple adjacent chips.
This is excellent for large datasets but can filter out real anomalies in small datasets.

When to use spatial filtering:
✓ Large datasets (100+ test chips)
✓ Looking for large structures (rover tracks, landing sites with equipment)
✓ High false positive rate from natural features

When to DISABLE spatial filtering (set use_spatial_filter=False):
✓ Small datasets (<20 test chips) ← YOUR CASE
✓ Looking for small/isolated features
✓ Features only appear in 1-2 chips
✓ Already using strong multi-metric scoring

Current configuration: use_spatial_filter=False (optimal for small datasets)

Usage:
    python xenarch_mk7_integrated.py
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
from scipy.ndimage import zoom, gaussian_filter, generic_filter
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
logger.add("logs/xenarch_mk7_{time}.log", rotation="10 MB")


# ============================================
# 1. VAE MODEL
# ============================================

class ConvolutionalVAE(nn.Module):
    """Variational Autoencoder with configurable latent dimension"""
    
    def __init__(self, latent_dim=64, input_size=256):
        super(ConvolutionalVAE, self).__init__()
        self.latent_dim = latent_dim
        self.input_size = input_size
        
        # Calculate final feature map size
        # 256 -> 128 -> 64 -> 32 -> 16 (4 downsamples)
        final_size = input_size // 16
        
        # Encoder
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        
        self.fc_mu = nn.Linear(256 * final_size * final_size, latent_dim)
        self.fc_logvar = nn.Linear(256 * final_size * final_size, latent_dim)
        
        # Decoder
        self.decoder_input = nn.Linear(latent_dim, 256 * final_size * final_size)
        self.final_size = final_size
        
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )
        
        logger.info(f"VAE initialized: latent_dim={latent_dim}, input_size={input_size}")
    
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
        h = h.view(-1, 256, self.final_size, self.final_size)
        return self.decoder_conv(h)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


# ============================================
# 2. DATA PROCESSING
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
        
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        img = img[np.newaxis, :]
        
        image_tensor = torch.from_numpy(img)
        if self.transform:
            image_tensor = self.transform(image_tensor)
            
        return image_tensor, path


# ============================================
# 3. MULTI-METRIC ANOMALY SCORER
# ============================================

class MultiMetricAnomalyScorer:
    """Combines multiple metrics to reduce false positives"""
    
    def __init__(self, model, device='cpu'):
        self.model = model.to(device)
        self.device = device
        self.model.eval()
    
    def compute_reconstruction_error(self, image):
        """MSE reconstruction error"""
        with torch.no_grad():
            recon, mu, logvar = self.model(image)
            mse = torch.mean((image - recon) ** 2, dim=[1, 2, 3])
            return mse.cpu().numpy(), recon
    
    def compute_latent_density(self, image):
        """Probability density in latent space (VAE specific)"""
        with torch.no_grad():
            mu, logvar = self.model.encode(image)
            distances = torch.sqrt(torch.sum(mu ** 2, dim=1))
            density_scores = torch.exp(-distances / self.model.latent_dim)
            return 1.0 - density_scores.cpu().numpy()
    
    def compute_contextual_anomaly(self, image):
        """
        NEW METRIC: Detect anomalies based on local context, not just geometry.
        
        Key insight: Natural features fit their surroundings. Artificial features
        (even circular ones like the lander) have unusual relationships with context:
        - Size anomaly: Too bright/dark for their size
        - Texture anomaly: Different texture than surroundings
        - Isolation anomaly: Suspiciously alone in otherwise uniform areas
        
        This catches both angular AND circular artificial features.
        """
        scores = []
        
        for i in range(image.shape[0]):
            img_np = image[i, 0].cpu().numpy()
            
            # === METRIC 1: Brightness Outlier Detection ===
            # Find bright spots and check if they're unusually bright
            mean_brightness = np.mean(img_np)
            std_brightness = np.std(img_np)
            
            # Threshold for "very bright" pixels
            bright_threshold = mean_brightness + 2 * std_brightness
            bright_pixels = img_np > bright_threshold
            
            if bright_pixels.sum() < 5:
                brightness_anomaly = 0.0
            else:
                # How much brighter than expected?
                bright_region_mean = np.mean(img_np[bright_pixels])
                brightness_anomaly = min((bright_region_mean - mean_brightness) / (std_brightness + 1e-8) / 3, 1.0)
            
            # === METRIC 2: Local Texture Contrast ===
            # Artificial objects have different texture than surroundings
            from scipy.ndimage import generic_filter
            
            # Local standard deviation (texture)
            try:
                local_std = generic_filter(img_np, np.std, size=9)
                
                # Find regions with unusual texture
                texture_mean = np.mean(local_std)
                texture_std = np.std(local_std)
                
                texture_outliers = np.abs(local_std - texture_mean) > 2 * texture_std
                texture_anomaly = texture_outliers.sum() / img_np.size
            except:
                texture_anomaly = 0.0
            
            # === METRIC 3: Size-Brightness Relationship ===
            # Natural craters: bigger = darker (shadows)
            # Lander: small but VERY bright (unusual!)
            
            # Find connected bright regions
            from scipy.ndimage import label
            bright_regions, num_regions = label(bright_pixels)
            
            size_brightness_anomaly = 0.0
            if num_regions > 0:
                for region_id in range(1, num_regions + 1):
                    region_mask = bright_regions == region_id
                    region_size = region_mask.sum()
                    region_brightness = np.mean(img_np[region_mask])
                    
                    # Small but very bright = suspicious
                    # Expected: large features are brighter
                    expected_brightness = mean_brightness + (region_size / img_np.size) * std_brightness
                    
                    if region_brightness > expected_brightness:
                        anomaly_strength = (region_brightness - expected_brightness) / (std_brightness + 1e-8)
                        size_brightness_anomaly = max(size_brightness_anomaly, min(anomaly_strength / 2, 1.0))
            
            # === METRIC 4: Compactness Score ===
            # Artificial objects are often compact and isolated
            # Natural features blend into surroundings
            
            if bright_pixels.sum() > 10:
                # Compute compactness of bright regions
                y_coords, x_coords = np.where(bright_pixels)
                
                # Bounding box area
                y_min, y_max = y_coords.min(), y_coords.max()
                x_min, x_max = x_coords.min(), x_coords.max()
                bbox_area = (y_max - y_min + 1) * (x_max - x_min + 1)
                
                # Actual bright pixel count
                actual_area = bright_pixels.sum()
                
                # Compactness: how much of bounding box is filled
                compactness = actual_area / (bbox_area + 1e-8)
                
                # High compactness (>0.6) + small size = potentially artificial
                if compactness > 0.6 and actual_area < (img_np.size * 0.05):  # Less than 5% of image
                    compactness_anomaly = compactness
                else:
                    compactness_anomaly = 0.0
            else:
                compactness_anomaly = 0.0
            
            # === COMBINE CONTEXTUAL METRICS ===
            contextual_score = (
                0.25 * brightness_anomaly +
                0.20 * texture_anomaly +
                0.35 * size_brightness_anomaly +  # Most important!
                0.20 * compactness_anomaly
            )
            
            scores.append(contextual_score)
        
        return np.array(scores)
    def compute_gradient_anomaly(self, image, recon):
        """Gradient pattern differences"""
        def compute_gradients(img_tensor):
            img_np = img_tensor.cpu().numpy()
            if len(img_np.shape) == 4:  # [batch, channel, H, W]
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
        """
        IMPROVED: Detect geometric/artificial edge patterns while avoiding boulder false positives.
        
        Key differences between boulders and artificial structures:
        - Boulders: Disconnected edges, random orientations, irregular spacing
        - Lander: Continuous straight edges, 90° angles, regular geometric shapes
        """
        scores = []
        
        for i in range(image.shape[0]):
            img_np = image[i, 0].cpu().numpy()
            
            # Compute gradients
            gx = np.gradient(img_np, axis=0)
            gy = np.gradient(img_np, axis=1)
            edge_strength = np.sqrt(gx**2 + gy**2)
            
            threshold = np.percentile(edge_strength, 90)
            strong_edges = edge_strength > threshold
            
            if strong_edges.sum() < 10:
                scores.append(0.0)
                continue
            
            # === METRIC 1: Basic Alignment ===
            row_alignment = np.max(strong_edges.sum(axis=1)) / strong_edges.sum()
            col_alignment = np.max(strong_edges.sum(axis=0)) / strong_edges.sum()
            alignment_score = max(row_alignment, col_alignment)
            
            # === METRIC 2: Edge Continuity ===
            # Boulders have fragmented edges, lander has long continuous edges
            from scipy.ndimage import label
            labeled_edges, num_features = label(strong_edges)
            
            if num_features == 0:
                scores.append(0.0)
                continue
            
            # Find longest continuous edge segment
            component_sizes = [np.sum(labeled_edges == j) for j in range(1, num_features + 1)]
            max_component_size = max(component_sizes) if component_sizes else 0
            continuity_score = max_component_size / strong_edges.sum()
            
            # Boulders: many small components (low continuity)
            # Lander: few large components (high continuity)
            
            # === METRIC 3: Angle Concentration ===
            # Straight edges cluster at specific angles (0°, 90°, 180°, 270°)
            edge_y_coords, edge_x_coords = np.where(strong_edges)
            edge_angles = np.arctan2(gy[strong_edges], gx[strong_edges])
            
            if len(edge_angles) > 0:
                # Convert to degrees and normalize to [0, 180)
                edge_angles_deg = np.degrees(edge_angles) % 180
                
                # Check for concentration at orthogonal angles (0°, 90°)
                # Bin into 18 bins (10° each)
                angle_hist, _ = np.histogram(edge_angles_deg, bins=18, range=(0, 180))
                
                # Artificial structures have peaks at 0° and 90°
                peak_at_0_or_180 = angle_hist[0]  # 0-10° and 170-180°
                peak_at_90 = angle_hist[9]  # 85-95°
                
                # Normalized concentration
                angle_concentration = (peak_at_0_or_180 + peak_at_90) / len(edge_angles)
            else:
                angle_concentration = 0.0
            
            # === METRIC 4: Edge Straightness ===
            # Measure how straight the longest edge component is
            if max_component_size > 20:  # Only if we have enough pixels
                # Get pixels in largest component
                largest_component_mask = (labeled_edges == (np.argmax(component_sizes) + 1))
                comp_y, comp_x = np.where(largest_component_mask)
                
                if len(comp_x) > 3:
                    # Fit a line and measure deviation
                    from numpy.polynomial import Polynomial
                    try:
                        # Fit line: y = mx + b
                        p = Polynomial.fit(comp_x, comp_y, 1)
                        y_predicted = p(comp_x)
                        deviation = np.sqrt(np.mean((comp_y - y_predicted) ** 2))
                        
                        # Normalize by image size
                        straightness_score = 1.0 / (1.0 + deviation / img_np.shape[0])
                    except:
                        straightness_score = 0.0
                else:
                    straightness_score = 0.0
            else:
                straightness_score = 0.0
            
            # === COMBINE METRICS ===
            # All four must be high for true artificial structures
            # Use multiplicative combination so boulders (low on some metrics) score low
            
            # Apply different weighting strategy
            regularity_score = (
                alignment_score ** 0.3 *           # 30% - basic alignment (boulders can have this)
                continuity_score ** 0.4 *          # 40% - CRITICAL: continuous edges
                angle_concentration ** 0.2 *       # 20% - orthogonal angles
                straightness_score ** 0.1          # 10% - straight line fit
            )
            
            scores.append(regularity_score)
        
        return np.array(scores)
    
    def compute_combined_score(self, image):
        """Weighted combination of all metrics - now includes contextual analysis"""
        mse_scores, recon = self.compute_reconstruction_error(image)
        density_scores = self.compute_latent_density(image)
        contextual_scores = self.compute_contextual_anomaly(image)  # NEW!
        gradient_scores = self.compute_gradient_anomaly(image, recon)
        regularity_scores = self.compute_edge_regularity(image)
        
        def normalize(scores):
            min_val, max_val = scores.min(), scores.max()
            if max_val - min_val < 1e-8:
                return scores * 0
            return (scores - min_val) / (max_val - min_val)
        
        mse_norm = normalize(mse_scores)
        density_norm = normalize(density_scores)
        contextual_norm = normalize(contextual_scores)  # NEW!
        gradient_norm = normalize(gradient_scores)
        regularity_norm = normalize(regularity_scores)
        
        # Weighted combination - UPDATED with contextual metric
        # This now catches BOTH angular AND circular artificial features
        combined = (
            0.30 * mse_norm +           # Reconstruction error
            0.20 * density_norm +       # Latent space position
            0.25 * contextual_norm +    # NEW! Context/size/brightness (catches circular lander)
            0.15 * gradient_norm +      # Gradient mismatch
            0.10 * regularity_norm      # Geometric edges (catches angular features)
        )
        
        metric_dict = {
            'mse': mse_scores,
            'density': density_scores,
            'contextual': contextual_scores,  # NEW!
            'gradient': gradient_scores,
            'regularity': regularity_scores,
            'mse_norm': mse_norm,
            'density_norm': density_norm,
            'contextual_norm': contextual_norm,  # NEW!
            'gradient_norm': gradient_norm,
            'regularity_norm': regularity_norm
        }
        
        return combined, metric_dict


# ============================================
# 4. SPATIAL CONSISTENCY FILTER
# ============================================

class SpatialConsistencyFilter:
    """Filters isolated false positives using spatial clustering"""
    
    def __init__(self, chip_metadata: List[Dict], anomaly_scores: np.ndarray):
        self.metadata = chip_metadata
        self.scores = anomaly_scores
        self.df = pd.DataFrame(chip_metadata)
        self.df['anomaly_score'] = anomaly_scores
    
    def apply_spatial_filtering(self, threshold_percentile=92, min_cluster_size=3):
        threshold = np.percentile(self.scores, threshold_percentile)
        self.df['is_anomaly_initial'] = self.df['anomaly_score'] > threshold
        
        anomalous_chips = self.df[self.df['is_anomaly_initial']].copy()
        
        if len(anomalous_chips) < min_cluster_size:
            self.df['is_anomaly_filtered'] = False
            self.df['cluster_id'] = -1
            logger.info("No clusters found - not enough anomalous chips")
            return self.df
        
        coords = anomalous_chips[['center_x', 'center_y']].values
        
        # Estimate chip spacing
        typical_width = self.df['bbox'].apply(lambda b: b[2] - b[0]).median()
        eps = typical_width * 1.5
        
        clustering = DBSCAN(eps=eps, min_samples=min_cluster_size).fit(coords)
        anomalous_chips['cluster_id'] = clustering.labels_
        
        valid_clusters = anomalous_chips[anomalous_chips['cluster_id'] >= 0]
        
        self.df['is_anomaly_filtered'] = False
        self.df['cluster_id'] = -1
        
        self.df.loc[valid_clusters.index, 'is_anomaly_filtered'] = True
        self.df.loc[valid_clusters.index, 'cluster_id'] = valid_clusters['cluster_id']
        
        # Boost clustered anomaly scores based on cluster size
        if 'cluster_id' in self.df.columns:
            for cluster_id in self.df[self.df['cluster_id'] >= 0]['cluster_id'].unique():
                cluster_mask = self.df['cluster_id'] == cluster_id
                cluster_size = cluster_mask.sum()
                
                # Larger clusters get bigger boost (lander spans more chips than boulders)
                # Small cluster (2-3 chips) = 1.1x boost
                # Medium cluster (4-6 chips) = 1.3x boost  
                # Large cluster (7+ chips) = 1.5x boost
                if cluster_size >= 7:
                    boost_factor = 1.5
                elif cluster_size >= 4:
                    boost_factor = 1.3
                else:
                    boost_factor = 1.1
                
                self.df.loc[cluster_mask, 'anomaly_score'] *= boost_factor
                
                logger.info(f"Cluster {cluster_id}: {cluster_size} chips, boost={boost_factor:.1f}x")
        
        num_filtered = self.df['is_anomaly_initial'].sum() - self.df['is_anomaly_filtered'].sum()
        num_clusters = self.df[self.df['cluster_id'] >= 0]['cluster_id'].nunique()
        
        logger.info(f"Spatial filtering: removed {num_filtered} isolated false positives")
        logger.info(f"Retained {self.df['is_anomaly_filtered'].sum()} anomalies in {num_clusters} clusters")
        
        return self.df


# ============================================
# 5. ADAPTIVE THRESHOLDING
# ============================================

class AdaptiveThreshold:
    """Context-aware thresholding based on local statistics"""
    
    def __init__(self, data: Union[List[Dict], pd.DataFrame], anomaly_scores: np.ndarray):
        self.scores = anomaly_scores
        if isinstance(data, pd.DataFrame):
            self.df = data.copy()
        else:
            self.df = pd.DataFrame(data)
        self.df['anomaly_score'] = anomaly_scores
    
    def compute_local_statistics(self, chip_path: str) -> Dict:
        with rasterio.open(chip_path) as src:
            img = src.read(1)
        
        return {
            'mean_intensity': np.mean(img),
            'std_intensity': np.std(img),
            'edge_density': self._compute_edge_density(img),
            'texture_complexity': self._compute_texture(img)
        }
    
    def _compute_edge_density(self, img: np.ndarray) -> float:
        gx = np.gradient(img.astype(float), axis=0)
        gy = np.gradient(img.astype(float), axis=1)
        edge_strength = np.sqrt(gx**2 + gy**2)
        threshold = np.percentile(edge_strength, 90)
        return (edge_strength > threshold).sum() / img.size
    
    def _compute_texture(self, img: np.ndarray) -> float:
        try:
            local_var = generic_filter(img.astype(float), np.var, size=5)
            return np.mean(local_var)
        except:
            return np.var(img)
    
    def apply_adaptive_threshold(self, base_percentile=92):
        logger.info("Computing local statistics for adaptive thresholding...")
        stats_list = []
        for _, row in tqdm(self.df.iterrows(), total=len(self.df), desc="Computing stats"):
            stats = self.compute_local_statistics(row['chip_path'])
            stats_list.append(stats)
        
        stats_df = pd.DataFrame(stats_list)
        self.df = pd.concat([self.df.reset_index(drop=True), stats_df], axis=1)
        
        # Normalize
        for col in ['mean_intensity', 'std_intensity', 'edge_density', 'texture_complexity']:
            mean_val = self.df[col].mean()
            std_val = self.df[col].std()
            self.df[f'{col}_norm'] = (self.df[col] - mean_val) / (std_val + 1e-8)
        
        base_threshold = np.percentile(self.scores, base_percentile)
        
        complexity_score = (
            self.df['edge_density_norm'] + 
            self.df['texture_complexity_norm']
        ) / 2
        
        threshold_adjustment = 1.0 + 0.2 * np.tanh(complexity_score)
        self.df['adaptive_threshold'] = base_threshold * threshold_adjustment
        
        self.df['is_anomaly_adaptive'] = (
            self.df['anomaly_score'] > self.df['adaptive_threshold']
        )
        
        logger.info(f"Adaptive thresholding identified {self.df['is_anomaly_adaptive'].sum()} anomalies")
        
        return self.df


# ============================================
# 6. INTEGRATED DETECTOR
# ============================================

class AdvancedAnomalyDetector:
    """Integrated detection pipeline"""
    
    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device
        self.scorer = MultiMetricAnomalyScorer(model, device)
    
    def detect_anomalies(
        self,
        test_loader,
        chip_metadata: List[Dict],
        use_spatial_filter=True,
        use_adaptive_threshold=True,
        base_percentile=92
    ) -> pd.DataFrame:
        
        logger.info("\n" + "="*70)
        logger.info("ADVANCED ANOMALY DETECTION PIPELINE")
        logger.info("="*70)
        
        # Determine if spatial filtering is appropriate
        num_chips = len(chip_metadata)
        
        # Auto-disable spatial filter for small datasets
        if num_chips < 20 and use_spatial_filter:
            logger.warning(f"Only {num_chips} test chips - spatial filtering may be too strict")
            logger.warning("Consider setting use_spatial_filter=False for small datasets")
            auto_disable_spatial = True
        else:
            auto_disable_spatial = False
        
        # Step 1: Multi-metric scoring
        logger.info("\n[1/4] Computing multi-metric anomaly scores...")
        all_combined_scores = []
        all_metrics = []
        all_paths = []
        
        with torch.no_grad():
            for images, paths in tqdm(test_loader, desc="Scoring"):
                images = images.to(self.device)
                combined, metrics = self.scorer.compute_combined_score(images)
                
                all_combined_scores.extend(combined)
                all_metrics.append(metrics)
                all_paths.extend(paths)
        
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
        
        # Step 2: Spatial filtering
        if use_spatial_filter and not auto_disable_spatial:
            logger.info("\n[2/4] Applying spatial consistency filter...")
            spatial_filter = SpatialConsistencyFilter(chip_metadata, combined_scores)
            results_df = spatial_filter.apply_spatial_filtering(
                threshold_percentile=base_percentile,
                min_cluster_size=2
            )
        else:
            if auto_disable_spatial:
                logger.info("\n[2/4] Spatial filter auto-disabled (small dataset)")
            else:
                logger.info("\n[2/4] Spatial filter disabled by configuration")
            
            threshold = np.percentile(combined_scores, base_percentile)
            results_df['is_anomaly_filtered'] = results_df['anomaly_score'] > threshold
            results_df['cluster_id'] = -1  # No clustering
            
            num_detected = results_df['is_anomaly_filtered'].sum()
            logger.info(f"Detected {num_detected} anomalies above {base_percentile}th percentile threshold")
        
        if use_adaptive_threshold:
            logger.info("\n[3/4] Applying adaptive thresholding...")
            adaptive = AdaptiveThreshold(results_df, results_df['anomaly_score'].values)
            results_df = adaptive.apply_adaptive_threshold(base_percentile=base_percentile)
            
            results_df['is_anomaly_final'] = (
                results_df['is_anomaly_filtered'] & 
                results_df['is_anomaly_adaptive']
            )
        else:
            logger.info("\n[3/4] Skipping adaptive threshold")
            results_df['is_anomaly_final'] = results_df['is_anomaly_filtered']
        
        # Step 4: Confidence scoring
        logger.info("\n[4/4] Computing confidence scores...")
        results_df['confidence'] = self._compute_confidence(results_df)
        
        results_df = results_df.sort_values('confidence', ascending=False)
        
        logger.info("\n" + "="*70)
        logger.info("DETECTION SUMMARY")
        logger.info("="*70)
        logger.info(f"Total chips: {len(results_df)}")
        logger.info(f"Anomalies detected: {results_df['is_anomaly_final'].sum()}")
        logger.info(f"High confidence (>0.8): {(results_df['confidence'] > 0.8).sum()}")
        
        if 'cluster_id' in results_df.columns:
            n_clusters = results_df[results_df['cluster_id'] >= 0]['cluster_id'].nunique()
            logger.info(f"Spatial clusters: {n_clusters}")
        
        return results_df
    
    def _compute_confidence(self, df: pd.DataFrame) -> np.ndarray:
        confidence = np.zeros(len(df))
        
        score_norm = (df['anomaly_score'] - df['anomaly_score'].min()) / \
                     (df['anomaly_score'].max() - df['anomaly_score'].min() + 1e-8)
        
        confidence += 0.4 * score_norm
        
        if 'cluster_id' in df.columns:
            in_cluster = (df['cluster_id'] >= 0).astype(float)
            confidence += 0.3 * in_cluster
        
        if 'metric_regularity_norm' in df.columns:
            confidence += 0.2 * df['metric_regularity_norm']
        
        if 'metric_mse_norm' in df.columns and 'metric_gradient_norm' in df.columns:
            metric_agreement = (df['metric_mse_norm'] + df['metric_gradient_norm']) / 2
            confidence += 0.1 * metric_agreement
        
        return np.clip(confidence, 0, 1)


# ============================================
# 7. HIGH-RES VISUALIZATION
# ============================================

class HighResVisualizer:
    def __init__(self, output_dir='./results'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        plt.rcParams['figure.dpi'] = 300
        plt.rcParams['savefig.dpi'] = 300
    
    def plot_reconstruction_examples(self, model, dataset, n_samples=6, device='cpu', upsample_factor=2):
        model.eval()
        fig, axes = plt.subplots(3, n_samples, figsize=(n_samples * 4, 12),
                                gridspec_kw={'hspace': 0.3, 'wspace': 0.2})
        
        indices = np.random.choice(len(dataset), min(n_samples, len(dataset)), replace=False)
        
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
                
                axes[0, i].imshow(original_np, cmap='gray', interpolation='bilinear', vmin=0, vmax=1)
                axes[0, i].axis('off')
                if i == 0: axes[0, i].set_ylabel('Original', fontsize=14, fontweight='bold')
                
                axes[1, i].imshow(recon_np, cmap='gray', interpolation='bilinear', vmin=0, vmax=1)
                axes[1, i].axis('off')
                if i == 0: axes[1, i].set_ylabel('Reconstructed', fontsize=14, fontweight='bold')
                
                im = axes[2, i].imshow(error_map, cmap='hot', interpolation='bilinear')
                axes[2, i].axis('off')
                if i == 0: axes[2, i].set_ylabel('Error', fontsize=14, fontweight='bold')
        
        plt.suptitle('High-Resolution Reconstructions (Mk7)', fontsize=16, fontweight='bold')
        plt.savefig(self.output_dir / 'reconstructions_mk7.png', dpi=300, bbox_inches='tight')
        logger.info(f"Saved reconstruction examples")
        plt.close()
    
    def plot_top_anomalies_with_confidence(self, results_df, n_samples=12):
        """Plot top anomalies sorted by confidence"""
        top_anomalies = results_df.nlargest(n_samples, 'confidence')
        
        fig, axes = plt.subplots(3, 4, figsize=(20, 15), 
                                gridspec_kw={'hspace': 0.4, 'wspace': 0.3})
        axes = axes.flatten()
        
        for idx, (_, row) in enumerate(top_anomalies.iterrows()):
            if idx >= len(axes): break
            
            with rasterio.open(row['chip_path']) as src:
                chip = src.read(1)
            
            chip_upsampled = zoom(chip, 2, order=1)
            axes[idx].imshow(chip_upsampled, cmap='gray', interpolation='bilinear')
            
            # Color by confidence
            conf = row['confidence']
            if conf > 0.8:
                color = '#e53935'  # High confidence - red
            elif conf > 0.6:
                color = '#fb8c00'  # Medium - orange
            else:
                color = '#fdd835'  # Lower - yellow
            
            title = (f"Confidence: {conf:.3f}\n"
                    f"Score: {row['anomaly_score']:.3f}\n"
                    f"Cluster: {row.get('cluster_id', -1)}")
            
            axes[idx].set_title(title, fontsize=9, color=color, fontweight='bold')
            axes[idx].axis('off')
            
            for spine in axes[idx].spines.values():
                spine.set_edgecolor(color)
                spine.set_linewidth(4)
                spine.set_visible(True)
        
        plt.suptitle('Top Anomalies by Confidence (Mk7)', fontsize=18, fontweight='bold')
        plt.savefig(self.output_dir / 'top_anomalies_mk7.png', dpi=300, bbox_inches='tight')
        logger.info(f"Saved top anomalies visualization")
        plt.close()
    
    def plot_detection_analysis(self, results_df):
        """Comprehensive analysis plots"""
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. Score distribution
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.hist(results_df['anomaly_score'], bins=50, alpha=0.7, color='blue')
        ax1.axvline(results_df['anomaly_score'].quantile(0.92), color='r', linestyle='--', label='Threshold')
        ax1.set_xlabel('Anomaly Score')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Score Distribution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. Confidence vs Score
        ax2 = fig.add_subplot(gs[0, 1])
        scatter = ax2.scatter(results_df['anomaly_score'], results_df['confidence'], 
                            c=results_df['is_anomaly_final'], cmap='RdYlGn', alpha=0.6)
        ax2.set_xlabel('Anomaly Score')
        ax2.set_ylabel('Confidence')
        ax2.set_title('Confidence vs Score')
        plt.colorbar(scatter, ax=ax2, label='Is Anomaly')
        ax2.grid(True, alpha=0.3)
        
        # 3. Metric correlation
        ax3 = fig.add_subplot(gs[0, 2])
        metrics = ['metric_mse_norm', 'metric_density_norm', 'metric_contextual_norm', 'metric_gradient_norm', 'metric_regularity_norm']
        if all(m in results_df.columns for m in metrics):
            corr_data = results_df[metrics].corr()
            sns.heatmap(corr_data, annot=True, fmt='.2f', cmap='coolwarm', ax=ax3, cbar_kws={'label': 'Correlation'})
            ax3.set_title('Metric Correlations')
        
        # 4. Spatial distribution
        ax4 = fig.add_subplot(gs[1, :])
        if 'center_x' in results_df.columns and 'center_y' in results_df.columns:
            anomalies = results_df[results_df['is_anomaly_final']]
            normal = results_df[~results_df['is_anomaly_final']]
            
            ax4.scatter(normal['center_x'], normal['center_y'], c='lightgray', s=10, alpha=0.5, label='Normal')
            if not anomalies.empty:
                scatter2 = ax4.scatter(anomalies['center_x'], anomalies['center_y'], 
                                      c=anomalies['confidence'], s=100, cmap='hot', edgecolors='black', linewidths=2)
                plt.colorbar(scatter2, ax=ax4, label='Confidence')
            ax4.set_xlabel('X Coordinate')
            ax4.set_ylabel('Y Coordinate')
            ax4.set_title('Spatial Distribution of Anomalies')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        # 5. Cluster analysis (if applicable)
        ax5 = fig.add_subplot(gs[2, 0])
        if 'cluster_id' in results_df.columns and (results_df['cluster_id'] >= 0).any():
            cluster_counts = results_df[results_df['cluster_id'] >= 0].groupby('cluster_id').size()
            ax5.bar(range(len(cluster_counts)), cluster_counts.values)
            ax5.set_xlabel('Cluster ID')
            ax5.set_ylabel('Number of Chips')
            ax5.set_title('Chips per Cluster')
            ax5.grid(True, alpha=0.3)
        else:
            ax5.text(0.5, 0.5, 'No Clusters Detected', ha='center', va='center')
            ax5.set_title('Cluster Analysis')

        # 6. Top metrics
        ax6 = fig.add_subplot(gs[2, 1])
        top_anomalies = results_df.nlargest(min(10, len(results_df)), 'confidence')
        metric_cols = [col for col in results_df.columns if col.startswith('metric_') and col.endswith('_norm')]
        if len(metric_cols) > 0 and len(top_anomalies) > 0:
            top_metrics = top_anomalies[metric_cols].mean()
            ax6.barh(range(len(top_metrics)), top_metrics.values)
            ax6.set_yticks(range(len(top_metrics)))
            ax6.set_yticklabels([m.replace('metric_', '').replace('_norm', '') for m in top_metrics.index])
            ax6.set_xlabel('Average Score')
            ax6.set_title('Top Anomalies - Average Metric Scores')
            ax6.grid(True, alpha=0.3)
        
        # 7. Confidence distribution
        ax7 = fig.add_subplot(gs[2, 2])
        if results_df['is_anomaly_final'].any():
            ax7.hist(results_df[results_df['is_anomaly_final']]['confidence'], bins=20, alpha=0.7, color='red', label='Anomalies')
        ax7.hist(results_df[~results_df['is_anomaly_final']]['confidence'], bins=20, alpha=0.5, color='blue', label='Normal')
        ax7.set_xlabel('Confidence')
        ax7.set_ylabel('Frequency')
        ax7.set_title('Confidence Distribution')
        ax7.legend()
        ax7.grid(True, alpha=0.3)
        
        plt.suptitle('Detection Analysis Dashboard (Mk12)', fontsize=18, fontweight='bold')
        plt.savefig(self.output_dir / 'analysis_dashboard_mk12.png', dpi=300, bbox_inches='tight')
        logger.info(f"Saved analysis dashboard")
        plt.close()

# ============================================
# 8. TRAINING
# ============================================
def vae_loss_function(recon_x, x, mu, logvar, beta=0.001):
    MSE = F.mse_loss(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return MSE + beta * KLD, MSE, KLD

class VAETrainer:
    def __init__(self, model, device='cpu', lr=1e-3):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def train_epoch(self, loader):
        self.model.train()
        total_loss, total_mse, total_kld = 0, 0, 0
        for images, _ in tqdm(loader, desc="Training"):
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

# ============================================
# 9. MAIN PIPELINE
# ============================================
def main():
    logger.info("="*70)
    logger.info("XENARCH Mk12: IMPROVED CONTEXTUAL DETECTION PIPELINE")
    logger.info("="*70)

    # Configuration
    config = {
        'chip_size': 256,
        'latent_dim': 56,
        'batch_size': 4,
        'num_epochs': 10,
        'learning_rate': 0.001,
        'base_percentile': 90, # Slightly more sensitive for Mk12
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }

    logger.info(f"Configuration: {json.dumps(config, indent=2)}")

    data_root = Path("data")
    results_dir = Path("results") / "mk12_advanced"
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

    # 3. Train model
    logger.info(f"\n[STEP 3/5] Training VAE on {config['device']}...")
    model = ConvolutionalVAE(
        latent_dim=config['latent_dim'],
        input_size=config['chip_size']
    )
    trainer = VAETrainer(model, device=config['device'], lr=config['learning_rate'])

    for epoch in range(config['num_epochs']):
        loss, mse, kld = trainer.train_epoch(train_loader)
        logger.info(f"Epoch {epoch+1}/{config['num_epochs']} - Loss: {loss:.2f}, MSE: {mse:.2f}, KLD: {kld:.2f}")

    # Save model
    model_path = data_root / "models" / "xenarch_mk12.pth"
    model_path.parent.mkdir(exist_ok=True, parents=True)
    torch.save(model.state_dict(), model_path)
    logger.info(f"Model saved: {model_path}")

    # 4. Advanced anomaly detection
    logger.info("\n[STEP 4/5] Running advanced anomaly detection...")
    detector = AdvancedAnomalyDetector(model, device=config['device'])

    results_df = detector.detect_anomalies(
        test_loader=test_loader,
        chip_metadata=all_test_chips,
        use_spatial_filter=False, # Optimal for small datasets as per Mk12
        use_adaptive_threshold=True,
        base_percentile=config['base_percentile']
    )

    # Save results
    results_df.to_csv(results_dir / "xenarch_mk12_results.csv", index=False)
    logger.info(f"Results saved: {results_dir / 'xenarch_mk12_results.csv'}")

    # Print top detections
    logger.info("\n" + "="*70)
    logger.info("TOP 10 DETECTIONS BY CONFIDENCE")
    logger.info("="*70)
    top_10 = results_df.nlargest(min(10, len(results_df)), 'confidence')
    for idx, row in top_10.iterrows():
        logger.info(f"Rank {list(top_10.index).index(idx) + 1}: "
                   f"Confidence={row['confidence']:.3f}, "
                   f"Score={row['anomaly_score']:.3f}, "
                   f"Cluster={row.get('cluster_id', -1)}, "
                   f"Path={Path(row['chip_path']).name}")

    # 5. Visualization
    logger.info("\n[STEP 5/5] Generating visualizations...")
    viz = HighResVisualizer(output_dir=results_dir)

    viz.plot_reconstruction_examples(model, test_ds, n_samples=min(6, len(test_ds)), device=config['device'])
    viz.plot_top_anomalies_with_confidence(results_df, n_samples=min(12, len(results_df)))
    viz.plot_detection_analysis(results_df)

    logger.info("\n" + "="*70)
    logger.info("XENARCH Mk12 PIPELINE COMPLETE!")
    logger.info("="*70)
    logger.info(f"Results directory: {results_dir}")
    logger.info(f"Total anomalies: {results_df['is_anomaly_final'].sum()}")
    logger.info(f"High confidence (>0.8): {(results_df['confidence'] > 0.8).sum()}")

    return results_df

if __name__ == "__main__":
    main()

