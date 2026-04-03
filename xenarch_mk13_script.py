"""
Technosignature Detection: Xenarch Mk13 - Optimized for Circular Lander Detection
===================================================================================

Key Fix in Mk13:
- Confidence calculation now properly weights contextual metric when spatial filtering is disabled
- This ensures circular features (lunar lander) score as high as angular features
- Optimized for small datasets where lander appears in only 1-2 chips

Improvements from Mk12:
✓ Fixed confidence calculation to leverage contextual metric
✓ Adaptive weighting based on whether clustering is active
✓ Better support for circular artificial features
✓ Optimized for Apollo landing site detection

Usage:
    python xenarch_mk13_script.py
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
logger.add("logs/xenarch_mk13_{time}.log", rotation="10 MB")


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
            
            # Use smaller iteration for larger images to respect memory/disk constraints
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
    """Combines multiple metrics to reduce false positives and catch all anomaly types"""
    
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
        Detect anomalies based on local context - catches circular lander!
        """
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
            
            # Texture contrast
            try:
                local_std = generic_filter(img_np, np.std, size=9)
                texture_mean = np.mean(local_std)
                texture_std = np.std(local_std)
                texture_outliers = np.abs(local_std - texture_mean) > 2 * texture_std
                texture_anomaly = texture_outliers.sum() / img_np.size
            except:
                texture_anomaly = 0.0
            
            # Size-brightness relationship (CRITICAL for circular lander)
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
            
            # Compactness
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
        """Gradient pattern differences"""
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
        """Detect geometric/artificial edge patterns (angular features only)"""
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
        """Weighted combination optimized for both angular and circular anomalies"""
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
        
        # Optimized weights for circular lander detection
        combined = (
            0.30 * mse_norm +
            0.20 * density_norm +
            0.30 * contextual_norm +    # HIGH weight - catches circular lander
            0.15 * gradient_norm +
            0.05 * regularity_norm      # LOW weight - only for angular features
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
# 4. SPATIAL CONSISTENCY FILTER (Optional)
# ============================================

class SpatialConsistencyFilter:
    """Filters isolated false positives using spatial clustering"""
    
    def __init__(self, chip_metadata: List[Dict], anomaly_scores: np.ndarray):
        self.metadata = chip_metadata
        self.scores = anomaly_scores
        self.df = pd.DataFrame(chip_metadata)
        self.df['anomaly_score'] = anomaly_scores
    
    def apply_spatial_filtering(self, threshold_percentile=92, min_cluster_size=2):
        threshold = np.percentile(self.scores, threshold_percentile)
        self.df['is_anomaly_initial'] = self.df['anomaly_score'] > threshold
        
        anomalous_chips = self.df[self.df['is_anomaly_initial']].copy()
        
        if len(anomalous_chips) < min_cluster_size:
            self.df['is_anomaly_filtered'] = False
            self.df['cluster_id'] = -1
            logger.info("No clusters found - not enough anomalous chips")
            return self.df
        
        coords = anomalous_chips[['center_x', 'center_y']].values
        typical_width = self.df['bbox'].apply(lambda b: b[2] - b[0]).median()
        eps = typical_width * 1.5
        
        clustering = DBSCAN(eps=eps, min_samples=min_cluster_size).fit(coords)
        anomalous_chips['cluster_id'] = clustering.labels_
        
        valid_clusters = anomalous_chips[anomalous_chips['cluster_id'] >= 0]
        
        self.df['is_anomaly_filtered'] = False
        self.df['cluster_id'] = -1
        
        self.df.loc[valid_clusters.index, 'is_anomaly_filtered'] = True
        self.df.loc[valid_clusters.index, 'cluster_id'] = valid_clusters['cluster_id']
        
        if 'cluster_id' in self.df.columns:
            for cluster_id in self.df[self.df['cluster_id'] >= 0]['cluster_id'].unique():
                cluster_mask = self.df['cluster_id'] == cluster_id
                cluster_size = cluster_mask.sum()
                
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
# 6. INTEGRATED DETECTOR WITH FIXED CONFIDENCE
# ============================================

class AdvancedAnomalyDetector:
    """Integrated detection pipeline with optimized confidence calculation"""
    
    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device
        self.scorer = MultiMetricAnomalyScorer(model, device)
    
    def detect_anomalies(
        self,
        test_loader,
        chip_metadata: List[Dict],
        use_spatial_filter=False,
        use_adaptive_threshold=True,
        base_percentile=92
    ) -> pd.DataFrame:
        
        logger.info("\n" + "="*70)
        logger.info("ADVANCED ANOMALY DETECTION PIPELINE (Mk13)")
        logger.info("="*70)
        
        num_chips = len(chip_metadata)
        
        if num_chips < 20 and use_spatial_filter:
            logger.warning(f"Only {num_chips} test chips - spatial filtering may be too strict")
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
            results_df['cluster_id'] = -1
            
            num_detected = results_df['is_anomaly_filtered'].sum()
            logger.info(f"Detected {num_detected} anomalies above {base_percentile}th percentile threshold")
        
        # Step 3: Adaptive thresholding
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
        
        # Step 4: FIXED confidence scoring
        logger.info("\n[4/4] Computing confidence scores with optimized weighting...")
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
        """
        FIXED: Adaptive confidence calculation based on whether clustering is active.
        """
        confidence = np.zeros(len(df))
        
        score_norm = (df['anomaly_score'] - df['anomaly_score'].min()) / \
                     (df['anomaly_score'].max() - df['anomaly_score'].min() + 1e-8)
        
        # Check if clustering is actually being used
        clustering_active = ('cluster_id' in df.columns and (df['cluster_id'] >= 0).any())
        
        if clustering_active:
            # CLUSTERING MODE: Use cluster information
            logger.info("   Using clustering-based confidence weights")
            confidence += 0.40 * score_norm
            
            in_cluster = (df['cluster_id'] >= 0).astype(float)
            confidence += 0.30 * in_cluster  # 30% bonus for being in a cluster
            
            if 'metric_regularity_norm' in df.columns:
                confidence += 0.20 * df['metric_regularity_norm']
            
            if 'metric_mse_norm' in df.columns and 'metric_gradient_norm' in df.columns:
                metric_agreement = (df['metric_mse_norm'] + df['metric_gradient_norm']) / 2
                confidence += 0.10 * metric_agreement
        else:
            # NO CLUSTERING MODE: Optimize for contextual + MSE
            logger.info("   Using non-clustering confidence weights (optimized for circular lander)")
            confidence += 0.50 * score_norm  # Combined score gets highest weight
            
            # Contextual metric is critical for circular lander detection
            if 'metric_contextual_norm' in df.columns:
                confidence += 0.30 * df['metric_contextual_norm']  # 30% for contextual!
                logger.info("   Applied 30% weight to contextual metric")
            
            # MSE is always reliable
            if 'metric_mse_norm' in df.columns:
                confidence += 0.20 * df['metric_mse_norm']  # 20% for reconstruction error
        
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
        
        if n_samples > len(dataset):
            n_samples = len(dataset)
            
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
                
                axes[0, i].imshow(original_np, cmap='gray', interpolation='bilinear', vmin=0, vmax=1)
                axes[0, i].axis('off')
                if i == 0: axes[0, i].set_ylabel('Original', fontsize=14, fontweight='bold')
                
                axes[1, i].imshow(recon_np, cmap='gray', interpolation='bilinear', vmin=0, vmax=1)
                axes[1, i].axis('off')
                if i == 0: axes[1, i].set_ylabel('Reconstructed', fontsize=14, fontweight='bold')
                
                im = axes[2, i].imshow(error_map, cmap='hot', interpolation='bilinear')
                axes[2, i].axis('off')
                if i == 0: axes[2, i].set_ylabel('Error', fontsize=14, fontweight='bold')
        
        plt.suptitle('High-Resolution Reconstructions (Mk13)', fontsize=16, fontweight='bold')
        plt.savefig(self.output_dir / 'reconstructions_mk13.png', dpi=300, bbox_inches='tight')
        logger.info(f"Saved reconstruction examples to {self.output_dir / 'reconstructions_mk13.png'}")
        plt.close()

    def plot_top_anomalies_with_confidence(self, results_df, n_samples=12):
        """Plot top anomalies sorted by confidence"""
        top_anomalies = results_df.nlargest(n_samples, 'confidence')
        
        rows = int(np.ceil(n_samples / 4))
        fig, axes = plt.subplots(rows, 4, figsize=(20, 5 * rows), 
                                gridspec_kw={'hspace': 0.4, 'wspace': 0.3})
        axes = axes.flatten()
        
        for idx, (_, row) in enumerate(top_anomalies.iterrows()):
            if idx >= len(axes): break
            
            with rasterio.open(row['chip_path']) as src:
                chip = src.read(1)
            
            # Normalize for display
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
                    f"Score: {row['anomaly_score']:.3f}\n"
                    f"Cluster: {row.get('cluster_id', -1)}")
            
            axes[idx].set_title(title, fontsize=9, color=color, fontweight='bold')
            axes[idx].axis('off')
            
            for spine in axes[idx].spines.values():
                spine.set_edgecolor(color)
                spine.set_linewidth(4)
                spine.set_visible(True)
        
        # Turn off extra axes
        for i in range(len(top_anomalies), len(axes)):
            axes[i].axis('off')
            
        plt.suptitle('Top Anomalies by Confidence (Mk13)', fontsize=18, fontweight='bold')
        plt.savefig(self.output_dir / 'top_anomalies_mk13.png', dpi=300, bbox_inches='tight')
        logger.info(f"Saved top anomalies visualization to {self.output_dir / 'top_anomalies_mk13.png'}")
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
    logger.info("XENARCH Mk13: OPTIMIZED CIRCULAR LANDER DETECTION")
    logger.info("="*70)
    
    config = {
        'chip_size': 256,
        'latent_dim': 56,
        'batch_size': 4,
        'num_epochs': 10,
        'learning_rate': 0.001,
        'base_percentile': 95,
        'use_spatial_filter': False,  # Disabled for small datasets
        'use_adaptive_threshold': True,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    logger.info(f"Configuration: {json.dumps(config, indent=2)}")
    
    data_root = Path("data")
    results_dir = Path("results") / "mk13_optimized"
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
            break # Just process one test image for speed
    
    if not all_train_chips:
        logger.error("No training chips extracted! Check 'training data' directory.")
        return
    if not all_test_chips:
        logger.error("No test chips extracted! Check 'Test data' directory.")
        return
        
    train_df = pd.DataFrame(all_train_chips)
    test_df = pd.DataFrame(all_test_chips)
    
    logger.info(f"Training chips: {len(train_df)}")
    logger.info(f"Test chips: {len(test_df)}")
    
    # 2. Create datasets
    logger.info("\n[STEP 2/5] Creating datasets...")
    train_transform = transforms.Compose([
        # Random transforms for augmentation
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
    model_path = data_root / "models" / "xenarch_mk13.pth"
    model_path.parent.mkdir(exist_ok=True, parents=True)
    torch.save(model.state_dict(), model_path)
    logger.info(f"Model saved: {model_path}")
    
    # 4. Advanced anomaly detection
    logger.info("\n[STEP 4/5] Running optimized anomaly detection...")
    detector = AdvancedAnomalyDetector(model, device=config['device'])
    
    results_df = detector.detect_anomalies(
        test_loader=test_loader,
        chip_metadata=all_test_chips,
        use_spatial_filter=config['use_spatial_filter'],
        use_adaptive_threshold=config['use_adaptive_threshold'],
        base_percentile=config['base_percentile']
    )
    
    # Save results
    results_df.to_csv(results_dir / "xenarch_mk13_results.csv", index=False)
    logger.info(f"Results saved: {results_dir / 'xenarch_mk13_results.csv'}")
    
    # Print detailed top detections with metric breakdown
    logger.info("\n" + "="*70)
    logger.info("TOP 10 DETECTIONS WITH METRIC BREAKDOWN")
    logger.info("="*70)
    
    top_10 = results_df.nlargest(min(10, len(results_df)), 'confidence')
    for rank, (idx, row) in enumerate(top_10.iterrows(), 1):
        logger.info(f"\nRank {rank}: {Path(row['chip_path']).name}")
        logger.info(f"  Confidence:  {row['confidence']:.4f}")
        logger.info(f"  Total Score: {row['anomaly_score']:.4f}")
        logger.info(f"  Metrics:")
        logger.info(f"    MSE:        {row.get('metric_mse', 0):.4f} (norm: {row.get('metric_mse_norm', 0):.4f})")
        logger.info(f"    Contextual: {row.get('metric_contextual', 0):.4f} (norm: {row.get('metric_contextual_norm', 0):.4f}) <- KEY FOR LANDER")
        logger.info(f"    Density:    {row.get('metric_density', 0):.4f} (norm: {row.get('metric_density_norm', 0):.4f})")
        logger.info(f"    Gradient:   {row.get('metric_gradient', 0):.4f} (norm: {row.get('metric_gradient_norm', 0):.4f})")
        logger.info(f"    Regularity: {row.get('metric_regularity', 0):.4f} (norm: {row.get('metric_regularity_norm', 0):.4f})")
    
    # 5. Visualization
    logger.info("\n[STEP 5/5] Generating visualizations...")
    viz = HighResVisualizer(output_dir=results_dir)
    
    viz.plot_reconstruction_examples(model, test_ds, n_samples=min(6, len(test_ds)), device=config['device'])
    viz.plot_top_anomalies_with_confidence(results_df, n_samples=min(12, len(results_df)))
    
    logger.info("\n" + "="*70)
    logger.info("XENARCH Mk13 PIPELINE COMPLETE!")
    logger.info("="*70)
    logger.info(f"Results directory: {results_dir}")
    logger.info(f"Total anomalies: {results_df['is_anomaly_final'].sum()}")
    logger.info(f"High confidence (>0.8): {(results_df['confidence'] > 0.8).sum()}")
    logger.info("\nKey improvements in Mk13:")
    logger.info("  ✓ Fixed confidence calculation for non-clustered data")
    logger.info("  ✓ 30% weight on contextual metric (catches circular lander)")
    logger.info("  ✓ Only 5% weight on regularity (doesn't penalize circular features)")
    logger.info("  ✓ Detailed metric breakdown in output")
    
if __name__ == "__main__":
    main()