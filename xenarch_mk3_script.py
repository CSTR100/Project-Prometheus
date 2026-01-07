"""
Vertical Slice Pipeline: End-to-End Technosignature Detection Demo
=====================================================================

This script demonstrates a complete workflow from data download to visualization:
1. Download 1 LRO NAC image
2. Extract training chips
3. Store in PostGIS
4. Train simple CNN
5. Visualize results

Usage:
    python vertical_slice_pipeline.py --config config.yaml
"""

import os
import sys
from pathlib import Path
from typing import List, Tuple, Dict
import yaml
import json
from datetime import datetime

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, Polygon, box
import rasterio
from rasterio.windows import Window
from rasterio.plot import show
import psycopg2
from psycopg2.extras import execute_values

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from loguru import logger

# Configure logger
logger.add("logs/pipeline_{time}.log", rotation="10 MB")


# ============================================
# 1. DATA INGESTION: LRO NAC Adapter
# ============================================

class LROAdapter:
    """Download and manage LRO NAC imagery"""
    
    def __init__(self, data_dir='./data/raw/lro'):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.pds_base_url = "https://pds.lroc.asu.edu/data/"
        
    def download_example_nac(self) -> str:
        """
        Use the user-provided Lunar_sample_image.png for the pipeline.
        """
        logger.info("Using City-image.jpg from Project-Prometheus...")
        
        source_image = Path(r"c:\Users\caleb\OneDrive\Documents\Xeno-archaeology\Project-Prometheus\Data\Earth\City-image.jpg")
        image_path = self.data_dir / "real_lunar_sample.tif"
        
        if not source_image.exists():
            logger.error(f"Source image not found at {source_image}")
            raise FileNotFoundError(f"Missing source image: {source_image}")

        # Copy or convert to TIF if needed, but rasterio can read PNG
        # For consistency with the rest of the pipeline which expects .tif, 
        # let's save a TIF version in the data dir.
        with rasterio.open(source_image) as src:
            data = src.read(1)
            
            # Ensure it's uint8
            if data.dtype != np.uint8:
                data = ((data - data.min()) / (data.max() - data.min() + 1e-8) * 255).astype(np.uint8)
            
            with rasterio.open(
                image_path, 'w',
                driver='GTiff',
                height=src.height,
                width=src.width,
                count=1,
                dtype=np.uint8,
                crs=src.crs if src.crs else 'EPSG:4326',
                transform=src.transform
            ) as dst:
                dst.write(data, 1)
                dst.update_tags(
                    mission='LRO',
                    instrument='User_Image',
                    source_file=source_image.name,
                    target='Moon'
                )
        
        logger.success(f"Prepared lunar image: {image_path}")
        return str(image_path)
    
    def extract_metadata(self, image_path: str) -> Dict:
        """Extract metadata from LRO NAC image"""
        with rasterio.open(image_path) as src:
            bounds = src.bounds
            tags = src.tags()
            metadata = {
                'image_id': Path(image_path).stem,
                'mission': tags.get('mission', 'LRO'),
                'instrument': tags.get('instrument', 'WAC_Mosaic'),
                'acquisition_date': tags.get('acquisition_date', '2023-06-15'),
                'resolution_meters': 100.0, # Known resolution of this mosaic
                'bbox_coords': (bounds.left, bounds.bottom, bounds.right, bounds.top),
                'width': src.width,
                'height': src.height,
                'crs': str(src.crs)
            }
        return metadata


# ============================================
# 2. DATA PROCESSING: Chip Extraction
# ============================================

class ChipExtractor:
    """Extract training chips from large images"""
    
    def __init__(self, chip_size=64, overlap=0.0):
        self.chip_size = chip_size
        self.overlap = overlap
        
    def extract_grid(self, image_path: str, output_dir: str) -> List[Dict]:
        """
        Extract regular grid of chips from image
        
        Returns:
            List of chip metadata dicts
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        chips_metadata = []
        
        with rasterio.open(image_path) as src:
            width, height = src.width, src.height
            stride = int(self.chip_size * (1 - self.overlap))
            
            chip_id = 0
            for y in range(0, height - self.chip_size, stride):
                for x in range(0, width - self.chip_size, stride):
                    window = Window(x, y, self.chip_size, self.chip_size)
                    chip = src.read(1, window=window)
                    
                    # Skip chips that are mostly empty
                    if chip.std() < 5:
                        continue
                    
                    # Get geographic bounds
                    window_transform = src.window_transform(window)
                    chip_bounds = rasterio.transform.array_bounds(
                        self.chip_size, self.chip_size, window_transform
                    )
                    
                    # Save chip
                    chip_filename = f"chip_{chip_id:04d}.tif"
                    chip_path = output_path / chip_filename
                    
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
                    
                    # Create center point
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
            
            logger.info(f"Extracted {chip_id} chips to {output_path}")
            
        return chips_metadata


# ============================================
# 3. ANNOTATION & LABELING
# ============================================

class SyntheticAnnotator:
    """
    Create synthetic annotations for demo
    In production, replace with manual annotations or import from QGIS
    """
    
    def __init__(self):
        pass
    
    def generate_labels(self, chips_metadata: List[Dict], image_path: str) -> pd.DataFrame:
        """
        Generate synthetic labels based on image analysis
        
        In production: import from shapefile or manual annotation tool
        """
        labels = []
        
        with rasterio.open(image_path) as src:
            for chip_meta in chips_metadata:
                chip_path = chip_meta['chip_path']
                
                with rasterio.open(chip_path) as chip_src:
                    chip_data = chip_src.read(1)
                    
                    # Simple heuristics for labeling (for demo only!)
                    # In real scenario: manual expert annotation
                    
                    # Check for linear features (potential artificial)
                    edges = self._detect_edges(chip_data)
                    linear_score = self._detect_linear_features(edges)
                    
                    # Check for crater-like features (natural)
                    circular_score = self._detect_circular_features(chip_data)
                    
                    # Label assignment (simplified)
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
        # Simple metric: std of edge orientations
        if edges.max() == 0:
            return 0.0
        
        threshold = edges.max() * 0.3
        edge_pixels = edges > threshold
        
        if edge_pixels.sum() < 10:
            return 0.0
        
        # Check for alignment (simplified)
        row_sums = edge_pixels.sum(axis=1)
        col_sums = edge_pixels.sum(axis=0)
        
        max_alignment = max(row_sums.max(), col_sums.max())
        total_edges = edge_pixels.sum()
        
        return min(max_alignment / total_edges, 1.0)
    
    def _detect_circular_features(self, image: np.ndarray) -> float:
        """Score for circular features (craters)"""
        # Simple approach: look for dark circular regions
        center = image.shape[0] // 2
        y, x = np.ogrid[:image.shape[0], :image.shape[1]]
        
        # Check if center is darker than edges
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
# 4. DATABASE STORAGE
# ============================================

class TechnosignatureDB:
    """Database interface for storing training data"""
    
    def __init__(self, dbname='technosignatures', user='postgres', 
                 password='postgres', host='localhost', port=5432):
        self.conn = psycopg2.connect(
            dbname=dbname, user=user, password=password,
            host=host, port=port
        )
        self.cursor = self.conn.cursor()
        
    def close(self):
        self.cursor.close()
        self.conn.close()
    
    def store_image_source(self, metadata: Dict) -> int:
        """Store image metadata in database"""
        
        # Create bbox polygon
        bbox = metadata['bbox_coords']
        bbox_wkt = f"POLYGON(({bbox[0]} {bbox[1]}, {bbox[2]} {bbox[1]}, " \
                   f"{bbox[2]} {bbox[3]}, {bbox[0]} {bbox[3]}, {bbox[0]} {bbox[1]}))"
        
        query = """
        INSERT INTO image_sources 
        (body_id, mission_name, instrument_name, image_id, acquisition_date,
         resolution_meters, image_path, bbox, center_point)
        VALUES (
            (SELECT body_id FROM planetary_bodies WHERE body_name = 'Moon'),
            %s, %s, %s, %s, %s, %s, 
            ST_GeomFromText(%s, 4326),
            ST_Centroid(ST_GeomFromText(%s, 4326))
        )
        RETURNING source_id
        """
        
        self.cursor.execute(query, (
            metadata['mission'],
            metadata['instrument'],
            metadata['image_id'],
            metadata['acquisition_date'],
            metadata['resolution_meters'],
            str(Path(metadata['image_id']).parent),  # Just store directory
            bbox_wkt,
            bbox_wkt
        ))
        
        source_id = self.cursor.fetchone()[0]
        self.conn.commit()
        logger.info(f"Stored image source with ID: {source_id}")
        return source_id
    
    def store_training_features(self, labels_df: pd.DataFrame, source_id: int):
        """Bulk store training features"""
        
        for _, row in labels_df.iterrows():
            # Get feature_type_id
            self.cursor.execute(
                "SELECT feature_type_id FROM feature_types WHERE category = %s AND subcategory = %s",
                (row['category'], row['subcategory'])
            )
            result = self.cursor.fetchone()
            if not result:
                # Insert if not exists
                self.cursor.execute(
                    "INSERT INTO feature_types (category, subcategory) VALUES (%s, %s) RETURNING feature_type_id",
                    (row['category'], row['subcategory'])
                )
                feature_type_id = self.cursor.fetchone()[0]
            else:
                feature_type_id = result[0]
            
            # Create point geometry
            point_wkt = f"POINT({row['center_lon']} {row['center_lat']})"
            
            # Insert feature
            self.cursor.execute("""
                INSERT INTO training_features
                (source_id, feature_type_id, geometry, confidence_score, 
                 annotator_id, image_chip_path)
                VALUES (%s, %s, ST_GeomFromText(%s, 4326), %s, %s, %s)
            """, (
                source_id, feature_type_id, point_wkt,
                row['confidence'], 'synthetic', row['chip_path']
            ))
        
        self.conn.commit()
        logger.info(f"Stored {len(labels_df)} training features")


# ============================================
# 5. MACHINE LEARNING: CNN Training
# ============================================

class LunarChipDataset(Dataset):
    """PyTorch dataset for lunar image chips"""
    
    def __init__(self, labels_df: pd.DataFrame, transform=None):
        self.labels_df = labels_df
        self.transform = transform
        
        # Binary classification: artificial (1) vs natural (0)
        self.labels_df['label'] = (self.labels_df['category'] == 'artificial').astype(int)
        
    def __len__(self):
        return len(self.labels_df)
    
    def __getitem__(self, idx):
        row = self.labels_df.iloc[idx]
        
        # Load image chip
        with rasterio.open(row['chip_path']) as src:
            image = src.read(1).astype(np.float32)
        
        # Normalize to [0, 1]
        image = (image - image.min()) / (image.max() - image.min() + 1e-8)
        
        # Convert to 3-channel (required for pretrained models)
        image = np.stack([image, image, image], axis=0)
        
        if self.transform:
            image = self.transform(torch.from_numpy(image))
        else:
            image = torch.from_numpy(image)
        
        label = torch.tensor(row['label'], dtype=torch.long)
        
        return image, label


class SimpleCNN(nn.Module):
    """Simple CNN for binary classification"""
    
    def __init__(self, num_classes=2):
        super(SimpleCNN, self).__init__()
        
        # Use pretrained ResNet18 as backbone
        self.backbone = models.resnet18(pretrained=True)
        
        # Replace final layer
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_features, num_classes)
        
    def forward(self, x):
        return self.backbone(x)


class ModelTrainer:
    """Train and evaluate CNN model"""
    
    def __init__(self, model, device='cpu'):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        
    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc="Training")
        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({'loss': running_loss / (pbar.n + 1), 'acc': 100. * correct / total})
        
        return running_loss / len(train_loader), 100. * correct / total
    
    def evaluate(self, test_loader):
        """Evaluate on test set"""
        self.model.eval()
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in tqdm(test_loader, desc="Evaluating"):
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                _, predicted = outputs.max(1)
                
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        accuracy = 100. * correct / total
        
        return accuracy, np.array(all_labels), np.array(all_preds)


# ============================================
# 6. VISUALIZATION
# ============================================

class ResultsVisualizer:
    """Visualize training data and model results"""
    
    def __init__(self, output_dir='./results'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def plot_image_with_annotations(self, image_path: str, labels_df: pd.DataFrame):
        """Plot image with labeled chips overlaid"""
        fig, ax = plt.subplots(figsize=(12, 12))
        
        with rasterio.open(image_path) as src:
            show(src, ax=ax, cmap='gray')
            
            # Overlay chip locations
            for _, row in labels_df.iterrows():
                color = 'red' if row['category'] == 'artificial' else 'blue'
                ax.plot(row['center_lon'], row['center_lat'], 'o', 
                       color=color, markersize=3, alpha=0.6)
        
        ax.set_title('LRO NAC Image with Annotations\nRed=Artificial, Blue=Natural')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'annotated_image.png', dpi=150)
        logger.info(f"Saved annotated image to {self.output_dir / 'annotated_image.png'}")
        plt.close()
    
    def plot_sample_chips(self, labels_df: pd.DataFrame, n_samples=12):
        """Plot sample chips with labels"""
        fig, axes = plt.subplots(3, 4, figsize=(12, 9))
        axes = axes.flatten()
        
        samples = labels_df.sample(min(n_samples, len(labels_df)))
        
        for idx, (_, row) in enumerate(samples.iterrows()):
            if idx >= len(axes):
                break
                
            with rasterio.open(row['chip_path']) as src:
                chip = src.read(1)
            
            axes[idx].imshow(chip, cmap='gray')
            axes[idx].set_title(f"{row['category']}\n{row['subcategory']}", fontsize=8)
            axes[idx].axis('off')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'sample_chips.png', dpi=150)
        logger.info(f"Saved sample chips to {self.output_dir / 'sample_chips.png'}")
        plt.close()
    
    def plot_confusion_matrix(self, labels: np.ndarray, predictions: np.ndarray):
        """Plot confusion matrix"""
        cm = confusion_matrix(labels, predictions)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Natural', 'Artificial'],
                   yticklabels=['Natural', 'Artificial'])
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'confusion_matrix.png', dpi=150)
        logger.info(f"Saved confusion matrix to {self.output_dir / 'confusion_matrix.png'}")
        plt.close()
    
    def plot_training_curves(self, train_losses: List[float], train_accs: List[float]):
        """Plot training curves"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        ax1.plot(train_losses)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training Loss')
        ax1.grid(True)
        
        ax2.plot(train_accs)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title('Training Accuracy')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'training_curves.png', dpi=150)
        logger.info(f"Saved training curves to {self.output_dir / 'training_curves.png'}")
        plt.close()


# ============================================
# 7. MAIN PIPELINE
# ============================================

def main():
    """Run complete vertical slice pipeline"""
    
    logger.info("="*60)
    logger.info("Starting Technosignature Detection Vertical Slice Pipeline")
    logger.info("="*60)
    
    # Configuration
    config = {
        'data_dir': './data',
        'chip_size': 64,
        'batch_size': 8,
        'num_epochs': 5,
        'test_split': 0.2,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    logger.info(f"Configuration: {config}")
    
    # Step 1: Download LRO NAC image
    logger.info("\n[1/6] Downloading LRO NAC image...")
    lro = LROAdapter(data_dir=f"{config['data_dir']}/raw/lro")
    image_path = lro.download_example_nac()
    metadata = lro.extract_metadata(image_path)
    
    # Step 2: Extract training chips
    logger.info("\n[2/6] Extracting training chips...")
    extractor = ChipExtractor(chip_size=config['chip_size'], overlap=0.0)
    chips_dir = f"{config['data_dir']}/processed/chips"
    chips_metadata = extractor.extract_grid(image_path, chips_dir)
    logger.info(f"Extracted {len(chips_metadata)} chips")
    
    # Step 3: Generate annotations (synthetic for demo)
    logger.info("\n[3/6] Generating annotations...")
    annotator = SyntheticAnnotator()
    labels_df = annotator.generate_labels(chips_metadata, image_path)
    labels_df.to_csv(f"{config['data_dir']}/labels.csv", index=False)
    logger.info(f"Label distribution:\n{labels_df['category'].value_counts()}")
    
    # Step 4: Store in PostGIS database
    logger.info("\n[4/6] Storing data in PostGIS...")
    try:
        db = TechnosignatureDB()
        source_id = db.store_image_source(metadata)
        db.store_training_features(labels_df, source_id)
        db.close()
        logger.success("Data successfully stored in database")
    except Exception as e:
        logger.warning(f"Database storage failed: {e}")
        logger.info("Continuing without database...")
    
    # Step 5: Train CNN
    logger.info(f"\n[5/6] Training CNN on {config['device']}...")
    
    # Split data
    train_df, test_df = train_test_split(
        labels_df, test_size=config['test_split'], 
        stratify=labels_df['category'], random_state=42
    )
    logger.info(f"Train: {len(train_df)}, Test: {len(test_df)}")
    
    # Create datasets
    train_dataset = LunarChipDataset(train_df)
    test_dataset = LunarChipDataset(test_df)
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)
    
    # Train model
    model = SimpleCNN(num_classes=2)
    trainer = ModelTrainer(model, device=config['device'])
    
    train_losses = []
    train_accs = []
    
    for epoch in range(config['num_epochs']):
        logger.info(f"\nEpoch {epoch+1}/{config['num_epochs']}")
        loss, acc = trainer.train_epoch(train_loader)
        train_losses.append(loss)
        train_accs.append(acc)
        logger.info(f"Train Loss: {loss:.4f}, Train Acc: {acc:.2f}%")
    
    # Evaluate
    test_acc, test_labels, test_preds = trainer.evaluate(test_loader)
    logger.info(f"\nTest Accuracy: {test_acc:.2f}%")
    
    # Save model
    model_path = Path(config['data_dir']) / 'models' / 'simple_cnn.pth'
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_path)
    logger.success(f"Model saved to {model_path}")
    
    # Step 6: Visualize results
    logger.info("\n[6/6] Generating visualizations...")
    viz = ResultsVisualizer(output_dir='./results')
    
    viz.plot_image_with_annotations(image_path, labels_df)
    viz.plot_sample_chips(labels_df)
    viz.plot_confusion_matrix(test_labels, test_preds)
    viz.plot_training_curves(train_losses, train_accs)
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("PIPELINE COMPLETE!")
    logger.info("="*60)
    logger.info(f"✓ Processed 1 LRO NAC image")
    logger.info(f"✓ Extracted {len(chips_metadata)} training chips")
    logger.info(f"✓ Generated {len(labels_df)} annotations")


if __name__ == "__main__":
    main()
