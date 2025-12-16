# Project-Prometheus
anomaly detection algorithm files

Key Features
1. Training Phase

Trains on natural Earth features (wilderness terrain) to learn what "normal" looks like
Uses transfer learning with ResNet50 to extract deep features
Employs Isolation Forest for unsupervised anomaly detection

2. Validation Phase

Tests on known anthropogenic features (buildings, roads, etc.)
Calculates detection rate to verify the model can distinguish artificial from natural

3. Planetary Analysis

Applies the trained model to Mars and Moon imagery
Ranks anomalies by their scores
Generates JSON catalogs and visualizations

4. Output

JSON files with complete anomaly catalogs
Ranked lists of most anomalous features
Visualizations of top candidates

Directory Structure You'll Need
project/
├── data/
│   ├── earth/
│   │   ├── natural/          # Wilderness, natural terrain
│   │   └── anthropogenic/    # Cities, roads, buildings
│   ├── mars/                 # HiRISE imagery
│   └── moon/                 # LRO NAC imagery
├── results/                  # Auto-created for outputs
└── anomaly_detector.py       # The script
Next Steps

Acquire Data: Download LRO NAC, HiRISE, and Landsat imagery
Install Dependencies: pip install torch torchvision numpy scikit-learn pillow matplotlib tqdm
Test on Known Sites: Include Viking/Apollo landing sites to validate detection
Tune Parameters: Adjust contamination rate and ANOMALY_THRESHOLD based on validation results
Expert Review: Have planetary scientists review top-ranked anomalies
