🌙 Lunar Anomaly Detection Pipeline
An unsupervised machine learning system for detecting artificial structures and anomalies on planetary surfaces using autoencoder-based reconstruction error analysis.
Show Image
Show Image
Show Image

🎯 Overview
This project implements a novel approach to technosignature detection on lunar and planetary surfaces. By training exclusively on natural geological features, the system learns what "normal" looks like and flags anything anomalous - including artificial structures like landing sites, rovers, and human-made equipment.
Key Features

Unsupervised Learning: Trains only on natural terrain, no labeled anomalies needed
High Resolution: Processes 128×128 pixel chips for detailed feature detection
Adaptive Architecture: Automatically adjusts to different input resolutions (64×64, 128×128, 256×256)
Variational Autoencoder Support: Mk5 implementation with probabilistic latent space
Real-World Validation: Tested on Apollo landing sites with actual human artifacts

Applications

Planetary surface analysis
Archaeological site detection
Infrastructure monitoring on Mars/Moon missions
Geological anomaly identification
Change detection in satellite imagery

🚀 Quick Start

Prerequisites
bashPython 3.8+
CUDA-capable GPU (optional, but recommended)
Installation
bash# Clone the repository
git clone https://github.com/yourusername/lunar-anomaly-detection.git
cd lunar-anomaly-detection

# Create virtual environment
```
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
Basic Usage
bash# Run the pipeline with default settings (64×64 resolution)
python anomaly_detection_pipeline.py

# Run with high resolution (128×128)
python anomaly_detection_pipeline_hires.py

# Run Mk5 with Variational Autoencoder
python xenarch_mk5_script.py
```
📁 Project Structure
```
lunar-anomaly-detection/
├── anomaly_detection_pipeline.py     # Main Mk4 pipeline (64×64)
├── anomaly_detection_pipeline_hires.py  # High-res version (128×128)
├── xenarch_mk5_script.py             # Mk5 with VAE architecture
├── requirements.txt                   # Python dependencies
├── README.md                          # This file
├── ARCHITECTURE.md                    # Detailed technical documentation
├── .gitignore                         # Git ignore rules
│
├── training data/                     # Natural lunar terrain (not included)
├── Test data/                         # Test images with potential anomalies
├── data/                              # Processed chips and labels
│   ├── processed/
│   │   ├── training_chips/
│   │   └── test_chips/
│   └── models/                        # Saved model checkpoints
│
├── results/                           # Generated visualizations
│   ├── reconstruction_examples.png
│   ├── top_anomalies.png
│   └── training_curve.png
│
└── logs/                              # Training logs
```
🔬 How It Works
1. Training Phase
The model trains exclusively on natural geological features:

Craters
Rocky terrain
Smooth regolith
Natural surface variations

2. Detection Phase
When shown test images, the model:

Attempts to reconstruct each chip
Calculates reconstruction error
Flags high-error regions as anomalies

3. Why It Works

Natural features → Low reconstruction error (model has seen similar patterns)
Artificial structures → High reconstruction error (model has never seen these patterns)

📊 Results

Performance Metrics:
```
Model Version Resolution Latent Dimensional Training Time 
Detection Rate (Mk4 Baseline)
```

Example Detections
The system successfully flags:

```
✅ Apollo Lunar Module descent stages
```
While correctly identifying as natural:
```
✅ Complex crater formations
✅ Boulder fields
✅ Unusual lighting conditions
✅ Natural linear features (rilles)
```
🛠️ Configuration
Key Hyperparameters
Edit these in the config dictionary:
pythonconfig = {
    'chip_size': 128,              # Resolution: 64, 128, or 256
    'latent_dim': 48,              # Bottleneck size: 32-128
    'batch_size': 16,              # Adjust based on GPU memory
    'num_epochs': 20,              # Training epochs
    'learning_rate': 0.001,        # Adam optimizer learning rate
    'anomaly_threshold_percentile': 95,  # Detection sensitivity
}
Tuning for Your Use Case
High Sensitivity (catch more anomalies):

latent_dim: 32
anomaly_threshold_percentile: 90

High Precision (fewer false positives):

latent_dim: 64
anomaly_threshold_percentile: 97

High Resolution (detailed features):

chip_size: 128 or 256
Reduce batch_size to fit GPU memory

📦 Data
Training Data
imagery of the Moon and Mars from Lunar Reconnaissance Orbiter (LROC) and Mars Reconnaissance orbiter (HiRISE) cameras.

# Configure for your data
config = {
    'data_dir': './my_data',
    'chip_size': 128,
    'latent_dim': 48,
    # ... other settings
}

# Run pipeline
main()
Using Pre-trained Models
pythonimport torch
from anomaly_detection_pipeline import AdaptiveConvolutionalAutoencoder

# Load model
model = AdaptiveConvolutionalAutoencoder(latent_dim=48, input_size=128)
model.load_state_dict(torch.load('data/models/autoencoder_128.pth'))
model.eval()

# Run inference on new image
Batch Processing
python# Process multiple test directories
test_dirs = ['Test data/apollo11', 'Test data/apollo17', 'Test data/chang_e']

for test_dir in test_dirs:
    # Extract and analyze chips
    # ... processing code
🛰️ Imagery agents
Two harvesters feed the pipeline from the public planetary archives, and they
are deliberate opposites.

lroc_fetch.py — negative set. Crawls LRO NAC (and optionally Chandrayaan-2
OHRC) for natural terrain, actively excluding a 2° box around every known
landing site, and writes reference chips the autoencoder trains on.

```
python lroc_fetch.py --source lroc --frames 10
```

apollo16_agent.py — positive control. Goes after one specific piece of human
hardware: the Apollo 16 LM "Orion" descent stage at Descartes (8.9730°S,
15.5002°E). These chips contain known artificial structure, so they are what
you score the detector against — anything that fails to flag them is a miss you
can measure.

```
python apollo16_agent.py                  # search, rank, cut out the best 3 frames
python apollo16_agent.py --dry-run        # see what it would fetch, and why
python apollo16_agent.py --self-test      # offline logic checks, no network
python apollo16_agent.py --emit-isis      # ISIS3 recipe for metre-level registration
```

The agent runs a fixed loop with a fallback at every stage and journals each
decision to journal.json:

```
PLAN    resolve the target point and the search box
SEARCH  ODE REST (rich geometry metadata), falling back to the PDS
        cumulative index, whose column offsets are read from CUMINDEX.LBL
        rather than hardcoded
PROBE   HTTP Range-read each candidate's 64 KB PDS3 label instead of pulling
        a 250 MB frame just to read its header
RANK    score on ground sample distance (55%), incidence angle (25%) and
        emission angle (20%) — weights overridable with --weights
LOCATE  invert the frame corner coordinates by Newton iteration; fall back to
        a tangent-plane projection about the frame centre
FETCH   Range-download only the line span around the target — roughly 5 MB
        instead of 250 MB per frame
VERIFY  heuristic check for a bright compact object with an attached shadow
REPORT  .npy + .png chips, manifest.json, journal.json
```

Output drops into the same manifest schema lroc_fetch.py uses:

```
manifest = json.load(open('apollo16_imagery/manifest.json'))
test_paths = [c['chip_path'] for c in manifest['chips']]
```

Scale check: the descent stage body is about 4.2 m across and the landing gear
spreads 9.4 m footpad to footpad. NAC resolves ~0.5 m/pixel from the 50 km
mapping orbit and better from the low-altitude campaign, so the target spans
roughly 20-40 pixels — enough for the 128×128 and 256×256 chip sizes, marginal
at 64×64.

Two caveats worth knowing before you trust an output. The locator is
approximate (corner interpolation, good to roughly a hundred metres), which is
why the default crop is ~1 km wide and why --emit-isis prints the campt recipe
for exact registration; feed the result back with --refine-line/--refine-sample.
And the VERIFY score is triage for ordering human review, not a detection —
sunlit boulders produce the same bright-plus-shadow signature, which is the
entire reason the autoencoder exists.

🤝 Contributing
Contributions are welcome! Please:

Fork the repository
Create a feature branch (git checkout -b feature/amazing-feature)
Commit your changes (git commit -m 'Add amazing feature')
Push to the branch (git push origin feature/amazing-feature)
Open a Pull Request

Development Roadmap

 Multi-scale analysis (combine 64×64, 128×128, 256×256)
 Attention mechanisms for spatial localization
 Transfer learning from Earth satellite imagery
 Real-time inference optimization
 Web interface for visualization
 Support for multi-spectral imagery

📝 Citation
If you use this work in your research, please cite:
bibtex@software{lunar_anomaly_detection,
  author = {Strom et al},
  title = {Planetary Anomaly Detection Pipeline},
  year = {2026},
  url = {https://github.com/yourusername/lunar-anomaly-detection}
}
📄 License
This project is licensed under the MIT License - see the LICENSE file for details.
🙏 Acknowledgments

NASA Lunar Reconnaissance Orbiter (LRO) team for imagery
NASA TREK for public data access
Apollo missions for validation data
PyTorch and scikit-learn communities

📧 Contact
Your Name

Email: your.email@example.com
GitHub: @yourusername
Project Link: https://github.com/yourusername/lunar-anomaly-detection

🐛 Known Issues

Large datasets (>1GB) may require batch processing
GPU memory constraints with 256×256 chips (reduce batch size)
Some natural linear features (fault lines, rilles) may trigger false positives

⚡ Performance Tips

Use GPU: 10-20× faster training with CUDA
Adjust batch size: Maximize GPU utilization without OOM errors
Pre-filter chips: Skip low-variance chips during extraction
Use mixed precision: Enable for faster training on modern GPUs

python# Enable mixed precision training
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()
🔗 Related Projects

Planetary Computer
Mars Rover Image Analysis
Lunar Mapping Tools


⭐ Star this repository if you find it useful!
Last Updated: January 2026

