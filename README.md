# In Silico Retrosynthetic Design of Fluorinated Dihydroartemisinin Analogues

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)](https://pytorch.org/)

> An AI-driven computational framework for designing novel fluorinated antimalarial drug candidates to overcome artemisinin resistance in *Plasmodium falciparum*.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Dataset](#dataset)
- [Usage](#usage)
- [Methodology](#methodology)
- [Results](#results)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Citation](#citation)
- [Authors](#authors)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## 🔬 Overview

Malaria remains a major global health burden, with increasing artemisinin resistance threatening current treatment strategies. This project implements a **machine learning-based drug design pipeline** to:

1. **Mine antimalarial bioassay data** from PubChem
2. **Train QSAR models** using molecular fingerprints
3. **Design fluorinated analogues** of dihydroartemisinin (DHA)
4. **Predict antimalarial activity** of novel compounds

By integrating cheminformatics and deep learning, this framework enables rapid virtual screening of drug candidates before expensive synthesis and testing.

## ✨ Features

- **Automated Data Collection**: PubChem bioassay mining and preprocessing
- **Molecular Fingerprinting**: Morgan (ECFP) fingerprint generation using RDKit
- **Neural Network QSAR Model**: PyTorch-based regression for IC₅₀ prediction
- **Systematic Fluorination**: Algorithm for generating fluorinated analogues
- **Physicochemical Analysis**: LogP, molecular weight, TPSA calculations
- **Visualization Tools**: Automated generation of publication-quality figures
- **Early Stopping**: Prevent overfitting with validation-based training control

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- (Optional) CUDA-capable GPU for faster training

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/fluorinated-dha-antimalarial.git
cd fluorinated-dha-antimalarial
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

### Google Colab Setup

Alternatively, run the notebook directly in Google Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yourusername/fluorinated-dha-antimalarial/blob/main/TermPaper.ipynb)

## 📊 Dataset

### Source
- **PubChem Bioassays**: IC₅₀ measurements against *Plasmodium falciparum*
- **Compounds**: Artemisinin, chloroquine, mefloquine, quinine, primaquine, atovaquone, and related antimalarials

### Data Files
- `pubchem_antimalarial_with_ic50.csv` - Curated dataset with SMILES, CID, IC₅₀ values, and molecular descriptors

### Data Preprocessing
- Remove entries with missing SMILES or IC₅₀
- Filter for exact measurements (standard relation = '=')
- Validate molecular structures with RDKit
- Calculate physicochemical descriptors

## 💻 Usage

### Quick Start

```python
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
import torch
from qsar_model import QSARModel

# Load trained model
model = QSARModel()
model.load_state_dict(torch.load('trained_qsar_model.pth'))
model.eval()

# Generate fingerprint for your molecule
smiles = "YOUR_SMILES_HERE"
mol = Chem.MolFromSmiles(smiles)
mfgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)
fp = mfgen.GetFingerprint(mol)

# Predict IC50
with torch.no_grad():
    prediction = model(torch.tensor(fp).float().unsqueeze(0))
    print(f"Predicted IC50: {prediction.item():.3f}")
```

### Running the Full Pipeline

1. **Data Collection**
```bash
python scripts/collect_pubchem_data.py
```

2. **Train QSAR Model**
```bash
python scripts/train_qsar_model.py --epochs 110 --lr 0.0005
```

3. **Generate Fluorinated Analogues**
```bash
python scripts/generate_analogues.py --input dha_smiles.txt
```

4. **Predict Activity**
```bash
python scripts/predict_activity.py --analogues fluorinated_dha.csv
```

### Jupyter Notebook

Open `TermPaper.ipynb` for an interactive walkthrough of the entire pipeline with visualizations.

## 🧪 Methodology

### 1. Data Collection & Preprocessing
- PubChem bioassay mining
- SMILES validation with RDKit
- Descriptor calculation (MolWt, LogP, TPSA)

### 2. Molecular Fingerprinting
- **Algorithm**: Morgan (ECFP) with radius=2
- **Size**: 1024 bits
- **Encoding**: Circular substructures around each atom

### 3. QSAR Model Architecture

```
Input Layer (1024) 
    ↓
Hidden Layer 1 (64) + ReLU
    ↓
Hidden Layer 2 (32) + ReLU
    ↓
Output Layer (1) → IC₅₀ prediction
```

**Training Parameters:**
- Loss: Mean Squared Error (MSE)
- Optimizer: Adam (lr=0.0005)
- Train/Test Split: 80/20
- Early Stopping: Patience=20

### 4. Fluorination Strategy
- Identify non-aromatic carbons with available H atoms
- Systematic H→F substitution
- Chemical validity verification
- Drug-likeness filtering (MW < 600 Da)

### 5. Evaluation Metrics
- IC₅₀ prediction (lower is better)
- LogP (lipophilicity)
- Molecular weight
- TPSA (permeability)

## 📈 Results

### Model Performance
- **Training Loss**: 3.37 → 2.05 (110 epochs)
- **Test Loss**: Stabilized at ~2.79
- **Status**: Converged without significant overfitting

### Predicted IC₅₀ Values

| Compound | IC₅₀ | Change vs DHA |
|----------|------|---------------|
| DHA (baseline) | 0.301 | - |
| Fluoro-1 | 0.486 | +61% |
| Fluoro-2 | 0.597 | +98% |
| Fluoro-3 | 0.668 | +122% |

### Key Findings
✅ QSAR model successfully learned structure-activity relationships  
✅ Fluorination systematically altered physicochemical properties  
✅ Fluoro-1 showed smallest activity decrease (most promising)  
✅ All analogues maintained nanomolar-range activity  
⚠️ Fluorination reduced predicted activity vs parent compound

### Physicochemical Properties

**LogP Changes:**
- DHA: 0.85
- Fluorinated analogues: 0.78-0.79 (slight decrease)

**Molecular Weight:**
- DHA: 190 g/mol
- Fluorinated analogues: 204-205 g/mol (+7-8%)

## 📁 Project Structure

```
fluorinated-dha-antimalarial/
├── README.md
├── requirements.txt
├── LICENSE
├── TermPaper.ipynb              # Main Jupyter notebook
├── term_paper_ieee_fixed.tex    # IEEE conference paper
│
├── dataset/
│   └── pubchem_antimalarial_with_ic50.csv
│
├── images/
│   ├── TrainTest_Comparison.png
│   ├── dha_structure.png
│   ├── retrosynthesis_placeholder.png
│   ├── IC50_Comparison.png
│   ├── logp_comparison.png
│   ├── molwt_comparison.png
│   └── ic50_comparison__1_.png
│
├── models/
│   ├── qsar_model.py            # PyTorch model definition
│   └── trained_qsar_model.pth   # Trained model weights
│
├── scripts/
│   ├── collect_pubchem_data.py
│   ├── train_qsar_model.py
│   ├── generate_analogues.py
│   └── predict_activity.py
│
└── utils/
    ├── fingerprint.py           # Fingerprint generation utilities
    ├── descriptors.py           # Molecular descriptor calculations
    └── visualization.py         # Plotting functions
```

## 📦 Requirements

```txt
python>=3.8
torch>=1.9.0
rdkit>=2021.09.1
pandas>=1.3.0
numpy>=1.21.0
matplotlib>=3.4.0
pubchempy>=1.0.4
scikit-learn>=0.24.0
jupyter>=1.0.0
```

## 📝 Citation

If you use this code or methodology in your research, please cite:

```bibtex
@inproceedings{alaba2025fluorinated,
  title={In Silico Retrosynthetic Design of a Novel Fluorinated Dihydroartemisinin Analogue as a Potential Antimalarial Lead to Overcome Drug Resistance},
  author={Alaba, Francesca Audrey L. and Cataluna, Geraldyn A.},
  booktitle={IEEE Conference Proceedings},
  year={2025},
  organization={University of Cebu}
}
```

## 👥 Authors

**Francesca Audrey L. Alaba**  
College of Computer Studies  
University of Cebu - Main Campus  
📧 francescaaudreyalaba26@gmail.com

**Geraldyn A. Cataluna**  
College of Computer Studies  
University of Cebu - Main Campus  
📧 geraldyncataluna2nd@gmail.com

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **RDKit** - Open-source cheminformatics toolkit
- **PyTorch** - Deep learning framework
- **PubChem** - National Library of Medicine database
- **World Health Organization** - Malaria statistics and research
- **Global scientific community** - Open-source tools and datasets

## 🔮 Future Work

- [ ] Experimental validation of predicted compounds
- [ ] Multi-target docking studies
- [ ] ADMET prediction integration
- [ ] Molecular dynamics simulations
- [ ] Resistance mutation profiling
- [ ] Expanded training dataset
- [ ] Transfer learning from larger chemical libraries
- [ ] 3D structural feature incorporation

## 🐛 Issues & Contributions

Found a bug or have a suggestion? Please open an issue on GitHub:

- **Bug Reports**: Include Python version, error messages, and reproduction steps
- **Feature Requests**: Describe the proposed enhancement
- **Pull Requests**: Welcome! Please follow PEP 8 style guidelines

## 📞 Contact

For questions or collaboration inquiries:
- Open an issue on GitHub
- Email: francescaaudreyalaba26@gmail.com

---

**⚠️ Disclaimer**: This is a computational research project. Predicted compounds require experimental validation before any therapeutic use. No medical claims are made.

**🌍 Impact**: This work contributes to global efforts in combating drug-resistant malaria through computational drug discovery.

---

<div align="center">
Made with ❤️ for advancing antimalarial drug discovery
</div>
