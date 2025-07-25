
# Efficient Semantic Segmentation of Radar Sounder Data

This repository provides code for performing efficient semantic segmentation on radar sounder data using the proposed **Efficient U²-Net** architecture. The model improves upon U²-Net by incorporating **Octave Convolution** to reduce redundancy and increase generalization in low-data scenarios.

---

## 🛰️ Overview

Radar sounders are used to map the subsurface of glaciers and planetary bodies. Deep learning has shown promise in analyzing such data, but high model complexity and limited labeled data remain challenges. This repository presents:

- **Efficient U²-Net**: A lightweight and accurate segmentation model.
- **Inference scripts** for standard and partitioned evaluation.
- **Support for datasets** like MCoRDS (Antarctica) and SHARAD (Mars).
- **Cross-validation and metrics** tracking per fold.

---

## 📄 Paper

**Title**: *Efficient Semantic Segmentation of Radar Sounder Data*  
**Authors**: Milkisa T. Yebasse and Lorenzo Bruzzone  
**Published in**: SPIE Conference Proceedings, Vol. 13196  
🔗 [DOI: 10.1117/12.3031783](https://www.spiedigitallibrary.org/conference-proceedings-of-spie/13196/131960G/Efficient-semantic-segmentation-of-radar-sounder-data/10.1117/12.3031783.short)

---

## 📁 Repository Structure

```
├── runu2net_test.py                  # Inference on a single radargram without partitioning
├── run_partition_u2net_test.py      # K-fold cross-validation with configurable dataset/model/partition
├── model/
│   └── model.py                      # U2NET and U2NETP architectures
├── data_loader/
│   └── data_loader.py               # Custom Dataset and Transform
├── implementation/
│   ├── metrics.py                   # Metric computation (precision, recall, accuracy)
│   ├── test.py                      # Per-fold inference logic
│   ├── output.py                    # Save predicted masks
│   └── dataset.py                   # Dataset configuration (e.g., MC10, SHARAD)
├── saved_models/
│   └── [pretrained_model.pth]       # Trained model checkpoints
├── test_data/
│   └── [results_dir/]               # Visualization of segmentation results
├── result_monthly.png               # Qualitative comparison of predictions
├── Efficent_semantic_segmentation_of_radar_sounder_data.pdf
└── README.md
```

---

## ⚙️ Setup

**Requirements:**
- Python 3.8+
- PyTorch ≥ 1.7
- torchvision
- numpy, pandas, matplotlib
- rasterio (if needed for your radargrams)

**Install dependencies:**
```bash
pip install -r requirements.txt
```

---

## 🚀 Usage

### A. Run Inference on a Fixed Radargram

```bash
python runu2net_test.py
```

This will:
- Load and preprocess a fixed radargram
- Run inference using a pretrained U²-NetP model
- Print per-class metrics (recall, precision, F1-score)
- Optionally save segmentation masks to `test_data/`

---

### B. Run Cross-Validation with Dataset Partitioning

```bash
python run_partition_u2net_test.py --dataset mc10 --model u2netp --partition 30_70
```

#### Arguments:
- `--dataset`: Dataset name, e.g., `mc10` or `sharad`
- `--model`: Model type, `u2net` or `u2netp`
- `--partition`: Train/test split ratio. Options:  
  - `70_30` (default): 70% training, 30% testing  
  - `30_70`: 30% training, 70% testing  
  - `10_90`: 10% training, 90% testing  

The script will:
- Load the selected dataset and split it according to the specified partition
- Run k-fold testing and print per-fold performance
- Print overall average metrics across folds

---

## 🖼️ Qualitative Results

The following figure shows a visual comparison of segmentation outputs on radargrams:

- **Column 1**: Original radargram  
- **Column 2**: Ground truth label  
- **Columns 3–6**: Predicted masks from different test folds

<p align="center">
  <img src="result_monthly.png" alt="Segmentation Results" width="750"/>
</p>

These results illustrate the model’s robustness and consistency across folds in identifying key subsurface structures such as **ice layers**, **bedrock**, and **noise**.

---

## 📊 Results Summary (From Paper)

| Model           | Augmentation | Recall  | Precision | F1-Score |
|----------------|--------------|---------|-----------|----------|
| Efficient U²-Net | ❌           | 0.9854  | 0.9818    | 0.9836   |
| Efficient U²-Net | ✅           | 0.9913  | 0.9840    | 0.9876   |

---

## 📌 Notes

- Model expects input patches of size 410×64 (range × azimuth).
- Ambiguous classes are ignored during training and evaluation.
- Color-mapped prediction masks can be saved using the `color_mapping()` function.
- The model is trained **from scratch**, not using pretrained backbones.

---

## 📚 Citation

```
@inproceedings{yebasse2025efficient,
  title = {Efficient semantic segmentation of radar sounder data},
  author = {Yebasse, Milkisa T. and Bruzzone, Lorenzo},
  booktitle = {Proc. SPIE 13196, Image and Signal Processing for Remote Sensing XXX},
  year = {2025},
  doi = {10.1117/12.3031783}
}
```

---

## 👤 Contact

For issues or collaboration:

**Milkisa T. Yebasse**  
PhD Researcher, University of Trento  
📧 milkisa.yebasse [at] unitn.it
