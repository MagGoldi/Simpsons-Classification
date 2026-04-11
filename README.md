# Simpsons Character Classification

> **Actual Public Score: `0.94580`**

Image classification pipeline for the [Journey to Springfield](https://www.kaggle.com/competitions/journey-springfield) Kaggle competition — 42-class character recognition from *The Simpsons* using a custom CNN with MLflow experiment tracking.

## Project Structure

```
simpsons_classification/
├── config.py                     # hyperparameters, paths, augmentation config
├── README.md
├── requirements.txt
├── .gitignore
│
├── src/                          # core library
│   ├── __init__.py
│   ├── dataset.py                # SimpsonsDataset, DataLoader factory
│   ├── models.py                 # SimpleCnn architecture
│   ├── trainer.py                # training loop, early stopping, MLflow logging
│   ├── metrics.py                # F1-score, per-class error analysis
│   ├── utils.py                  # data loading, label encoding
│   ├── visualization.py          # EDA, training curves, confusion matrices
│   └── logger.py                 # centralized logging setup
│
├── scripts/                      # entry points
│   ├── train.py                  # end-to-end training pipeline
│   ├── evaluate.py               # standalone model evaluation
│   └── submit_kaggle.py          # generate submission CSV
│
├── notebooks/                    # exploratory analysis
│   └── hw_5_1.ipynb
│
├── data/                         # auto-downloaded on first run
│   ├── train/
│   ├── testset/
│   └── checkpoints/
│
├── reports/                      # generated artifacts
│   ├── logs/                     # persistent log files (append-only)
│   ├── *.png                     # training & evaluation charts
│   ├── *.csv                     # error statistics, submission
│   └── *.json                    # evaluation summaries
│
└── mlruns/                       # MLflow tracking (auto-generated)
```

## Quick Start

### Prerequisites

- Python 3.12+
- CUDA-capable GPU (recommended)

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd simpsons_classification

# Install dependencies (using uv)
uv pip install -r requirements.txt

# Or with pip
pip install -r requirements.txt
```

### Training

```bash
uv run python scripts/train.py
```

This will:
1. Download and extract the dataset (first run only)
2. Generate EDA reports in `reports/`
3. Train the model with early stopping
4. Log all metrics to MLflow
5. Save the best checkpoint to `data/checkpoints/`
6. Generate post-training visualizations

### Evaluation

```bash
uv run python scripts/evaluate.py
```

Runs a full evaluation on the validation set and produces:
- Classification report (text + JSON)
- Confusion matrix
- Per-class error analysis
- Misclassified examples visualization

**Options:**

```bash
uv run python scripts/evaluate.py --checkpoint path/to/model.pth --output-dir reports/
```

### Kaggle Submission

```bash
uv run python scripts/submit_kaggle.py
```

Generates `reports/submission.csv` ready for Kaggle upload.

### MLflow UI

```bash
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001 --host 0.0.0.0
```

Open [http://localhost:5001](http://localhost:5001) to compare experiments.

## Model Architecture

**SimpleCnn** — lightweight 5-block convolutional network:

```
5 × (Conv2d → BatchNorm2d → ReLU → MaxPool2d) → Dropout(0.2) → Linear
```

- Input: `(B, 3, 224, 224)` RGB images
- Output: `(B, 42)` class logits
- Augmentations: random horizontal flip, rotation, affine translation
- Normalization: ImageNet statistics

## Training Configuration

| Parameter       | Value       |
|-----------------|-------------|
| Optimizer       | Adam        |
| Learning Rate   | 1e-3        |
| Weight Decay    | 1e-4        |
| Batch Size      | 64          |
| Scheduler       | ReduceLROnPlateau |
| Early Stopping  | patience=7  |
| Image Size      | 224 × 224   |

## Tech Stack

| Library        | Version   |
|----------------|-----------|
| Python         | 3.12.3    |
| PyTorch        | 2.11.0    |
| torchvision    | 0.26.0    |
| MLflow         | 3.11.1    |
| scikit-learn   | 1.8.0     |
| matplotlib     | 3.10.8    |
| seaborn        | 0.13.2    |
| pandas         | 2.3.3     |
| NumPy          | 2.4.3     |
| Pillow         | 12.1.1    |
| tqdm           | 4.67.3    |
| gdown          | 5.2.1     |

## Reports

All artifacts are saved to `reports/`:

| File | Description |
|------|-------------|
| `01_augmentations.png` | Sample augmented training images |
| `02_sample_images.png` | Random grid from the dataset |
| `03_training_history.png` | Loss & accuracy curves |
| `04_predictions_grid.png` | Model predictions with confidence |
| `05_error_statistics.csv` | Per-class error rates |
| `06_error_analysis.png` | Error distribution charts |
| `07_confusion_matrix.png` | Top-error confusion matrix |
| `08_misclassified_examples.png` | Misclassified image samples |
| `training_history.csv` | Epoch-level metrics |
| `logs/app.log` | Persistent application log |

## License

This project is intended for educational and competition purposes.
