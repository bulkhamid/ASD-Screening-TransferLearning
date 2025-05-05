# ASD Screening using Deep Learning and Transfer Learning on Video Data

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the code for a deep learning project aimed at early screening for Autism Spectrum Disorder (ASD) using video data. The approach leverages transfer learning and integrates spatial and temporal feature analysis from videos. The project explores different model architectures and training strategies, including cross-validation, regularization, and model calibration.

## Table of Contents

1.  [Overview](#overview)
2.  [Key Features](#key-features)
3.  [Repository Structure](#repository-structure)
4.  [Methodology](#methodology)
    * [Data Preparation](#data-preparation)
    * [Model Architectures](#model-architectures)
    * [Training](#training)
    * [Evaluation & Calibration](#evaluation--calibration)
5.  [Setup Instructions](#setup-instructions)
6.  [Usage](#usage)
    * [Data Preprocessing](#1-data-preprocessing-datasave_numpy_datasetpy)
    * [Training (Cross-Validation or Hold-Out)](#2-training-trainpy)
    * [Regularization Experiments](#3-regularization-experiments-experimentstrain_regularizedpy)
    * [Label Shuffle Sanity Check](#4-label-shuffle-sanity-check-experimentsdebug_shuffle_labelspy)
    * [Inspecting Data Splits](#5-inspecting-data-splits-notebooksinspect_splitipynb)
7.  [Experiments and Results](#experiments-and-results)
8.  [Literature Review](#literature-review)
9.  [Branches](#branches)
10. [License](#license)

## Overview

Early detection of Autism Spectrum Disorder (ASD) is crucial for timely intervention. This project investigates the use of deep learning models to screen for ASD based on analyzing videos of children's behavior. We utilize techniques like 3D Convolutional Neural Networks (CNNs) and Recurrent Neural Networks (LSTMs), along with transfer learning from pre-trained models (R3D-18), to capture relevant spatio-temporal patterns. The goal is to develop a robust and reliable automated screening tool. While the initial README mentioned integrating gaze features, the current implementation focuses primarily on spatial and temporal information extracted directly from video frames.

## Key Features

* **Models Implemented:**
    * `Simple3DLSTM`: A custom model combining 3D CNN layers for spatial feature extraction across time and an LSTM layer for temporal sequence modeling. Includes optional 3D dropout.
    * `PretrainedR3D`: Utilizes a ResNet-based 3D CNN (R3D-18) pre-trained on the Kinetics-400 action recognition dataset, adapted for binary classification. Supports head-only training and fine-tuning (unfreezing deeper layers).
* **Data Handling:**
    * Supports training directly from raw video files (`.mp4`, `.webm`, `.avi`).
    * Supports training from pre-processed, cached `.npz` tensors for faster experimentation.
    * Includes a script (`data/save_numpy_dataset.py`) for preprocessing raw videos into `.npz` files, featuring:
        * **Deduplication:** Uses SHA1 hashing on the first 1MB of video files to detect and handle exact duplicates, preventing data leakage across splits. Reports conflicting duplicates (same video, different labels).
        * **Frame Sampling/Padding:** Selects a fixed number of frames (`MAX_FRAMES`) randomly or uses all frames if fewer exist, padding if necessary.
        * **Splitting:** Creates stratified train/validation/test splits (approx. 70%/10%/20%).
* **Training Strategies:**
    * **Cross-Validation:** Supports 5-fold `GroupKFold` cross-validation, ensuring subjects are not split across train/validation/test sets within a fold (`train.py --cv`).
    * **Hold-Out Validation:** Supports a single 70%/10%/20% train/validation/test split (`train.py` default).
    * **Optimization:** Uses AdamW optimizer with Cosine Annealing learning rate scheduler.
    * **Mixed Precision:** Leverages Automatic Mixed Precision (AMP) via `torch.amp` and `GradScaler` for faster training and reduced memory usage.
    * **Regularization:** Includes weight decay (AdamW) and optional 3D Dropout (for `Simple3DLSTM`).
    * **Early Stopping:** Monitors validation loss and stops training if no improvement is seen after a defined patience (`--patience`), saving the best model checkpoint.
* **Evaluation & Calibration:**
    * **Metrics:** Computes standard classification metrics (AUC, Accuracy, Precision, Recall, F1-Score, BCE Loss).
    * **Thresholding:** Automatically selects the optimal classification threshold based on the best F1-score achieved on the validation set.
    * **Calibration:** Implements optional Temperature Scaling (`--temp`) to calibrate model prediction probabilities using the validation set (`src/utils/temperature_scaling.py`).
    * **Visualization:** Generates and saves plots for training/validation loss and AUC curves, as well as ROC curves for each fold/run.
* **Experiment Management:**
    * Integrates with Weights & Biases (`wandb`) for logging metrics, configurations, and results (requires setup).
    * Includes scripts for specific experiments (regularization tuning, sanity checks).
* **Sanity Check:** Provides a label shuffling script (`experiments/debug_shuffle_labels.py`) to verify that the model cannot learn effectively when labels are randomized, helping to rule out data leakage or implementation bugs.

## Repository Structure
```
└── bulkhamid-asd-screening-transferlearning/
    ├── README.md               # This file
    ├── LICENSE                 # Project license (MIT)
    ├── requirements.txt        # Python dependencies
    ├── train.py                # Main training script (CV or Hold-out, Raw or Cached Data)
    ├── data/                   # Data handling scripts and potentially raw/processed data
    │   ├── __init__.py
    │   ├── dataloader.py       # Defines VideoDataset and augmentations for raw videos
    │   └── save_numpy_dataset.py # Script to preprocess videos into cached .npz files (handles deduplication)
    ├── docs/                   # Documentation
    │   ├── LiteratureReview.md # Summary of related research papers
    │   └── references.bib      # Bibliography for the literature review
    ├── experiments/            # Experiment scripts, logs, and results
    │   ├── __init__.py
    │   ├── debug_shuffle_labels.py # Sanity check script (trains on shuffled labels)
    │   ├── reg_results_r3d.txt     # Results from regularization experiments on R3D
    │   ├── reg_results_simple3dlstm.txt # Results from regularization experiments on Simple3DLSTM
    │   ├── shuffling_results.txt # Output logs from the label shuffle check
    │   ├── train results         # Detailed logs from train.py runs (CV results)
    │   └── train_regularized.py  # Script for focused regularization experiments (uses cached data)
    ├── notebooks/              # Jupyter notebooks for exploration
    │   └── inspect_splits.ipynb # Notebook to visually inspect video frames in .npz splits
    ├── plots/                  # Directory where training plots (loss, AUC, ROC) are saved
    └── src/                    # Source code
        ├── __init__.py
        ├── models/             # Model definitions
        │   ├── __init__.py
        │   ├── pretrained_r3d.py # Defines the PretrainedR3D model wrapper
        │   └── simple3dlstm.py # Defines the Simple3DLSTM model
        └── utils/              # Utility functions and classes
            ├── __init__.py
            ├── data_cached.py  # Data loading and augmentation pipeline for cached .npz data
            ├── earlystop.py    # EarlyStopper class
            ├── metrics.py      # Function to compute evaluation metrics
            ├── temperature_scaling.py # TemperatureScaler class for calibration
            └── wandb_logger.py # Helper for wandb logging (though train.py logs directly)
```

## Methodology

The project follows these general steps:

### Data Preparation

1.  **Collection:** Raw video data is expected in a root directory (e.g., `dataset/`) with subfolders `positive/` and `negative/` containing videos corresponding to ASD positive and negative cases, respectively.
2.  **Preprocessing (`data/save_numpy_dataset.py`):**
    * Videos are scanned, and hashes are computed (first 1MB) to identify duplicates.
    * Unique videos are selected, handling potential label conflicts for duplicates.
    * A fixed number of frames (`MAX_FRAMES`) are sampled/padded per video.
    * Frames are resized to `TARGET_SIZE`.
    * Data is split into stratified train/validation/test sets (approx. 70%/10%/20%).
    * Each split is saved as a compressed NumPy file (`data/train.npz`, `data/val.npz`, `data/test.npz`) containing video tensors (`X`) of shape `(N, T, H, W, C)` (uint8) and labels (`y`).
3.  **Loading:**
    * **Raw Video Pipeline (`data/dataloader.py`):** Reads video files on-the-fly, performs frame sampling/padding, applies augmentations (random crop, jitter, flip, rotation, erasing, normalization). Used by `train.py` when `--cached` is *not* specified.
    * **Cached Data Pipeline (`src/utils/data_cached.py`):** Loads `.npz` files using memory mapping (`mmap_mode`), permutes tensor dimensions to `(N, C, T, H, W)`, applies augmentations via a custom `collate_fn`, and normalizes. Used by `train.py --cached`, `train_regularized.py`, and `debug_shuffle_labels.py`.

### Model Architectures

* **Simple3DLSTM (`src/models/simple3dlstm.py`):**
    * Uses a stack of 3D Convolutional layers with ReLU, Max Pooling, Batch Normalization, and optional 3D Dropout to extract spatio-temporal features.
    * Flattens the features for each time step and feeds them into an LSTM layer to model temporal dependencies.
    * Uses the final hidden state of the LSTM as input to a linear classification head (outputting raw logits).
* **Pretrained R3D-18 (`src/models/pretrained_r3d.py`):**
    * Loads the R3D-18 model pre-trained on Kinetics-400 from `torchvision`.
    * Replaces the final fully connected layer with a new `nn.Linear` layer for binary classification (outputting raw logits).
    * Supports two training modes via `train.py`:
        1.  **Head-Only:** Freezes all backbone layers and trains only the final classification layer.
        2.  **Fine-Tuning (`--finetune`):** After an initial head-only training phase, unfreezes the later layers (e.g., `layer4` and `fc`) and trains them with a smaller learning rate.

### Training

* The `train.py` script orchestrates the training process.
* It can perform either k-fold cross-validation (`--cv`) based on subject groups or a single hold-out split.
* Uses AdamW optimizer and Cosine Annealing LR scheduler.
* Employs Automatic Mixed Precision (AMP) for efficiency.
* Implements early stopping based on validation loss.
* Logs metrics using `wandb` (if configured).
* Saves best model checkpoints and plots.

### Evaluation & Calibration

* After training, the best model (based on validation loss) is loaded.
* **Calibration (Optional):** If `--temp` is enabled, `TemperatureScaler` learns an optimal temperature `T` on the validation set logits to improve probability calibration.
* **Threshold Selection:** The optimal decision threshold is determined by maximizing the F1-score on the (potentially calibrated) validation set predictions.
* **Final Evaluation:** The model (potentially calibrated) and the chosen threshold are used to evaluate performance on the held-out test set. Metrics (AUC, Acc, Prec, Rec, F1) are reported.
* For cross-validation, test metrics are aggregated across folds (mean ± std).

## Setup Instructions

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/bulkhamid/ASD-Screening-TransferLearning.git](https://github.com/bulkhamid/ASD-Screening-TransferLearning.git)
    cd ASD-Screening-TransferLearning
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    # Using conda
    conda create -n asd_screening python=3.9 # Or your preferred Python version
    conda activate asd_screening

    # Or using venv
    python -m venv venv
    source venv/bin/activate # On Windows use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **(Optional) Weights & Biases Setup:** If you want to use `wandb` logging, make sure you have an account and are logged in:
    ```bash
    pip install wandb
    wandb login
    ```
    The `train.py` script will automatically log runs if `wandb` is installed and configured. You might need to uncomment the `wandb` import and calls in the script if they are commented out.

5.  **Prepare Data:**
    * Place your raw video data in a directory structure like `dataset/positive/` and `dataset/negative/`. Update the path in `data/save_numpy_dataset.py` or modify the script to accept a path argument.
    * Run the preprocessing script (see [Usage](#1-data-preprocessing-datasave_numpy_datasetpy)) to generate the `.npz` files in the `data/` directory.

## Usage

### 1. Data Preprocessing (`data/save_numpy_dataset.py`)

This script converts raw videos into cached `.npz` files for faster training and performs deduplication.

```bash
# Example: Specify the root directory containing 'positive' and 'negative' subfolders
# First, modify the hardcoded path in the script or adapt it to use command-line arguments.
# Then run:
python data/save_numpy_dataset.py 
```
- This will create `data/train.npz`, `data/val.npz`, and `data/test.npz`.

### 2. Training (`train.py`)

This is the main script for training models using either cross-validation or a hold-out split, and optionally using raw or cached data.

**Common Arguments:**

* `--model`: Choose the model (`simple3dlstm` or `r3d`). (Required)
* `--data_path`: Path to the directory containing raw videos or the generated `.npz` files (defaults to `data/`).
* `--cached`: Use pre-processed `.npz` files (requires running `save_numpy_dataset.py` first).
* `--cv`: Perform 5-fold Group Cross-Validation (default is True in the script). Use `--no-cv` or similar based on argparse implementation if you want hold-out.
* `--splits`: Number of CV splits (default: 5).
* `--batch_size`: Training batch size (default: 8).
* `--epochs`: Total number of epochs (default: 50). For R3D, this is split between head-only and fine-tuning stages if `--finetune` is used.
* `--patience`: Epochs to wait for improvement before early stopping (default: 10).
* `--wd`: Weight decay for AdamW optimizer (default: 1e-4).
* `--temp`: Enable temperature scaling for calibration.
* `--max_frames`: Maximum frames to sample per video (default: 60).
* `--target_size`: Target HxW resolution for frames (default: 112 112).
* `--finetune`: For `r3d` model, unfreeze `layer4` and `fc` for fine-tuning after the head-only phase.
* `--drop`: 3D dropout probability for `simple3dlstm` (default: 0.0).

**Examples:**

```bash
# Train Simple3DLSTM using 5-fold CV with cached data and temperature scaling
python train.py --model simple3dlstm --cached --cv --temp --epochs 50 --patience 10 --batch_size 8

# Train R3D (head-only) using a single hold-out split with cached data
# (Assuming --cv is default True, you might need to add a flag like --no-cv if implemented, or modify the script default)
python train.py --model r3d --cached --no-cv --epochs 50 --patience 10 --batch_size 8 # Add --no-cv if needed

# Train R3D with fine-tuning using 5-fold CV with cached data and temperature scaling
python train.py --model r3d --cached --cv --finetune --temp --epochs 50 --patience 10 --batch_size 8

# Train Simple3DLSTM directly from raw videos (slower) with dropout
# (Assumes raw videos are in 'data/positive' and 'data/negative')
# python train.py --model simple3dlstm --cv --drop 0.3 --epochs 50 --patience 10 --batch_size 4 --data_path data
```
- Plots and checkpoints will be saved in `plots/` and `checkpoints/` respectively.
- Final cross-validation results are printed to the console and logged to `wandb`.

### 3. Regularization Experiments (experiments/train_regularized.py)
This script appears tailored for running quick experiments on a single hold-out split using cached data only, focusing on the effects of regularization (weight decay, dropout) and calibration (temperature scaling). It was used to generate the results in `experiments/reg_results_*.txt`.

```bash
# Example: Test R3D with fine-tuning, specific weight decay, and temperature scaling
python experiments/train_regularized.py --model r3d --finetune --wd 5e-4 --temp --epochs 50 --patience 10

# Example: Test Simple3DLSTM with 30% dropout and temperature scaling
python experiments/train_regularized.py --model simple3dlstm --drop 0.3 --temp --epochs 25 --patience 10
```

### 4. Label Shuffle Sanity Check (experiments/debug_shuffle_labels.py)
Run this script to ensure the model performs near random chance when trained on shuffled labels. This helps detect potential data leakage or fundamental issues in the pipeline. It uses cached data only.

```bash
# Run for Simple3DLSTM
python experiments/debug_shuffle_labels.py --model simple3dlstm --epochs 10

# Run for R3D
python experiments/debug_shuffle_labels.py --model r3d --epochs 10
```

- Expect AUC scores around 0.5 and loss around 0.69 (log(2)). The script will print a warning if AUC is significantly higher. Results are logged in `experiments/shuffling_results.txt`.

### 5. Inspecting Data Splits (notebooks/inspect_splits.ipynb)

Open and run this Jupyter notebook to visually check frames from the generated `.npz` files and verify the class balance within each split.

```bash
# Ensure you have jupyter installed (pip install jupyter notebook)
jupyter notebook notebooks/inspect_splits.ipynb
```

## Experiments and Results

Several experiments have been conducted and logged in the `experiments/` directory.

* **Cross-Validation Performance (`experiments/train results`):**
    * **`Simple3DLSTM`:** Achieved strong performance with 5-fold CV using cached data, resulting in a mean Test AUC of 1.000 ± 0.000 and F1-score of 0.991 ± 0.017. Temperature scaling was applied.
    * **`R3D (Fine-tuned)`:** Also demonstrated excellent performance with 5-fold CV using cached data and fine-tuning, achieving perfect mean Test metrics (AUC=1.0, F1=1.0, Acc=1.0) across folds. Temperature scaling was applied.

* **Regularization Effects (`experiments/reg_results_*.txt`):**
    * For **`R3D`**, fine-tuning (`--finetune`) significantly boosted performance compared to head-only training, reaching perfect test scores on the hold-out split used in these experiments. Temperature scaling (`--temp`) consistently improved calibration (lower validation loss after scaling) without hurting discriminative performance (AUC). Performance was robust across tested weight decay values (`1e-4`, `5e-5`, `5e-4`) when fine-tuning.
    * For **`Simple3DLSTM`**, adding 3D dropout (`--drop`) seemed detrimental, significantly reducing performance, especially with higher dropout rates (0.3, 0.4). The best results were obtained with no dropout (`--drop 0.0`). Temperature scaling (`--temp`) again proved beneficial, improving the F1 score (0.909 -> 0.957) and accuracy (0.909 -> 0.955) for the no-dropout case.

* **Sanity Check (`experiments/shuffling_results.txt`):**
    * The label shuffle experiment confirmed that both `Simple3DLSTM` and `R3D` models performed close to random chance (Test AUCs of 0.500 and 0.570, respectively) when trained on randomly shuffled labels. This increases confidence that the models are learning meaningful patterns from the data and not exploiting data leakage or biases.

*For detailed epoch-by-epoch logs, command lines, and specific metrics, please refer to the files within the `experiments/` directory.*

## Literature Review

A review of relevant academic papers on video-based autism screening, deep learning approaches, transfer learning, and multimodal systems can be found in `docs/LiteratureReview.md`. The corresponding bibliography is available in `docs/references.bib` (marked as needing updates).

## Branches

* **`master`**: Finalized code.
* **`Zhamshidbek-updates`**: Updates and experiments from Zhamshidbek.
* **`Zhansaya-updates`**: Updates and experiments from Zhansaya.
* **`Saya-updates`**: Updates and experiments from Saya.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.