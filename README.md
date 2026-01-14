# FedBKD: Federated Knowledge Distillation for Glaucoma Diagnosis

This project implements a glaucoma diagnosis system based on federated learning and knowledge distillation, supporting collaborative training and privacy protection for multi-center medical imaging data.

## Project Overview

### Core Features

- **Federated Learning Architecture**: Supports multiple medical institutions to collaboratively train models without sharing original data
- **Knowledge Distillation Technology**: Implements knowledge transfer between heterogeneous models through teacher-student framework
- **Multi-modal Fusion**: Supports joint modeling of multiple modal data such as fundus images (OCT), fundus photographs (CLI), visual field examination (VF)
- **Multi-dataset Support**: Compatible with multiple glaucoma datasets including gamma, papila, zhongshan, gongli, airogs

### Algorithm Flow

1. **Local Forward Knowledge Distillation**: Each client uses local multi-modal models to guide the training of unimodal fundus models
2. **Global Model Aggregation**: The server aggregates all clients' unimodal models through federated averaging algorithm
3. **Reverse Knowledge Distillation**: Distill the knowledge of the global model back to the local multi-modal model to enhance its performance
4. **Iterative Optimization**: Repeat the above process until the preset number of communication rounds is reached

## Directory Structure

```
FedBKD/
├── models.py          # Model definitions, including various network architectures
├── utils.py           # Utility functions, including evaluation metrics, federated averaging, etc.
├── dataloader.py      # Data loading module, supporting multi-modal data
├── FedBKD_train.py    # Main training script
├── test.py            # Model testing script
├── run.sh             # Training run script
├── data/              # Data directory
│   ├── Train/         # Training data
│   └── Test/          # Testing data
├── checkpoint/        # Model checkpoint directory
└── README.md          # this documentation
```

## Installation

### Python Version Requirements

- Python 3.7+

### Experimental Environment (Reference)

```
Python 3.9
PyTorch 1.13.1+cu117
NumPy 1.26.4
Pandas 2.3.3
Scikit-learn 1.6.1
```

### Package Dependencies

```bash
pip install torch torchvision
pip install numpy pandas
pip install scikit-learn
pip install tqdm
```

## Quick Start

### Basic Training

```bash
python FedBKD_train.py \
    --num_rounds 50 \
    --local_epochs 2 \
    --batch_size 16 \
    --lr 1e-4
```

### Enable Early Stopping and Save Only Best Models

```bash
python FedBKD_train.py \
    --num_rounds 50 \
    --local_epochs 2 \
    --batch_size 16 \
    --lr 1e-4 \
    --best_only
```

**Early Stopping Mechanism Explanation**:
- When `--best_only` is enabled, the system calculates the mean of monitoring metrics across all participating datasets
- Only saves models when the mean improves
- Stops training when the mean doesn't improve for `--patience` consecutive rounds
- The monitoring metric can be selected via the `--monitor_metric` parameter (default: auroc)

### Multi-seed Experiments

```bash
python FedBKD_train.py \
    --run_multiple_seeds \
    --num_seeds 5 \
    --seed 42 \
    --best_only
```

## Parameter Description

### Federated Learning Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--num_rounds` | 30 | Number of communication rounds for federated learning |
| `--local_epochs` | 2 | Number of local training epochs for each client |
| `--kd_epochs` | 1 | Number of training epochs for knowledge distillation |

### Dataset Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--datasets` | gamma,zhongshan,gongli,airogs_* | List of datasets participating in federated learning |
| `--batch_size` | 16 | Batch size |
| `--num_workers` | 2 | Number of worker processes for data loading |

### Model Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--backbone` | resnet | Model backbone network (resnet/mobilenet) |
| `--num_classes` | 2 | Number of classes |

### Knowledge Distillation Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--temperature` | 4.0 | Temperature parameter for knowledge distillation |
| `--alpha` | 0.7 | Loss balancing coefficient for knowledge distillation |
| `--beta` | 0.5 | Loss weight for reverse knowledge distillation |

### Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--lr` | 1e-4 | Learning rate |
| `--weight_decay` | 1e-5 | Weight decay coefficient |
| `--seed` | 42 | Random seed |
| `--output_dir` | ./checkpoint | Output directory for training results and model checkpoints |
| `--best_only` | False | Enable early stopping mechanism and only save the best performing models |
| `--patience` | 5 | Patience value for early stopping, i.e., number of consecutive rounds with no performance improvement before stopping training |
| `--monitor_metric` | auroc | Metric to monitor for early stopping, options: auroc, kappa, f1 |
| `--monitor_dataset` | gamma | Dataset to monitor for early stopping (deprecated, now calculates mean across all participants) |

## Model Architecture

### Unified Model Interface

This project uses a unified model creation interface, dynamically creating required models through the `get_model()` function based on parameters.

```python
from models import get_model

# Parameter description:
# - model_type: Model type ('mm' for Multi-Modal / 'um' for Uni-Modal)
# - backbone: Backbone network ('resnet' / 'mobilenet')
# - dataset: Dataset name ('gamma' / 'papila' / 'zhongshan' / 'gongli' / 'airogs')
# - num_classes: Number of classes (default 2)

# Multi-modal model (ResNet18, gamma dataset)
mm_model = get_model(model_type='mm', backbone='resnet', dataset='gamma', num_classes=2)

# Single-modal model (MobileNetV2, zhongshan dataset)
um_model = get_model(model_type='um', backbone='mobilenet', dataset='zhongshan', num_classes=2)
```

### Model Types

| model_type | Description | Supported Datasets |
|------------|-------------|--------------------|
| `mm` | Multi-modal model, supporting joint modeling of fundus images, OCT and other modal data | gamma, papila, zhongshan, gongli |
| `um` | Uni-modal model, only using fundus images for diagnosis | gamma, zhongshan, gongli, airogs |

### Backbone Networks

| backbone | Description |
|----------|-------------|
| `resnet` | ResNet18 standard residual network |
| `mobilenet` | MobileNetV2 lightweight network |

### Internal Model Classes

The unified interface `get_model()` internally calls the following classes:

| Class Name | Description |
|------------|-------------|
| `Fundus_CrossAttention` | Single-modal fundus image classification model (unified implementation) |
| `GAMMA_Dual_CrossAttention` | GAMMA dual-modal fusion model |
| `Zhongshan_Dual_CrossAttention` | Zhongshan dual-modal fusion model |
| `Gongli_Dual_CrossAttention` | Gongli dual-modal fusion model |
| `LightweightOCTBranch` | Lightweight OCT branch |
| `CrossAttentionFusion` | Cross-attention fusion module |
| `MultiModalFusion` | Multi-modal fusion module |
| `GatedFusion` | Gated fusion module |

## Data Format

### Training Data Directory Structure

```
data/
├── Train/
│   ├── Train_gamma_0001.pkl
│   ├── Train_gamma_0002.pkl
│   └── ...
└── Test/
    ├── Test_gamma_0001.pkl
    ├── Test_gamma_0002.pkl
    └── ...
```

### PKL File Format

Each .pkl file contains the following data keys:

| Key | Data Type | Description |
|-----|-----------|-------------|
| `fundus` | Tensor | Fundus image data |
| `oct` | Tensor | OCT image data |
| `cli` | Tensor | Fundus photograph data |
| `oct_dev` | Tensor | OCT_dev image data |
| `oct_pie` | Tensor | OCT_pie image data |
| `vf` | Tensor | Visual field examination data |
| `oct_gl` | Tensor | Glaucoma OCT data |
| `label` | Tensor | Classification label |

## Evaluation Metrics

The system supports the following evaluation metrics:

- **AUROC**: Area Under the Receiver Operating Characteristic Curve
- **Kappa**: Cohen's Kappa coefficient
- **F1**: F1 score

## Output Files

After training is completed, the output directory will contain:

### Model Saving Structure

```
checkpoint/
└── seed_{seed}/
    ├── global_model_best.pth      # Best global single-modal model checkpoint
    ├── local_mm_model_gamma_best.pth      # Best local multi-modal model for GAMMA dataset
    ├── local_mm_model_zhongshan_best.pth  # Best local multi-modal model for Zhongshan dataset
    ├── local_mm_model_gongli_best.pth     # Best local multi-modal model for Gongli dataset
    └── test_results.csv           # Test results
```

### File Description

| File Name | Description |
|-----------|-------------|
| `global_model_best.pth` | Best global single-modal model checkpoint (only saved when using `--best_only`) |
| `local_mm_model_{dataset}_best.pth` | Best local multi-modal model checkpoint for each dataset (only saved when using `--best_only`) |
| `global_model_final.pth` | Final global single-modal model checkpoint (only saved when not using `--best_only`) |
| `local_mm_model_{dataset}_final.pth` | Final local multi-modal model checkpoint for each dataset (only saved when not using `--best_only`) |
| `test_results.csv` | Test results (generated by `test.py`) |

CSV result files contain the following columns:

| Column Name | Description |
|-------------|-------------|
| round | Communication round |
| seed | Random seed |
| dataname | Dataset name |
| model_type | Model type (local_mm/global_fundus) |
| AUROC | AUROC metric |
| Kappa | Kappa metric |
| F1 | F1 score |

## Model Testing

After training is completed, you can use the `test.py` script to independently evaluate the model:

### Usage

#### 1. Directly Specify Model Path

```bash
python test.py --model_path ./checkpoint/seed_42/global_model_best.pth --dataset gamma

python test.py --model_path ./checkpoint/seed_42/local_mm_model_gamma_best.pth --dataset gamma
```

#### 2. Automatically Build Path via seed and Model Type (Recommended)

```bash
# Test global model
python test.py --seed 42 --model_type global --dataset gamma

# Test local multi-modal model
python test.py --seed 42 --model_type local_mm --dataset gamma

# Test on different dataset
python test.py --seed 42 --model_type global --dataset zhongshan
```

### Test Script Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--model_path` | None | Model checkpoint path (.pth file), if not specified, path will be automatically built via --seed and --model_type |
| `--seed` | 42 | Random seed, also used for building model path |
| `--model_type` | global | Model type: global (global model) or local_mm (local multi-modal model) |
| `--dataset` | gamma | Test dataset name |
| `--data_dir` | ../data | Data directory path |
| `--batch_size` | 16 | Batch size |
| `--num_workers` | 2 | Number of worker processes for data loading |
| `--device` | cuda | Computing device (cuda/cpu) |
| `--backbone` | None | Backbone network type (auto-detected if not specified) |
