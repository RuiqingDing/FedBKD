# -*- coding: utf-8 -*-
"""
Model Testing Script

Loads a trained model checkpoint and evaluates performance on the test set,
outputting evaluation metrics including AUROC, Kappa, and F1.

Created: 2025-01-11
"""

import argparse
import os
import sys
from typing import Tuple
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

warnings.filterwarnings('ignore')

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import (
    calculate_metrics,
    print_metrics,
    test_model,
    set_seed
)
from dataloader import FullDataLoader, FullDataset
from models import get_model


def parse_args():
    parser = argparse.ArgumentParser(description='Model testing script')
    
    parser.add_argument('--model_path', type=str, default=None,
                        help='Model checkpoint path (.pth). If not specified, it will be auto-constructed using --seed and --model_type')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed; also used to construct model path')
    parser.add_argument('--model_type', type=str, default='global',
                        choices=['global', 'local_mm'],
                        help='Model type: global (global model) or local_mm (local multi-modal model)')
    parser.add_argument('--dataset', type=str, default='gamma',
                        choices=['gamma', 'zhongshan', 'gongli', 'airogs_0', 'airogs_1', 'airogs_2', 'airogs_3', 'airogs_4'],
                        help='Test dataset name')
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='Data directory path')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--num_workers', type=int, default=2,
                        help='Number of data loader workers')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Compute device (cuda/cpu)')
    parser.add_argument('--backbone', type=str, default=None,
                        choices=['resnet', 'mobilenet', None],
                        help='Backbone type (auto-detected if not specified)')
    
    return parser.parse_args()


def detect_backbone_from_checkpoint(checkpoint):
    """Auto-detect backbone type from checkpoint keys"""
    state_dict_keys = list(checkpoint.get('model_state_dict', checkpoint).keys())
    
    has_fundus_branch = any('fundus_branch' in key for key in state_dict_keys)
    has_features = any('features.' in key and 'classifier' not in key for key in state_dict_keys)
    
    if has_fundus_branch:
        return 'resnet'
    elif has_features:
        return 'mobilenet'
    else:
        return 'resnet'


def load_test_dataset(args: argparse.Namespace) -> Tuple[FullDataLoader, str]:
    """
    Load test dataset
    
    Args:
        args: Command-line arguments
        
    Returns:
        test_dataloader: Test dataloader
        dataset_name: Dataset name
    """
    dataset_name = args.dataset
    
    test_dir = os.path.join(args.data_dir, 'Test')
    
    if not os.path.exists(test_dir):
        raise FileNotFoundError(f"Test data directory does not exist: {test_dir}")
    
    test_dataset = FullDataset(
        data_dir=test_dir,
        dataset_type=dataset_name,
        cache_data=False,
        verbose=False
    )
    
    test_dataloader = FullDataLoader(
        test_dataset,
        batch_size=args.batch_size,
        seed=args.seed,
        num_workers=args.num_workers,
        shuffle=False
    )
    
    return test_dataloader, dataset_name


def detect_model_type_from_checkpoint(checkpoint):
    """Auto-detect model type (single/dual-modal) from checkpoint"""
    state_dict_keys = list(checkpoint.get('model_state_dict', checkpoint).keys())
    
    has_oct_branch = any('oct_branch' in key for key in state_dict_keys)
    has_cross_attention = any('cross_attention' in key for key in state_dict_keys)
    has_fundus_features = any('fundus_features' in key for key in state_dict_keys)
    
    if has_oct_branch or has_cross_attention or has_fundus_features:
        return 'mm'
    else:
        return 'um'


def remap_state_dict_keys(state_dict: dict, backbone: str, model_type: str) -> dict:
    """
    Remap state_dict keys to fit the target model architecture
    
    Args:
        state_dict: Original state dict
        backbone: Backbone type
        model_type: Model type ('um' or 'mm')
        
    Returns:
        Remapped state dict
    """
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    
    has_features = any('features.' in key for key in state_dict.keys())
    has_fundus_branch = any('fundus_branch.' in key for key in state_dict.keys())
    
    if model_type == 'mm':
        if backbone == 'mobilenet' and has_features and not has_fundus_branch:
            for key, value in state_dict.items():
                new_key = key.replace('features.', 'fundus_branch.')
                new_state_dict[new_key] = value
        else:
            new_state_dict = state_dict
    else:
        new_state_dict = state_dict
    
    return new_state_dict


def load_model(args: argparse.Namespace) -> nn.Module:
    """
    Load model
    
    Args:
        args: Command-line arguments
        
    Returns:
        model: Model loaded with weights
    """
    checkpoint = torch.load(args.model_path, map_location='cpu', weights_only=False)
    
    if args.backbone is not None:
        backbone = args.backbone
    elif 'backbone' in checkpoint:
        backbone = checkpoint['backbone']
    else:
        backbone = detect_backbone_from_checkpoint(checkpoint)
        print(f"Auto-detected backbone type: {backbone}")
    
    num_classes = checkpoint.get('num_classes', 2)
    
    detected_model_type = detect_model_type_from_checkpoint(checkpoint)
    if args.dataset.startswith('airogs'):
        model_type = 'um'
        data_type = 'single'
    elif detected_model_type == 'um':
        model_type = 'um'
        data_type = 'single'
    else:
        model_type = 'mm'
        data_type = 'dual'
    
    print(f"Detected model type: {model_type} ({'single-modal' if model_type == 'um' else 'multi-modal'})")
    
    model = get_model(
        model_type=model_type,
        backbone=backbone,
        dataset=args.dataset,
        num_classes=num_classes
    )
    
    state_dict = checkpoint['model_state_dict']
    state_dict = remap_state_dict_keys(state_dict, backbone, model_type)
    
    model.load_state_dict(state_dict)
    
    return model, data_type


def main() -> None:
    args = parse_args()
    
    if args.model_path is None:
        
        base_output_dir = './checkpoint'
        
        seed_output_dir = os.path.join(base_output_dir, f'seed_{args.seed}')
        
        if args.model_type == 'global':
            model_filename = 'global_model_best.pth'
        elif args.model_type == 'local_mm':
            model_filename = f'local_mm_model_{args.dataset}_best.pth'
        else:
            raise ValueError(f"Unsupported model type: {args.model_type}")
        
        args.model_path = os.path.join(seed_output_dir, model_filename)
        print(f"Auto-constructed model path: {args.model_path}")
    
    set_seed(args.seed)
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Model path: {args.model_path}")
    print(f"Test dataset: {args.dataset}")
    
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model file does not exist: {args.model_path}")
    
    model, data_type = load_model(args)
    model = model.to(device)
    model.eval()
    
    test_dataloader, dataset_name = load_test_dataset(args)
    print(f"Number of test samples: {len(test_dataloader.dataset)}")
    
    print(f"\nStarting model evaluation...")
    metrics = test_model(
        model, test_dataloader, device,
        data_type=data_type, dataset_type=dataset_name
    )
    
    print(f"\nEvaluation results:")
    print_metrics(metrics, prefix="  ")
    
    results = {
        'model_path': args.model_path,
        'dataset': args.dataset,
        'AUROC': metrics.get('auroc', 0),
        'Kappa': metrics.get('kappa', 0),
        'F1': metrics.get('f1_score', 0),
        'optimal_threshold': metrics.get('optimal_threshold', 0.5)
    }
    
    output_dir = os.path.dirname(args.model_path)
    csv_path = os.path.join(output_dir, 'test_results.csv')
    df = pd.DataFrame([results])
    df.to_csv(csv_path, index=False, encoding='utf-8')
    print(f"\nResults saved to: {csv_path}")
    
    print("\nTesting completed!")


if __name__ == "__main__":
    main()
