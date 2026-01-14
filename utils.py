#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FedBKD Utility Functions Module

Provides data processing, model evaluation, and other functions for federated learning and local training,
supporting multiple datasets such as GAMMA, Zhongshan, Gongli.

Main functions:
    - Random seed setting
    - Data transformation definition
    - Model evaluation metric calculation
    - Model validation and testing
    - Model saving and loading
    - Result saving and loading
    - Federated averaging algorithm
"""

import os
import numpy as np
import cv2
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from PIL import Image
import json
import random
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union
from sklearn.metrics import (
    f1_score, roc_auc_score, cohen_kappa_score, confusion_matrix
)
import warnings
warnings.filterwarnings('ignore')


def set_seed(seed: int) -> None:
    """
    Set random seed to ensure experimental reproducibility
    
    Args:
        seed: Random seed value, 42 is recommended
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Setting random seed: {seed}")


def oct_to_pil(oct_img: np.ndarray) -> Image.Image:
    """
    Convert OCT image numpy array to PIL format
    
    Args:
        oct_img: OCT image numpy array, may be 2D or 3D array
        
    Returns:
        PIL image object
    """
    if oct_img is None:
        return Image.new('L', (224, 224), 0)
    
    if isinstance(oct_img, np.ndarray):
        if oct_img.dtype != np.uint8:
            oct_img = (oct_img * 255).astype(np.uint8)
        
        if len(oct_img.shape) == 3:
            middle_slice = oct_img.shape[2] // 2
            oct_img = oct_img[:, :, middle_slice]
        
        if len(oct_img.shape) == 2:
            return Image.fromarray(oct_img, mode='L')
        else:
            return Image.fromarray(oct_img)
    return Image.new('L', (224, 224), 0)


def gamma_oct_to_tensor(oct_img: np.ndarray) -> torch.Tensor:
    """
    GAMMA dataset specific: Convert OCT 3D array directly to tensor
    
    Args:
        oct_img: OCT image numpy array
        
    Returns:
        PyTorch tensor, shape [C, H, W] or [1, H, W]
    """
    if oct_img is None:
        return torch.zeros(1, 224, 224, dtype=torch.float32)
    
    if isinstance(oct_img, np.ndarray):
        if oct_img.dtype != np.uint8:
            oct_img = (oct_img * 255).astype(np.uint8)
        
        if len(oct_img.shape) == 3:
            oct_tensor = torch.from_numpy(oct_img.transpose(2, 0, 1)).float() / 255.0
            return oct_tensor
        else:
            oct_tensor = torch.from_numpy(oct_img).unsqueeze(0).float() / 255.0
            return oct_tensor
    return torch.zeros(1, 224, 224, dtype=torch.float32)


def get_transforms(image_size: int, dataset_type: str = 'gamma'):
    """
    Get data transformations for the specified dataset
    
    Args:
        image_size: Target image size
        dataset_type: Dataset type, optional 'gamma', 'papila', 'zhongshan', 'gongli'
        
    Returns:
        Corresponding data transformation tuple
        
    Raises:
        ValueError: Unsupported dataset type
    """
    if dataset_type == 'gamma':
        return get_gamma_transforms(image_size)
    elif dataset_type == 'papila':
        return get_papila_transforms(image_size)
    elif dataset_type == 'zhongshan':
        return get_zhongshan_transforms(image_size)
    elif dataset_type == 'gongli':
        return get_gongli_transforms(image_size)
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")


def get_gamma_transforms(image_size: int, oct_img_size: tuple = (224, 224)):
    """
    Get data transformations for GAMMA dataset
    
    Args:
        image_size: Fundus image target size
        oct_img_size: OCT image target size, default (224, 224)
        
    Returns:
        (img_train, img_val, oct_train, oct_val) transformation tuple
    """
    img_train_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    img_val_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    oct_train_transforms = transforms.Compose([
        transforms.Lambda(oct_to_pil),
        transforms.Resize(oct_img_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ToTensor()
    ])
    
    oct_val_transforms = transforms.Compose([
        transforms.Lambda(oct_to_pil),
        transforms.Resize(oct_img_size),
        transforms.ToTensor()
    ])
    
    return img_train_transforms, img_val_transforms, oct_train_transforms, oct_val_transforms


def get_papila_transforms(image_size: int):
    """
    Get data transformations for PAPILA dataset
    
    Args:
        image_size: Target image size
        
    Returns:
        (img_train, img_val) transformation tuple
    """
    img_train_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    img_val_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return img_train_transforms, img_val_transforms


def get_zhongshan_transforms(img_size: int):
    """
    Get data transformations for Zhongshan dataset
    
    Args:
        img_size: Target image size
        
    Returns:
        (img_train, img_val, oct_train, oct_val, vf_train, vf_val) transformation tuple
    """
    img_train_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    img_val_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    oct_train_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    oct_val_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    vf_train_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    vf_val_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return (
        img_train_transforms, img_val_transforms,
        oct_train_transforms, oct_val_transforms,
        vf_train_transforms, vf_val_transforms
    )


def get_gongli_transforms(img_size: int):
    """
    Get data transformations for Gongli dataset
    
    Args:
        img_size: Target image size
        
    Returns:
        (img_train, img_val, oct_train, oct_val) transformation tuple
    """
    img_train_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    img_val_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    oct_train_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ToTensor()
    ])
    
    oct_val_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor()
    ])
    
    return img_train_transforms, img_val_transforms, oct_train_transforms, oct_val_transforms


def find_optimal_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    method: str = 'youden'
) -> float:
    """
    Find optimal classification threshold
    
    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        method: Optimization method, optional 'f1', 'youden', 'balanced_accuracy'
        
    Returns:
        Optimal threshold
    """
    thresholds = np.linspace(0.01, 0.99, 99)
    best_threshold = 0.5
    best_score = 0
    
    for threshold in thresholds:
        y_pred_thresh = (y_prob >= threshold).astype(int)
        
        if method == 'f1':
            score = f1_score(y_true, y_pred_thresh)
        elif method == 'youden':
            cm = confusion_matrix(y_true, y_pred_thresh)
            if cm.shape == (1, 1):
                if y_true[0] == 0:
                    tn, fp, fn, tp = cm[0, 0], 0, 0, 0
                else:
                    tn, fp, fn, tp = 0, 0, 0, cm[0, 0]
            elif cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()
            else:
                tn, fp, fn, tp = 0, 0, 0, 0
            
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            score = sensitivity + specificity - 1
        elif method == 'balanced_accuracy':
            from sklearn.metrics import balanced_accuracy_score
            score = balanced_accuracy_score(y_true, y_pred_thresh)
        
        if score > best_score:
            best_score = score
            best_threshold = threshold
    
    return best_threshold


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray = None,
    find_optimal: bool = True
) -> Dict[str, float]:
    """
    Calculate classification evaluation metrics
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Predicted probabilities
        find_optimal: Whether to find optimal threshold
        
    Returns:
        Dictionary containing various metrics:
        - auroc: AUROC
        - kappa: Cohen's Kappa coefficient
        - f1_score: F1 score
        - optimal_threshold: Optimal threshold
    """
    if find_optimal and y_prob is not None:
        optimal_threshold = find_optimal_threshold(y_true, y_prob, 'youden')
        y_pred_optimal = (y_prob >= optimal_threshold).astype(int)
    else:
        y_pred_optimal = y_pred
        optimal_threshold = 0.5
    
    auroc = roc_auc_score(y_true, y_prob) if y_prob is not None else 0.0
    kappa = cohen_kappa_score(y_true, y_pred_optimal)
    f1 = f1_score(y_true, y_pred_optimal, zero_division=0)
    
    return {
        'auroc': auroc,
        'kappa': kappa,
        'f1_score': f1,
        'optimal_threshold': optimal_threshold if find_optimal else 0.5
    }


def print_metrics(metrics: Dict[str, float], prefix: str = "") -> None:
    """
    Print evaluation metrics
    
    Args:
        metrics: Dict containing evaluation metrics
        prefix: Prefix for printed content
    """
    print(f"{prefix} AUROC={metrics['auroc']:.4f} Kappa={metrics['kappa']:.4f} "
          f"F1={metrics['f1_score']:.4f}")


def validate_model(
    model: nn.Module,
    val_dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    data_type: str = 'dual',
    dataset_type: str = 'gamma'
) -> tuple:
    """
    Validate model
    
    Args:
        model: Model
        val_dataloader: Validation dataloader
        criterion: Loss function
        device: Compute device
        data_type: Data type, 'dual' or 'single'
        dataset_type: Dataset type
        
    Returns:
        (avg_val_loss, metrics) tuple
    """
    model.eval()
    val_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(val_dataloader):
            if data_type == 'dual':
                if dataset_type == 'gamma':
                    fundus_img = batch_data['fundus'].to(device)
                    oct_img = batch_data['oct'].to(device)
                    labels = batch_data['label'].to(device)
                    inputs = (fundus_img, oct_img)
                elif dataset_type == 'papila':
                    fundus_img = batch_data['fundus'].to(device)
                    clinical_data = batch_data['clinical'].to(device)
                    labels = batch_data['label'].to(device)
                    inputs = (fundus_img, clinical_data)
                elif dataset_type == 'zhongshan':
                    fundus_img = batch_data['fundus'].to(device)
                    oct_dev_img = batch_data['oct_dev'].to(device)
                    oct_pie_img = batch_data['oct_pie'].to(device)
                    vf_img = batch_data['vf'].to(device)
                    labels = batch_data['label'].to(device)
                    inputs = (fundus_img, oct_dev_img, oct_pie_img, vf_img)
                elif dataset_type == 'gongli':
                    fundus_img = batch_data['fundus'].to(device)
                    oct_img = batch_data['oct_gl'].to(device)
                    labels = batch_data['label'].to(device)
                    inputs = (fundus_img, oct_img)
            else:
                if isinstance(batch_data, dict):
                    inputs = batch_data['fundus'].to(device)
                    labels = batch_data['label'].to(device)
                else:
                    inputs, labels = batch_data
                    inputs = inputs.to(device)
                    labels = labels.to(device)

            if inputs.dim() == 1:
                raise ValueError(f"Detected 1D input tensor (shape: {inputs.shape}). This usually indicates a data loading or collate function issue. Please check data file formats.")
            if inputs.dim() == 3:
                inputs = inputs.unsqueeze(0)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item()
            
            probs = torch.softmax(outputs, dim=1)[:, 1]
            preds = torch.argmax(outputs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    avg_val_loss = val_loss / len(val_dataloader)
    metrics = calculate_metrics(np.array(all_labels), np.array(all_preds), np.array(all_probs))
    
    return avg_val_loss, metrics


def test_model(
    model: nn.Module,
    test_dataloader: DataLoader,
    device: torch.device,
    data_type: str = 'dual',
    dataset_type: str = 'gamma'
) -> Dict[str, float]:
    """
    Test model
    
    Args:
        model: Model
        test_dataloader: Test dataloader
        device: Compute device
        data_type: Data type, 'dual' or 'single'
        dataset_type: Dataset type
        
    Returns:
        Dict containing evaluation metrics
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(test_dataloader):
            if data_type == 'dual':
                if dataset_type == 'gamma':
                    fundus_img = batch_data['fundus'].to(device)
                    oct_img = batch_data['oct'].to(device)
                    labels = batch_data['label'].to(device)
                    inputs = (fundus_img, oct_img)
                elif dataset_type == 'papila':
                    fundus_img = batch_data['fundus'].to(device)
                    clinical_data = batch_data['clinical'].to(device)
                    labels = batch_data['label'].to(device)
                    inputs = (fundus_img, clinical_data)
                elif dataset_type == 'zhongshan':
                    fundus_img = batch_data['fundus'].to(device)
                    oct_dev_img = batch_data['oct_dev'].to(device)
                    oct_pie_img = batch_data['oct_pie'].to(device)
                    vf_img = batch_data['vf'].to(device)
                    labels = batch_data['label'].to(device)
                    inputs = (fundus_img, oct_dev_img, oct_pie_img, vf_img)
                elif dataset_type == 'gongli':
                    fundus_img = batch_data['fundus'].to(device)
                    oct_img = batch_data['oct_gl'].to(device)
                    labels = batch_data['label'].to(device)
                    inputs = (fundus_img, oct_img)
            else:
                if isinstance(batch_data, dict):
                    inputs = batch_data['fundus'].to(device)
                    labels = batch_data['label'].to(device)
                else:
                    inputs, labels = batch_data
                    inputs = inputs.to(device)
                    labels = labels.to(device)
            
            outputs = model(inputs)
            
            probs = torch.softmax(outputs, dim=1)[:, 1]
            preds = torch.argmax(outputs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    metrics = calculate_metrics(np.array(all_labels), np.array(all_preds), np.array(all_probs))
    
    return metrics


def save_model(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    metrics: Dict[str, float],
    save_path: str
) -> None:
    """
    Save model checkpoint
    
    Args:
        model: Model
        optimizer: Optimizer
        epoch: Current epoch
        loss: Loss value
        metrics: Evaluation metrics
        save_path: Save path
    """
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'metrics': metrics
    }, save_path)


def load_model(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    load_path: str,
    device: torch.device
) -> tuple:
    """
    Load model checkpoint
    
    Args:
        model: Model
        optimizer: Optimizer
        load_path: Load path
        device: Compute device
        
    Returns:
        (epoch, loss, metrics) tuple
    """
    checkpoint = torch.load(load_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch'], checkpoint['loss'], checkpoint.get('metrics', {})


def save_fedmkd_results(results: Dict, save_path: str) -> None:
    """
    Save FedMKD experiment results
    
    Args:
        results: Experiment results dict
        save_path: Save path
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {key: convert_numpy(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(item) for item in obj]
        return obj
    
    results_serializable = convert_numpy(results)
    
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(results_serializable, f, indent=2, ensure_ascii=False)
    
    print(f"Results saved to: {save_path}")


def load_fedmkd_results(load_path: str) -> Dict:
    """
    Load FedMKD experiment results
    
    Args:
        load_path: Load path
        
    Returns:
        Experiment results dict
    """
    with open(load_path, 'r', encoding='utf-8') as f:
        results = json.load(f)
    return results


def save_results(results: Dict, save_path: str) -> None:
    """
    Save training results
    
    Args:
        results: Results dict
        save_path: Save path
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {save_path}")


def plot_training_curves(results: Dict, save_path: str = None) -> None:
    """
    Plot training curves (placeholder function)
    
    Args:
        results: Training results
        save_path: Save path
    """
    print("Training curve plotting is not implemented yet")
    if save_path:
        print(f"Curve figure should be saved to: {save_path}")


def federated_averaging(
    client_models: list,
    client_weights: list = None
) -> nn.Module:
    """
    Federated averaging algorithm
    
    Perform weighted averaging across multiple client models to produce a global model.
    
    Args:
        client_models: List of client models
        client_weights: List of client weights, default None (equal weighting)
        
    Returns:
        Aggregated global model
    """
    if not client_models:
        return None
    
    if client_weights is None:
        client_weights = [1.0 / len(client_models)] * len(client_models)
    
    client_weights = [float(w) for w in client_weights]
    total_weight = sum(client_weights)
    client_weights = [w / total_weight for w in client_weights]
    
    global_model = client_models[0]
    global_state_dict = global_model.state_dict()
    
    for key in global_state_dict.keys():
        global_state_dict[key] = torch.zeros_like(global_state_dict[key]).float()
        
        for i, client_model in enumerate(client_models):
            client_state_dict = client_model.state_dict()
            param_value = client_state_dict[key].float()
            weight_value = float(client_weights[i])
            global_state_dict[key] += weight_value * param_value
    
    global_model.load_state_dict(global_state_dict)
    return global_model
