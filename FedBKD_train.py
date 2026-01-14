# -*- coding: utf-8 -*-
"""
FedBKD Federated Knowledge Distillation Training Script

Implements a federated learning based knowledge distillation algorithm that supports heterogeneous federated training on multi-modal medical imaging data.
This script implements the complete federated training pipeline, including client local training, global model aggregation, and reverse knowledge distillation.

Main features:
1. Multi-dataset federated learning: supports gamma, zhongshan, gongli, airogs datasets
2. Heterogeneous model training:
   - gamma, zhongshan, gongli: local multi-modal → single-modal fundus knowledge distillation
   - AIROGS: direct single-modal training
3. Global model aggregation: the server aggregates all clients' single-modal fundus models
4. Reverse knowledge distillation: global model → gamma, zhongshan, gongli multi-modal models
5. Model evaluation and result saving

Author: Glaucoma Diagnosis Team
Created: 2025-01-11
Copyright © 2025 Glaucoma Diagnosis Team. All rights reserved.
"""

import argparse
import os
import sys
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

warnings.filterwarnings('ignore')

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import (
    calculate_metrics,
    federated_averaging,
    print_metrics,
    save_model,
    set_seed,
    test_model,
    validate_model
)
from dataloader import FullDataLoader, FullDataset
from models import KnowledgeDistillationLoss, get_model


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments

    Returns:
        Namespace object containing all command-line arguments
    """
    parser = argparse.ArgumentParser(
        description='FedBKD Federated Knowledge Distillation Training',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--num_rounds',
        type=int,
        default=30,
        help='Number of federated communication rounds'
    )
    parser.add_argument(
        '--local_epochs',
        type=int,
        default=2,
        help='Local training epochs per client'
    )
    parser.add_argument(
        '--kd_epochs',
        type=int,
        default=1,
        help='Knowledge distillation epochs'
    )
    parser.add_argument(
        '--datasets',
        type=str,
        default='gamma,zhongshan,gongli,airogs_0,airogs_1,airogs_2,airogs_3,airogs_4',
        help='Comma-separated list of datasets participating in federated learning'
    )
    parser.add_argument(
        '--backbone',
        type=str,
        default='resnet',
        choices=['resnet', 'mobilenet'],
        help='Model backbone type'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=16,
        help='Batch size'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=1e-4,
        help='Learning rate'
    )
    parser.add_argument(
        '--weight_decay',
        type=float,
        default=1e-5,
        help='Weight decay coefficient'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=4.0,
        help='Temperature for KD to soften probability distribution'
    )
    parser.add_argument(
        '--alpha',
        type=float,
        default=0.7,
        help='Alpha for KD to balance hard and soft losses'
    )
    parser.add_argument(
        '--beta',
        type=float,
        default=0.5,
        help='Beta for reverse knowledge distillation'
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=2,
        help='Number of data loader workers'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    parser.add_argument(
        '--run_multiple_seeds',
        action='store_true',
        help='Run multiple seeds for more stable results'
    )
    parser.add_argument(
        '--num_seeds',
        type=int,
        default=5,
        help='Number of seeds when multi-seed run is enabled'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda:0',
        help='Compute device, supports CUDA or CPU'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./checkpoint',
        help='Output directory for training results and checkpoints'
    )
    parser.add_argument(
        '--best_only',
        action='store_true',
        help='Enable early stopping and save only the best model'
    )
    parser.add_argument(
        '--patience',
        type=int,
        default=5,
        help='Patience for early stopping: stop after N non-improving rounds'
    )
    parser.add_argument(
        '--monitor_metric',
        type=str,
        default='auroc',
        choices=['auroc', 'kappa', 'f1'],
        help='Metric to monitor for early stopping'
    )
    parser.add_argument(
        '--monitor_dataset',
        type=str,
        default='gamma',
        help='Dataset monitored for early stopping'
    )

    return parser.parse_args()


def create_client_datasets(
    args: argparse.Namespace,
    data_dir: str = '../data/'
) -> Tuple[Dict[str, FullDataset], Dict[str, FullDataset]]:
    """
    Create client training and test datasets for each dataset

    Args:
        args: Command-line arguments including dataset list
        data_dir: Root data directory, default '../data/'

    Returns:
        client_datasets: Dict of training datasets keyed by dataset name
        client_test_datasets: Dict of test datasets keyed by dataset name
    """
    datasets_list = args.datasets.split(',')
    client_datasets = {}
    client_test_datasets = {}

    for dataset_name in datasets_list:
        train_dir = os.path.join(data_dir, 'Train')
        client_datasets[dataset_name] = FullDataset(
            data_dir=train_dir,
            dataset_type=dataset_name,
            cache_data=True,
            verbose=True
        )
        print(f"Dataset {dataset_name} training samples: {len(client_datasets[dataset_name])}")

        test_dir = os.path.join(data_dir, 'Test')
        client_test_datasets[dataset_name] = FullDataset(
            data_dir=test_dir,
            dataset_type=dataset_name,
            cache_data=True,
            verbose=True
        )
        print(f"Dataset {dataset_name} test samples: {len(client_test_datasets[dataset_name])}")

    return client_datasets, client_test_datasets


def create_client_dataloaders(
    client_datasets: Dict[str, FullDataset],
    args: argparse.Namespace
) -> Dict[str, FullDataLoader]:
    """
    Create dataloaders for each client dataset

    Args:
        client_datasets: Dict of client datasets
        args: Command-line args including batch_size, seed, num_workers

    Returns:
        client_dataloaders: Dict of dataloaders
    """
    client_dataloaders = {}

    for dataset_name, dataset in client_datasets.items():
        client_dataloaders[dataset_name] = FullDataLoader(
            dataset,
            batch_size=args.batch_size,
            seed=args.seed,
            num_workers=args.num_workers,
            shuffle=True
        )

    return client_dataloaders


def train_knowledge_distillation(
    mm_model: nn.Module,
    um_model: nn.Module,
    train_loader: FullDataLoader,
    criterion: KnowledgeDistillationLoss,
    optimizer: torch.optim.Optimizer,
    epochs: int,
    device: torch.device,
    dataset_type: str
) -> Tuple[nn.Module, List[float]]:
    """
    Perform local knowledge distillation training

    Use the multi-modal model as the teacher and the single-modal fundus model as the student,
    transferring multi-modal knowledge to the single-modal model via distillation.

    Args:
        mm_model: Multi-modal teacher model
        um_model: Single-modal student model
        train_loader: Training dataloader
        criterion: Knowledge distillation loss
        optimizer: Optimizer
        epochs: Number of training epochs
        device: Compute device
        dataset_type: Dataset type determining input organization

    Returns:
        um_model: Trained single-modal model
        local_losses: List of epoch losses
    """
    mm_model.eval()
    um_model.train()

    local_losses = []

    for epoch in range(epochs):
        epoch_loss = 0.0
        num_batches = 0

        for batch_data in train_loader:
            teacher_inputs, student_inputs, labels = _prepare_inputs(
                batch_data, dataset_type, device
            )

            optimizer.zero_grad()

            with torch.no_grad():
                teacher_logits = mm_model(teacher_inputs)

            student_logits = um_model(student_inputs)

            loss, hard_loss, soft_loss = criterion(
                student_logits, teacher_logits, labels
            )

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        avg_epoch_loss = epoch_loss / num_batches
        local_losses.append(avg_epoch_loss)
        print(
            f"  Knowledge distillation epoch {epoch+1}/{epochs}, "
            f"loss: {avg_epoch_loss:.4f} "
            f"(hard loss: {hard_loss:.4f}, soft loss: {soft_loss:.4f})"
        )

    return um_model, local_losses


def train_airogs_local(
    local_model: nn.Module,
    train_loader: FullDataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    epochs: int,
    device: torch.device
) -> Tuple[nn.Module, List[float]]:
    """
    Local training for AIROGS clients

    The AIROGS dataset contains only single-modal data and is trained directly with cross-entropy loss.

    Args:
        local_model: Local model
        train_loader: Training dataloader
        criterion: Loss function
        optimizer: Optimizer
        epochs: Number of training epochs
        device: Compute device

    Returns:
        local_model: Trained model
        local_losses: List of epoch losses
    """
    local_model.train()
    local_losses = []

    for epoch in range(epochs):
        epoch_loss = 0.0
        num_batches = 0

        for batch_data in train_loader:
            inputs = batch_data['fundus'].to(device)
            labels = batch_data['label'].to(device)

            optimizer.zero_grad()
            outputs = local_model(inputs)

            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        avg_epoch_loss = epoch_loss / num_batches
        local_losses.append(avg_epoch_loss)
        print(f"  Local training epoch {epoch+1}/{epochs}, loss: {avg_epoch_loss:.4f}")

    return local_model, local_losses


def train_reverse_knowledge_distillation(
    mm_model: nn.Module,
    global_model: nn.Module,
    train_loader: FullDataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    epochs: int,
    device: torch.device,
    dataset_type: str,
    beta: float
) -> Tuple[nn.Module, List[float]]:
    """
    Perform reverse knowledge distillation

    Use the global single-modal fundus model as the teacher and distill knowledge back to the local multi-modal model to enhance its performance.

    Args:
        mm_model: Multi-modal student model
        global_model: Global teacher model
        train_loader: Training dataloader
        criterion: Loss function
        optimizer: Optimizer
        epochs: Number of training epochs
        device: Compute device
        dataset_type: Dataset type
        beta: Weight of reverse distillation loss

    Returns:
        mm_model: Trained multi-modal model
        local_losses: List of epoch losses
    """
    global_model.eval()
    mm_model.train()

    local_losses = []

    for epoch in range(epochs):
        epoch_loss = 0.0
        num_batches = 0

        for batch_data in train_loader:
            student_inputs, teacher_inputs, labels = _prepare_inputs(
                batch_data, dataset_type, device
            )

            optimizer.zero_grad()

            with torch.no_grad():
                teacher_logits = global_model(teacher_inputs)

            student_logits = mm_model(student_inputs)

            ce_loss = nn.CrossEntropyLoss()(student_logits, labels)
            kl_loss = nn.KLDivLoss(reduction='batchmean')(
                nn.functional.log_softmax(student_logits, dim=1),
                nn.functional.softmax(teacher_logits, dim=1)
            )
            loss = beta * kl_loss + (1 - beta) * ce_loss

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        avg_epoch_loss = epoch_loss / num_batches
        local_losses.append(avg_epoch_loss)
        print(f"  Reverse KD epoch {epoch+1}/{epochs}, loss: {avg_epoch_loss:.4f}")

    return mm_model, local_losses


def _prepare_inputs(
    batch_data: Dict[str, torch.Tensor],
    dataset_type: str,
    device: torch.device
) -> Tuple[Tuple[torch.Tensor, ...], torch.Tensor, torch.Tensor]:
    """
    Prepare model inputs based on dataset type

    Args:
        batch_data: Batch data dict
        dataset_type: Dataset type
        device: Compute device

    Returns:
        teacher_inputs: Tuple of teacher model inputs
        student_inputs: Student model input
        labels: Label tensor
    """
    if dataset_type == 'gamma':
        fundus_img = batch_data['fundus'].to(device)
        oct_img = batch_data['oct'].to(device)
        labels = batch_data['label'].to(device)
        teacher_inputs = (fundus_img, oct_img)
        student_inputs = fundus_img

    elif dataset_type == 'zhongshan':
        fundus_img = batch_data['fundus'].to(device)
        oct_dev_img = batch_data['oct_dev'].to(device)
        oct_pie_img = batch_data['oct_pie'].to(device)
        vf_img = batch_data['vf'].to(device)
        labels = batch_data['label'].to(device)
        teacher_inputs = (fundus_img, oct_dev_img, oct_pie_img, vf_img)
        student_inputs = fundus_img

    elif dataset_type == 'gongli':
        fundus_img = batch_data['fundus'].to(device)
        oct_img = batch_data['oct_gl'].to(device)
        labels = batch_data['label'].to(device)
        teacher_inputs = (fundus_img, oct_img)
        student_inputs = fundus_img

    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")

    return teacher_inputs, student_inputs, labels


def evaluate_models(
    global_model: nn.Module,
    client_mm_models: Dict[str, nn.Module],
    test_datasets: Dict[str, FullDataset],
    args: argparse.Namespace
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Evaluate model performance on all test sets

    Use different evaluation strategies for different datasets:
    - gamma, zhongshan, gongli: evaluate both local multi-modal and global single-modal models
    - airogs: evaluate only the global single-modal model

    Args:
        global_model: Global single-modal model
        client_mm_models: Dict of local multi-modal models
        test_datasets: Dict of test datasets
        args: Command-line arguments

    Returns:
        results: Dict of evaluation results
    """
    results = {}

    for dataset_name, test_dataset in test_datasets.items():
        test_dataloader = FullDataLoader(
            test_dataset,
            batch_size=args.batch_size,
            seed=args.seed,
            num_workers=args.num_workers,
            shuffle=False
        )

        print(f"\nTest results for dataset {dataset_name}:")

        if dataset_name.startswith('airogs'):
            print("  Using global fundus model:")
            metrics = test_model(
                global_model, test_dataloader, args.device,
                data_type='single', dataset_type=dataset_name
            )
            results[dataset_name] = {'global_fundus': metrics}
            print_metrics(metrics)

        else:
            print("  Using local multi-modal model:")
            mm_model = client_mm_models[dataset_name]
            metrics_mm = test_model(
                mm_model, test_dataloader, args.device,
                data_type='dual', dataset_type=dataset_name
            )

            print("  Using global fundus model:")
            metrics_global = test_model(
                global_model, test_dataloader, args.device,
                data_type='single', dataset_type=dataset_name
            )

            results[dataset_name] = {
                'local_mm': metrics_mm,
                'global_fundus': metrics_global
            }

            print("  Local multi-modal model results:")
            print_metrics(metrics_mm)
            print("  Global fundus model results:")
            print_metrics(metrics_global)

    return results


def save_results_to_csv(
    results: Dict[str, Dict[str, Dict[str, float]]],
    output_dir: str,
    seed: int,
    round_num: int = 0
) -> None:
    """
    Save evaluation results to CSV

    Args:
        results: Evaluation results dict
        output_dir: Output directory
        seed: Current random seed
        round_num: Current communication round
    """
    csv_data = []

    for dataset_name, dataset_results in results.items():
        if dataset_name.startswith('airogs'):
            metrics = dataset_results['global_fundus']
            row = {
                'round': round_num,
                'seed': seed,
                'dataname': dataset_name,
                'model_type': 'global_fundus',
                'AUROC': metrics.get('auroc', 0),
                'Kappa': metrics.get('kappa', 0),
                'F1': metrics.get('f1_score', 0)
            }
            csv_data.append(row)

        else:
            metrics_mm = dataset_results['local_mm']
            row_mm = {
                'round': round_num,
                'seed': seed,
                'dataname': dataset_name,
                'model_type': 'local_mm',
                'AUROC': metrics_mm.get('auroc', 0),
                'Kappa': metrics_mm.get('kappa', 0),
                'F1': metrics_mm.get('f1_score', 0)
            }
            csv_data.append(row_mm)

            metrics_global = dataset_results['global_fundus']
            row_global = {
                'round': round_num,
                'seed': seed,
                'dataname': dataset_name,
                'model_type': 'global_fundus',
                'AUROC': metrics_global.get('auroc', 0),
                'Kappa': metrics_global.get('kappa', 0),
                'F1': metrics_global.get('f1_score', 0)
            }
            csv_data.append(row_global)

    df = pd.DataFrame(csv_data)

    csv_path = os.path.join(output_dir, f'results_seed_{seed}.csv')
    df.to_csv(csv_path, index=False, encoding='utf-8')
    print(f"\nResults saved to: {csv_path}")

    summary_path = os.path.join(output_dir, 'all_results.csv')

    if not os.path.exists(summary_path):
        df.to_csv(summary_path, index=False, encoding='utf-8', mode='w')
    else:
        df.to_csv(summary_path, index=False, encoding='utf-8', mode='a', header=False)


def train_with_seed(args: argparse.Namespace, seed: int) -> Tuple[nn.Module, Dict[str, nn.Module]]:
    """
    Run the full training pipeline with a specific random seed
    
    Includes dataset loading, model initialization, federated training, and result evaluation.
    
    Args:
        args: Command-line arguments
        seed: Random seed
        
    Returns:
        global_model: Trained global model
        client_mm_models: Trained client multi-modal models
    """
    set_seed(seed)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    print(f"Using random seed: {seed}\n")

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading client datasets...")
    client_datasets, client_test_datasets = create_client_datasets(args)
    client_dataloaders = create_client_dataloaders(client_datasets, args)

    print("Initializing global model...")
    global_model = get_model(model_type='um', backbone=args.backbone, dataset='gamma', num_classes=2)
    global_model = global_model.to(device)

    client_mm_models = {}
    client_um_models = {}

    print("Initializing client models...")

    for dataset_name in client_datasets.keys():
        if dataset_name.startswith('airogs'):
            um_model = get_model(model_type='um', backbone=args.backbone, dataset='airogs', num_classes=2)
            client_um_models[dataset_name] = um_model.to(device)

        else:
            mm_model = get_model(
                model_type='mm', backbone=args.backbone, dataset=dataset_name, num_classes=2
            )
            um_model = get_model(
                model_type='um', backbone=args.backbone, dataset=dataset_name, num_classes=2
            )
            client_mm_models[dataset_name] = mm_model.to(device)
            client_um_models[dataset_name] = um_model.to(device)

    kd_criterion = KnowledgeDistillationLoss(
        temperature=args.temperature, alpha=args.alpha
    )
    ce_criterion = nn.CrossEntropyLoss()

    # Early stopping variables
    best_score = -float('inf')
    patience_counter = 0
    best_round = 0
    
    print("\nStarting federated training...")

    for round_num in range(args.num_rounds):
        print(f"\n========== Communication round {round_num+1}/{args.num_rounds} ==========")

        selected_clients = list(client_datasets.keys())
        print(f"Selected clients: {selected_clients}")

        client_models = []
        client_weights = []

        for client_id in selected_clients:
            print(f"\nClient {client_id} local training...")

            local_um_model = type(global_model)().to(device)
            local_um_model.load_state_dict(global_model.state_dict())

            if client_id.startswith('airogs'):
                print("  AIROGS client direct single-modal training...")

                optimizer = torch.optim.AdamW(
                    local_um_model.parameters(),
                    lr=args.lr,
                    weight_decay=args.weight_decay
                )

                train_loader = client_dataloaders[client_id]
                local_um_model, local_losses = train_airogs_local(
                    local_um_model, train_loader, ce_criterion,
                    optimizer, args.local_epochs, device
                )

                client_um_models[client_id].load_state_dict(
                    local_um_model.state_dict()
                )

            else:
                print("  Local KD: multi-modal → single-modal...")

                local_mm_model = client_mm_models[client_id]

                optimizer = torch.optim.AdamW(
                    local_um_model.parameters(),
                    lr=args.lr,
                    weight_decay=args.weight_decay
                )

                train_loader = client_dataloaders[client_id]
                local_um_model, local_losses = train_knowledge_distillation(
                    local_mm_model, local_um_model, train_loader,
                    kd_criterion, optimizer, args.kd_epochs, device, client_id
                )

            client_models.append(local_um_model)
            client_weights.append(len(client_datasets[client_id]))

        print("\nAggregating global model...")
        global_model = federated_averaging(client_models, client_weights)
        global_model = global_model.to(device)

        print("\nReverse KD: global model → multi-modal model...")

        for client_id in selected_clients:
            if not client_id.startswith('airogs'):
                print(f"  Client {client_id} reverse knowledge distillation...")

                local_mm_model = client_mm_models[client_id]

                optimizer = torch.optim.AdamW(
                    local_mm_model.parameters(),
                    lr=args.lr,
                    weight_decay=args.weight_decay
                )

                train_loader = client_dataloaders[client_id]
                local_mm_model, local_losses = train_reverse_knowledge_distillation(
                    local_mm_model, global_model, train_loader,
                    kd_criterion, optimizer, args.local_epochs,
                    device, client_id, args.beta
                )

                client_mm_models[client_id] = local_mm_model

        print("\nEvaluating model performance...")
        results = evaluate_models(
            global_model, client_mm_models, client_test_datasets, args
        )

        # 禁用CSV文件保存
        # save_results_to_csv(results, args.output_dir, seed, round_num + 1)

        # Early stopping and model saving logic
        if args.best_only:
            # Compute average of monitored metric across participants
            scores = []
            for dataset_name, dataset_results in results.items():
                if 'local_mm' in dataset_results:
                    score = dataset_results['local_mm'].get(args.monitor_metric, 0)
                    scores.append(score)
                elif 'global_fundus' in dataset_results:
                    score = dataset_results['global_fundus'].get(args.monitor_metric, 0)
                    scores.append(score)
            
            # Compute mean
            if scores:
                current_score = sum(scores) / len(scores)
            else:
                current_score = 0
            
            print(f"Monitored metric ({args.monitor_metric}) mean across participants: {current_score:.4f}, best: {best_score:.4f}")
            
            # Check for improvement
            if current_score > best_score:
                best_score = current_score
                best_round = round_num + 1
                patience_counter = 0
                print(f"Performance improved, updating best model (round: {best_round})")
                
                # Save best model
                if args.best_only:
                    print("Saving best model...")
                    # Create subfolder per seed
                    seed_output_dir = os.path.join(args.output_dir, f'seed_{seed}')
                    os.makedirs(seed_output_dir, exist_ok=True)
                    
                    # Save global model
                    global_model_path = os.path.join(seed_output_dir, f'global_model_best.pth')
                    torch.save({
                        'model_state_dict': global_model.state_dict(),
                        'backbone': args.backbone,
                        'num_classes': 2,
                        'round': best_round,
                        'score': best_score
                    }, global_model_path)
                    print(f"Best global model saved to: {global_model_path}")
                    
                    # Save multi-modal models
                    for dataset_name, mm_model in client_mm_models.items():
                        model_path = os.path.join(seed_output_dir, f'local_mm_model_{dataset_name}_best.pth')
                        torch.save({
                            'dataset': dataset_name,
                            'model_state_dict': mm_model.state_dict(),
                            'backbone': args.backbone,
                            'num_classes': 2,
                            'round': best_round,
                            'score': best_score
                        }, model_path)
                        print(f"Best local multi-modal model ({dataset_name}) saved to: {model_path}")
            else:
                patience_counter += 1
                print(f"No improvement, patience counter: {patience_counter}/{args.patience}")
                
                # Check early stopping condition
                if args.best_only and patience_counter >= args.patience:
                    print(f"Early stopping triggered, terminating training (best round: {best_round})")
                    break

        print(f"\nCommunication round {round_num+1} completed!")
    
    return global_model, client_mm_models


def main() -> None:
    """
    Main entry point
    
    Decide whether to run single-seed or multi-seed experiments based on command-line arguments and invoke the corresponding training pipeline.
    """
    args = parse_args()

    # Handle --best_only argument
    if args.best_only:
        print("Early stopping enabled and only best model will be saved (--best_only)")

    final_global_model = None
    final_client_mm_models = {}

    if args.run_multiple_seeds:
        seeds = list(range(args.num_seeds))
        print(f"\nStarting experiments with {args.num_seeds} seeds...")
        print(f"Seed list: {seeds}")

        for seed in seeds:
            print(f"\n{'='*60}")
            print(f"Current seed: {seed}")
            print('='*60)
            global_model, client_mm_models = train_with_seed(args, seed)
            # 保存最后一个种子的模型
            final_global_model = global_model
            final_client_mm_models = client_mm_models

        print(f"\nTraining completed for all {args.num_seeds} seeds!")
        # 禁用CSV文件保存
        # print(f"汇总结果已保存到: {os.path.join(args.output_dir, 'all_results.csv')}")

    else:
        final_global_model, final_client_mm_models = train_with_seed(args, args.seed)

    print("\nTraining completed...")

    # Save final models (only when best_only is not enabled)
    if not args.best_only:
        print("Saving final models...")
        # Create subfolder per seed
        seed_output_dir = os.path.join(args.output_dir, f'seed_{args.seed}')
        os.makedirs(seed_output_dir, exist_ok=True)
        
        if final_global_model is not None:
            global_model_path = os.path.join(seed_output_dir, 'global_model_final.pth')
            torch.save({
                'model_state_dict': final_global_model.state_dict(),
                'backbone': args.backbone,
                'num_classes': 2
            }, global_model_path)
            print(f"Global model saved to: {global_model_path}")

        if final_client_mm_models:
            for dataset_name, mm_model in final_client_mm_models.items():
                model_path = os.path.join(seed_output_dir, f'local_mm_model_{dataset_name}.pth')
                torch.save({
                    'dataset': dataset_name,
                    'model_state_dict': mm_model.state_dict(),
                    'backbone': args.backbone,
                    'num_classes': 2
                }, model_path)
                print(f"Local multi-modal model ({dataset_name}) saved to: {model_path}")
    else:
        print("Best models saved via early stopping, skipping final model saving")

    print("\nAll models saved!")


if __name__ == "__main__":
    main()
