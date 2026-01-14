# -*- coding: utf-8 -*-
"""
Data Loading Module

Provides enhanced PKL dataset class and deterministic DataLoader implementation,
supporting batch filename return, caching mechanism, and data integrity validation.

Created: 2025-01-11
"""

import os
import random
import pickle
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, RandomSampler


class FullDataset(Dataset):
    """
    Enhanced PKL Dataset Class

    Supports loading of multi-modal medical imaging data, providing data caching,
    integrity validation, and flexible data transformation capabilities.
    Each sample contains multiple modal data such as fundus images (OCT), 
    fundus photographs (CLI), visual field examination (VF), and corresponding labels.

    Args:
        data_dir: Data folder path containing .pkl format sample files
        dataset_type: Dataset type name used to filter files for specific datasets
        file_list: Specified file path list, automatically found from data_dir if None
        required_keys: List of required data keys, including various modal data and labels
        transform: Data preprocessing transformation function
        cache_data: Whether to cache data in memory for faster access
        verbose: Whether to display detailed loading information

    Raises:
        ValueError: Raised when no .pkl files are found in data_dir

    Attributes:
        data_dir: Data directory path (Path object)
        file_paths: List of paths to all valid data files
        data_cache: Data cache dictionary, only exists when cache_data=True
        required_keys: List of required data keys
        invalid_files: List of invalid files and error reasons

    Example:
        >>> dataset = FullDataset(
        ...     data_dir="./data",
        ...     dataset_type="gamma",
        ...     cache_data=True
        ... )
        >>> sample = dataset[0]
        >>> print(sample.keys())
    """

    def __init__(
        self,
        data_dir: str,
        dataset_type: str = 'gamma',
        file_list: Optional[List[str]] = None,
        required_keys: Optional[List[str]] = None,
        transform: Optional[Callable] = None,
        cache_data: bool = False,
        verbose: bool = True
    ) -> None:
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.cache_data = cache_data
        self.verbose = verbose
        self.required_keys = required_keys or [
            'fundus', 'oct', 'cli', 'oct_dev', 'oct_pie', 'vf', 'oct_gl',
            'OCT_mask', 'CLI_mask', 'OCT_dev_mask', 'OCT_pie_mask',
            'VF_mask', 'OCT_gl_mask', 'label'
        ]

        self.file_paths = self._get_file_paths(dataset_type, file_list)

        if len(self.file_paths) == 0:
            raise ValueError(f"No .pkl files found in folder {data_dir}")

        self.data_cache: Dict[Path, Dict[str, Any]] = {} if cache_data else None
        self.invalid_files: List[tuple] = []

        self._validate_dataset()

        if verbose:
            print(f"Dataset initialized successfully, {len(self.file_paths)} samples in total")
            self._print_dataset_info()

    def _get_file_paths(self, dataset_type: str, file_list: Optional[List[str]]) -> List[Path]:
        """
        Get Data File Path List

        Args:
            dataset_type: Dataset type name
            file_list: Specified file list

        Returns:
            List of matching .pkl file paths
        """
        if file_list is not None:
            return [self.data_dir / f for f in file_list if (self.data_dir / f).exists()]

        if dataset_type == 'all':
            return list(self.data_dir.glob('*.pkl'))

        normalized_type = self._normalize_dataset_type(dataset_type)
        return list(self.data_dir.glob(f'*{normalized_type}*.pkl'))

    def _normalize_dataset_type(self, dataset_type: str) -> str:
        """
        Normalize Dataset Type Name

        Convert user-input dataset type name to actual file matching format.

        Args:
            dataset_type: Original dataset type name

        Returns:
            Normalized dataset type name
        """
        if dataset_type == 'zhongshan':
            return 'Zhongshan'
        elif dataset_type.startswith('airogs'):
            return dataset_type.replace('airogs', 'AIROGS')
        return dataset_type

    def _validate_dataset(self) -> None:
        """
        Validate Dataset Integrity and Validity

        Check if each .pkl file contains all required data keys,
        record files with missing keys or loading failures for subsequent processing.
        """
        valid_files = []

        for file_path in self.file_paths:
            try:
                with open(file_path, 'rb') as f:
                    data = pickle.load(f)

                missing_keys = [key for key in self.required_keys if key not in data]

                if missing_keys:
                    if self.verbose:
                        print(f"Warning: File {file_path.name} missing keys: {missing_keys}")
                    self.invalid_files.append((file_path, f"Missing keys: {missing_keys}"))
                else:
                    valid_files.append(file_path)

            except Exception as e:
                if self.verbose:
                    print(f"Error: Failed to load file {file_path.name}: {e}")
                self.invalid_files.append((file_path, str(e)))

        self.file_paths = valid_files

        if self.verbose and self.invalid_files:
            print(f"Found {len(self.invalid_files)} invalid files")

    def _print_dataset_info(self) -> None:
        """
        Print Dataset Detailed Information

        Includes shape and data type of each modal data,
        helping users understand the basic structure of the data.
        """
        if not self.file_paths:
            return

        sample_data = self._load_single_sample(self.file_paths[0])
        print("Data shape information:")

        for key, value in sample_data.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: {value.shape} {value.dtype}")
            else:
                print(f"  {key}: {type(value).__name__}")

    def _load_single_sample(self, file_path: Path) -> Dict[str, Any]:
        """
        Load Single Data Sample

        Read data from .pkl file and convert to PyTorch tensor format.

        Args:
            file_path: Data file path

        Returns:
            Dictionary containing various modal data and labels
        """
        if self.data_cache is not None and file_path in self.data_cache:
            return self.data_cache[file_path]

        with open(file_path, 'rb') as f:
            data = pickle.load(f)

        tensor_data = {}

        for key, value in data.items():
            if value is None:
                tensor_data[key] = self._create_default_tensor(key)
            elif isinstance(value, np.ndarray):
                tensor_data[key] = torch.from_numpy(value).float()
            elif isinstance(value, (list, tuple)):
                tensor_data[key] = torch.tensor(value, dtype=torch.float32)
            elif isinstance(value, (int, float)):
                tensor_data[key] = torch.tensor([value], dtype=torch.float32)
            else:
                tensor_data[key] = value

        if self.data_cache is not None:
            self.data_cache[file_path] = tensor_data

        return tensor_data

    def _create_default_tensor(self, key: str) -> torch.Tensor:
        """Create default tensor based on key name"""
        if key in ['fundus', 'cli']:
            return torch.zeros(3, 224, 224, dtype=torch.float32)
        elif key.startswith('oct') or key.startswith('OCT'):
            return torch.zeros(1, 224, 224, dtype=torch.float32)
        elif key.startswith('vf') or key.startswith('VF'):
            return torch.zeros(3, 224, 224, dtype=torch.float32)
        elif key == 'label':
            return torch.tensor(0, dtype=torch.long)
        elif key == 'clinical':
            return torch.zeros(10, dtype=torch.float32)
        else:
            return torch.tensor(0.0)

    def __len__(self) -> int:
        """
        Return Number of Samples in Dataset

        Returns:
            Number of valid data files
        """
        return len(self.file_paths)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get Data Sample at Specified Index

        Args:
            idx: Sample index, range [0, len(dataset))

        Returns:
            Dictionary containing data modalities, labels, and file names
        """
        file_path = self.file_paths[idx]

        try:
            sample = self._load_single_sample(file_path)

            if self.transform:
                sample = self.transform(sample)

            sample['file_name'] = file_path.name

            return sample

        except Exception as e:
            print(f"Error loading file {file_path}: {e}")
            empty_sample = self._get_empty_sample()
            empty_sample['file_name'] = file_path.name
            return empty_sample

    def _get_empty_sample(self) -> Dict[str, torch.Tensor]:
        """
        Create Empty Sample Dictionary

        Used to handle data loading failure cases, returns a dictionary with default values.

        Returns:
            Sample dictionary with default values and invalid file marker
        """
        sample = {}
        for key in self.required_keys:
            if key in ['fundus', 'cli']:
                sample[key] = torch.zeros(3, 224, 224, dtype=torch.float32)
            elif key.startswith('oct') or key.startswith('OCT'):
                sample[key] = torch.zeros(1, 224, 224, dtype=torch.float32)
            elif key.startswith('vf') or key.startswith('VF'):
                sample[key] = torch.zeros(3, 224, 224, dtype=torch.float32)
            elif key == 'label':
                sample[key] = torch.tensor(0, dtype=torch.long)
            elif key == 'clinical':
                sample[key] = torch.zeros(10, dtype=torch.float32)
            else:
                sample[key] = torch.tensor(0.0)
        sample['file_name'] = 'invalid_file'
        return sample

    def get_file_names(self) -> List[str]:
        """
        Get List of All File Names in Dataset

        Returns:
            List of all .pkl file names
        """
        return [path.name for path in self.file_paths]


def custom_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Custom Collate Function

    Combines multiple data samples into a batch tensor,
    specially handling non-tensor data like file names.

    Args:
        batch: List of samples in a batch

    Returns:
        Combined batch data dictionary

    Example:
        >>> for batch in dataloader:
        >>>     print(batch['fundus'].shape)  # [batch_size, channels, height, width]
        >>>     print(batch['file_name'])      # ['sample1.pkl', 'sample2.pkl', ...]
    """
    if len(batch) == 0:
        return {}

    keys = batch[0].keys()
    batched_data = {}

    for key in keys:
        if key == 'file_name':
            batched_data[key] = [sample[key] for sample in batch]
            continue

        values = [sample[key] for sample in batch]

        if all(isinstance(v, torch.Tensor) for v in values):
            shapes = [v.shape for v in values]

            if all(shape == shapes[0] for shape in shapes):
                batched_data[key] = torch.stack(values)
            else:
                batched_data[key] = _pad_variable_tensors(values)
        else:
            batched_data[key] = values

    return batched_data


def _pad_variable_tensors(values: List[torch.Tensor]) -> torch.Tensor:
    """
    Pad Tensors with Inconsistent Dimensions

    Specifically handles padding of image tensors, maintaining channel dimensions unchanged,
    only padding in spatial dimensions (H, W).

    Args:
        values: List of tensors with incomplete same shapes

    Returns:
        Padded tensor, shape [N, C, H_max, W_max]
    """
    if len(values) == 0:
        return torch.tensor([])

    if all(v.shape == values[0].shape for v in values):
        return torch.stack(values)

    max_h = max(v.shape[-2] for v in values)
    max_w = max(v.shape[-1] for v in values)

    padded_tensors = []
    for v in values:
        if v.shape[-2] == max_h and v.shape[-1] == max_w:
            padded_tensors.append(v)
        else:
            h_pad = max_h - v.shape[-2]
            w_pad = max_w - v.shape[-1]
            padding = (0, w_pad, 0, h_pad)
            padded = F.pad(v, padding, mode='constant', value=0)
            padded_tensors.append(padded)

    return torch.stack(padded_tensors)


class FullDataLoader:
    """
    Deterministic Data Loader

    Adds deterministic sampling and filename tracking functionality on top of standard DataLoader,
    ensuring reproducibility of experimental results.

    Args:
        dataset: Dataset object
        batch_size: Number of samples per batch, default 32
        seed: Random seed, ensuring deterministic data shuffling, default 42
        num_workers: Number of worker processes for data loading, default 4
        shuffle: Whether to shuffle data at the start of each epoch, default True
        drop_last: Whether to drop the last incomplete batch, default False
        collate_fn: Collate function, default uses custom_collate_fn

    Attributes:
        dataloader: Internal PyTorch DataLoader object

    Example:
        >>> dataloader = FullDataLoader(
        ...     dataset,
        ...     batch_size=16,
        ...     seed=42,
        ...     shuffle=True
        ... )
        >>> for batch in dataloader:
        ...     filenames = dataloader.get_batch_filenames(batch)
    """

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int = 32,
        seed: int = 42,
        num_workers: int = 4,
        shuffle: bool = True,
        drop_last: bool = False,
        collate_fn: Callable = custom_collate_fn
    ) -> None:
        self.seed = seed
        self.batch_size = batch_size
        self.dataset = dataset
        self.num_workers = num_workers
        self.drop_last = drop_last
        self.collate_fn = collate_fn

        generator = torch.Generator()
        generator.manual_seed(seed)

        if shuffle:
            self.sampler = RandomSampler(dataset, generator=generator)
        else:
            self.sampler = None

        self.dataloader = self._create_dataloader(generator)

    def _create_dataloader(self, generator: torch.Generator) -> DataLoader:
        """
        Create Internal DataLoader Instance

        Args:
            generator: Random number generator

        Returns:
            Configured DataLoader object
        """
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            sampler=self.sampler,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            generator=generator,
            drop_last=self.drop_last,
            pin_memory=True
        )

    def __iter__(self):
        """
        Return Data Iterator

        Returns:
            DataLoader iterator, can be iterated to get batch data
        """
        return iter(self.dataloader)

    def __len__(self) -> int:
        """
        Return Number of Batches

        Returns:
            Number of batches in the dataset
        """
        return len(self.dataloader)

    def get_batch_filenames(self, batch: Dict[str, Any]) -> List[str]:
        """
        Get Filename List from Batch Data

        Args:
            batch: Batch data dictionary

        Returns:
            List of filenames corresponding to each sample in the batch
        """
        return batch.get('file_name', [])


if __name__ == "__main__":
    data_dir = "./data/Train"

    dataset = FullDataset(
        data_dir=data_dir,
        dataset_type='papila',
        cache_data=False,
        verbose=True
    )

    dataloader = FullDataLoader(
        dataset,
        batch_size=16,
        seed=42,
        num_workers=2,
        shuffle=True
    )

    print("\nTesting data loading:")

    for i, batch in enumerate(dataloader):
        print(f"Batch {i + 1}:")

        filenames = dataloader.get_batch_filenames(batch)
        print(f"  Filenames: {filenames}")

        for key, value in batch.items():
            if key != 'file_name':
                if isinstance(value, torch.Tensor):
                    print(f"  {key}: {value.shape}")
                else:
                    print(f"  {key}: {len(value)} items")

        if i == 2:
            break
