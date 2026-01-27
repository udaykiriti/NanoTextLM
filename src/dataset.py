import torch
from torch.utils.data import Dataset
import numpy as np
import os
from typing import Tuple, Optional

class TextDataset(Dataset):
    """
    Efficient memory-mapped Text Dataset for Large Language Models.
    
    Args:
        data_path (str): Path to the binary (.bin) file containing token ids.
        block_size (int): The context length (sequence length) for training.
        split (str): 'train' or 'val'.
        split_ratio (float): The ratio of data to use for training (default 0.9).
    """
    def __init__(self, data_path: str, block_size: int, split: str = 'train', split_ratio: float = 0.9):
        self.block_size = block_size
        
        if not os.path.exists(data_path):
             print(f"Error: {data_path} not found.")
             self.data = np.array([], dtype=np.uint16)
             self.num_samples = 0
             return

        # Load data using memory mapping (efficient for large files)
        try:
            raw_data = np.memmap(data_path, dtype=np.uint16, mode='r')
            total_len = len(raw_data)
            split_idx = int(total_len * split_ratio)
            
            if split == 'train':
                self.data = raw_data[:split_idx]
            else:
                self.data = raw_data[split_idx:]
                
            self.num_samples = (len(self.data) - 1) // self.block_size
            print(f"[{split.upper()}] Loaded from {data_path}. Tokens: {len(self.data):,}. Samples: {self.num_samples:,}")
        except Exception as e:
            print(f"Error loading memmap: {e}")
            self.data = np.array([], dtype=np.uint16)
            self.num_samples = 0
        
    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        start = idx * self.block_size
        end = start + self.block_size + 1
        
        # Get chunk from memmap
        chunk = self.data[start:end].astype(np.int64)
        
        # Handle edge case (should not happen with correct len calculation, but safety first)
        if len(chunk) < self.block_size + 1:
             chunk = np.pad(chunk, (0, self.block_size + 1 - len(chunk)), constant_values=0)
             
        x = torch.from_numpy(chunk[:-1])
        y = torch.from_numpy(chunk[1:])
        return x, y