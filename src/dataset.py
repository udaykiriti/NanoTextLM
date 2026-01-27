import torch
from torch.utils.data import Dataset
import numpy as np
import os

class TextDataset(Dataset):
    def __init__(self, data_path, block_size, split='train', split_ratio=0.9):
        self.block_size = block_size
        
        if not os.path.exists(data_path):
             print(f"Error: {data_path} not found.")
             # Fallback just to avoid crash, but training will fail
             self.data = np.array([], dtype=np.uint16)
             self.num_samples = 0
             return

        # Load data using memory mapping
        # We assume uint16 because our vocab size is usually < 65535 (50257)
        try:
            raw_data = np.memmap(data_path, dtype=np.uint16, mode='r')
            total_len = len(raw_data)
            split_idx = int(total_len * split_ratio)
            
            if split == 'train':
                self.data = raw_data[:split_idx]
            else:
                self.data = raw_data[split_idx:]
                
            self.num_samples = (len(self.data) - 1) // self.block_size
            print(f"Loaded {split} dataset from {data_path}. Tokens: {len(self.data):,}. Samples: {self.num_samples:,}")
        except Exception as e:
            print(f"Error loading memmap: {e}")
            self.data = np.array([], dtype=np.uint16)
            self.num_samples = 0
        
    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        start = idx * self.block_size
        end = start + self.block_size + 1
        
        # Get chunk from memmap (this is efficiently loaded from disk by OS)
        chunk = self.data[start:end].astype(np.int64) # Convert to int64 for PyTorch
        
        # We shouldn't need padding if we define length correctly, 
        # but handled just in case last block is short (which we usually drop in __len__)
        if len(chunk) < self.block_size + 1:
             # This case essentially shouldn't be hit with valid logic, but for safety:
             # We would need a padding token ID. Passing 0 for now or handling it.
             # Ideally we just don't return partial blocks for training.
             chunk = np.pad(chunk, (0, self.block_size + 1 - len(chunk)), constant_values=0)
             
        x = torch.from_numpy(chunk[:-1])
        y = torch.from_numpy(chunk[1:])
        return x, y
