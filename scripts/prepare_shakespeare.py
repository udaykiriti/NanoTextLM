import os
import requests
import numpy as np
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, processors

def prepare_shakespeare():
    # Paths
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(PROJECT_ROOT, "data", "processed")
    raw_dir = os.path.join(PROJECT_ROOT, "data", "raw")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(raw_dir, exist_ok=True)
    
    input_file_path = os.path.join(raw_dir, 'shakespeare.txt')
    if not os.path.exists(input_file_path):
        print("Downloading Tiny Shakespeare...")
        data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
        with open(input_file_path, 'w', encoding='utf-8') as f:
            f.write(requests.get(data_url).text)
    
    print("Training Tokenizer on Shakespeare...")
    # Train a BPE tokenizer specifically for this dataset
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    
    # Smaller vocab for small dataset
    trainer = trainers.BpeTrainer(
        vocab_size=5000, 
        special_tokens=["<|endoftext|>", "<|padding|>"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
    )
    
    tokenizer.train([input_file_path], trainer)
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
    
    tokenizer_path = os.path.join(PROJECT_ROOT, "tokenizer.json")
    tokenizer.save(tokenizer_path)
    print(f"Saved tokenizer to {tokenizer_path}")
    
    print("Tokenizing data...")
    with open(input_file_path, 'r', encoding='utf-8') as f:
        text = f.read()
        
    encoded = tokenizer.encode(text)
    data = np.array(encoded.ids, dtype=np.uint16)
    
    # Split 90/10
    n = len(data)
    train_data = data[:int(n*0.9)]
    val_data = data[int(n*0.9):]
    
    train_data.tofile(os.path.join(data_dir, "train.bin"))
    val_data.tofile(os.path.join(data_dir, "val.bin"))
    
    print(f"Saved train.bin ({len(train_data)} tokens) and val.bin ({len(val_data)} tokens)")

if __name__ == "__main__":
    prepare_shakespeare()
