import os
import tqdm
import numpy as np
from tokenizers import Tokenizer
import multiprocessing as mp
from functools import partial

def tokenize_chunk(chunk, tokenizer_path):
    # Worker function for parallel processing
    tokenizer = Tokenizer.from_file(tokenizer_path)
    encoded = tokenizer.encode_batch(chunk)
    return [enc.ids for enc in encoded]

def tokenize_stream(input_path, output_dir, tokenizer_path, shard_size=100_000_000):
    """
    Tokenizes a large text file in a streaming fashion and saves to sharded binary files.
    """
    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found.")
        return
    if not os.path.exists(tokenizer_path):
        print(f"Error: {tokenizer_path} not found.")
        return

    os.makedirs(output_dir, exist_ok=True)
    print(f"Tokenizing {input_path} -> {output_dir}/train_*.bin")

    tokenizer = Tokenizer.from_file(tokenizer_path)
    vocab_size = tokenizer.get_vocab_size()
    dtype = np.uint16 if vocab_size < 65535 else np.uint32
    
    # Configuration
    chunk_size = 1000 # Lines per chunk for processing
    shard_idx = 0
    token_buffer = []
    total_tokens = 0
    
    # Initialize first shard
    current_shard_path = os.path.join(output_dir, f"train_{shard_idx:03d}.bin")
    f_out = open(current_shard_path, "wb")
    
    print(f"Writing shard {shard_idx}...")
    
    with open(input_path, 'r', encoding='utf-8') as f:
        lines_buffer = []
        
        pbar = tqdm.tqdm(desc="Processing Lines", unit="lines")
        
        for line in f:
            lines_buffer.append(line)
            
            if len(lines_buffer) >= chunk_size:
                # Tokenize batch
                encoded_batch = tokenizer.encode_batch(lines_buffer)
                for enc in encoded_batch:
                    token_buffer.extend(enc.ids)
                
                # Flush to disk if buffer is large enough or shard is full
                if len(token_buffer) > 0:
                    arr = np.array(token_buffer, dtype=dtype)
                    f_out.write(arr.tobytes())
                    total_tokens += len(token_buffer)
                    token_buffer = []
                    
                    # Check shard rotation
                    if f_out.tell() > shard_size * 2: # heuristic: bytes check
                        f_out.close()
                        shard_idx += 1
                        current_shard_path = os.path.join(output_dir, f"train_{shard_idx:03d}.bin")
                        f_out = open(current_shard_path, "wb")
                        print(f"Rotated to shard {shard_idx}")
                
                pbar.update(len(lines_buffer))
                lines_buffer = []
        
        # Process remaining
        if lines_buffer:
            encoded_batch = tokenizer.encode_batch(lines_buffer)
            for enc in encoded_batch:
                token_buffer.extend(enc.ids)
            if token_buffer:
                arr = np.array(token_buffer, dtype=dtype)
                f_out.write(arr.tobytes())
                total_tokens += len(token_buffer)
            pbar.update(len(lines_buffer))

    f_out.close()
    print(f"Done. Total tokens: {total_tokens:,}. Saved to {output_dir}")

if __name__ == "__main__":
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_txt = os.path.join(PROJECT_ROOT, "data", "processed", "train.txt")
    output_dir = os.path.join(PROJECT_ROOT, "data", "shards")
    tokenizer_json = os.path.join(PROJECT_ROOT, "tokenizer.json")
    
    tokenize_stream(input_txt, output_dir, tokenizer_json)
