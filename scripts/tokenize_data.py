import os
import numpy as np
from tokenizers import Tokenizer
import tqdm

def tokenize_file(input_path, output_path, tokenizer_path):
    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found.")
        return
    if not os.path.exists(tokenizer_path):
        print(f"Error: {tokenizer_path} not found.")
        return

    print(f"Loading tokenizer from {tokenizer_path}...")
    tokenizer = Tokenizer.from_file(tokenizer_path)

    # We will read line by line (or chunk by chunk) to avoid loading everything
    print(f"Tokenizing {input_path} to {output_path}...")
    
    # First, let's count lines for progress bar (optional, but nice)
    # with open(input_path, 'r', encoding='utf-8') as f:
    #     num_lines = sum(1 for _ in f)
    # Actually counting lines is slow on 500MB. Let's just stream.

    token_dtype = np.uint16 # 0 to 65535, efficient storage
    
    # We'll use a temp list and flush periodically
    batch_size = 10000 
    batch = []
    
    # Initialize output file (clearing it)
    with open(output_path, "wb") as f_out:
        pass

    total_tokens = 0
    
    with open(input_path, 'r', encoding='utf-8') as f:
        # We can read line by line since our format has newlines
        # Or read in binary chunks. Text reading is safer for encoding.
        
        all_tokens = []
        
        # Optimization: use tokenizer.encode_batch if possible, but streaming
        # file reading is the bottleneck.
        
        pbar = tqdm.tqdm(desc="Tokenizing")
        
        # Process in chunks of lines
        chunk = []
        for line in f:
            chunk.append(line)
            if len(chunk) >= batch_size:
                # Batch encode
                encoded_batch = tokenizer.encode_batch(chunk)
                for enc in encoded_batch:
                    all_tokens.extend(enc.ids)
                
                # Flush to disk if getting too big (e.g., > 100MB in RAM)
                # 100M tokens * 2 bytes = 200MB. 
                if len(all_tokens) > 10_000_000: # Flush every ~10M tokens
                    arr = np.array(all_tokens, dtype=token_dtype)
                    with open(output_path, "ab") as f_out:
                        f_out.write(arr.tobytes())
                    total_tokens += len(all_tokens)
                    all_tokens = []
                
                pbar.update(len(chunk))
                chunk = []
        
        # Process remaining chunk
        if chunk:
            encoded_batch = tokenizer.encode_batch(chunk)
            for enc in encoded_batch:
                all_tokens.extend(enc.ids)
            pbar.update(len(chunk))
            
        # Flush remaining tokens
        if all_tokens:
            arr = np.array(all_tokens, dtype=token_dtype)
            with open(output_path, "ab") as f_out:
                f_out.write(arr.tobytes())
            total_tokens += len(all_tokens)
            
    print(f"Done. Dictionary size: {tokenizer.get_vocab_size()}. Total tokens: {total_tokens}")
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_path = os.path.join(PROJECT_ROOT, "data", "processed", "train.txt")
    output_path = os.path.join(PROJECT_ROOT, "data", "processed", "train.bin")
    tokenizer_path = os.path.join(PROJECT_ROOT, "tokenizer.json")
    
    tokenize_file(input_path, output_path, tokenizer_path)
