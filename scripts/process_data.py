import pandas as pd
import os
import argparse

def process_data(input_path, output_path):
    print(f"Processing {input_path}...")
    
    if not os.path.exists(input_path):
        print(f"Error: Input file {input_path} not found.")
        return

    try:
        df = pd.read_parquet(input_path)
    except Exception as e:
        print(f"Failed to read parquet: {e}")
        return

    print(f"Loaded {len(df)} rows.")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    print(f"Writing to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        for text in df['text']:
            if isinstance(text, str) and len(text.strip()) > 0:
                f.write(text.strip() + "\n<|endoftext|>\n")
                
    print("Done.")

if __name__ == "__main__":
    # Default paths
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    default_input = os.path.join(PROJECT_ROOT, "data", "raw", "openwebtext", "train-00074-of-00080.parquet")
    default_output = os.path.join(PROJECT_ROOT, "data", "processed", "train.txt")
    
    process_data(default_input, default_output)
