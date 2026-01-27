from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, processors
import os

def train_tokenizer(files, save_path="tokenizer.json", vocab_size=50257):
    print("Training tokenizer...")
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size, 
        special_tokens=["<|endoftext|>", "<|padding|>"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
    )
    
    tokenizer.train(files, trainer)
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
    
    tokenizer.save(save_path)
    print(f"Saved tokenizer to {save_path}")

if __name__ == "__main__":
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_file = os.path.join(PROJECT_ROOT, "data", "processed", "train.txt")
    save_path = os.path.join(PROJECT_ROOT, "tokenizer.json")
    
    if os.path.exists(data_file):
        train_tokenizer([data_file], save_path=save_path)
    else:
        print("Data file not found.")
