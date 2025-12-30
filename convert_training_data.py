#!/usr/bin/env python3
"""
Convert FunctionGemma training data to MLX format.
MLX expects a 'text' field containing the full prompt + completion.
"""

import json
import random
from pathlib import Path


def convert_to_mlx_format(input_file: str, output_dir: str, train_split: float = 0.9):
    """
    Convert JSONL from prompt/completion format to MLX 'text' format.
    Also splits into train.jsonl and valid.jsonl.
    """
    input_path = Path(input_file)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Read all entries
    entries = []
    with open(input_path, 'r') as f:
        for line in f:
            entry = json.loads(line.strip())
            # Combine prompt + completion into 'text' field
            mlx_entry = {
                "text": entry["prompt"] + entry["completion"]
            }
            entries.append(mlx_entry)
    
    print(f"Loaded {len(entries)} entries from {input_path}")
    
    # Shuffle
    random.shuffle(entries)
    
    # Split into train/valid
    split_idx = int(len(entries) * train_split)
    train_entries = entries[:split_idx]
    valid_entries = entries[split_idx:]
    
    # Write train.jsonl
    train_file = output_path / "train.jsonl"
    with open(train_file, 'w') as f:
        for entry in train_entries:
            f.write(json.dumps(entry) + "\n")
    print(f"✅ Created {train_file} ({len(train_entries)} examples)")
    
    # Write valid.jsonl
    valid_file = output_path / "valid.jsonl"
    with open(valid_file, 'w') as f:
        for entry in valid_entries:
            f.write(json.dumps(entry) + "\n")
    print(f"✅ Created {valid_file} ({len(valid_entries)} examples)")
    
    # Show sample
    print("\n📋 Sample entry (text field):")
    sample_text = train_entries[0]["text"]
    # Show truncated version
    if len(sample_text) > 500:
        print(f"  {sample_text[:250]}...")
        print(f"  ...{sample_text[-200:]}")
    else:
        print(f"  {sample_text}")
    
    return train_file, valid_file


def main():
    print("=" * 60)
    print("MLX Training Data Converter")
    print("=" * 60)
    
    input_file = Path(__file__).parent / "training_data" / "functiongemma_training.jsonl"
    output_dir = Path(__file__).parent / "training_data"
    
    if not input_file.exists():
        print(f"❌ Input file not found: {input_file}")
        print("   Run generate_training_data.py first!")
        return
    
    convert_to_mlx_format(str(input_file), str(output_dir))
    
    print("\n" + "=" * 60)
    print("Done! Your data is ready for MLX fine-tuning.")
    print("\nNext step:")
    print("  python -m mlx_lm.lora --model google/functiongemma-270m-it --data ./training_data")
    print("=" * 60)


if __name__ == "__main__":
    main()
