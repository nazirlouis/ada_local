"""
Train FunctionGemma following Google's official documentation exactly.
https://ai.google.dev/gemma/docs/functiongemma/finetuning-with-functiongemma
"""

import torch
import json
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.utils import get_json_schema
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, PeftModel

# Configuration
MODEL_ID = "google/functiongemma-270m-it"
OUTPUT_DIR = "functiongemma-270m-ft"
MERGED_OUTPUT_DIR = "merged_model"
DATA_FILE = "training_dataset.jsonl"

# --- Tool Definitions (using get_json_schema like Google's example) ---
def thinking(prompt: str) -> str:
    """
    Use this function for complex queries that require reasoning, math, coding, or multi-step analysis.
    
    Args:
        prompt: The user's original prompt
    """
    return "Thinking result"

def nonthinking(prompt: str) -> str:
    """
    Use this function for simple queries, greetings, factual questions, or tasks that do not require deep reasoning.
    
    Args:
        prompt: The user's original prompt
    """
    return "Non-thinking result"

# Generate tool schemas using HF's utility (same as Google's example)
TOOLS = [get_json_schema(thinking), get_json_schema(nonthinking)]

DEFAULT_SYSTEM_MSG = "You are a model that can do function calling with the following functions"

def rebuild_with_proper_schema(sample):
    """Rebuild sample with properly formatted tool schemas."""
    messages = sample["messages"]
    
    # Find the tool call
    tool_name = None
    tool_args = None
    user_content = None
    
    for msg in messages:
        if msg["role"] == "user":
            user_content = msg["content"]
        elif msg["role"] == "assistant" and "tool_calls" in msg:
            tc = msg["tool_calls"][0]["function"]
            tool_name = tc["name"]
            tool_args = tc["arguments"]
    
    if not all([user_content, tool_name, tool_args]):
        return sample
    
    return {
        "messages": [
            {"role": "developer", "content": DEFAULT_SYSTEM_MSG},
            {"role": "user", "content": user_content},
            {"role": "assistant", "tool_calls": [{"type": "function", "function": {"name": tool_name, "arguments": tool_args}}]},
        ],
        "tools": TOOLS  # Use properly formatted tools
    }

def train():
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    
    print("Loading dataset...")
    raw_dataset = load_dataset("json", data_files=DATA_FILE, split="train")
    print(f"Dataset size: {len(raw_dataset)}")
    
    # Rebuild with proper tool schemas
    print("Rebuilding with proper tool schemas...")
    dataset = raw_dataset.map(rebuild_with_proper_schema, remove_columns=raw_dataset.column_names)
    dataset = dataset.map(lambda x: x)  # Ensure proper format
    
    # Debug: show formatted prompt
    print("\n--- Dataset input ---")
    print(json.dumps(dataset[0], indent=2, default=str))
    
    debug_msg = tokenizer.apply_chat_template(
        dataset[0]["messages"], 
        tools=dataset[0]["tools"], 
        add_generation_prompt=False, 
        tokenize=False
    )
    print("\n--- Formatted prompt (what model sees) ---")
    print(debug_msg[:1500] + "..." if len(debug_msg) > 1500 else debug_msg)
    print("---\n")
    
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager"
    )
    
    print(f"Device: {model.device}")
    print(f"DType: {model.dtype}")
    
    # LoRA config
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )
    
    # Training config (following Google's example)
    args = SFTConfig(
        output_dir=OUTPUT_DIR,
        max_length=512,
        packing=False,
        num_train_epochs=8,
        per_device_train_batch_size=4,
        gradient_checkpointing=False,
        optim="adamw_torch_fused",
        logging_steps=1,
        save_strategy="epoch",
        learning_rate=2e-5,
        bf16=True,
        lr_scheduler_type="constant",
        overwrite_output_dir=True,
    )
    
    # Create trainer
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        peft_config=peft_config,
        processing_class=tokenizer,
    )
    
    print("Starting training...")
    trainer.train()
    
    print("Saving adapter...")
    trainer.save_model(OUTPUT_DIR)
    
    # Free memory
    del model
    del trainer
    torch.cuda.empty_cache()
    
    # Merge
    print("\nMerging adapter into base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    
    merged_model = PeftModel.from_pretrained(base_model, OUTPUT_DIR)
    merged_model = merged_model.merge_and_unload()
    
    print(f"Saving merged model to: {MERGED_OUTPUT_DIR}")
    merged_model.save_pretrained(MERGED_OUTPUT_DIR, safe_serialization=True)
    tokenizer.save_pretrained(MERGED_OUTPUT_DIR)
    
    print("\nDone!")
    print("Next: ollama create functiongemma-finetuned -f Modelfile")

if __name__ == "__main__":
    train()
