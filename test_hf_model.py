"""
Test the fine-tuned model directly with HuggingFace (no Ollama).
This helps diagnose if the issue is with training or Ollama conversion.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.utils import get_json_schema
import json

# Models to test
BASE_MODEL_ID = "google/functiongemma-270m-it"
FINETUNED_MODEL_DIR = "./merged_model"

# Tool definitions (same as training)
def thinking(prompt: str) -> str:
    """
    Use this function for complex queries that require reasoning, math, coding, or multi-step analysis.
    Args:
        prompt: The user's original prompt
    """
    return "result"

def nonthinking(prompt: str) -> str:
    """
    Use this function for simple queries, greetings, factual questions, or tasks that do not require deep reasoning.
    Args:
        prompt: The user's original prompt
    """
    return "result"

TOOLS = [get_json_schema(thinking), get_json_schema(nonthinking)]

# Test questions
TEST_QUESTIONS = [
    ("What is the capital of France?", "nonthinking"),
    ("Hello there!", "nonthinking"),
    ("Write a Python function to reverse a string", "thinking"),
    ("What is 2 + 2?", "nonthinking"),
    ("Explain the theory of relativity", "thinking"),
]

def test_model(model, tokenizer, model_name):
    print(f"\n{'='*60}")
    print(f"TESTING: {model_name}")
    print(f"{'='*60}")
    
    correct = 0
    
    for question, expected in TEST_QUESTIONS:
        # Format the message
        messages = [
            {"role": "developer", "content": "You are a model that can do function calling with the following functions"},
            {"role": "user", "content": question},
        ]
        
        # Apply chat template
        prompt = tokenizer.apply_chat_template(
            messages,
            tools=TOOLS,
            add_generation_prompt=True,
            tokenize=False
        )
        
        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        
        # Decode only new tokens
        new_tokens = outputs[0][inputs['input_ids'].shape[1]:]
        response = tokenizer.decode(new_tokens, skip_special_tokens=False)
        
        # Check for function call
        called = "none"
        if "<start_function_call>" in response and "call:" in response:
            # Parse function name
            if "call:thinking" in response:
                called = "thinking"
            elif "call:nonthinking" in response:
                called = "nonthinking"
        
        match = "OK" if called == expected else "MISS"
        if called == expected:
            correct += 1
        
        print(f"\nQ: {question}")
        print(f"  Expected: {expected}")
        print(f"  Called:   {called} [{match}]")
        print(f"  Raw output: {response[:150]}...")
    
    accuracy = 100 * correct / len(TEST_QUESTIONS)
    print(f"\n{'='*60}")
    print(f"ACCURACY: {correct}/{len(TEST_QUESTIONS)} ({accuracy:.1f}%)")
    print(f"{'='*60}")
    return correct

def main():
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    
    print("\n" + "="*60)
    print("DIRECT HUGGINGFACE MODEL TEST (No Ollama)")
    print("="*60)
    
    # Test base model
    print("\nLoading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    base_correct = test_model(base_model, tokenizer, "Base FunctionGemma")
    del base_model
    torch.cuda.empty_cache()
    
    # Test fine-tuned model
    print("\nLoading fine-tuned model...")
    try:
        finetuned_model = AutoModelForCausalLM.from_pretrained(
            FINETUNED_MODEL_DIR,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        ft_correct = test_model(finetuned_model, tokenizer, "Fine-tuned FunctionGemma")
    except Exception as e:
        print(f"Error loading fine-tuned model: {e}")
        ft_correct = 0
    
    # Summary
    print("\n" + "="*60)
    print("FINAL COMPARISON")
    print("="*60)
    print(f"Base model:      {base_correct}/{len(TEST_QUESTIONS)}")
    print(f"Fine-tuned:      {ft_correct}/{len(TEST_QUESTIONS)}")
    
    if ft_correct > base_correct:
        print(f"\n>>> Fine-tuned improved by +{ft_correct - base_correct}! <<<")
    elif ft_correct < base_correct:
        print(f"\n>>> Fine-tuned decreased by {ft_correct - base_correct} <<<")
    else:
        print(f"\n>>> Same performance <<<")

if __name__ == "__main__":
    main()
