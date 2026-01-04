"""
Fast FunctionGemma Router - Optimized version.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.utils import get_json_schema
from typing import Literal
import time

# Define tools at module level (required for get_json_schema)
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

# Pre-compute tool schemas
TOOLS = [get_json_schema(thinking), get_json_schema(nonthinking)]
SYSTEM_MSG = "You are a model that can do function calling with the following functions"


class FunctionGemmaRouter:
    """Routes user prompts to 'thinking' or 'nonthinking' using fine-tuned FunctionGemma."""
    
    def __init__(self, model_path: str = "./merged_model", compile_model: bool = True):
        print("Loading FunctionGemma Router...")
        start = time.time()
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
        )
        self.model.eval()
        
        # Compile for speed (PyTorch 2.0+)
        if compile_model:
            try:
                self.model = torch.compile(self.model, mode="reduce-overhead")
                print("Model compiled with torch.compile()")
            except Exception as e:
                print(f"torch.compile() not available: {e}")
        
        print(f"Router loaded in {time.time() - start:.2f}s")
        print(f"Device: {self.model.device}, Dtype: {self.model.dtype}")
    
    @torch.inference_mode()
    def route(self, user_prompt: str) -> Literal["thinking", "nonthinking"]:
        """Route a user prompt to the appropriate function."""
        # Build messages
        messages = [
            {"role": "developer", "content": SYSTEM_MSG},
            {"role": "user", "content": user_prompt},
        ]
        
        # Apply chat template
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tools=TOOLS,
            add_generation_prompt=True,
            tokenize=False
        )
        
        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        # Generate with minimal settings for speed
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=30,  # Function call is very short
            do_sample=False,
            use_cache=True,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        
        # Decode new tokens only
        new_tokens = outputs[0][inputs['input_ids'].shape[1]:]
        response = self.tokenizer.decode(new_tokens, skip_special_tokens=False)
        
        # Parse function call
        if "call:thinking" in response:
            return "thinking"
        elif "call:nonthinking" in response:
            return "nonthinking"
        
        # Default fallback
        return "nonthinking"
    
    def route_with_timing(self, user_prompt: str) -> tuple[Literal["thinking", "nonthinking"], float]:
        """Route with timing info."""
        start = time.time()
        result = self.route(user_prompt)
        elapsed = time.time() - start
        return result, elapsed


if __name__ == "__main__":
    router = FunctionGemmaRouter(compile_model=False)  # Disable compile for first test
    
    test_prompts = [
        ("What is the capital of France?", "nonthinking"),
        ("Hello there!", "nonthinking"),
        ("Write a Python function to reverse a string", "thinking"),
        ("What is 2 + 2?", "nonthinking"),
        ("Explain the theory of relativity", "thinking"),
        ("Design a database schema for e-commerce", "thinking"),
        ("Who wrote Harry Potter?", "nonthinking"),
        ("Solve the differential equation dy/dx = y", "thinking"),
    ]
    
    print("\n" + "="*60)
    print("ROUTING TEST")
    print("="*60)
    
    total_time = 0
    correct = 0
    for prompt, expected in test_prompts:
        result, elapsed = router.route_with_timing(prompt)
        total_time += elapsed
        match = "OK" if result == expected else "MISS"
        if result == expected:
            correct += 1
        print(f"\n[{result:12}] ({elapsed*1000:.0f}ms) [{match}] {prompt}")
    
    avg_time = total_time / len(test_prompts)
    print(f"\n{'='*60}")
    print(f"Accuracy: {correct}/{len(test_prompts)} ({100*correct/len(test_prompts):.0f}%)")
    print(f"Average routing time: {avg_time*1000:.0f}ms per prompt")
    print(f"Total time: {total_time:.2f}s for {len(test_prompts)} prompts")
