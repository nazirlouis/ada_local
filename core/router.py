"""
FunctionGemma Router - Routes user prompts to appropriate functions.
Supports 9 functions: 6 actions, 1 context, 2 passthrough.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.utils import get_json_schema
from typing import Literal, Tuple, Dict, Any
import time
import re
import json

# Debug flag - set to True to see Gemma's raw response
DEBUG_ROUTER = True


# --- Tool Definitions (all 9 functions) ---

def control_light(action: str, device_name: str = None, brightness: int = None, color: str = None) -> str:
    """
    Control smart lights - turn on, off, dim, or change color.
    
    Args:
        action: Action to perform: on, off, dim, toggle
        device_name: Name of the light or room
        brightness: Brightness level 0-100
        color: Color name or hex code
    """
    return "result"

def set_timer(duration: str, label: str = None) -> str:
    """
    Set a countdown timer.
    
    Args:
        duration: Duration like '5 minutes' or '1 hour'
        label: Optional label for the timer
    """
    return "result"

def set_alarm(time: str, label: str = None) -> str:
    """
    Set an alarm for a specific time.
    
    Args:
        time: Time for alarm like '7am' or '14:30'
        label: Optional label
    """
    return "result"

def create_calendar_event(title: str, date: str = None, time: str = None, duration: int = None) -> str:
    """
    Create a calendar event.
    
    Args:
        title: Event title
        date: Date like 'tomorrow' or '2024-01-15'
        time: Time like '3pm'
        duration: Duration in minutes
    """
    return "result"

def add_task(text: str, priority: str = None) -> str:
    """
    Add a task to the to-do list.
    
    Args:
        text: Task description
        priority: Priority level
    """
    return "result"

def web_search(query: str) -> str:
    """
    Search the web for information.
    
    Args:
        query: Search query
    """
    return "result"

def get_system_info() -> str:
    """
    Get current system state including timers, calendar, tasks, devices, and weather.
    """
    return "result"

def thinking(prompt: str) -> str:
    """
    Use for complex queries requiring reasoning, math, coding, or multi-step analysis.
    
    Args:
        prompt: The user's original prompt
    """
    return "result"

def nonthinking(prompt: str) -> str:
    """
    Use for simple queries, greetings, factual questions not requiring deep reasoning.
    
    Args:
        prompt: The user's original prompt
    """
    return "result"


# Pre-compute tool schemas
TOOLS = [
    get_json_schema(control_light),
    get_json_schema(set_timer),
    get_json_schema(set_alarm),
    get_json_schema(create_calendar_event),
    get_json_schema(add_task),
    get_json_schema(web_search),
    get_json_schema(get_system_info),
    get_json_schema(thinking),
    get_json_schema(nonthinking),
]

SYSTEM_MSG = "You are a model that can do function calling with the following functions"

# All valid function names
VALID_FUNCTIONS = {
    "control_light", "set_timer", "set_alarm", "create_calendar_event",
    "add_task", "web_search", "get_system_info", "thinking", "nonthinking"
}


class FunctionGemmaRouter:
    """Routes user prompts to appropriate functions using fine-tuned FunctionGemma."""
    
    def __init__(self, model_path: str = "./merged_model", compile_model: bool = False):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading FunctionGemma Router on {device.upper()}...")
        start = time.time()
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # CPU often doesn't support bfloat16 natively
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map=device,
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
    def route(self, user_prompt: str) -> Tuple[str, Dict[str, Any]]:
        """
        Route a user prompt to the appropriate function.
        
        Returns:
            Tuple of (function_name, arguments_dict)
        """
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
            max_new_tokens=100,  # Increased for function args
            do_sample=False,
            use_cache=True,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        
        # Decode new tokens only
        new_tokens = outputs[0][inputs['input_ids'].shape[1]:]
        response = self.tokenizer.decode(new_tokens, skip_special_tokens=False)
        
        # Debug: Print raw Gemma response
        if DEBUG_ROUTER:
            print(f"\n{'='*50}")
            print(f"[Router DEBUG] User prompt: {user_prompt}")
            print(f"[Router DEBUG] Raw Gemma response:")
            print(f"  {repr(response)}")
            print(f"{'='*50}")
        
        # Parse function call
        return self._parse_function_call(response, user_prompt)
    
    def _parse_function_call(self, response: str, user_prompt: str) -> Tuple[str, Dict[str, Any]]:
        """Parse the model's response to extract function name and arguments."""
        
        # Try to find function call pattern: call:function_name
        for func_name in VALID_FUNCTIONS:
            if f"call:{func_name}" in response:
                # Try to extract arguments
                args = self._extract_arguments(response, func_name, user_prompt)
                return func_name, args
        
        # Fallback to nonthinking if no function found
        return "nonthinking", {"prompt": user_prompt}
    
    def _extract_arguments(self, response: str, func_name: str, user_prompt: str) -> Dict[str, Any]:
        """Extract arguments from the response."""
        
        # Default arguments for passthrough functions
        if func_name in ("thinking", "nonthinking"):
            return {"prompt": user_prompt}
        
        # For get_system_info, no args needed
        if func_name == "get_system_info":
            return {}
        
        # Parse the model's custom format: {key:<escape>value<escape>,key2:<escape>value2<escape>}
        # Find the arguments block after the function name
        pattern = rf"call:{func_name}\{{([^}}]+)\}}"
        match = re.search(pattern, response)
        
        if match:
            args_str = match.group(1)
            args = {}
            
            # Split by comma, but handle values with commas inside <escape> tags
            # Pattern: key:<escape>value<escape>
            arg_pattern = r'(\w+):<escape>([^<]*)<escape>'
            for arg_match in re.finditer(arg_pattern, args_str):
                key = arg_match.group(1)
                value = arg_match.group(2)
                
                # Try to convert to appropriate type
                if value.isdigit():
                    args[key] = int(value)
                elif value.lower() in ('true', 'false'):
                    args[key] = value.lower() == 'true'
                else:
                    args[key] = value
            
            if args:
                return args
        
        # Fallback: return user prompt as main argument
        if func_name == "control_light":
            return {"action": "toggle", "device_name": user_prompt}
        elif func_name == "set_timer":
            return {"duration": user_prompt}
        elif func_name == "set_alarm":
            return {"time": user_prompt}
        elif func_name == "create_calendar_event":
            return {"title": user_prompt}
        elif func_name == "add_task":
            return {"text": user_prompt}
        elif func_name == "web_search":
            return {"query": user_prompt}
        
        return {}
    
    def route_with_timing(self, user_prompt: str) -> Tuple[Tuple[str, Dict], float]:
        """Route with timing info."""
        start = time.time()
        result = self.route(user_prompt)
        elapsed = time.time() - start
        return result, elapsed


if __name__ == "__main__":
    router = FunctionGemmaRouter(compile_model=False)
    
    test_prompts = [
        # Action functions
        ("Turn on the living room lights", "control_light"),
        ("Set a timer for 10 minutes", "set_timer"),
        ("Wake me up at 7am", "set_alarm"),
        ("Schedule meeting tomorrow at 3pm", "create_calendar_event"),
        ("Add buy groceries to my list", "add_task"),
        ("Search for Italian recipes", "web_search"),
        # Context function
        ("How much time is left on my timer?", "get_system_info"),
        ("What's on my calendar today?", "get_system_info"),
        ("Are any lights on?", "get_system_info"),
        # Passthrough functions
        ("Explain quantum computing", "thinking"),
        ("Write a Python function to sort a list", "thinking"),
        ("Hello there!", "nonthinking"),
        ("What's the capital of France?", "nonthinking"),
    ]
    
    print("\n" + "="*70)
    print("FUNCTION CALLING ROUTER TEST")
    print("="*70)
    
    total_time = 0
    correct = 0
    
    for prompt, expected in test_prompts:
        (func_name, args), elapsed = router.route_with_timing(prompt)
        total_time += elapsed
        match = "✓" if func_name == expected else "✗"
        if func_name == expected:
            correct += 1
        
        print(f"\n[{match}] {prompt}")
        print(f"    → {func_name}({args}) [{elapsed*1000:.0f}ms]")
    
    avg_time = total_time / len(test_prompts)
    print(f"\n{'='*70}")
    print(f"Accuracy: {correct}/{len(test_prompts)} ({100*correct/len(test_prompts):.0f}%)")
    print(f"Average routing time: {avg_time*1000:.0f}ms per prompt")
    print(f"Total time: {total_time:.2f}s for {len(test_prompts)} prompts")
