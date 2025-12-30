import requests
import json
import time
import sys
import psutil

# Model configurations: (model_name, think_enabled, display_name)
MODELS = [
    {"model": "qwen3:0.6b", "think": False, "name": "qwen3:0.6b"},
    #{"model": "qwen3:0.6b", "think": True, "name": "qwen3:0.6b (think)"},
    {"model": "qwen3:1.7b", "think": False, "name": "qwen3:1.7b"},
    #{"model": "qwen3:1.7b", "think": True, "name": "qwen3:1.7b (think)"},
    {"model": "gemma3:1b", "think": False, "name": "gemma3:1b"},
    {"model": "gemma3:270m", "think": False, "name": "gemma3:270m"},
    {"model": "deepseek-r1:1.5b", "think": False, "name": "deepseek-r1:1.5b"},
    #{"model": "deepseek-r1:1.5b", "think": True, "name": "deepseek-r1:1.5b (think)"},
] 

# Ground truth Q&A pairs for accuracy testing
QA_PAIRS = [
    # Math
    {"prompt": "What is 15 + 27? Answer with just the number.", "expected": ["42"]},
    {"prompt": "What is 144 divided by 12? Answer with just the number.", "expected": ["12"]},
    {"prompt": "What is the square root of 81? Answer with just the number.", "expected": ["9"]},
    {"prompt": "How many prime numbers are between 1 and 10? Answer with just the number.", "expected": ["4"]},
    {"prompt": "What is 7 x 8? Answer with just the number.", "expected": ["56"]},
    
    # Geography
    {"prompt": "What is the capital of France? Answer with just the city name.", "expected": ["Paris"]},
    {"prompt": "What is the capital of Japan? Answer with just the city name.", "expected": ["Tokyo"]},
    {"prompt": "What is the largest ocean on Earth? Answer with just the name.", "expected": ["Pacific"]},
    {"prompt": "What continent is Egypt in? Answer with just the continent name.", "expected": ["Africa"]},
    {"prompt": "What is the longest river in the world? Answer with just the name.", "expected": ["Nile", "Amazon"]},  # Both accepted
    
    # Science
    {"prompt": "What is the chemical symbol for gold? Answer with just the symbol.", "expected": ["Au"]},
    {"prompt": "What is the chemical symbol for water? Answer with just the formula.", "expected": ["H2O"]},
    {"prompt": "How many planets are in our solar system? Answer with just the number.", "expected": ["8"]},
    {"prompt": "What is the boiling point of water in Celsius? Answer with just the number.", "expected": ["100"]},
    {"prompt": "What gas do plants absorb from the atmosphere? Answer with just the name.", "expected": ["carbon dioxide", "CO2"]},
    
    # History & General Knowledge  
    {"prompt": "In what year did World War II end? Answer with just the year.", "expected": ["1945"]},
    {"prompt": "Who wrote Romeo and Juliet? Answer with just the name.", "expected": ["Shakespeare", "William Shakespeare"]},
    {"prompt": "How many sides does a hexagon have? Answer with just the number.", "expected": ["6"]},
    {"prompt": "What is the freezing point of water in Fahrenheit? Answer with just the number.", "expected": ["32"]},
    {"prompt": "How many days are in a leap year? Answer with just the number.", "expected": ["366"]},
]

URL_GENERATE = "http://localhost:11434/api/chat"
URL_PS = "http://localhost:11434/api/ps"

def get_ram_usage(model_name):
    """
    Checks System RAM and Model VRAM/Size via Ollama API.
    """
    # 1. Get Total System RAM Usage (in GB)
    mem = psutil.virtual_memory()
    system_used_gb = mem.used / (1024 ** 3)
    
    # 2. Get Model Specific Memory from Ollama API
    model_mem_gb = 0.0
    try:
        response = requests.get(URL_PS)
        if response.status_code == 200:
            models_data = response.json().get('models', [])
            for m in models_data:
                # Find our model in the list of running models
                if m['name'] == model_name or m['model'] == model_name:
                    # 'size' is usually bytes in VRAM/RAM
                    model_mem_gb = m.get('size', 0) / (1024 ** 3)
                    break
    except:
        pass

    return system_used_gb, model_mem_gb

def run_benchmark(model_name, prompt, think=False):
    """
    Streams the response to calculate TTFT, TPS, and captures response text.
    """
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,    # MUST be True for TTFT
        "think": think,    # Thinking mode toggle
        "options": {
            "temperature": 0.5, 
            "seed": 42
        }
    }

    ttft = 0
    total_tokens = 0
    eval_duration_ns = 0
    response_text = ""
    
    start_time = time.time()
    first_token_received = False
    
    try:
        # Send Request
        with requests.post(URL_GENERATE, json=payload, stream=True) as r:
            r.raise_for_status()
            
            for line in r.iter_lines():
                if not line: continue
                
                # Parse Chunk
                try:
                    chunk = json.loads(line.decode('utf-8'))
                except:
                    continue
                
                # Show thinking content in real-time (for thinking models)
                thinking_content = chunk.get('message', {}).get('thinking', '')
                if thinking_content and think:
                    sys.stdout.write(f"\033[90m{thinking_content}\033[0m")  # Gray text
                    sys.stdout.flush()
                
                # Accumulate response text
                content = chunk.get('message', {}).get('content', '')
                response_text += content
                
                # 1. Measure TTFT (Time to First Token)
                if not first_token_received:
                    # As soon as we get the first chunk with content, stop the timer
                    if content:
                        ttft = time.time() - start_time
                        first_token_received = True

                # 2. Capture Final Stats (Ollama sends these in the last chunk)
                if chunk.get('done') is True:
                    total_tokens = chunk.get('eval_count', 0)
                    eval_duration_ns = chunk.get('eval_duration', 0)

        # Calculate TPS (Tokens Per Second)
        # We use the API's reported duration for precision, falling back to wall time if needed
        if eval_duration_ns > 0:
            tps = total_tokens / (eval_duration_ns / 1_000_000_000)
        else:
            # Fallback if API doesn't report duration (rare)
            total_time = time.time() - start_time
            tps = total_tokens / total_time if total_time > 0 else 0

        return tps, ttft, total_tokens, response_text

    except Exception as e:
        print(f"\nError: {e}")
        return 0, 0, 0, ""


def check_accuracy(response, expected_answers):
    """
    Check if any of the expected answers appear in the response.
    Case-insensitive matching with unicode subscript normalization.
    """
    # Normalize unicode subscripts to ASCII (e.g., H₂O → H2O, CO₂ → CO2)
    subscript_map = str.maketrans('₀₁₂₃₄₅₆₇₈₉', '0123456789')
    response_normalized = response.translate(subscript_map).lower()
    
    for answer in expected_answers:
        if answer.lower() in response_normalized:
            return True
    return False

def main():
    model_names = [m["name"] for m in MODELS]
    print(f"Benchmark: {', '.join(model_names)}")
    print(f"Testing with {len(QA_PAIRS)} ground truth Q&A pairs\n")
    print(f"{'MODEL':<25} | {'METRIC':<12} | {'VALUE':<10} | {'NOTE'}")
    print("-" * 85)

    for config in MODELS:
        model = config["model"]
        think = config["think"]
        display_name = config["name"]
        
        # --- 1. WARMUP & RAM CHECK ---
        # We run a silent request to force the model into memory
        sys.stdout.write(f"Loading {display_name}...\r")
        sys.stdout.flush()
        
        try:
            # Simple warmup
            run_benchmark(model, "hi", think) 
            
            # Now that it's loaded, check RAM
            sys_ram, model_ram = get_ram_usage(model)
        except Exception as e:
            print(f"Could not load {display_name}: {e}")
            continue

        # --- 2. RUN TESTS ---
        total_tps = 0
        total_ttft = 0
        count = 0
        correct = 0
        wrong_answers = []  # Track which ones it got wrong
        
        print(f"{display_name:<25} | {'RAM (Sys)':<12} | {sys_ram:.1f} GB     | Total System Memory Used")
        print(f"{display_name:<25} | {'VRAM/Size':<12} | {model_ram:.1f} GB     | Model Size in Memory")

        for i, qa in enumerate(QA_PAIRS):
            sys.stdout.write(f"  Testing {display_name}: {i+1}/{len(QA_PAIRS)}...\r")
            sys.stdout.flush()
            
            tps, ttft, tokens, response = run_benchmark(model, qa["prompt"], think)
            
            if tps > 0:
                total_tps += tps
                total_ttft += ttft
                count += 1
                
                # Check accuracy
                if check_accuracy(response, qa["expected"]):
                    correct += 1
                else:
                    wrong_answers.append({
                        "q": qa["prompt"][:40] + "...",
                        "expected": qa["expected"],
                        "got": response.strip()[:50]
                    })

        # --- 3. RESULTS ---
        if count > 0:
            avg_tps = total_tps / count
            avg_ttft = total_ttft / count
            accuracy = (correct / len(QA_PAIRS)) * 100
            
            # Print Final Stats for this model
            print(f"{display_name:<25} | {'Avg Speed':<12} | {avg_tps:<6.1f} T/s | Generation Speed")
            print(f"{display_name:<25} | {'Avg TTFT':<12} | {avg_ttft*1000:<6.0f} ms   | Time to First Token")
            print(f"{display_name:<25} | {'Accuracy':<12} | {correct}/{len(QA_PAIRS)} ({accuracy:.0f}%) | Ground Truth Score")
            
            # Show wrong answers if any (optional verbose)
            if wrong_answers and len(wrong_answers) <= 5:
                for w in wrong_answers:
                    print(f"  ❌ Expected {w['expected']}, got: {w['got']}")
            elif wrong_answers:
                print(f"  ❌ {len(wrong_answers)} incorrect answers")
                
            print("-" * 85)
        else:
            print(f"{display_name:<25} | Failed to run tests.")

if __name__ == "__main__":
    main()