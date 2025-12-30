#!/usr/bin/env python3
"""
FunctionGemma Baseline Test (Pre-Fine-Tuning)
Uses proper FunctionGemma control tokens for accurate measurement.
"""

import requests
import re
import sys
import json
from datetime import datetime
from pathlib import Path
from collections import defaultdict

MODEL = "functiongemma:270m"
URL = "http://localhost:11434/api/generate"

# Official FunctionGemma JSON Schema format for function definitions
FUNCTIONS = [
    {
        "type": "function",
        "function": {
            "name": "control_light",
            "description": "Controls smart lights - turn on, off, or dim lights in a room",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {"type": "string", "description": "The action to perform: on, off, or dim"},
                    "room": {"type": "string", "description": "The room name where the light is located"}
                },
                "required": ["action", "room"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Searches the web for information using Google",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query string"}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "set_timer",
            "description": "Sets a countdown timer for a specified duration",
            "parameters": {
                "type": "object",
                "properties": {
                    "duration": {"type": "string", "description": "Time duration like 5 minutes or 1 hour"},
                    "label": {"type": "string", "description": "Optional timer name or label"}
                },
                "required": ["duration"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "create_calendar_event",
            "description": "Creates a new calendar event or appointment",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "The event title"},
                    "date": {"type": "string", "description": "The date of the event"},
                    "time": {"type": "string", "description": "The time of the event"},
                    "description": {"type": "string", "description": "Optional event details"}
                },
                "required": ["title", "date"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_calendar",
            "description": "Reads and retrieves calendar events for a date or time range",
            "parameters": {
                "type": "object",
                "properties": {
                    "date": {"type": "string", "description": "The date or date range to check"},
                    "filter": {"type": "string", "description": "Optional filter like meetings or appointments"}
                },
                "required": ["date"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "passthrough",
            "description": "Routes the query to a conversational AI model when no specific tool action is needed. Use this for greetings, general questions, explanations, and conversations that don't require lights, timers, search, or calendar actions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "thinking": {"type": "boolean", "description": "Set to true for complex reasoning, math, logic, or multi-step problems. Set to false for simple greetings, chitchat, or straightforward questions."}
                },
                "required": ["thinking"]
            }
        }
    }
]

# Helper to get function names for extraction
FUNCTION_NAMES = [f["function"]["name"] for f in FUNCTIONS]

TEST_CASES = [
    # Light control - on/off/dim
    {"prompt": "Turn on the bedroom lights", "expected": "control_light"},
    {"prompt": "Switch off the kitchen light", "expected": "control_light"},
    {"prompt": "Turn off all the lights", "expected": "control_light"},
    {"prompt": "Dim the living room lights", "expected": "control_light"},
    {"prompt": "Can you turn on the lights in the office", "expected": "control_light"},
    {"prompt": "Lights on in the bathroom please", "expected": "control_light"},
    
    # Web search
    {"prompt": "Search for the best restaurants in New York", "expected": "web_search"},
    {"prompt": "Look up how to make pasta", "expected": "web_search"},
    {"prompt": "Google the weather in Paris", "expected": "web_search"},
    {"prompt": "Find information about quantum computing", "expected": "web_search"},
    {"prompt": "What are the latest news on AI?", "expected": "web_search"},
    {"prompt": "Search the web for Python tutorials", "expected": "web_search"},
    
    # Timer
    {"prompt": "Set a timer for 5 minutes", "expected": "set_timer"},
    {"prompt": "Start a 30 second timer", "expected": "set_timer"},
    {"prompt": "Timer for 1 hour please", "expected": "set_timer"},
    {"prompt": "Set a 10 minute timer for the oven", "expected": "set_timer"},
    {"prompt": "Can you set a timer for 2 hours", "expected": "set_timer"},
    {"prompt": "Start a pomodoro timer for 25 minutes", "expected": "set_timer"},
    
    # Create calendar event
    {"prompt": "Add a meeting to my calendar for tomorrow at 3pm", "expected": "create_calendar_event"},
    {"prompt": "Schedule a dentist appointment for next Monday", "expected": "create_calendar_event"},
    {"prompt": "Create a calendar event for my birthday party on Saturday", "expected": "create_calendar_event"},
    {"prompt": "Add dinner with Sarah to my calendar for Friday at 7pm", "expected": "create_calendar_event"},
    {"prompt": "Schedule a team standup for tomorrow morning at 9am", "expected": "create_calendar_event"},
    {"prompt": "Put a reminder on my calendar for the doctor visit next week", "expected": "create_calendar_event"},
    
    # Read calendar
    {"prompt": "What do I have on my calendar today?", "expected": "read_calendar"},
    {"prompt": "Show me my schedule for tomorrow", "expected": "read_calendar"},
    {"prompt": "What meetings do I have this week?", "expected": "read_calendar"},
    {"prompt": "Check my calendar for next Monday", "expected": "read_calendar"},
    {"prompt": "Am I free on Friday afternoon?", "expected": "read_calendar"},
    {"prompt": "What's on my schedule for the rest of the day?", "expected": "read_calendar"},
    
    # Passthrough - no thinking (simple greetings, chitchat)
    {"prompt": "Hello", "expected": "passthrough"},
    {"prompt": "Hi there!", "expected": "passthrough"},
    {"prompt": "Good morning", "expected": "passthrough"},
    {"prompt": "How are you?", "expected": "passthrough"},
    {"prompt": "Thanks for your help", "expected": "passthrough"},
    {"prompt": "What's your name?", "expected": "passthrough"},
    
    # Passthrough - with thinking (complex reasoning, math, logic)
    {"prompt": "What are the prime numbers from 1 to 10?", "expected": "passthrough"},
    {"prompt": "Explain how photosynthesis works", "expected": "passthrough"},
    {"prompt": "If I have 3 apples and give away 2, how many do I have?", "expected": "passthrough"},
    {"prompt": "What are the pros and cons of electric cars?", "expected": "passthrough"},
    {"prompt": "Write a haiku about the ocean", "expected": "passthrough"},
    {"prompt": "Solve this riddle: I have cities but no houses", "expected": "passthrough"},
]


def esc(s):
    """Wrap string with <escape> tokens."""
    return f"<escape>{s}<escape>"


def build_function_declarations():
    """
    Build function declarations using FunctionGemma control tokens.
    Uses official JSON schema format converted to FunctionGemma's token format.
    """
    declarations = ""
    for func_def in FUNCTIONS:
        func = func_def["function"]
        name = func["name"]
        desc = func["description"]
        params = func["parameters"]
        
        # Build properties string
        props_parts = []
        for prop_name, prop_spec in params["properties"].items():
            props_parts.append(
                f"{prop_name}:{{description:{esc(prop_spec['description'])},type:{esc(prop_spec['type'].upper())}}}"
            )
        props_str = ",".join(props_parts)
        
        # Build required array
        required_str = ",".join([esc(r) for r in params.get("required", [])])
        
        # Build full declaration
        declarations += (
            f"<start_function_declaration>"
            f"declaration:{name}{{"
            f"description:{esc(desc)},"
            f"parameters:{{properties:{{{props_str}}},required:[{required_str}],type:{esc('OBJECT')}}}"
            f"}}<end_function_declaration>"
        )
    return declarations


def build_prompt(user_query):
    """Build prompt using FunctionGemma's exact format."""
    declarations = build_function_declarations()
    
    return (
        f"<start_of_turn>developer "
        f"You are a model that can do function calling with the following functions"
        f"{declarations}"
        f"<end_of_turn>\n"
        f"<start_of_turn>user {user_query}<end_of_turn>\n"
        f"<start_of_turn>model"
    )


def extract_function(response):
    """Extract function name from FunctionGemma response."""
    # Pattern: <start_function_call>call:function_name{...}
    match = re.search(r"call:(\w+)", response)
    if match:
        return match.group(1)
    
    # Fallback: check for function names in response
    for func_name in FUNCTION_NAMES:
        if func_name in response:
            return func_name
    return None


def run_test(prompt, verbose=False):
    """Run test using generate API for raw token control."""
    full_prompt = build_prompt(prompt)
    
    if verbose:
        print(f"\n--- PROMPT ---\n{full_prompt[:500]}...\n")
    
    payload = {
        "model": MODEL,
        "prompt": full_prompt,
        "stream": False,
        "raw": True,  # Use raw mode to preserve tokens
        "options": {
            "temperature": 0.0,
            "seed": 42,
            "num_predict": 150,
            "stop": ["<end_of_turn>", "<start_function_response>"]
        }
    }
    
    try:
        r = requests.post(URL, json=payload, timeout=30)
        return r.json().get("response", "")
    except Exception as e:
        return f"ERROR: {e}"


def save_log(results, accuracy, model):
    """Save test results to a timestamped log file."""
    log_dir = Path(__file__).parent / "logs"
    log_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"function_test_{timestamp}.json"
    
    # Build log entries with full details
    log_entries = []
    for r in results:
        log_entries.append({
            "input": r["full_prompt"],
            "output": r["raw"],
            "expected_function": r["expected"],
            "detected_function": r["called"],
            "correct": r["correct"],
            "prompt_text": r["prompt"]
        })
    
    log_data = {
        "timestamp": datetime.now().isoformat(),
        "model": model,
        "accuracy": accuracy,
        "total_tests": len(results),
        "correct_count": sum(1 for r in results if r["correct"]),
        "results": log_entries
    }
    
    with open(log_file, "w") as f:
        json.dump(log_data, f, indent=2)
    
    return log_file


def main():
    print("=" * 70)
    print(f"FunctionGemma Baseline Test | Model: {MODEL}")
    print("Using proper FunctionGemma control tokens")
    print("=" * 70)
    
    # Show sample prompt/response
    print("\n📋 Sample (first test case):")
    sample = run_test(TEST_CASES[0]["prompt"], verbose=True)
    print(f"Response: {sample[:200]}")
    print("-" * 70)
    
    correct = 0
    results = []
    by_func = defaultdict(lambda: {"total": 0, "correct": 0})
    
    for i, test in enumerate(TEST_CASES):
        sys.stdout.write(f"\rTesting {i+1}/{len(TEST_CASES)}...")
        sys.stdout.flush()
        
        full_prompt = build_prompt(test["prompt"])
        response = run_test(test["prompt"])
        called = extract_function(response)
        expected = test["expected"]
        is_correct = called == expected
        
        if is_correct:
            correct += 1
        
        by_func[expected]["total"] += 1
        if is_correct:
            by_func[expected]["correct"] += 1
            
        results.append({
            "prompt": test["prompt"],
            "full_prompt": full_prompt,
            "expected": expected,
            "called": called, 
            "correct": is_correct, 
            "raw": response
        })
    
    print("\r" + " " * 30)
    
    # Results table with full output
    print(f"\n{'='*80}")
    for i, r in enumerate(results):
        status = "✓" if r["correct"] else "✗"
        print(f"\n[{i+1}] {status} {r['prompt']}")
        print(f"    Expected: {r['expected']:<16} Got: {r['called'] or 'None'}")
        print(f"    Raw: {r['raw']}")
    
    accuracy = (correct / len(TEST_CASES)) * 100
    print("-" * 80)
    print(f"\n🎯 BASELINE ACCURACY: {correct}/{len(TEST_CASES)} ({accuracy:.1f}%)")
    
    print("\nBy Function:")
    for func, stats in sorted(by_func.items()):
        pct = (stats["correct"] / stats["total"]) * 100 if stats["total"] > 0 else 0
        bar = "█" * int(pct / 10) + "░" * (10 - int(pct / 10))
        print(f"  {func:<16} [{bar}] {stats['correct']}/{stats['total']}")
    
    # Show some raw responses for debugging
    print("\n📝 Sample raw responses:")
    for r in results[:3]:
        print(f"  '{r['prompt'][:30]}...' → '{r['raw']}'")
    
    # Save log file
    log_file = save_log(results, accuracy, MODEL)
    print(f"\n📁 Log saved to: {log_file}")
    
    print("\n" + "=" * 70)
    print("📝 Save this baseline, then run again after fine-tuning!")
    print("=" * 70)


if __name__ == "__main__":
    main()
