"""
Model Manager - Utilities for loading/unloading Ollama models.
"""

import requests
import threading
from config import OLLAMA_URL, GRAY, RESET


def sync_unload_model(model_name: str):
    """
    Synchronously unload a model from Ollama.
    """
    try:
        # Send a request with keep_alive=0 to unload
        response = requests.post(
            f"{OLLAMA_URL}/generate",
            json={
                "model": model_name,
                "prompt": "",
                "keep_alive": 0  # Immediately unload
            },
            timeout=5
        )
        if response.status_code == 200:
            print(f"{GRAY}[ModelManager] Unloaded model: {model_name}{RESET}")
        else:
            print(f"{GRAY}[ModelManager] Failed to unload {model_name}: {response.status_code}{RESET}")
    except Exception as e:
        print(f"{GRAY}[ModelManager] Error unloading {model_name}: {e}{RESET}")


def unload_model(model_name: str):
    """
    Unload a model from Ollama to free up VRAM.
    Uses keep_alive=0 to immediately unload.
    """
    # Run in background to not block UI
    threading.Thread(target=sync_unload_model, args=(model_name,), daemon=True).start()


def unload_all_models(sync: bool = False):
    """Unload all running models in Ollama."""
    try:
        response = requests.get(f"{OLLAMA_URL}/ps", timeout=2)
        if response.status_code == 200:
            data = response.json()
            models = data.get("models", [])
            for model in models:
                model_name = model.get("name", "")
                if model_name:
                    if sync:
                        sync_unload_model(model_name)
                    else:
                        unload_model(model_name)
    except Exception as e:
        print(f"{GRAY}[ModelManager] Error getting running models: {e}{RESET}")


def get_running_models() -> list:
    """Get list of currently running model names."""
    try:
        response = requests.get(f"{OLLAMA_URL}/ps", timeout=2)
        if response.status_code == 200:
            data = response.json()
            return [m.get("name", "") for m in data.get("models", [])]
    except:
        pass
    return []


def ensure_exclusive_qwen(target_model: str):
    """
    Ensure that no other Qwen models are running except for the target.
    Helpful for VRAM constrained systems.
    """
    try:
        running = get_running_models()
        # Find all qwen models that aren't the target
        # Note: Ollama model names might have tags, so we check for 'qwen' in name
        to_unload = [
            m for m in running 
            if "qwen" in m.lower() and m != target_model and not target_model.startswith(m)
        ]
        
        for m in to_unload:
            unload_model(m)
            
    except Exception as e:
        print(f"{GRAY}[ModelManager] Error in exclusion logic: {e}{RESET}")
