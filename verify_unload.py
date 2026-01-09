
import requests

OLLAMA_URL = "http://localhost:11434/api"
GRAY = "\033[90m"
RESET = "\033[0m"

def sync_unload_model(model_name: str):
    try:
        response = requests.post(
            f"{OLLAMA_URL}/generate",
            json={
                "model": model_name,
                "prompt": "",
                "keep_alive": 0
            },
            timeout=5
        )
        if response.status_code == 200:
            print(f"{GRAY}[ModelManager] Unloaded model: {model_name}{RESET}")
        else:
            print(f"{GRAY}[ModelManager] Failed to unload {model_name}: {response.status_code}{RESET}")
    except Exception as e:
        print(f"{GRAY}[ModelManager] Error unloading {model_name}: {e}{RESET}")

def unload_all_models():
    try:
        response = requests.get(f"{OLLAMA_URL}/ps", timeout=2)
        if response.status_code == 200:
            data = response.json()
            models = data.get("models", [])
            if not models:
                print("No models running.")
                return
            for model in models:
                model_name = model.get("name", "")
                if model_name:
                    sync_unload_model(model_name)
    except Exception as e:
        print(f"{GRAY}[ModelManager] Error getting running models: {e}{RESET}")

if __name__ == "__main__":
    unload_all_models()
