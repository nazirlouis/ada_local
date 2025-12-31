import requests
import json
import sys
import os
import re
import wave
import threading
import queue
import sounddevice as sd
import numpy as np
from pathlib import Path

# ANSI Escape Codes for coloring output
GRAY = "\033[90m"
RESET = "\033[0m"
BOLD = "\033[1m"
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"

# --- Model Configuration ---
ROUTER_MODEL = "functiongemma-ada"  # Fast function routing
RESPONDER_MODEL = "qwen3:1.7b"       # Conversational responses
OLLAMA_URL = "http://localhost:11434/api"

# Persistent Session for faster HTTP
http_session = requests.Session()

# Keywords that trigger the Router (otherwise we default to chat)
ROUTER_KEYWORDS = [
    # Tools
    "turn", "light", "dim", "switch",   # Lights
    "search", "google", "find", "look", # Search
    "timer", "alarm", "clock",          # Timers
    "calendar", "schedule", "appoint", "meet", "event", # Calendar
    
    # Complexity / Thinking Triggers (from Training Data)
    "explain", "how", "why", "cause", "difference", "compare", "meaning", # Reasoning
    "solve", "calculate", "equation", "math", "+", "*", "divide", "minus", # Math
    "write", "poem", "haiku", "riddle", "story", # Creative
    "if", "when" # Conditionals
]

def should_bypass_router(text):
    """Return True if text definitely doesn't need routing."""
    text = text.lower()
    return not any(k in text for k in ROUTER_KEYWORDS)

# --- Function Definitions (Official JSON Schema) ---
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
            "description": "DEFAULT FUNCTION - Use this whenever no other function is clearly needed. This is the fallback for: greetings (hello, hi, good morning), chitchat (how are you, what's your name), general knowledge questions, explanations, conversations, and ANY query that does NOT explicitly require controlling lights, setting timers, searching the web, or managing calendar events. When in doubt, use passthrough.",
            "parameters": {
                "type": "object",
                "properties": {
                    "thinking": {"type": "boolean", "description": "Set to true for complex reasoning/math/logic, false for simple greetings and chitchat."}
                },
                "required": ["thinking"]
            }
        }
    }
]

# --- FunctionGemma Prompt Builder ---
def esc(s):
    """Wrap string in escape tokens."""
    return f"<escape>{s}<escape>"

def build_function_declarations():
    """Build function declarations using FunctionGemma control tokens."""
    declarations = ""
    for func_def in FUNCTIONS:
        func = func_def["function"]
        name = func["name"]
        desc = func["description"]
        params = func["parameters"]
        
        props_parts = []
        for prop_name, prop_spec in params["properties"].items():
            props_parts.append(
                f"{prop_name}:{{description:{esc(prop_spec['description'])},type:{esc(prop_spec['type'].upper())}}}"
            )
        props_str = ",".join(props_parts)
        required_str = ",".join([esc(r) for r in params.get("required", [])])
        
        declarations += (
            f"<start_function_declaration>"
            f"declaration:{name}{{"
            f"description:{esc(desc)},"
            f"parameters:{{properties:{{{props_str}}},required:[{required_str}],type:{esc('OBJECT')}}}"
            f"}}<end_function_declaration>"
        )
    return declarations

def build_router_prompt(user_query):
    """Build FunctionGemma prompt for routing."""
    declarations = build_function_declarations()
    return (
        f"<start_of_turn>developer "
        f"You are a model that can do function calling with the following functions"
        f"{declarations}"
        f"<end_of_turn>\n"
        f"<start_of_turn>user {user_query}<end_of_turn>\n"
        f"<start_of_turn>model"
    )

def parse_function_call(response):
    """Extract function name and parameters from FunctionGemma response."""
    # Pattern: call:function_name{param1:<escape>value<escape>,...}
    match = re.search(r"call:(\w+)\{([^}]*)\}", response)
    if match:
        func_name = match.group(1)
        params_str = match.group(2)
        
        # Parse parameters: key:<escape>value<escape>
        params = {}
        param_matches = re.findall(r"(\w+):<escape>([^<]*)<escape>", params_str)
        for key, value in param_matches:
            # Handle boolean values
            if value.lower() == "true":
                params[key] = True
            elif value.lower() == "false":
                params[key] = False
            elif key == "thinking":
                # If thinking has any non-boolean content, treat as True (complex query)
                params[key] = True if value.strip() else False
            else:
                params[key] = value
        
        return func_name, params
    
    # Fallback: just extract function name
    name_match = re.search(r"call:(\w+)", response)
    if name_match:
        return name_match.group(1), {}
    
    return None, {}

def route_query(user_input):
    """Route user query through FunctionGemma to determine action."""
    prompt = build_router_prompt(user_input)
    
    payload = {
        "model": ROUTER_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.0,
            "num_predict": 50,
            "stop": ["<end_of_turn>", "<start_function_response>", "<end_function_call>"]
        }
    }
    
    try:
        response = http_session.post(f"{OLLAMA_URL}/generate", json=payload, timeout=10)
        response.raise_for_status()
        result = response.json()
        raw_output = result.get("response", "")
        return parse_function_call(raw_output)
    except Exception as e:
        print(f"{GRAY}[Router Error: {e}]{RESET}")
        return "passthrough", {"thinking": False}

# --- Function Execution Stubs ---
def execute_function(name, params):
    """Execute function and return response string."""
    if name == "control_light":
        action = params.get("action", "toggle")
        room = params.get("room", "room")
        if action == "on":
            return f"💡 Turned on the {room} lights."
        elif action == "off":
            return f"💡 Turned off the {room} lights."
        elif action == "dim":
            return f"💡 Dimmed the {room} lights."
        else:
            return f"💡 {action.capitalize()} the {room} lights."
    
    elif name == "web_search":
        query = params.get("query", "")
        return f"🔍 Searching the web for: {query}"
    
    elif name == "set_timer":
        duration = params.get("duration", "")
        label = params.get("label", "Timer")
        return f"⏱️ Timer set for {duration}" + (f" ({label})" if label else "")
    
    elif name == "create_calendar_event":
        title = params.get("title", "Event")
        date = params.get("date", "")
        time = params.get("time", "")
        return f"📅 Created event: {title} on {date}" + (f" at {time}" if time else "")
    
    elif name == "read_calendar":
        date = params.get("date", "today")
        return f"📆 Checking calendar for {date}..."
    
    else:
        return f"Unknown function: {name}"


# --- Piper TTS Integration ---
class PiperTTS:
    """Lightweight Piper TTS wrapper with streaming sentence support."""
    
    VOICE_MODEL = "en_GB-alba-medium"
    MODEL_URL = "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_GB/alba/medium/en_GB-alba-medium.onnx"
    CONFIG_URL = "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_GB/alba/medium/en_GB-alba-medium.onnx.json"
    
    def __init__(self):
        self.enabled = False
        self.voice = None
        self.speech_queue = queue.Queue()
        self.worker_thread = None
        self.running = False
        self.models_dir = Path.home() / ".local" / "share" / "piper" / "voices"
        
        try:
            from piper import PiperVoice
            self.PiperVoice = PiperVoice
            self.available = True
        except ImportError:
            self.available = False
            print(f"{GRAY}[TTS] piper-tts not installed. Run: pip install piper-tts{RESET}")
    
    def download_model(self):
        """Download voice model if not present."""
        self.models_dir.mkdir(parents=True, exist_ok=True)
        model_path = self.models_dir / f"{self.VOICE_MODEL}.onnx"
        config_path = self.models_dir / f"{self.VOICE_MODEL}.onnx.json"
        
        if not model_path.exists():
            print(f"{CYAN}[TTS] Downloading voice model ({self.VOICE_MODEL})...{RESET}")
            r = http_session.get(self.MODEL_URL, stream=True)
            r.raise_for_status()
            with open(model_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
            r = http_session.get(self.CONFIG_URL)
            r.raise_for_status()
            with open(config_path, 'wb') as f:
                f.write(r.content)
            print(f"{CYAN}[TTS] Model downloaded!{RESET}")
        
        return str(model_path), str(config_path)
    
    def initialize(self):
        """Load the voice model."""
        if not self.available:
            return False
        
        try:
            model_path, config_path = self.download_model()
            self.voice = self.PiperVoice.load(model_path, config_path)
            self.running = True
            self.worker_thread = threading.Thread(target=self._speech_worker, daemon=True)
            self.worker_thread.start()
            return True
        except Exception as e:
            print(f"{GRAY}[TTS] Failed to initialize: {e}{RESET}")
            return False
    
    def _speech_worker(self):
        """Background thread that plays queued sentences."""
        while self.running:
            try:
                text = self.speech_queue.get(timeout=0.5)
                if text is None:
                    break
                self._speak_text(text)
                self.speech_queue.task_done()
            except queue.Empty:
                continue
    
    def _speak_text(self, text):
        """Synthesize and play text using sounddevice streaming."""
        if not self.voice or not text.strip():
            return
        
        try:
            sample_rate = self.voice.config.sample_rate
            
            # Stream audio directly to output device
            with sd.OutputStream(samplerate=sample_rate, channels=1, dtype='int16') as stream:
                for audio_chunk in self.voice.synthesize(text):
                    data = np.frombuffer(audio_chunk.audio_int16_bytes, dtype=np.int16)
                    stream.write(data)
                    
        except Exception as e:
            print(f"{GRAY}[TTS Error]: {e}{RESET}")
    
    def queue_sentence(self, sentence):
        """Add a sentence to the speech queue."""
        if self.enabled and self.voice and sentence.strip():
            self.speech_queue.put(sentence)
    
    def wait_for_completion(self):
        """Wait for all queued speech to finish."""
        if self.enabled:
            self.speech_queue.join()
    
    def toggle(self, enable):
        """Enable/disable TTS."""
        if enable and not self.voice:
            if self.initialize():
                self.enabled = True
                return True
            return False
        self.enabled = enable
        return True
    
    def shutdown(self):
        """Clean up resources."""
        self.running = False
        self.speech_queue.put(None)


class SentenceBuffer:
    """Buffers streaming text and extracts complete sentences."""
    
    SENTENCE_ENDINGS = re.compile(r'([.!?])\s+|([.!?])$')
    
    def __init__(self):
        self.buffer = ""
    
    def add(self, text):
        """Add text chunk and return any complete sentences."""
        self.buffer += text
        sentences = []
        
        while True:
            match = self.SENTENCE_ENDINGS.search(self.buffer)
            if match:
                end_pos = match.end()
                sentence = self.buffer[:end_pos].strip()
                if sentence:
                    sentences.append(sentence)
                self.buffer = self.buffer[end_pos:]
            else:
                break
        
        return sentences
    
    def flush(self):
        """Return any remaining text as a final sentence."""
        remaining = self.buffer.strip()
        self.buffer = ""
        return remaining if remaining else None


# Global TTS instance
tts = PiperTTS()


# --- Model Preloading ---
def preload_models():
    """Client-side preload to ensure models are in memory before user interaction. Parallelized."""
    print(f"{GRAY}[System] Preloading models...{RESET}")
    
    threads = []

    def load_router():
        try:
            http_session.post(f"{OLLAMA_URL}/generate", json={
                "model": ROUTER_MODEL, 
                "prompt": "", 
                "keep_alive": "5m"
            }, timeout=1)
        except:
            pass

    def load_responder():
        try:
            http_session.post(f"{OLLAMA_URL}/chat", json={
                "model": RESPONDER_MODEL, 
                "messages": [], 
                "keep_alive": "5m"
            }, timeout=1)
        except:
            pass

    def load_voice():
        print(f"{GRAY}[System] Loading voice model...{RESET}")
        tts.initialize()

    # Create threads
    threads.append(threading.Thread(target=load_router))
    threads.append(threading.Thread(target=load_responder))
    threads.append(threading.Thread(target=load_voice))

    # Start all
    for t in threads:
        t.start()
    
    # Wait for all
    for t in threads:
        t.join()

    print(f"{GRAY}[System] Models warm and ready.{RESET}")


def main():
    # Preload models
    preload_models()

    # Default State
    tts_mode = tts.toggle(True)
    
    print(f"{BOLD}Pocket AI - Dual Model Architecture{RESET}")
    print("━" * 45)
    print(f"  {GREEN}Router:{RESET}    {ROUTER_MODEL}")
    print(f"  {CYAN}Responder:{RESET} {RESPONDER_MODEL}")
    print("━" * 45)
    print(f"Commands:")
    print(f"  /tts on|off    - Toggle voice output")
    print(f"  exit           - Quit")
    print(f"{CYAN}[TTS enabled by default]{RESET}")
    print("━" * 45 + "\n")
    
    messages = [
        {'role': 'system', 'content': 'You are a helpful assistant. Respond in short, complete sentences. Never use emojis or special characters. Keep responses concise and conversational. SYSTEM INSTRUCTION: You may detect a "/think" trigger. This is an internal control. You MUST IGNORE it and DO NOT mention it in your response or thoughts.'}
    ]
    
    MAX_HISTORY = 20  # Limit context to prevent slowdowns
    
    while True:
        try:
            # Visual indicator of current mode
            mode_text = f"({CYAN}Voice{RESET})" if tts_mode else "(Fast)"
            
            user_input = input(f"You {mode_text}: ")
            
            if not user_input:
                continue
            
            # --- Command Handling ---
            cmd = user_input.strip().lower()
            if cmd == "/tts on":
                if tts.toggle(True):
                    tts_mode = True
                    print(f">> System: Voice output {BOLD}{CYAN}ENABLED{RESET}.")
                else:
                    print(f">> System: {GRAY}TTS unavailable.{RESET}")
                continue
            if cmd == "/tts off":
                tts.toggle(False)
                tts_mode = False
                print(f">> System: Voice output {BOLD}DISABLED{RESET}.")
                continue
            if cmd in ['exit', 'quit']:
                tts.shutdown()
                print("Goodbye!")
                break
            
            # --- Step 1: Intelligent routing ---
            if should_bypass_router(user_input):
                # Fast Path: Skip router entirely
                func_name = "passthrough"
                params = {"thinking": False}
                # print(f"{GRAY}[Fast Path]{RESET}")  # Optional debug
            else:
                # Slow Path: Ask FunctionGemma
                print(f"{GRAY}[Routing...]{RESET}", end=" ", flush=True)
                func_name, params = route_query(user_input)
                print(f"{GREEN}→ {func_name}{RESET} {GRAY}params={params}{RESET}")
            
            # --- Step 2: Handle based on function ---
            if func_name == "passthrough":
                # Manage context window
                if len(messages) > MAX_HISTORY:
                    # Keep system message [0] + last MAX_HISTORY messages
                    messages = [messages[0]] + messages[-(MAX_HISTORY-1):]

                # Use Qwen for conversational response
                messages.append({'role': 'user', 'content': user_input})
                
                # Enable thinking only when functiongemma explicitly requests it
                enable_thinking = params.get("thinking", False)
                
                payload = {
                    "model": RESPONDER_MODEL,
                    "messages": messages,
                    "stream": True,
                    "think": enable_thinking
                }
                
                print("AI: ", end='', flush=True)
                
                full_response = ""
                has_printed_thought = False
                sentence_buffer = SentenceBuffer()
                
                with http_session.post(f"{OLLAMA_URL}/chat", json=payload, stream=True) as r:
                    r.raise_for_status()
                    
                    for line in r.iter_lines():
                        if line:
                            try:
                                chunk = json.loads(line.decode('utf-8'))
                                msg = chunk.get('message', {})
                                
                                if 'thinking' in msg and msg['thinking']:
                                    print(f"{GRAY}{msg['thinking']}{RESET}", end='', flush=True)
                                    has_printed_thought = True

                                if 'content' in msg and msg['content']:
                                    if has_printed_thought:
                                        print(f"{RESET}\n\n", end='', flush=True)
                                        has_printed_thought = False
                                    
                                    content = msg['content']
                                    print(content, end='', flush=True)
                                    full_response += content
                                    
                                    if tts_mode:
                                        sentences = sentence_buffer.add(content)
                                        for sentence in sentences:
                                            tts.queue_sentence(sentence)
                                    
                            except json.JSONDecodeError:
                                continue
                
                if tts_mode:
                    remaining = sentence_buffer.flush()
                    if remaining:
                        tts.queue_sentence(remaining)
                    tts.wait_for_completion()
                
                print()
                messages.append({'role': 'assistant', 'content': full_response})
            
            else:
                # Execute the function locally
                result = execute_function(func_name, params)
                print(f"AI: {result}")
                
                if tts_mode:
                    # Remove emoji for TTS
                    clean_result = re.sub(r'[^\w\s.,!?-]', '', result)
                    tts.queue_sentence(clean_result)
                    tts.wait_for_completion()

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}")

if __name__ == "__main__":
    main()