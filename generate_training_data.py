#!/usr/bin/env python3
"""
Training Data Generator for FunctionGemma Fine-tuning
Generates 1050 training examples in JSONL format (150 per function category).
"""

import json
import random
from pathlib import Path

# === FUNCTION DEFINITIONS (same as test file) ===
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
            "description": "Routes the query to a conversational AI model when no specific tool action is needed.",
            "parameters": {
                "type": "object",
                "properties": {
                    "thinking": {"type": "boolean", "description": "Set to true for complex reasoning, false for simple chat."}
                },
                "required": ["thinking"]
            }
        }
    }
]


# === TRAINING DATA TEMPLATES ===

# control_light: 75 examples
LIGHT_TEMPLATES = [
    # Turn on variations
    ("Turn on the {room} lights", "on", "{room}"),
    ("Switch on the lights in the {room}", "on", "{room}"),
    ("Lights on in the {room}", "on", "{room}"),
    ("Can you turn on the {room} light", "on", "{room}"),
    ("Please turn on the lights in the {room}", "on", "{room}"),
    ("Turn the {room} lights on", "on", "{room}"),
    ("I need the lights on in the {room}", "on", "{room}"),
    ("Light up the {room}", "on", "{room}"),
    ("Illuminate the {room}", "on", "{room}"),
    ("Enable the {room} lights", "on", "{room}"),
    # Turn off variations
    ("Turn off the {room} lights", "off", "{room}"),
    ("Switch off the lights in the {room}", "off", "{room}"),
    ("Lights off in the {room}", "off", "{room}"),
    ("Can you turn off the {room} light", "off", "{room}"),
    ("Please turn off the lights in the {room}", "off", "{room}"),
    ("Turn the {room} lights off", "off", "{room}"),
    ("Kill the lights in the {room}", "off", "{room}"),
    ("Shut off the {room} lights", "off", "{room}"),
    ("Disable the {room} lights", "off", "{room}"),
    ("I need the {room} lights off", "off", "{room}"),
    # Dim variations
    ("Dim the {room} lights", "dim", "{room}"),
    ("Lower the lights in the {room}", "dim", "{room}"),
    ("Can you dim the {room} light", "dim", "{room}"),
    ("Make the {room} lights dimmer", "dim", "{room}"),
    ("Turn down the {room} lights", "dim", "{room}"),
]

ROOMS = [
    "bedroom", "kitchen", "living room", "bathroom", "office", "garage",
    "dining room", "hallway", "basement", "attic", "guest room", "nursery",
    "master bedroom", "kids room", "study", "den", "patio", "porch",
    "laundry room", "closet", "pantry", "mudroom", "sunroom", "loft",
    "foyer", "entryway", "game room", "home office", "media room", "theater room",
    "walk-in closet", "bonus room", "playroom", "workshop", "studio", "conservatory"
]


# web_search: 75 examples
SEARCH_TEMPLATES = [
    "Search for {query}",
    "Look up {query}",
    "Google {query}",
    "Find information about {query}",
    "Search the web for {query}",
    "Can you search for {query}",
    "I need to find {query}",
    "What can you find about {query}",
    "Do a web search for {query}",
    "Look online for {query}",
    "Find me information on {query}",
    "Search online for {query}",
    "I want to know about {query}",
    "Research {query} for me",
    "Find out about {query}",
]

SEARCH_QUERIES = [
    "best restaurants in New York", "how to make pasta", "weather in Paris",
    "Python programming tutorials", "latest news on AI", "quantum computing explained",
    "healthy breakfast recipes", "cheap flights to Tokyo", "how to change a tire",
    "best laptops 2024", "stock market today", "COVID vaccine side effects",
    "how to meditate", "best books of 2024", "iPhone vs Android comparison",
    "climate change facts", "how to start a business", "learn Spanish online",
    "home workout routines", "vegan meal prep ideas", "cryptocurrency explained",
    "best hiking trails near me", "how to invest in stocks", "machine learning basics",
    "electric car reviews", "how to fix a leaky faucet", "travel tips for Europe",
    "remote work best practices", "healthy snack ideas", "how to sleep better",
    "best coffee shops nearby", "how to learn guitar", "yoga for beginners",
    "history of the Roman Empire", "best Netflix shows", "how to bake bread",
    "Mars exploration news", "Taylor Swift tour dates", "DIY home projects",
    "best running shoes 2024", "how to train a dog", "sustainable living tips"
]


# set_timer: 75 examples
TIMER_TEMPLATES = [
    "Set a timer for {duration}",
    "Start a {duration} timer",
    "Timer for {duration}",
    "Can you set a timer for {duration}",
    "I need a {duration} timer",
    "Set a {duration} countdown",
    "Please set a timer for {duration}",
    "Start a countdown for {duration}",
    "Set timer {duration}",
    "Create a {duration} timer",
    "Set a {duration} timer for {label}",
    "Start a {duration} timer for the {label}",
    "{duration} timer please",
    "I need a timer for {duration}",
    "Countdown {duration}",
]

DURATIONS = [
    "5 minutes", "10 minutes", "15 minutes", "30 minutes", "1 hour",
    "2 hours", "45 seconds", "90 seconds", "3 minutes", "7 minutes",
    "20 minutes", "25 minutes", "40 minutes", "1 minute", "2 minutes",
    "8 minutes", "12 minutes", "18 minutes", "35 minutes", "50 minutes"
]

TIMER_LABELS = [
    "oven", "laundry", "pasta", "eggs", "workout", "break", "nap",
    "meeting", "meditation", "stretch", "walk", "reading", "pomodoro"
]


# create_calendar_event: 75 examples
CALENDAR_CREATE_TEMPLATES = [
    "Add {event} to my calendar for {date}",
    "Schedule {event} for {date}",
    "Create a calendar event for {event} on {date}",
    "Put {event} on my calendar for {date}",
    "Add {event} to my schedule for {date}",
    "I need to schedule {event} for {date}",
    "Can you add {event} to my calendar on {date}",
    "Book {event} for {date}",
    "Set up {event} on {date}",
    "Create an appointment for {event} on {date}",
    "Schedule {event} on {date} at {time}",
    "Add {event} for {date} at {time}",
    "I have {event} on {date}",
    "Remember {event} on {date}",
    "Calendar event: {event} on {date}",
]

EVENTS = [
    "a meeting", "dentist appointment", "doctor visit", "team standup",
    "lunch with Sarah", "dinner with friends", "birthday party", "job interview",
    "gym session", "yoga class", "haircut", "car service", "flight to Miami",
    "conference call", "project deadline", "parent teacher conference",
    "wedding anniversary", "date night", "book club", "movie night"
]

DATES = [
    "tomorrow", "next Monday", "Friday", "next week", "this Saturday",
    "December 15th", "January 5th", "the 20th", "next Tuesday",
    "this Thursday", "Monday morning", "Wednesday afternoon", "Sunday evening"
]

TIMES = [
    "3pm", "9am", "2:30pm", "10am", "4pm", "7pm", "noon", "6pm",
    "8am", "11am", "1pm", "5pm", "3:30pm", "9:30am"
]


# read_calendar: 75 examples
CALENDAR_READ_TEMPLATES = [
    "What do I have on my calendar {date}?",
    "Show me my schedule for {date}",
    "What's on my calendar {date}?",
    "Check my calendar for {date}",
    "What meetings do I have {date}?",
    "Am I free {date}?",
    "Do I have anything scheduled {date}?",
    "What's my schedule {date}?",
    "Any appointments {date}?",
    "What events do I have {date}?",
    "Show my appointments for {date}",
    "Read my calendar for {date}",
    "What's happening {date}?",
    "Calendar for {date}",
    "My schedule {date}",
]

READ_DATES = [
    "today", "tomorrow", "this week", "next week", "Monday", "Tuesday",
    "on Friday", "this weekend", "next Monday", "for December",
    "in the morning", "this afternoon", "tonight", "for the rest of the day"
]


# passthrough (thinking=false): 75 examples - simple greetings/chitchat
PASSTHROUGH_SIMPLE = [
    "Hello", "Hi", "Hey", "Hi there", "Hello!", "Hey there",
    "Good morning", "Good afternoon", "Good evening", "Good night",
    "How are you?", "How's it going?", "What's up?", "How are you doing?",
    "Thanks", "Thank you", "Thanks a lot", "Thank you so much", "Appreciate it",
    "What's your name?", "Who are you?", "What can you do?", "Help",
    "Bye", "Goodbye", "See you", "Later", "Take care", "Have a nice day",
    "Yes", "No", "Maybe", "Sure", "Okay", "Alright", "Got it", "I understand",
    "Please", "Sorry", "Excuse me", "Pardon", "My bad", "Oops",
    "Cool", "Nice", "Great", "Awesome", "Perfect", "Wonderful", "Excellent",
    "I see", "Interesting", "Really?", "Oh", "Ah", "Hmm", "Wow",
    "That's nice", "That's great", "That's cool", "That's interesting",
    "Tell me more", "Go on", "Continue", "Keep going", "And then?",
    "What do you mean?", "Can you repeat that?", "I don't understand",
    "Never mind", "Forget it", "It's okay", "No problem", "No worries",
    "You're welcome", "My pleasure", "Anytime", "Happy to help",
    "Haha", "Lol", "That's funny", "You're funny", "Nice one"
]


# passthrough (thinking=true): 75 examples - complex reasoning
PASSTHROUGH_COMPLEX = [
    "What are the prime numbers from 1 to 100?",
    "Explain how photosynthesis works",
    "If I have 3 apples and give away 2, how many do I have?",
    "What are the pros and cons of electric cars?",
    "Write a haiku about the ocean",
    "Solve this riddle: I have cities but no houses",
    "Explain quantum entanglement in simple terms",
    "What's the difference between RAM and ROM?",
    "How does the stock market work?",
    "Explain the theory of relativity",
    "What causes rainbows?",
    "How do vaccines work?",
    "Explain machine learning to a 5 year old",
    "What's the meaning of life?",
    "Compare Python and JavaScript",
    "How does encryption work?",
    "Explain blockchain technology",
    "What causes earthquakes?",
    "How does WiFi work?",
    "Explain the water cycle",
    "What's the difference between weather and climate?",
    "How do airplanes fly?",
    "Explain how batteries work",
    "What causes the northern lights?",
    "How does GPS work?",
    "Explain the digestive system",
    "What's the difference between a virus and bacteria?",
    "How do magnets work?",
    "Explain how the internet works",
    "What causes seasons?",
    "If a train leaves at 2pm going 60mph, when does it arrive 120 miles away?",
    "Solve: 15% of 240",
    "What's 17 times 23?",
    "Calculate the area of a circle with radius 5",
    "If x + 5 = 12, what is x?",
    "Compare democracy and autocracy",
    "Explain the scientific method",
    "What are the causes of World War 1?",
    "How do black holes form?",
    "Explain the greenhouse effect",
    "What's the difference between mitosis and meiosis?",
    "How does sound travel?",
    "Explain how 3D printing works",
    "What causes inflation?",
    "How do solar panels work?",
    "Explain the immune system",
    "What's the difference between AC and DC current?",
    "How does memory work in the brain?",
    "Explain supply and demand",
    "What causes tides?",
    "How do touchscreens work?",
    "Explain the carbon cycle",
    "What's compound interest?",
    "How do nuclear reactors work?",
    "Explain evolution by natural selection",
    "What causes wind?",
    "How do microwaves heat food?",
    "Explain the Pythagorean theorem",
    "What's the difference between DNA and RNA?",
    "How do computers process information?",
    "Explain what causes thunder and lightning",
    "What are the layers of the atmosphere?",
    "How do antibiotics work?",
    "Explain the difference between mass and weight",
    "What causes volcanoes to erupt?",
    "How does refrigeration work?",
    "Explain the concept of infinity",
    "What's the difference between speed and velocity?",
    "How do lasers work?",
    "Explain the concept of paradoxes",
    "What causes dreams?",
    "How does the heart pump blood?",
    "Explain the Doppler effect",
    "What's the difference between socialism and capitalism?",
    "How do catalysts work in chemistry?"
]


def esc(s):
    """Wrap string with <escape> tokens."""
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


def build_prompt(user_query):
    """Build the full prompt."""
    declarations = build_function_declarations()
    return (
        f"<start_of_turn>developer "
        f"You are a model that can do function calling with the following functions"
        f"{declarations}"
        f"<end_of_turn>\n"
        f"<start_of_turn>user {user_query}<end_of_turn>\n"
        f"<start_of_turn>model"
    )


def build_completion(func_name, **params):
    """Build the expected completion (function call)."""
    params_str = ",".join([f"{k}:{esc(str(v))}" for k, v in params.items()])
    return f"<start_function_call>call:{func_name}{{{params_str}}}<end_function_call>"


def generate_training_data():
    """Generate 1050 training examples (150 per category)."""
    examples = []
    
    # === control_light: 150 examples ===
    count = 0
    while count < 150:
        template, action, room_template = random.choice(LIGHT_TEMPLATES)
        room = random.choice(ROOMS)
        prompt_text = template.format(room=room)
        
        entry = {
            "prompt": build_prompt(prompt_text),
            "completion": build_completion("control_light", action=action, room=room),
            "function": "control_light",
            "prompt_text": prompt_text
        }
        examples.append(entry)
        count += 1
    
    # === web_search: 150 examples ===
    count = 0
    while count < 150:
        template = random.choice(SEARCH_TEMPLATES)
        query = random.choice(SEARCH_QUERIES)
        prompt_text = template.format(query=query)
        
        entry = {
            "prompt": build_prompt(prompt_text),
            "completion": build_completion("web_search", query=query),
            "function": "web_search",
            "prompt_text": prompt_text
        }
        examples.append(entry)
        count += 1
    
    # === set_timer: 150 examples ===
    count = 0
    while count < 150:
        template = random.choice(TIMER_TEMPLATES)
        duration = random.choice(DURATIONS)
        label = random.choice(TIMER_LABELS)
        prompt_text = template.format(duration=duration, label=label)
        
        # Some have labels, some don't
        if "{label}" in template:
            entry = {
                "prompt": build_prompt(prompt_text),
                "completion": build_completion("set_timer", duration=duration, label=label),
                "function": "set_timer",
                "prompt_text": prompt_text
            }
        else:
            entry = {
                "prompt": build_prompt(prompt_text),
                "completion": build_completion("set_timer", duration=duration),
                "function": "set_timer",
                "prompt_text": prompt_text
            }
        examples.append(entry)
        count += 1
    
    # === create_calendar_event: 150 examples ===
    count = 0
    while count < 150:
        template = random.choice(CALENDAR_CREATE_TEMPLATES)
        event = random.choice(EVENTS)
        date = random.choice(DATES)
        time = random.choice(TIMES)
        prompt_text = template.format(event=event, date=date, time=time)
        
        if "{time}" in template:
            entry = {
                "prompt": build_prompt(prompt_text),
                "completion": build_completion("create_calendar_event", title=event, date=date, time=time),
                "function": "create_calendar_event",
                "prompt_text": prompt_text
            }
        else:
            entry = {
                "prompt": build_prompt(prompt_text),
                "completion": build_completion("create_calendar_event", title=event, date=date),
                "function": "create_calendar_event",
                "prompt_text": prompt_text
            }
        examples.append(entry)
        count += 1
    
    # === read_calendar: 150 examples ===
    count = 0
    while count < 150:
        template = random.choice(CALENDAR_READ_TEMPLATES)
        date = random.choice(READ_DATES)
        prompt_text = template.format(date=date)
        
        entry = {
            "prompt": build_prompt(prompt_text),
            "completion": build_completion("read_calendar", date=date),
            "function": "read_calendar",
            "prompt_text": prompt_text
        }
        examples.append(entry)
        count += 1
    
    # === passthrough (thinking=false): 150 examples ===
    # Need to repeat simple prompts since we only have ~75 unique ones
    simple_prompts = PASSTHROUGH_SIMPLE * 2  # Double the list
    random.shuffle(simple_prompts)
    for i, prompt_text in enumerate(simple_prompts[:150]):
        entry = {
            "prompt": build_prompt(prompt_text),
            "completion": build_completion("passthrough", thinking="false"),
            "function": "passthrough_simple",
            "prompt_text": prompt_text
        }
        examples.append(entry)
    
    # === passthrough (thinking=true): 150 examples ===
    # Need to repeat complex prompts since we only have ~75 unique ones
    complex_prompts = PASSTHROUGH_COMPLEX * 2  # Double the list
    random.shuffle(complex_prompts)
    for i, prompt_text in enumerate(complex_prompts[:150]):
        entry = {
            "prompt": build_prompt(prompt_text),
            "completion": build_completion("passthrough", thinking="true"),
            "function": "passthrough_complex",
            "prompt_text": prompt_text
        }
        examples.append(entry)
    
    # Shuffle all examples
    random.shuffle(examples)
    
    return examples


def validate_data(examples):
    """Validate the training data."""
    print("\n=== VALIDATION ===")
    
    # Count by function
    counts = {}
    for ex in examples:
        func = ex["function"]
        counts[func] = counts.get(func, 0) + 1
    
    print(f"Total examples: {len(examples)}")
    print("\nBy function:")
    for func, count in sorted(counts.items()):
        print(f"  {func}: {count}")
    
    # Sample entries
    print("\n=== SAMPLE ENTRIES ===")
    for i in range(3):
        ex = examples[i]
        print(f"\n[{i+1}] Function: {ex['function']}")
        print(f"    Prompt: {ex['prompt_text']}")
        print(f"    Completion: {ex['completion']}")
    
    return True


def main():
    print("=" * 60)
    print("FunctionGemma Training Data Generator")
    print("=" * 60)
    
    # Generate
    print("\nGenerating 1050 training examples (150 per category)...")
    examples = generate_training_data()
    
    # Validate
    validate_data(examples)
    
    # Save JSONL (just prompt and completion for training)
    output_dir = Path(__file__).parent / "training_data"
    output_dir.mkdir(exist_ok=True)
    
    jsonl_path = output_dir / "functiongemma_training.jsonl"
    with open(jsonl_path, "w") as f:
        for ex in examples:
            # Only save prompt and completion for training
            training_entry = {
                "prompt": ex["prompt"],
                "completion": ex["completion"]
            }
            f.write(json.dumps(training_entry) + "\n")
    
    print(f"\n✅ Training data saved to: {jsonl_path}")
    
    # Also save a readable version with metadata
    readable_path = output_dir / "functiongemma_training_readable.json"
    with open(readable_path, "w") as f:
        json.dump(examples, f, indent=2)
    
    print(f"📋 Readable version saved to: {readable_path}")
    
    print("\n" + "=" * 60)
    print("Done! Ready for fine-tuning.")
    print("=" * 60)


if __name__ == "__main__":
    main()
