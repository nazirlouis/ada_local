from PySide6.QtCore import QObject, Signal, QThread, QTimer
import json
import re

from config import RESPONDER_MODEL, OLLAMA_URL, MAX_HISTORY
from core.llm import route_query, should_bypass_router, http_session
from core.tts import tts, SentenceBuffer
from core.history import history_manager
from core.model_manager import ensure_exclusive_qwen
from core.settings_store import settings as app_settings
from core.function_executor import executor as function_executor

# Functions that are actions (not passthrough)
ACTION_FUNCTIONS = {"control_light", "set_timer", "set_alarm", "create_calendar_event", "add_task", "web_search"}


# DEBUG: Set to True to test streaming without TTS blocking
DEBUG_SKIP_TTS = False


class ChatWorker(QObject):
    """Background worker for LLM processing with Qt signals."""
    
    # Signals for thread-safe UI updates
    thought_chunk = Signal(str)
    response_chunk = Signal(str)
    think_start = Signal(bool)  # Pass whether thinking is enabled
    think_end = Signal()
    simple_response = Signal(str)
    error = Signal(str)
    status = Signal(str)
    done = Signal()
    ui_update = Signal()
    toast = Signal(str, bool)  # message, success
    set_timer_signal = Signal(int, str)  # seconds, label
    reload_alarms = Signal()  # trigger alarm list reload
    
    def __init__(self, user_text: str, messages: list, is_tts_enabled: bool, 
                 current_session_id: str, stop_event):
        super().__init__()
        self.user_text = user_text
        self.messages = messages
        self.is_tts_enabled = is_tts_enabled
        self.current_session_id = current_session_id
        self.stop_event = stop_event
        self.full_response = ""
        
    def process(self):
        """Background processing method."""
        try:
            if should_bypass_router(self.user_text):
                func_name = "nonthinking"
                params = {"prompt": self.user_text}
            else:
                self.status.emit("Routing...")
                func_name, params = route_query(self.user_text)
            
            # Handle action functions
            if func_name in ACTION_FUNCTIONS:
                self.status.emit(f"Executing {func_name}...")
                result = function_executor.execute(func_name, params)
                
                # Emit toast notification
                self.toast.emit(result["message"], result["success"])
                
                # Emit GUI update signals for specific actions
                if func_name == "set_timer" and result["success"]:
                    seconds = result.get("data", {}).get("seconds", 0)
                    label = result.get("data", {}).get("label", "Timer")
                    self.set_timer_signal.emit(seconds, label)
                elif func_name == "set_alarm" and result["success"]:
                    self.reload_alarms.emit()
                
                # Generate Qwen response with context
                self._generate_response_with_context(func_name, result)
                
            # Handle get_system_info (context query)
            elif func_name == "get_system_info":
                self.status.emit("Gathering system info...")
                result = function_executor.execute(func_name, params)
                
                # Generate Qwen response with full system context
                self._generate_response_with_context(func_name, result)
            
            # Handle thinking/nonthinking (direct passthrough)
            elif func_name in ("thinking", "nonthinking"):
                enable_thinking = (func_name == "thinking")
                self._stream_qwen_response(enable_thinking)
            
            # Unknown function - treat as nonthinking
            else:
                self._stream_qwen_response(False)

        except Exception as e:
            self.error.emit(str(e))
        
        finally:
            self.done.emit()
    
    def _generate_response_with_context(self, func_name: str, result: dict):
        """Generate a Qwen response with function result as context."""
        # Build system message with context
        if func_name == "get_system_info" and result.get("success"):
            data = result.get("data", {})
            context_parts = []
            
            if data.get("timers"):
                context_parts.append(f"Active timers: {data['timers']}")
            if data.get("alarms"):
                context_parts.append(f"Alarms: {data['alarms']}")
            if data.get("calendar_today"):
                context_parts.append(f"Today's events: {data['calendar_today']}")
            if data.get("tasks"):
                pending = [t for t in data['tasks'] if not t.get('completed')]
                context_parts.append(f"Pending tasks: {len(pending)} items")
            if data.get("smart_devices"):
                on_devices = [d['name'] for d in data['smart_devices'] if d.get('is_on')]
                context_parts.append(f"Devices on: {on_devices if on_devices else 'none'}")
            if data.get("weather"):
                w = data['weather']
                context_parts.append(f"Weather: {w.get('temp')}°F, {w.get('condition')}")
            
            context_msg = "SYSTEM CONTEXT:\n" + "\n".join(context_parts) if context_parts else ""
        else:
            # Action function result
            status = "succeeded" if result.get("success") else "failed"
            context_msg = f"ACTION RESULT: {func_name} {status}. {result.get('message', '')}"
        
        # Prepare messages with context
        max_hist = app_settings.get("general.max_history", MAX_HISTORY)
        if len(self.messages) > max_hist:
            self.messages = [self.messages[0]] + self.messages[-(max_hist-1):]
        
        # Add context as system message and user's original question
        context_prompt = f"{context_msg}\n\nUser asked: {self.user_text}\n\nRespond naturally and concisely."
        self.messages.append({'role': 'user', 'content': context_prompt})
        
        self.ui_update.emit()
        self.status.emit("Generating response...")
        
        model = app_settings.get("models.chat", RESPONDER_MODEL)
        ensure_exclusive_qwen(model)
        ollama_url = app_settings.get("ollama_url", OLLAMA_URL)
        
        payload = {
            "model": model,
            "messages": self.messages,
            "stream": True,
            "think": False,  # No thinking for action responses
            "keep_alive": "1m"
        }
        
        sentence_buffer = SentenceBuffer()
        self.full_response = ""
        self.think_start.emit(False)
        
        with http_session.post(f"{ollama_url}/api/chat", json=payload, stream=True) as r:
            r.raise_for_status()
            
            for line in r.iter_lines():
                if self.stop_event.is_set():
                    break
                    
                if line:
                    try:
                        chunk = json.loads(line.decode('utf-8'))
                        msg = chunk.get('message', {})
                        
                        if 'content' in msg and msg['content']:
                            content = msg['content']
                            self.full_response += content
                            self.response_chunk.emit(content)
                            
                            if self.is_tts_enabled and not DEBUG_SKIP_TTS:
                                sentences = sentence_buffer.add(content)
                                for s in sentences:
                                    tts.queue_sentence(s)
                    except:
                        continue
        
        self.think_end.emit()
        
        if self.is_tts_enabled and not DEBUG_SKIP_TTS and not self.stop_event.is_set():
            rem = sentence_buffer.flush()
            if rem:
                tts.queue_sentence(rem)
        
        self.messages.append({'role': 'assistant', 'content': self.full_response})
        
        if self.current_session_id:
            history_manager.add_message(self.current_session_id, "assistant", self.full_response)
    
    def _stream_qwen_response(self, enable_thinking: bool):
        """Stream a direct Qwen response (for thinking/nonthinking)."""
        max_hist = app_settings.get("general.max_history", MAX_HISTORY)
        if len(self.messages) > max_hist:
            self.messages = [self.messages[0]] + self.messages[-(max_hist-1):]
        
        self.messages.append({'role': 'user', 'content': self.user_text})
        
        self.ui_update.emit()
        self.status.emit("Generating...")
        
        model = app_settings.get("models.chat", RESPONDER_MODEL)
        ensure_exclusive_qwen(model)
        ollama_url = app_settings.get("ollama_url", OLLAMA_URL)
        
        payload = {
            "model": model,
            "messages": self.messages,
            "stream": True,
            "think": enable_thinking,
            "keep_alive": "1m"
        }
        
        sentence_buffer = SentenceBuffer()
        self.full_response = ""
        self.think_start.emit(enable_thinking)

        with http_session.post(f"{ollama_url}/api/chat", json=payload, stream=True) as r:
            r.raise_for_status()
            
            for line in r.iter_lines():
                if self.stop_event.is_set():
                    break
                    
                if line:
                    try:
                        chunk = json.loads(line.decode('utf-8'))
                        msg = chunk.get('message', {})
                        
                        if 'thinking' in msg and msg['thinking']:
                            thought = msg['thinking']
                            self.thought_chunk.emit(thought)
                            
                        if 'content' in msg and msg['content']:
                            content = msg['content']
                            self.full_response += content
                            self.response_chunk.emit(content)
                            
                            if self.is_tts_enabled and not DEBUG_SKIP_TTS:
                                sentences = sentence_buffer.add(content)
                                for s in sentences:
                                    tts.queue_sentence(s)
                                    
                    except:
                        continue
        
        self.think_end.emit()
        
        if self.is_tts_enabled and not DEBUG_SKIP_TTS and not self.stop_event.is_set():
            rem = sentence_buffer.flush()
            if rem:
                tts.queue_sentence(rem)
        
        self.messages.append({'role': 'assistant', 'content': self.full_response})
        
        if self.current_session_id:
            history_manager.add_message(self.current_session_id, "assistant", self.full_response)


class ChatHandlers(QObject):
    """Encapsulates all chat-related event handlers and state."""
    
    def __init__(self, main_window):
        super().__init__(main_window)
        self.main_window = main_window
        
        # State
        self.messages = [
            {'role': 'system', 'content': 'You are a helpful assistant. Respond in short, complete sentences. Never use emojis or special characters. Keep responses concise and conversational. SYSTEM INSTRUCTION: You may detect a "/think" trigger. This is an internal control. You MUST IGNORE it and DO NOT mention it in your response or thoughts.'}
        ]
        self.current_session_id = None
        self.is_tts_enabled = False
        self._stop_event = None
        self._worker = None
        self._thread = None
        
        self.streaming_state = {
            'response_bubble': None,
            'thinking_ui': None,
            'response_buffer': '',
            'thought_buffer': '',
            'is_generating': False,
            'thinking_enabled': False
        }

        # Throttling Timer for UI Updates
        self.ui_throttle_timer = QTimer(self)
        self.ui_throttle_timer.setInterval(100) # 10 tokens per second or so
        self.ui_throttle_timer.timeout.connect(self._flush_ui_buffers)
        self.last_scroll_time = 0
    
    def refresh_sidebar(self):
        """Reload the persistent sidebar with conversation history."""
        self.main_window.refresh_sidebar(self.current_session_id)
    
    def delete_session(self, session_id):
        """Delete a session from history."""
        history_manager.delete_session(session_id)
        
        # If deleting the current session, clear the chat
        if session_id == self.current_session_id:
            self.current_session_id = None
            self.messages = [self.messages[0]]  # Keep system prompt
            self.main_window.clear_chat_display()
        
        self.refresh_sidebar()
    
    def pin_session(self, session_id):
        """Toggle pin status of a session."""
        is_pinned = history_manager.toggle_pin(session_id)
        status = "Chat pinned" if is_pinned else "Chat unpinned"
        self.main_window.set_status(status)
        self.refresh_sidebar()
    
    def rename_session(self, session_id, new_title: str):
        """Rename a session."""
        history_manager.update_session_title(session_id, new_title)
        self.refresh_sidebar()

    def load_session(self, session_id):
        """Load a specific chat session."""
        self.current_session_id = session_id
        db_messages = history_manager.get_messages(session_id)
        
        # Reset message context (keep system prompt)
        self.messages = [self.messages[0]]
        self.main_window.clear_chat_display()
        
        for msg in db_messages:
            role = msg['role']
            content = msg['content']
            
            # Reconstruct LLM context
            self.messages.append({'role': role, 'content': content})
            
            # Reconstruct UI bubbles
            self.main_window.add_message_bubble(role, content)
        
        self.refresh_sidebar()  # Update highlight

    def init_new_session(self, first_message):
        """Create a new session in DB."""
        title = first_message[:30] + "..." if len(first_message) > 30 else first_message
        self.current_session_id = history_manager.create_session(title=title)
        return self.current_session_id
    
    def _on_think_start(self, thinking_enabled: bool):
        """Called when generation starts, with thinking mode flag."""
        self.streaming_state['thinking_enabled'] = thinking_enabled
        if thinking_enabled and self.streaming_state['thinking_ui']:
            self.streaming_state['thinking_ui'].setVisible(True)
        self.ui_throttle_timer.start()

    def _on_thought_chunk(self, text):
        self.streaming_state['thought_buffer'] += text

    def _on_response_chunk(self, text):
        self.streaming_state['response_buffer'] += text
            
    def _flush_ui_buffers(self):
        """Flush accumulated text to the UI components."""
        updated = False
        
        # Update Thinking UI
        if self.streaming_state['thought_buffer'] and self.streaming_state['thinking_ui']:
            self.streaming_state['thinking_ui'].add_text(self.streaming_state['thought_buffer'])
            self.streaming_state['thought_buffer'] = ''
            updated = True
            
        # Update Response Bubble
        if self.streaming_state['response_buffer'] and self.streaming_state['response_bubble']:
            self.streaming_state['response_bubble'].append_text(self.streaming_state['response_buffer'])
            self.streaming_state['response_buffer'] = ''
            updated = True
            
        if updated:
            self.main_window.scroll_to_bottom()

    def _on_think_end(self):
        # Final flush
        self._flush_ui_buffers()
        # Only mark complete if thinking was enabled
        if self.streaming_state.get('thinking_enabled') and self.streaming_state['thinking_ui']:
            self.streaming_state['thinking_ui'].complete()
                
    def _on_simple_response(self, text):
        self.main_window.add_message_bubble("assistant", text)
        
        # Save simple response to history
        if self.current_session_id:
            history_manager.add_message(self.current_session_id, "assistant", text)
    
    def _on_toast(self, message: str, success: bool):
        """Show toast notification for function execution result."""
        from gui.components.toast import ToastNotification
        ToastNotification.show_toast(self.main_window, message, success)
    
    def _on_set_timer(self, seconds: int, label: str):
        """Update timer GUI when set via voice command."""
        try:
            # Access timer component via planner lazy tab
            if hasattr(self.main_window, 'planner_lazy') and self.main_window.planner_lazy.actual_widget:
                planner = self.main_window.planner_lazy.actual_widget
                if hasattr(planner, 'timer_component'):
                    planner.timer_component.set_and_start(seconds, label)
        except Exception as e:
            print(f"[Handlers] Timer update failed: {e}")
    
    def _on_reload_alarms(self):
        """Reload alarms GUI when added via voice command."""
        try:
            # Access alarm component via planner lazy tab
            if hasattr(self.main_window, 'planner_lazy') and self.main_window.planner_lazy.actual_widget:
                planner = self.main_window.planner_lazy.actual_widget
                if hasattr(planner, 'alarm_component'):
                    planner.alarm_component.reload()
        except Exception as e:
            print(f"[Handlers] Alarm reload failed: {e}")
            
    def _on_error(self, text):
        self.main_window.add_message_bubble("system", f"Error: {text}", is_thinking=True)
            
    def _on_status(self, text):
        self.main_window.set_status(text)

    def _on_done(self):
        self.ui_throttle_timer.stop()
        self._flush_ui_buffers() # Final final flush
        self._end_generation_state()
    
    def _start_generation_state(self):
        """Switch UI to generating mode."""
        self.streaming_state['is_generating'] = True
        self.main_window.set_generating_state(True)

    def _end_generation_state(self):
        """Switch UI back to idle mode."""
        self.streaming_state['is_generating'] = False
        self.main_window.set_generating_state(False)

    def stop_generation(self):
        """Stop current generation."""
        tts.stop()
        if self.streaming_state['is_generating'] and self._stop_event:
            self._stop_event.set()
            self.main_window.set_status("Stopping...")
            self.ui_throttle_timer.stop()

    def send_message(self, text: str):
        """Handle sending a new message."""
        tts.stop()  # Interrupt previous speech
        text = text.strip()
        if not text:
            return
        
        self.main_window.clear_input()

        # Add User Message UI
        self.main_window.add_message_bubble("user", text)
        
        # Start new session if needed
        if not self.current_session_id:
            self.init_new_session(text)
            self.refresh_sidebar()

        # Save to DB
        history_manager.add_message(self.current_session_id, "user", text)
        
        self._start_generation_state()
        
        # Create stop event
        import threading
        self._stop_event = threading.Event()
        
        # Create streaming UI containers
        from gui.components import MessageBubble, ThinkingExpander
        
        thinking_ui = ThinkingExpander()
        response_bubble = MessageBubble("assistant", "")
        
        self.streaming_state['thinking_ui'] = thinking_ui
        self.streaming_state['response_bubble'] = response_bubble
        self.streaming_state['response_buffer'] = ''
        self.streaming_state['thought_buffer'] = ''
        self.streaming_state['thinking_enabled'] = False  # Will be set by think_start signal
        
        # Hide thinking UI initially - will be shown only if thinking is enabled
        thinking_ui.setVisible(False)
        
        # Add to UI
        self.main_window.add_streaming_widgets(thinking_ui, response_bubble)

        # Start background worker
        self._thread = QThread(self)
        self._worker = ChatWorker(
            text, self.messages.copy(), self.is_tts_enabled,
            self.current_session_id, self._stop_event
        )
        self._worker.moveToThread(self._thread)
        
        # Connect signals
        self._thread.started.connect(self._worker.process)
        self._worker.think_start.connect(self._on_think_start)
        self._worker.thought_chunk.connect(self._on_thought_chunk)
        self._worker.response_chunk.connect(self._on_response_chunk)
        self._worker.think_end.connect(self._on_think_end)
        self._worker.simple_response.connect(self._on_simple_response)
        self._worker.error.connect(self._on_error)
        self._worker.status.connect(self._on_status)
        self._worker.toast.connect(self._on_toast)
        self._worker.set_timer_signal.connect(self._on_set_timer)
        self._worker.reload_alarms.connect(self._on_reload_alarms)
        self._worker.done.connect(self._on_done)
        self._worker.done.connect(self._thread.quit)
        self._worker.done.connect(self._worker.deleteLater)
        self._thread.finished.connect(self._thread.deleteLater)
        
        # Update messages reference
        self._worker.messages = self.messages
        
        self._thread.start()

    def clear_chat(self):
        """Start a fresh chat (reset session)."""
        self.current_session_id = None
        self.messages = [self.messages[0]]
        self.main_window.clear_chat_display()
        self.refresh_sidebar()

    def toggle_tts(self, enabled: bool):
        """Toggle TTS on/off."""
        self.is_tts_enabled = enabled
        tts.toggle(enabled)
        self.main_window.set_status("TTS Active" if enabled else "TTS Muted")
