from pythonosc.udp_client import SimpleUDPClient
from rasa_sdk import Action
from rasa_sdk.events import SlotSet
import google.generativeai as genai
import os
from typing import Optional, Dict, List, Union
import spacy
import re
import socket
import struct
from word2number import w2n

# Load spaCy's English tokenizer and models
nlp = spacy.load("en_core_web_sm")


# WING console IP and TCP/UDP ports (static values)
WING_IP = '192.168.1.183'
TCP_PORT = 2222  # TCP port used by the WING console
UDP_PORT = 14135  # Port where you want to receive meter data
OSC_PORT = 2223  # Port for sending OSC messages
WING_PORT = 2223

# Ensure API key is retrieved from environment variable
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise ValueError("Missing Gemini API key. Set GOOGLE_API_KEY in your environment.")

genai.configure(api_key=API_KEY)
google_model = genai.GenerativeModel(model_name='gemini-2.0-flash')


class ActionControlMixer(Action):
    def name(self):
        return "action_control_mixer"
    
    def run(self, dispatcher, tracker, domain):
        user_text = tracker.latest_message.get("text", "")  # Get raw text input
        print("User Text:", user_text)
        
        refined_user_text = validate_voice_command(user_text)
        print("Refined User Text:", refined_user_text)
        
        # Add a check for None before further processing
        if refined_user_text is None:
            dispatcher.utter_message(text="I couldn't understand that command. Could you please repeat?")
            return []
        
        command_interpreter = CommandInterpreter()  # Create an instance
        parsed_command = command_interpreter.interpret_command(refined_user_text)  # Call the method
        
        if not parsed_command:
            dispatcher.utter_message(text="I couldn't understand that command.")
            return []
        
        if parsed_command:
            execute_command(parsed_command)
            response = self.get_gemini_response(f"Convert this mixer command '{parsed_command}' into a confirmation phrase that confirms the action has been completed, similar to 'Channel 1 muted.' Keep it short and direct.")
            dispatcher.utter_message(text=response)

        else:
            # For unknown commands
            response = self.get_gemini_response(f"Convert this mixer command '{parsed_command}' into a confirmation phrase that confirms the action has been completed. Keep it short and direct.")
            dispatcher.utter_message(text=response)
        
        return [SlotSet("mixer_command", None), SlotSet("volume_level", None)]
    
    def get_gemini_response(self, prompt):
        """Generates a response using Gemini for mixer command confirmations."""
        try:
            response = google_model.generate_content(prompt)
            return response.text.strip() if response and response.text else "Command executed successfully."
        except Exception as e:
            return f"Error: {str(e)}"

class ActionUseGemini(Action):
    def name(self):
        return "action_use_gemini"

    def run(self, dispatcher, tracker, domain):
        user_message = tracker.latest_message.get("text")
        last_audio_topic = tracker.get_slot("last_audio_topic")

        if not self.is_audio_related(user_message):
            dispatcher.utter_message(
                text="I'm here to assist with **audio-related topics** like:\n"
                     "- 🎚️ Mixing\n"
                     "- 🎛️ Music production\n"
                     "Please ask a relevant question."
            )
            return []

        # Handle case where there's no previous topic
        if last_audio_topic is None:
            last_audio_topic = "audio basics"  # Default topic

        response = self.get_gemini_response(f"{user_message}. Previously, we discussed {last_audio_topic}.")
        
        dispatcher.utter_message(text=response)
        return [SlotSet("last_audio_topic", user_message)]

    def is_follow_up(self, prompt):
        """Determines if a question is a follow-up (e.g., 'Why is that?')."""
        follow_up_keywords = ["why", "how", "what about", "explain", "tell me more", "elaborate"]
        return any(word in prompt.lower() for word in follow_up_keywords)

    def is_audio_related(self, prompt):
        """Uses Gemini to classify whether a question is about audio-related topics."""
        try:
            classification_prompt = (
                "Determine if the following question is related to audio, sound engineering, music production, or audio technology.\n"
                "Respond ONLY with 'yes' or 'no'. If unsure, default to 'no'.\n\n"
                f"Question: {prompt}\n"
                "Answer:"
            )

            response = google_model.generate_content(classification_prompt)
            
            if response and response.text:
                clean_response = response.text.strip().lower()
                print(f"Gemini Response: {clean_response}")  # Debugging output
                
                return clean_response.startswith("yes")

            return False

        except Exception as e:
            print(f"[ERROR] Gemini classification failed: {str(e)}")
            return None  # Can return False if you prefer

    def get_gemini_response(self, prompt):
        """Generates a response using Gemini with a concise format."""
        try:
            concise_prompt = f"""
            Provide a **direct and clear** answer to the following **audio-related** question.
            **Keep it under 2 sentences, removing unnecessary details.**\n\n
            User Question: "{prompt}"
            """
            response = google_model.generate_content(concise_prompt)
            return response.text.strip() if response and response.text else "No response generated."

        except Exception as e:
            return f"Error: {str(e)}"
            
            
class CommandInterpreter:
    _instance = None  # Static variable to hold a single instance

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(CommandInterpreter, cls).__new__(cls)
            cls._instance.nlp = spacy.load("en_core_web_sm")  # Load NLP only once
        return cls._instance

    def __init__(self):
        # Ensure that re-initialization doesn't reload the model
        if not hasattr(self, "command_words"):
            self.command_words = {
                'mute': ['mute', 'unmute'],
                'phantom_power': ['phantom', 'phantom power', '48v'],
                'solo': ['solo'],
                'gate': ['gate'],
                'clear_solo': ['clear solo'],  # Add clear solo here
                'fader': ['fader', 'volume', 'level'],
                'gain': ['gain', 'input gain', 'preamp'],
                'threshold': ['threshold', 'gate threshold'],
                'select': ['select'],
                'load': ['load'],  # Add the "load" command
                'equalizer': ['eq', 'equalizer'],  # Added Equalizer
                'compressor': ['compressor', 'compression'],  # Added Compressor
            }
            
            self.action_verbs = {
                'increase': ['increase', 'raise', 'boost', 'turn up', 'bump up'],
                'decrease': ['decrease', 'lower', 'reduce', 'turn down'],
                'set': ['set', 'change', 'adjust', 'make'],
                'toggle': ['toggle', 'switch', 'turn'],
                'turn_on': ['turn on', 'enable', 'activate', 'on'],
                'turn_off': ['turn off', 'disable', 'deactivate', 'off']
            }

            # self.gain_stage_regex = r'gain stage (?:on|for)? channel (\d+)'
            # self.gain_stage_regex = r'gain stage (?:on|for)? channel (\d+)(?:\s*(?:and|,|through|-)\s*(\d+))?'
            self.gain_stage_regex = r'gain stage (?:on|for)? channel (\d+)(?:\s*(?:to|-|through)\s*(\d+))?'

            # self.gain_stage_regex = r'gain stage (?:on|for)? channel (\d+)(?:\s*(?:and|,|through|-)\s*(\d+))?'



    def extract_channels_and_value(self, doc) -> tuple[Optional[List[int]], Optional[float]]:
        channels = set()  # Use a set to prevent duplicates
        value = None
        adjustment = None
        tokens = list(doc)
        
        # Combine the text of the entire command for matching multi-word channel names
        text = doc.text.lower()
        
        # Reference pre-initialized CHANNEL_MAP
        # if not hasattr(self, "CHANNEL_MAP"):
            # raise RuntimeError("CHANNEL_MAP is not initialized. Call initialize_channel_map first.")
        
        is_fader_or_gain_command = any(keyword in text for keyword in ['fader', 'volume', 'level', 'gain'])
        
        # Track which tokens are part of channel specifications to avoid reusing them
        channel_tokens = set()

        for i, token in enumerate(tokens):
            # Handle "channel <number>" syntax
            try:
                if token.text.lower() == "channel" and i + 1 < len(tokens) and tokens[i + 1].like_num:
                    # Convert token to a number using word2number library if it's a word, otherwise use int()
                    channel_num = w2n.word_to_num(tokens[i + 1].text) if not tokens[i + 1].text.isdigit() else int(tokens[i + 1].text)
                    channels.add(channel_num)
                    channel_tokens.add(i + 1)  # Mark this token as used for channel
            except ValueError:
                print(f"Invalid number: {tokens[i + 1].text}")
            
            # Handle number ranges like "1 to 3" or "1 through 3"
            if token.like_num and i + 1 < len(tokens) and tokens[i + 1].text in ["to", "-", "through"] and i + 2 < len(tokens) and tokens[i + 2].like_num:
                try:
                    start = w2n.word_to_num(token.text) if not token.text.isdigit() else int(token.text)
                    end = w2n.word_to_num(tokens[i + 2].text) if not tokens[i + 2].text.isdigit() else int(tokens[i + 2].text)
                    channels.update(range(start, end + 1))
                    channel_tokens.update([i, i+1, i+2])  # Mark these tokens as used for channel range
                except ValueError:
                    continue

        # After processing channels, look for value specifications
        for i, token in enumerate(tokens):
            # Skip tokens already used for channel specifications
            if i in channel_tokens:
                continue
                
            # Handle "to <value>" for setting commands
            if is_fader_or_gain_command and token.text.lower() == "to" and i + 1 < len(tokens) and i + 1 not in channel_tokens:
                next_token = tokens[i + 1].text.lower()
                try:
                    # Check if next token is a valid number or ends with "db"
                    if '.' in next_token or 'db' in next_token:
                        value = float(next_token.replace("db", ""))
                    else:
                        # Check if next token is a numeric word
                        value = w2n.word_to_num(next_token) if not next_token.isdigit() else float(next_token)
                except ValueError:
                    pass
        
            # Handle adjustments like "by 3db"
            if is_fader_or_gain_command and token.text.lower() == "by" and i + 1 < len(tokens) and i + 1 not in channel_tokens:
                next_token = tokens[i + 1].text.lower()
                try:
                    adjustment = float(next_token.replace("db", "")) if '.' in next_token or 'db' in next_token else w2n.word_to_num(next_token) if not next_token.isdigit() else float(next_token)
                except ValueError:
                    pass
        
        # Default value handling for phrases like "increase" or "decrease" without specific values
        if value is None and adjustment is None:
            if any(token.text.lower() in ["increase", "raise", "boost"] for token in tokens):
                adjustment = 3.0  # Default increase adjustment
            elif any(token.text.lower() in ["decrease", "lower", "reduce"] for token in tokens):
                adjustment = -3.0  # Default decrease adjustment
        
        return (sorted(channels) if channels else None, value if value is not None else adjustment)

    

    def _identify_command_type(self, text: str) -> str:
        text = text.lower()
        if any(word in text for word in ['by']):
            return 'adjust'
        elif any(word in text for word in ['to']):
            if any(word in text for word in ['enable', 'disable', 'on', 'off']):
                return 'toggle'
            return 'set'
        return 'toggle'

    def determine_action(self, doc) -> Optional[str]:
        text = doc.text.lower()
    
        # Check for 'gain stage' command using regular expression
        if re.search(r'gain stage on (\w+)', text):  # Check for "gain stage on <channel>"
            return 'gain_stage'
        if re.search(self.gain_stage_regex, text):  # Check for range-based command
            return 'gain_stage'

        # Special handling for "clear solo" without requiring channel numbers
        if 'clear solo' in text:
            return 'clear_solo'
    
        # Look for direct command words like 'load', 'solo', etc.
        for command, variations in self.command_words.items():
            if any(var in text for var in variations):
                if command in ['load', 'phantom_power', 'solo', 'gate', 'select', 'mute', 'equalizer',  'compressor']:
                    return command  # Direct commands like 'load', 'solo', etc.
                elif command in ['fader', 'gain']:
                    # Check for specific action verbs (e.g., increase, decrease)
                    for action_type, verbs in self.action_verbs.items():
                        if any(verb in text for verb in verbs):
                            if action_type in ['increase', 'decrease']:
                                return f'adjust_{command}'
                            return f'set_{command}'
                elif command == 'threshold':
                    # Handle gate threshold adjustments
                    for action_type, verbs in self.action_verbs.items():
                        if any(verb in text for verb in verbs):
                            if action_type in ['increase', 'decrease']:
                                return 'adjust_gate_threshold'
                            return 'set_gate_threshold'

        # # Special handling for "clear solo" without requiring channel numbers
        # if 'clear solo' in text:
        #     return 'clear_solo'
    
        # Fallback for unrecognized actions
        return None


    
    def determine_state(self, doc) -> Optional[int]:
        text = doc.text.lower()
        
        if any(word in text for word in ['unmute', 'off', 'disable', 'deactivate', 'unsolo']):
            return 0
        if any(word in text for word in ['mute', 'on', 'enable', 'activate', 'select', 'solo']):
            return 1
            
        return None


    def extract_item_to_load(self, doc) -> Optional[str]:
        # Look for nouns or proper nouns after the command word "load" and include "preset"
        item_tokens = []
        load_found = False
        preset_found = False
    
        for token in doc:
            if token.text.lower() == "load":
                load_found = True
            elif load_found and token.text.lower() == "preset":
                preset_found = True
                break
            elif load_found:
                item_tokens.append(token.text)
    
        if item_tokens and preset_found:
            return " ".join(item_tokens)
        return None



    def interpret_command(self, text: str) -> Optional[Dict[str, Union[str, List[int], float, None]]]:
        doc = self.nlp(text.lower())
        
        # Determine the action
        action = self.determine_action(doc)
        print(action)
        if not action:
            print(f"Could not determine action from command: {text}")
            return None
        
        # Special handling for 'load' action
        if action == 'load':
            # Extract the item to load (e.g., "keyboard", "kick")
            # item = next((token.text for token in doc if token.text.lower() not in ['load', 'the']), None)
            item = self.extract_item_to_load(doc)
            channel = read_selected_channel()
            if not item:
                print(f"Could not determine what to load in command: {text}")
                return None
            
            return {
                'action': 'load',
                'item': item.lower(),
                'channel': channel,
            }

        # Special handling for 'clear solo' action
        if action == 'clear_solo':
            channel = read_selected_channel()
            return {
                'action': 'clear_solo',
                'channel': channel,
            }
        
        channels, value = self.extract_channels_and_value(doc)
        
        # For other actions, extract channels and values
        channels, value = self.extract_channels_and_value(doc)

        # Check for default channels if not explicitly mentioned
        if not channels and action in ['solo', 'gate', 'equalizer', 'compressor']:
            channel = read_selected_channel()
            if not channel:
                print("No channel specified and no selected channel available")
                return None
            channels = [channel]

        if not channels:
            print("No channel numbers or names found in command")
            return None



        # Handle toggle actions
        if action in ['mute', 'phantom_power', 'solo', 'gate', 'select', 'equalizer', 'compressor']:
            state = self.determine_state(doc)
            if state is None:
                print(f"Could not determine state for {action} command")
                return None
            
            return {
                'action': action,
                'channels': channels,
                'state': state
            }
        
        # Handle set or adjust actions
        elif action.startswith('set_') or action.startswith('adjust_'):
            if action.startswith('adjust_'):
                adjustment = value if value is not None else 3.0
                if any(word in text.lower() for word in ['decrease', 'lower', 'reduce', 'down']):
                    adjustment = -adjustment
                
                return {
                    'action': action.split('_')[1],
                    'channels': channels,
                    'value': None,
                    'adjustment': adjustment
                }
            else:
                if value is None:
                    print(f"No value specified for {action} command")
                    return None
                
                return {
                    'action': action.split('_')[1],
                    'channels': channels,
                    'value': value,
                    'adjustment': None
                }
        
        # Handle 'gain_stage' action (if applicable)
        elif action == 'gain_stage':
            matches = re.findall(self.gain_stage_regex, text.lower())  # Ensure case-insensitivity
            if matches:
                channels = set()
                for match in matches:
                    start = int(match[0])  # First channel (always present)
                    end = int(match[1]) if match[1] else start  # Second channel (optional)
                    channels.update(range(start, end + 1))  # Add the full range
                return {
                    'action': 'gain_stage',
                    'channels': sorted(channels),
                }

            # channel_name_match = re.search(r'gain stage on (\w+)', text.lower())
            # if channel_name_match:
                # channel_name = channel_name_match.group(1).strip().lower()
                # print(f"Channel name '{channel_name}' not found in CHANNEL_MAP.")
                # return None
        
        return None

def execute_command(command):
    try:
        """Execute the interpreted command."""
        if command is None:
            return

        action = command['action']
        
        # Handle commands with multiple channels
        if 'channels' in command:
            for channel in command['channels']:
                if action == 'phantom_power':
                    set_channel_phantom_power(channel, command['state'])
                elif action == 'solo':
                    set_channel_solo(channel, command['state'])
                elif action == 'select':
                    select_channel(channel)
                elif action == 'gate':
                    set_channel_gate(channel, command['state'])
                elif action == 'mute':
                    if command['state'] == 1:
                        mute_channel(channel)
                    else:
                        unmute_channel(channel)
                elif action == 'fader':
                    if command.get('value') is not None:
                        set_fader_value(channel, command['value'])
                    elif command.get('adjustment') is not None:
                        print(command['adjustment'])
                        current_value = read_fader_value(channel)
                        if current_value is not None:
                            new_value = current_value + command['adjustment']
                            set_fader_value(channel, new_value)
                elif action == 'gain':
                    source_type_ch = read_channel_source(WING_IP, OSC_PORT, channel)
                    if source_type_ch in ['LCL', 'A', 'B', 'C']:
                        if command.get('value') is not None:
                            set_input_gain(channel, command['value'])
                        elif command.get('adjustment') is not None:
                            current_gain = read_input_gain(channel)
                            if current_gain is not None:
                                new_gain = max(min(current_gain + command['adjustment'], 40), 0)
                                set_input_gain(channel, new_gain)
                    else:
                        if command.get('value') is not None:
                            set_input_trim(channel, command['value'])
                        elif command.get('adjustment') is not None:
                            current_gain = read_input_trim(channel)
                            if current_gain is not None:
                                new_gain = max(min(current_gain + command['adjustment'], 40), 0)
                                set_input_trim(channel, new_gain)
                elif action == 'gate_threshold':
                    pass
                    # set_channel_gate_threshold(channel, 
                                             # value=command.get('value'),
                                             # adjustment=command.get('adjustment'))
                elif action == "gain_stage":
                    pass
                    # try:
                        # run_metering('channel', channel=channel)
                    # except ValueError:
                        # print("Invalid channel number provided.")
                elif action == 'equalizer':
                    set_channel_eq(channel, command['state'])
                elif action == 'compressor':
                    set_channel_compressor(channel, command['state'])
                else:
                    print(f"Unknown action: {action}")
                
            return

        # Existing single channel command logic remains the same
        channel = command['channel']
        
        if action == 'fader':
            if command.get('value') is not None:
                set_fader_value(channel, command['value'])
            elif command.get('adjustment') is not None:
                current_value = read_fader_value(channel)
                if current_value is not None:
                    new_value = current_value + command['adjustment']
                    set_fader_value(channel, new_value)
        elif action == 'gain':
            if command.get('value') is not None:
                set_input_gain(WING_IP, OSC_PORT, channel, command['value'])
            elif command.get('adjustment') is not None:
                current_gain = read_input_gain(channel)
                if current_gain is not None:
                    new_gain = max(min(current_gain + command['adjustment'], 40), 0)
                    set_input_gain(WING_IP, OSC_PORT, channel, new_gain)
        elif action == 'gate_threshold':
            pass
            # set_channel_gate_threshold(channel, 
                                     # value=command.get('value'),
                                     # adjustment=command.get('adjustment'))
        elif action == "gain_stage":
            pass
            # try:
                # run_metering('channel', channel=channel)
            # except ValueError:
                # print("Invalid channel number provided.")
        elif action == 'clear_solo':  # Handle clear solo here
            clear_all_solos()
        elif action == 'load':
            pass
            # Extract the item to load from the command
            # item_to_load = command.get('item')
            # item_to_load = item_to_load.upper()
            # if item_to_load and channel <= 40:
                # print(f"Loading {item_to_load}...")
                # manager = WingOSCChannelManager()
                # Add your logic to handle loading (e.g., loading samples or presets)
                # manager.apply_channel_settings(channel, item_to_load)
                # load_item(item_to_load)
            # elif channel > 40:
                # print(f"Cannot load on this channel {channel}.")
            # else:
                # print("No item specified to load.")
            # return
        elif action == 'equalizer':
            set_channel_equalizer(channel, command['state'])
        elif action == 'compressor':
            set_channel_compressor(channel, command['state'])
        else:
            print(f"Unknown action: {action}")
    except Exception as e:
        print(f"An exception occurred: {e} or Mixer not connected")
        
        
def mute_channel(channel):
    """Mute a channel if not already muted."""
    current_status = read_mute_status(WING_IP, OSC_PORT, channel)
    print(current_status)
    
    if current_status == 1:
        print(f"Channel {channel} already muted")
        return f"Channel {channel} already muted"
        
    else:
        send_osc_message(WING_IP, OSC_PORT, f'/ch/{channel}/mute', 'i', 1)
        print(f"Channel {channel} muted")
        return f"Channel {channel} muted"

def unmute_channel(channel):
    """Unmute a channel if not already unmuted."""
    current_status = read_mute_status(WING_IP, OSC_PORT, channel)
    
    if current_status == 0:
        return f"Channel {channel} already unmuted"
    else:
        send_osc_message(WING_IP, OSC_PORT, f'/ch/{channel}/mute', 'i', 0)
        return f"Channel {channel} unmuted"

def read_mute_status(wing_ip, wing_port, channel):
    """Read the current mute status of a channel."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        osc_message = f'/ch/{channel}/mute'.encode()
        osc_message = pad_osc_message(osc_message)
        sock.sendto(osc_message, (wing_ip, wing_port))
        sock.settimeout(1.0)
        response, _ = sock.recvfrom(1024)
        
        # Parse the integer response
        # OSC format: address + type tag + value
        # Skip past the address part (the original message)
        data = response[len(osc_message):]
        
        # Check if we have an integer response (type tag ,i)
        if data.startswith(b',i'):
            # Extract the integer value (4 bytes after the type tag)
            value_bytes = data[4:8]
            value = struct.unpack('>i', value_bytes)[0]
            return value
        else:
            return None
    except socket.timeout:
        return None
    except Exception as e:
        return None
    finally:
        sock.close()

# Function to send OSC messages
def send_osc_message(ip, port, address, type_tag=None, value=None):
    """Send OSC message to the WING console.
    
    Args:
        ip (str): IP address of the WING console
        port (int): Port number
        address (str): OSC address pattern
        type_tag (str, optional): Type tag ('i' for integer, 'f' for float, etc)
        value: The value to send
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        msg = address.encode()
        msg += b'\0' * (4 - (len(msg) % 4))
        
        if type_tag and value is not None:
            msg += f',{type_tag}'.encode()
            msg += b'\0' * (4 - (len(msg) % 4))
            
            if type_tag == 'f':
                msg += struct.pack('>f', float(value))
            elif type_tag == 'i':
                msg += struct.pack('>i', int(value))
            elif type_tag == 's':
                msg += value.encode()
                msg += b'\0' * (4 - (len(value.encode()) % 4))
        
        sock.sendto(msg, (ip, port))
        return sock
    except Exception as e:
        print(f"Error sending OSC message: {e}")
        sock.close()
        return None

def select_channel(channel, wing_ip=WING_IP, wing_port=OSC_PORT):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        osc_message = b'/$ctl/$stat/selidx'
        osc_message += b'\0' * (4 - (len(osc_message) % 4))
        osc_message += struct.pack('>4si', b',i\0\0', channel)
        sock.sendto(osc_message, (wing_ip, wing_port))
        print(f"Sent request to select channel {channel}")
    except Exception as e:
        print(f'Error selecting channel: {e}')
    finally:
        sock.close()

# Add the functions for selecting and reading the selected channel
def read_selected_channel(wing_ip=WING_IP, wing_port=OSC_PORT):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        osc_message = b'/$ctl/$stat/selidx'
        osc_message += b'\0' * (4 - (len(osc_message) % 4))
        sock.sendto(osc_message, (wing_ip, wing_port))
        sock.settimeout(5.0)
        response, _ = sock.recvfrom(1024)
        if response.startswith(osc_message):
            response_data = response[len(osc_message):]
            if len(response_data) >= 20:
                type_tag = response_data[:4].decode('ascii').strip('\x00')
                if type_tag == ',sfi':
                    int_value = struct.unpack('>i', response_data[16:20])[0]
                    return int_value + 1 
                else:
                    print(f'Unexpected type tag: {type_tag}')
            else:
                print('Response too short')
        else:
            print('Unexpected response format')
    except socket.timeout:
        print('No response from WING')
    except Exception as e:
        print(f'Error: {e}')
    finally:
        sock.close()
    return None

def read_fader_value(channel, wing_ip=WING_IP, wing_port=OSC_PORT):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # Ensure zero-padded channel string for consistency
        channel_str = f'{channel:02d}'
        
        # OSC message for reading fader value
        osc_message = f'/ch/{channel_str}/fdr'.encode() + b'\0\0\0'
        sock.sendto(osc_message, (wing_ip, wing_port))
        sock.settimeout(5.0)
        response, _ = sock.recvfrom(1024)
        
        # Verify response format
        if response.startswith(osc_message[:-3]):
            response_data = response[len(osc_message)-3:]
            
            # Try to extract float value using struct
            try:
                fader_value = struct.unpack('>f', response_data[-4:])[0]
                if fader_value < -90:
                    fader_value = -90.0
                return fader_value
            except Exception:
                # Fallback parsing
                try:
                    fader_str = response_data.decode('ascii', errors='ignore').strip()
                    
                    if fader_str == '-oo':
                        return -90.0

                    # Remove any non-numeric characters
                    fader_str = ''.join(c for c in fader_str if c in '.-0123456789')
                    return float(fader_str)
                except:
                    pass
        return None
    except Exception:
        return None
    finally:
        sock.close()

def set_fader_value(channel, fader_value, wing_ip=WING_IP, wing_port=OSC_PORT):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # Ensure channel is zero-padded and clean
        channel_str = f'{channel:02d}'
        
        # OSC message for setting fader value
        osc_message = f'/ch/{channel_str}/fdr'.encode()
        
        # Pad to 4-byte alignment
        padding_length = 4 - (len(osc_message) % 4)
        osc_message += b'\0' * padding_length
        
        # Add type tag and value
        osc_message += b',f\0\0'
        osc_message += struct.pack('>f', fader_value)
        
        sock.sendto(osc_message, (wing_ip, wing_port))
    except Exception as e:
        print(f'Error setting fader value for channel {channel}: {e}')
    finally:
        sock.close()

# Function to set input trim
def set_input_trim(channel, trim_value, wing_ip=WING_IP, wing_port=OSC_PORT):
    if not -18 <= trim_value <= 18:
        print("Trim value must be between -18 and 18 dB")
        return False
    sock = send_osc_message(wing_ip, wing_port, f'/ch/{channel}/in/set/trim', 'f', float(trim_value))
    sock.close()
    return True

def set_input_gain(channel, gain_value, wing_ip=WING_IP, wing_port=OSC_PORT):
    if not 0 <= gain_value <= 40:
        print("Gain value must be between 0 and 40 dB")
        return False
    sock = send_osc_message(wing_ip, wing_port, f'/ch/{channel}/in/set/$g', 'f', float(gain_value))
    sock.close()
    return True

def set_channel_phantom_power(channel, state):
    """Set phantom power state for a channel."""
    send_osc_message(WING_IP, OSC_PORT, f'/ch/{channel}/in/set/$vph', 'i', state)
    print(f"Channel {channel} phantom power set to {'on' if state == 1 else 'off'}")

def set_channel_solo(channel, state):
    """Set solo state for a channel."""
    send_osc_message(WING_IP, OSC_PORT, f'/ch/{channel}/$solo', 'i', state)
    print(f"Channel {channel} solo set to {'on' if state == 1 else 'off'}")

def set_channel_gate(channel, state):
    """Set gate state for a channel."""
    send_osc_message(WING_IP, OSC_PORT, f'/ch/{channel}/gate/on', 'i', state)
    print(f"Channel {channel} gate set to {'on' if state == 1 else 'off'}")

def set_channel_eq(channel, state):
    """Set gate state for a channel."""
    send_osc_message(WING_IP, OSC_PORT, f'/ch/{channel}/eq/on', 'i', state)
    print(f"Channel {channel} eq set to {'on' if state == 1 else 'off'}")

def set_channel_compressor(channel, state):
    """Set gate state for a channel."""
    send_osc_message(WING_IP, OSC_PORT, f'/ch/{channel}/dyn/on', 'i', state)
    print(f"Channel {channel} compressor set to {'on' if state == 1 else 'off'}")

def clear_all_solos():
    """Clear solo state for all channels."""
    solo = 1
    for channel in range(1, 42):  # Assuming 40 channels, check your console's channel range
        send_osc_message(WING_IP, OSC_PORT, f'/ch/{solo}/$solo', 'i', 0)
        solo += 1 

# Function to read input gain
def read_input_gain(channel, wing_ip=WING_IP, wing_port=OSC_PORT):
    sock = send_osc_message(wing_ip, wing_port, f'/ch/{channel}/in/set/$g')
    try:
        sock.settimeout(5.0)
        response, _ = sock.recvfrom(1024)
        gain_value = struct.unpack('>f', response[-4:])[0]
        print(gain_value)
        return gain_value
    except socket.timeout:
        print("Timeout occurred while reading input gain")
    except Exception as e:
        print(f"Error reading input gain: {e}")
    finally:
        sock.close()

# Function to read input gain
def read_input_trim(channel, wing_ip=WING_IP, wing_port=OSC_PORT):
    sock = send_osc_message(wing_ip, wing_port, f'/ch/{channel}/in/set/trim')
    try:
        sock.settimeout(5.0)
        response, _ = sock.recvfrom(1024)
        trim_value = struct.unpack('>f', response[-4:])[0]
        print(trim_value, 54)
        return trim_value
    except socket.timeout:
        print("Timeout occurred while reading input gain")
    except Exception as e:
        print(f"Error reading input gain: {e}")
    finally:
        sock.close()
def read_channel_source(wing_ip, wing_port, channel):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        osc_message = f'/ch/{channel:02d}/in/conn/grp'.encode()
        osc_message = pad_osc_message(osc_message)
        sock.sendto(osc_message, (wing_ip, wing_port))
        sock.settimeout(5.0)
        response, _ = sock.recvfrom(1024)
        response_data = response[len(osc_message):].decode('ascii', errors='ignore')
        if response_data.startswith(',s'):
            source_value = response_data[2:].strip('\x00')
            return source_value
        else:
            print(f"Unexpected response format: {response_data}")
            return None
    except socket.timeout:
        print("Socket timeout occurred")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        sock.close()
        
def pad_osc_message(osc_message):
    """Pad OSC message to be a multiple of 4 bytes."""
    padding = (4 - len(osc_message) % 4) % 4
    return osc_message + b'\0' * padding
    

        
command_prompt ='''
You are a command generator for a digital audio mixer. Analyze the text input and process digital mixer commands.
IMPORTANT RULES:
DO NOT generate a command unless the input is 100% clear and matches a known format.
If a command is incomplete, ambiguous return:
HEARD: [transcription of what was heard]
COMMAND: NO_COMMAND_DETECTED
COMMAND FORMATS:
For channel operations:
"mute channel [number/range]" or "unmute [channel name/range]"
"solo channel [number/range]" or "solo [channel name/range]"
"clear solo"
"select channel [number/range]" or "select [channel name/range]"
For level controls:
"increase/decrease fader/volume/level on channel [number/range] by [value]dB"
"set fader/volume/level on [channel name/range] to [value]dB"
"increase/decrease gain on channel [number/range] by [value]dB"
"set gain on [channel name/range] to [value]dB"
"gain stage on channel [number/range]" or "gain stage on channel [number] to [number]"
For processing:
"turn on/off phantom/48v on channel [number/range]"
"enable/disable gate for [channel name/range]"
"set threshold on channel [number/range] to [value]dB"
"turn on/off eq for channel [number/range]"
"enable/disable compressor for [channel name/range]"
For presets:
"load preset [name]"
MULTI-CHANNEL SUPPORT:
Commands can specify multiple channels using:
Explicit lists: "mute channels 1, 2, and 3"
Ranges: "increase volume on channels 1 to 5 by 3dB"
Ranges: Set fader channel 1 to 10 to 0 dB.
When detecting multiple channels:
Convert text numbers to integers (e.g., "two" → 2)
Support "to", "through", "-" for ranges (e.g., "channels 3 to 7")
Handle channel names using the predefined mapping below
text PATTERN CORRECTIONS:
Apply the following corrections when detecting commands:
- "news" → "mute" (when before "channel")
- "peter", "speaker", "speed", "feeder", "further" → "fader"
- "tom" → "turn" (when before "phantom")
- "clay", "play" → "clear" (when before "solo")
- "solo." → "solo"
- "feeder", "feeders" → "fader" (when in context of mixer controls)
- "gin", "against", "game" → "gain"
- "it", "stitch", "state" → "stage" (when after "gain")
- "guest", "get", "gift", "kids" → "gate"
- "special" → "threshold" (when after "gate")
- "sex" → "set" (when before "fader")
- "gd", "gb", "dd", "td", "seconds" → "db" (when after numbers)
- "near" → "snare"
- "find" → "phantom" (when before "power")
- "equaliser" → "equalizer"
- "on mute" → "unmute"

RESPONSE FORMAT:
TEXT: [Verbatim transcript of what was detected]
COMMAND: [The formatted command OR "NO_COMMAND_DETECTED" if no valid command was heard]
Examples:
✅ Valid command:
TEXT: increase the fader on channels two through five by three
COMMAND: increase fader on channels 2-5 by 3dB
✅ Valid command with misheard word:
TEXT: increase the feeder on channels two through five by three
COMMAND: increase fader on channels 2-5 by 3dB
✅ Valid command using channel name:
HEATEXTRD: mute kick
COMMAND: mute channel 1
✅ Valid command with misheard DB format:
TEXT: set fader on channel 3 to minus 10 gb
COMMAND: set fader on channel 3 to -10dB
❌ Unclear or incomplete text:
TEXT: gain stage on... [trailing off]
COMMAND: NO_COMMAND_DETECTED
❌ Background noise detected:
HEARD: [background conversation]
COMMAND: NO_COMMAND_DETECTED
❌ Silence detected:
HEARD: [silence]
COMMAND: NO_COMMAND_DETECTED
❌ Unintelligible input:
HEARD: [mumbling, distorted audio]
COMMAND: NO_COMMAND_DETECTED
DO NOT assume missing words. If unsure, return "NO_COMMAND_DETECTED"
Input command: {user_text}
Corrected command:
'''

def validate_voice_command(user_text, command_prompt=command_prompt):
    try:
        # If input is None or empty, return None
        if not user_text:
            print("Empty input received")
            return None

        # Format the prompt with the user's text
        formatted_prompt = command_prompt.format(user_text=user_text)  # Explicitly map 'user_text'
        # print("Formatted Prompt:", formatted_prompt)
        
        # Generate content with Gemini
        if not google_model:
            print("Google model not initialized")
            return None

        response = google_model.generate_content(formatted_prompt)
        
        # Parse the Gemini response
        gemini_output = response.text.strip()
        print("Gemini Output:", gemini_output)
        
        # Extract command section
        if 'COMMAND:' in gemini_output:
            command_section = gemini_output.split('COMMAND:')[1].strip()
            print("Extracted Command:", command_section)
            
            # If command is NO_COMMAND_DETECTED, return None
            if command_section == 'NO_COMMAND_DETECTED':
                print("No valid command detected")
                return None
            
            return command_section
        else:
            print("No COMMAND section found in Gemini output")
            return None

    except KeyError as e:
        print(f"KeyError in validate_voice_command: {e}")
        return None
    except Exception as e:
        print(f"Error in validate_voice_command: {e}")
        return None
