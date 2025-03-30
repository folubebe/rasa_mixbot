Wing Mixer Voice Control
A voice-controlled interface for Behringer Wing digital mixer, allowing hands-free control through natural language commands. The system uses Rasa for intent recognition and Google's Gemini AI to process and refine voice commands.
🎚️ Features

Natural Language Processing: Control your mixer with conversational commands
Multiple Command Types:

Channel selection, muting, and unmuting
Fader level adjustments
Gain control
Phantom power toggling
Solo and gate controls
EQ and compressor activation
Clear all solos


Intelligent Command Interpretation: Understands variations in command phrasing
Audio Knowledge Base: Ask general audio engineering questions through the integrated AI assistant

🔧 System Architecture
The system consists of the following components:

Rasa NLU: For intent classification and entity extraction
Command Interpreter: Processes and translates natural language into structured mixer commands
OSC Communication Layer: Sends commands to the Wing console via OSC protocol
Google Gemini AI: Refines ambiguous voice commands and provides audio knowledge responses

📋 Requirements

Python 3.8+
Rasa (for NLU)
google-generativeai
spaCy with English language model (en_core_web_sm)
pythonosc
word2number
Behringer Wing digital mixer with network connectivity

💻 Setup

Clone this repository:
Copygit clone https://github.com/yourusername/wing-mixer-voice-control.git
cd wing-mixer-voice-control

Install the required dependencies:
Copypip install -r requirements.txt
python -m spacy download en_core_web_sm

Set up your environment variables:
Copyexport GOOGLE_API_KEY="your_gemini_api_key"

Configure your Wing mixer IP and port in the code:
pythonCopyWING_IP = '192.168.1.183'  # Change to your Wing console IP
TCP_PORT = 2222  # TCP port used by the Wing console
UDP_PORT = 14135  # Port where you want to receive meter data
OSC_PORT = 2223  # Port for sending OSC messages

Train the Rasa model:
Copyrasa train


🎯 Usage

Start the Rasa server:
Copyrasa run

In a separate terminal, start the actions server:
Copyrasa run actions

Connect a microphone and start the voice input client (not included in this repo):
Copypython voice_client.py

Speak commands such as:

"Mute channel 5"
"Raise the fader on channel 1 by 3 dB"
"Set the gain on channel 7 to 20 dB"
"Enable phantom power on channel 3"
"Solo channel 2"
"Clear all solos"
"Turn on the gate for channel 8"
"Enable EQ on channel 4"
"Turn on compressor for channel 6"


Ask audio-related questions:

"What's the best way to mic a drum kit?"
"How should I set up compression for vocals?"
"Explain the difference between shelving and peaking EQs"



🧠 Command Interpreter Logic
The system processes commands through multiple stages:

Voice Command Validation: Using Gemini AI to clean up speech recognition results
Command Type Detection: Identifying the action (mute, fader adjustment, etc.)
Channel Extraction: Finding channel numbers in the command
Parameter Extraction: Determining values for settings
Command Execution: Translating to OSC messages for the mixer

📚 Classes and Functions
Key components:

ActionControlMixer: Main Rasa action for interpreting and executing mixer commands
ActionUseGemini: Handles general audio knowledge questions using Gemini AI
CommandInterpreter: Parses natural language into structured commands
send_osc_message(): Sends OSC messages to the Wing console
OSC message utility functions for each mixer operation

🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

Fork the repository
Create your feature branch (git checkout -b feature/AmazingFeature)
Commit your changes (git commit -m 'Add some AmazingFeature')
Push to the branch (git push origin feature/AmazingFeature)
Open a Pull Request

📄 License
This project is licensed under the MIT License - see the LICENSE file for details.
🙏 Acknowledgements

Behringer for the WING OSC implementation
Google for Gemini AI
The Rasa community