# Idilig

Idilig is an Interactive DIfferential LIne Growth implementation using numpy and scikit.

## Usage

Clone the repository: `git clone https://github.com/valentinlageard/differential_line_growth.git`

Go into the folder: `cd differential_line_growth`

Install the requirements: `pip install -r requirements.py`

Start the program: `python3 main.py`

Upon launch, if MIDI is enabled, it will prompt you to select a midi device. Enter the number associated and press Enter.

## Interactions

The preferred way to interact with the program is to use a midi controller. By default, the MIDI configuration maps to an Akai MidiMix

You can configure the `config.ini` file to map the MIDI interactions to custom CCs and note events.

Else the keyboard is used to interact with the following mapping:

Key|Interaction
-|-
Space|Play/Pause
A|Trace
R|Reset
I|Debug info
O|Debug color
P|Node drawing
UP/DOWN|Increase/Decrease scale
Q/W|Increase/Decrease growth
S/X|Increase/Decrease attraction
D/C|Increase/Decrease repulsion
F/V|Increase/Decrease alignement