#!/bin/bash

# Exit on error
set -e

# ANSI colors for styling
CYAN='\033[96m'
GREEN='\033[92m'
YELLOW='\033[93m'
DIM='\033[2m'
BOLD='\033[1m'
RESET='\033[0m'

echo -e "\n${BOLD}🎤 Live Transcriber macOS Installer${RESET}\n"

# 1. Check for Homebrew
if ! command -v brew &> /dev/null; then
    echo -e "${YELLOW}Installing Homebrew...${RESET}"
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    
    # Add brew to PATH for Apple Silicon just in case
    if [[ -d /opt/homebrew/bin ]]; then
        eval "$(/opt/homebrew/bin/brew shellenv)"
    fi
else
    echo -e "${GREEN}✓ Homebrew already installed.${RESET}"
fi

# 2. Update Homebrew silently
echo -e "${DIM}Updating Homebrew...${RESET}"
brew update --quiet >/dev/null 2>&1

# 3. Install PortAudio
if ! brew ls --versions portaudio > /dev/null; then
    echo -e "${YELLOW}Installing PortAudio...${RESET}"
    brew install portaudio
else
    echo -e "${GREEN}✓ PortAudio already installed.${RESET}"
fi

# 4. Install BlackHole (System Audio Driver)
if ! system_profiler SPAudioDataType | grep -q "BlackHole"; then
    echo -e "${YELLOW}Installing BlackHole 2ch (for system audio capture)...${RESET}"
    brew install blackhole-2ch
else
    echo -e "${GREEN}✓ BlackHole 2ch already installed.${RESET}"
fi

# 5. Check Python
if ! command -v python3 &> /dev/null; then
    echo -e "${YELLOW}Python3 not found. Installing via Homebrew...${RESET}"
    brew install python
else
    echo -e "${GREEN}✓ Python3 already installed.${RESET}"
fi

# 6. Install Python Dependencies
echo -e "${YELLOW}Installing Python dependencies (mlx-whisper, pyaudio, numpy)...${RESET}"
python3 -m pip install -r requirements.txt --quiet

echo -e "\n${GREEN}${BOLD}🎉 Installation Complete!${RESET}"
echo -e "\nTo start the transcriber, run:"
echo -e "  ${CYAN}python3 transcriber.py${RESET}\n"

# Check if BlackHole was freshly installed and they need to make a Multi-Output device
if system_profiler SPAudioDataType | grep -q "BlackHole 2ch" && [[ ! -f ".blackhole_setup_done" ]]; then
    echo -e "${BOLD}Important: To capture Zoom/system audio:${RESET}"
    echo -e "  1. Open ${CYAN}Audio MIDI Setup${RESET} (press Cmd+Space, type it in)"
    echo -e "  2. Click the '+' button and select 'Create Multi-Output Device'"
    echo -e "  3. Check the boxes for your Speakers and BlackHole 2ch"
    echo -e "  4. In System Settings > Sound, set your output to that Multi-Output Device"
    touch .blackhole_setup_done
fi
