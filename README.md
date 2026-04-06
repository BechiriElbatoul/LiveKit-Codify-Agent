# Hospital Voice Agent

A voice AI agent for hospital patient intake, built with LiveKit. It collects patient information, checks a services catalog (PDF), stores data locally (SQLite), and optionally pushes to Google Sheets.

##  Features

- **Natural voice conversation** – powered by LiveKit inference (Deepgram STT, OpenAI/Gemini LLM, Cartesia TTS)
- **PDF catalog lookup** – reads hospital services from a PDF file
- **Patient data collection** – name, age, gender, city, phone, medical issue
- **Local database** – SQLite stores all intakes
- **Google Sheets export** – optional, real-time logging to a spreadsheet
- **Consent management** – asks permission before recording

##  Setup

1. **Clone the repo**
   ```bash
   git clone https://github.com/BechiriElbatoul/LiveKit-Codify-Agent.git

2. **Create virtual environment**
    ```bash
    python3 -m venv venv
    source venv/bin/activate  

3. **Install dependencies**
    ```bash
    pip install -r requirements.txt

3. **Run locally**
    ```bash
    python main.py dev