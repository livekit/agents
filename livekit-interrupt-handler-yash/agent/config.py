import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    def __init__(self):
        self.livekit_url = os.getenv("LIVEKIT_URL")
        self.api_key = os.getenv("LIVEKIT_API_KEY")
        self.api_secret = os.getenv("LIVEKIT_API_SECRET")
        self.openai_key = os.getenv("OPENAI_API_KEY")
        self.cartesia_key = os.getenv("CARTESIA_API_KEY")
        self.deepgram_key = os.getenv("DEEPGRAM_API_KEY")

def load_config():
    return Config()
