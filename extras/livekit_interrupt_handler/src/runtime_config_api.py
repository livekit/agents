# runtime_config_api.py
"""
Small FastAPI server to update the ignored-words list at runtime.
Run: python runtime_config_api.py
"""

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from filler_filter import FillerFilter

app = FastAPI()
filter_instance = FillerFilter()

class WordsPayload(BaseModel):
    words: list[str]

@app.post("/config/ignored-words")
async def update_ignored_words(payload: WordsPayload):
    await filter_instance.update_ignored_words(payload.words)
    return {"ok": True, "words": payload.words}

@app.get("/config")
async def get_config():
    return {
        "ignored_words": list(filter_instance._ignored_set),
        "confidence_threshold": filter_instance._confidence_threshold
    }

def run_api():
    uvicorn.run(app, host="127.0.0.1", port=8081, log_level="info")

if __name__ == "__main__":
    run_api()
