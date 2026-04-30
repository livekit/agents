# FonadaLabs TTS plugin for Livekit Agents

A LiveKit plugin that integrates FonadaLabs TTS API using WebSocket, enabling high-quality text-to-speech synthesis with multiple Indian voices in LiveKit agents.

## Features

- 🎙️ **High-Quality TTS**: Uses FonadaLabs TTS API for natural-sounding speech
- 🇮🇳 **Rich Indian Voices**: 110+ voices across Hindi, Tamil, Telugu, and English
- ⚡ **Real-time Streaming**: WebSocket-based streaming for low-latency audio
- 🔒 **Secure**: API key authentication
- 🚀 **Easy Integration**: Simple plugin interface for LiveKit agents
- 🔄 **Dynamic Catalog**: Supported languages and voices are fetched live from the API

## Installation

```bash
pip install livekit-plugins-fonadalabs
```

## Pre-requisites

You'll need an API key from FonadaLabs (https://fonadalabs.ai). It can be set as an environment variable: `FONADALABS_API_KEY`



## Supported Languages & Voices

> **Note:** The full list of supported languages and voices is fetched dynamically from the FonadaLabs API at runtime (`https://api.fonada.ai/supported-voices`). The list below reflects the currently available voices.

### 🇮🇳 Hindi (default) — 18 voices

| | | | | |
|---|---|---|---|---|
| Dhruv | Vaanee *(default)* | Swastik | Laksh | Raag |
| Sarvagya | Komal | Meghra | Pancham | Tara |
| Sharad | Kritika | Mandra | Karn | Gauri |
| Ruhi | Roshini | Parikshit | | |

---

### 🎵 Tamil — 15 voices

| | | | | |
|---|---|---|---|---|
| Vaani | Isai | Thalam | Swaram | Madhuri |
| Naadham | Rachna | Pallavi | Mrityunjay | Malika |
| Yamini | Tilak | Dhruv | Sanket | Rudraksh |

---

### 🎶 Telugu — 60 voices

| | | | | |
|---|---|---|---|---|
| Ansh | Dhruv | Aadhira | Aahana | Aakriti |
| Ridhima | Vaani | Shaury | Bhavyaa | Tanuj |
| Utkarsh | Adyaa | Raagini | Kanika | Madhav |
| Malini | Priya | Mala | Dhairy | Shruti |
| Tara | Shubhra | Mandara | Tanya | Sara |
| Rudrika | Ruhi | Sameer | Geetika | Naitik |
| Kartik | Naira | Meghvani | Sneha | Sonika |
| Nikunj | Sonia | Mridul | Shobha | Gunika |
| Yuvaan | Abhishek | Dhruvika | Hardik | Raj |
| Raghav | Sanatan | Tanmay | Sanjana | Niharika |
| Ruchika | Sonakshi | Aaliya | Aanchal | Vartika |
| Rihaan | Divya | Taarini | Divyansh | Pavani |

---

### 🌐 English — 18 voices

| | | | | |
|---|---|---|---|---|
| Dhruv | Vaanee *(default)* | Swastik | Laksh | Raag |
| Sarvagya | Komal | Meghra | Pancham | Tara |
| Sharad | Kritika | Mandra | Karn | Gauri |
| Ruhi | Roshini | Parikshit | | |

---

## Language Specification

You can specify the language either by **display name**:

| Display Name |
|---|
| Hindi |
| English |
| Tamil |
| Telugu |
