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

### 🎵 Tamil — 16 voices

| | | | | |
|---|---|---|---|---|
| Vaani | Isai | Thalam | Swaram | Madhuram |
| Naadham | Ragam | Pallavi | Komalam | Raagamalika |
| Geetham | Taalam | Dhruv | Sangeetham | Raagaratna |
| Pancham | | | | |

---

### 🎶 Telugu — 60 voices

| | | | | |
|---|---|---|---|---|
| Naadamu | Dhruv | Taalam | Geetamu | Raagamalika |
| Sangeetamu | Vaani | Swaramu | Layamu | Taalabaddha |
| Raagapriya | Swarajathi | Raagini | Karn | Naada |
| Meghamalini | Sangeetapriya | Raagamala | Dhairy | Shruti |
| Tara | Komalavani | Mandara | Taana | Swarajati |
| Raagaanjali | Ruhi | Shweta | Geetika | Swaramala |
| Aalapana | Raagaratnam | Meghavani | Sanskar | Geetavani |
| Taala | Hardik | Mridul | Sangeetavani | Geetamala |
| Naadapriya | Abhishek | Dhwanimala | Sangeetanjali | Gamaka |
| Raagasudha | Sangeetaratna | Taalamani | Sangeetasundari | Nishchay |
| Raagavalli | Swarasudha | Sangeetaswarna | Raagamanjari | Swaravara |
| Naadeshwara | Dhwanividya | Taalapala | Dhwanipala | Swarapala |

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
