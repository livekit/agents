# FonadaLabs TTS plugin for Livekit Agents

A LiveKit plugin that integrates FonadaLabs TTS API using WebSocket, enabling high-quality text-to-speech synthesis with multiple Indian voices in LiveKit agents.

## Features

- 🎙️ **High-Quality TTS**: Uses FonadaLabs TTS API for natural-sounding speech
- 🇮🇳 **Rich Indian Voices**: 70+ voices across Hindi, Tamil, Telugu, and English
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

### 🇮🇳 Hindi (default) — 70 voices

| | | | | |
|---|---|---|---|---|
| Naad | Dhwani | Vaanee *(default)* | Swara | Taal |
| Laya | Raaga | Geetika | Swarini | Geet |
| Sangeeta | Raagini | Madhura | Komal | Sangeet |
| Meghra | Gandhar | Madhyam | Shruti | Pancham |
| Dhaivat | Nishad | Tara | Shadja | Komalika |
| Rishabh | Mandra | Tarana | Swarika | Komala |
| Geetini | Teevra | Chaitra | Madhur | Raagika |
| Swarita | Vibhaag | Gitanjali | Aalap | Sangeeti |
| Taan | Meend | Raagita | Gamak | Murki |
| Khatka | Andolan | Sparsh | Kampan | Shrutika |
| Swaranjali | Nada | Lahar | Tarang | Dhwaniya |
| Shrutini | Swar | Geetanjali | Raaginika | Sangeetika |
| Ninada | Swaroopa | Geetimala | Naadayana | Swarayana |
| Layakari | Taalayana | Raag | Swaranjana | Naadanika |
| Dhwanika | Swaraka | Sangeetara | Layabaddha | |

---

### 🎵 Tamil — 14 voices

| | | | | |
|---|---|---|---|---|
| Vaani | Isai | Thalam | Swaram | Madhuram |
| Naadham | Ragam | Pallavi | Komalam | Raagamalika |
| Geetham | Taalam | Dhwani | Sangeetham | Raagaratna |
| Shruti | | | | |

---

### 🎶 Telugu — 60 voices

| | | | | |
|---|---|---|---|---|
| Naadamu | Dhwani | Taalam | Geetamu | Raagamalika |
| Sangeetamu | Vaani | Swaramu | Layamu | Taalabaddha |
| Raagapriya | Swarajathi | Raagini | Komala | Naada |
| Meghamalini | Sangeetapriya | Raagamala | Dhwaniya | Shruti |
| Tara | Komalavani | Mandara | Taana | Swarajati |
| Raagaanjali | Raagika | Swaranjali | Geetika | Swaramala |
| Aalapana | Raagaratnam | Meghavani | Swarita | Geetavani |
| Taala | Layakari | Murki | Sangeetavani | Geetamala |
| Naadapriya | Dhwanika | Dhwanimala | Sangeetanjali | Gamaka |
| Raagasudha | Sangeetaratna | Taalamani | Sangeetasundari | Naadayana |
| Raagavalli | Swarasudha | Sangeetaswarna | Raagamanjari | Swaravara |
| Naadeshwara | Dhwanividya | Taalapala | Dhwanipala | Swarapala |

---

### 🌐 English — 70 voices

| | | | | |
|---|---|---|---|---|
| Naad | Dhwani | Vaanee *(default)* | Swara | Taal |
| Laya | Raaga | Geetika | Swarini | Geet |
| Sangeeta | Raagini | Madhura | Komal | Sangeet |
| Meghra | Gandhar | Madhyam | Shruti | Pancham |
| Dhaivat | Nishad | Tara | Shadja | Komalika |
| Rishabh | Mandra | Tarana | Swarika | Komala |
| Geetini | Teevra | Chaitra | Madhur | Raagika |
| Swarita | Vibhaag | Gitanjali | Aalap | Sangeeti |
| Taan | Meend | Raagita | Gamak | Murki |
| Khatka | Andolan | Sparsh | Kampan | Shrutika |
| Swaranjali | Nada | Lahar | Tarang | Dhwaniya |
| Shrutini | Swar | Geetanjali | Raaginika | Sangeetika |
| Meghra | Swaroopa | Geetimala | Naadayana | Swarayana |
| Layakari | Taalayana | Raag | Swaranjana | Naadanika |
| Dhwanika | Swaraka | Sangeetara | Layabaddha | |

---

## Language Specification

You can specify the language either by **display name**:

| Display Name
| Hindi 
| English
| Tamil 
| Telugu
