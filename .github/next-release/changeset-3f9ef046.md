---
"livekit-plugins-langchain": patch
---

feat: add RemoteGraph support to LangGraph plugin (#2589)

While working on this plugin, I encountered a naming conflict between the local langgraph.py file and the external langgraph package. This caused Python to attempt to load the local file instead of the package, breaking imports like from langgraph.pregel.protocol import PregelProtocol.

To avoid shadowing the package and improve clarity, I've renamed the file to langgraph_plugin.py. This aligns with the naming pattern of other plugin files and eliminates any future ambiguity
