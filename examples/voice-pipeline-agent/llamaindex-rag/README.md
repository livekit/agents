# RAG Example using LlamaIndex

This repository showcases three ways to build a voice assistant with Retrieval-Augmented Generation (RAG) using LlamaIndex:

1. **`chat_engine.py`**: Utilizes LlamaIndex's `as_chat_engine` for a straightforward, integrated solution. **Trade-off**: Lacks function calling support, limiting advanced interactions.

2. **`query_engine.py`**: Uses an LLM that supports function calling (e.g., OpenAI's models) to define custom functions like `query_info` for retrieval. **Trade-off**: Requires additional setup but offers greater flexibility.

3. **`retrieval.py`**: Manually injects retrieved context into the system prompt using LlamaIndex's retriever. **Trade-off**: Provides fine-grained control but involves complex prompt engineering.

**Current recommended way**: Use **`query_engine.py`** for its balance of flexibility and control, enabling function calling and custom behaviors without excessive complexity.
