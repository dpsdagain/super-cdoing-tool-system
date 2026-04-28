# Antigravity Super Coding Agent

Antigravity is an advanced, production-grade agentic coding assistant designed to maintain architectural parity with state-of-the-art enterprise tools like Anthropic Claude Code. Through a highly modular and extensible pipeline, Antigravity acts as an autonomous pair-programmer capable of editing files, executing terminal commands, researching codebases via a powerful hybrid RAG system, and managing its context intelligently.

## 🚀 Key Features

### 5-Layer Context Fortress
To prevent context saturation and high API costs, the **ContextManager** employs a rigorous 5-tiered preservation system:
1. **Lossless Tool Result Budgeting:** Heavy tool outputs (like `ls -R` or deep `git diff`) are offloaded seamlessly to a local `result_archive` rather than flooding the active context window.
2. **Micro-Compaction:** Aggressive stripping of dead syntactical noise and whitespace.
3. **Context Collapse:** Infinite-loop protection via automatic merging of identical operational loops.
4. **History Snipping:** Safe redaction of middle-turn conversational filler.
5. **Autocompact with State Restoration:** Pure distillation of engineering progress coupled with a reinjection of active `.git` status and project plans.

### Universal Hybrid RAG (Retrieval-Augmented Generation)
Antigravity ships with a specialized dual-path search engine built over `ChromaDB` and `SQLiteFTS5`:
- **Deep Code Understanding**: Incorporates an AST syntax chunker and regex-based extraction.
- **Provider-Aware Caching**: Natively supports deterministic block sorting and prompt-caching headers across Claude and DeepSeek endpoints. 
- **Semantic Caching**: Zero-latency cache hits for repeated queries via `SemanticCache`.

### Zero-Latency Vector Routing
Eliminates overhead by using mathematical and local semantic routing to determine if an incoming user prompt is a related follow-up or an entirely new concept, intelligently prioritizing cached chunk retrieval.

### Unified Model Intelligence
A centralized `llm_factory.py` manages an agnostic endpoint solution seamlessly across:
- **Local Ollama** models (for completely private inference)
- **Ollama Cloud** (for high-availability OSS endpoints)
- **OpenRouter** (for frontier intelligence like Claude 3.5 Sonnet, Gemini 2.0, or DeepSeek)

Includes intelligent semantic aliases (e.g. `fast`, `coder`, `opus`) that inject enhanced config constraints and token limits (like the `[1m]` expanded sequence).

## 🛠 Directory Architecture

After the rigorous Phase-3 refactor, the central agentic processing loop lives cleanly defined beneath `/new_sysy/`.

| File / Module | Purpose |
| ------------- | ------- |
| `query_engine.py` | The main execution loop; handles incoming queries and tool dispatches. |
| `context_manager.py` | The sophisticated pipeline for compressing and pruning chat history. |
| `permissions.py` | Secure `PermissionDecision` governance blocking dangerous OS executions. |
| `llm_factory.py` | The unified factory governing all LLM client instantiations and cache headers. |
| `search_engine.py` | RAG retrieval, chunk reranking, and semantic lookup. |
| `cache_engine.py` | Semantic inference cache for bypassing repeated prompts. |
| `prompt_builder.py` | Master system instructions and role framing blocks. |
| `*__tools.py` | Extracted and refined modular tools controlling Git, File I/O, Web, and Sub-agent execution. |
| `backend.py` | Facade for codebase ingestion components (`ingestion.py`, `chunkers.py`). |

## 📦 Requirements & Installation

1. Clone the repository
2. Ensure you have python dependencies configured (e.g., `langchain`, `langchain-community`, `beautifulsoup4`, `chromadb`).
3. Set your environment variables in `.env`:

```env
OPENROUTER_API_KEY="sk-or-v1-..."
OLLAMA_CLOUD_API_KEY="..."

# Example Tweaks:
LLM_TEMPERATURE=0.0
ENABLE_PROMPT_CACHING=True
```

## 🔐 Security 

A Fail-Closed security paradigm powers the agent's interaction via the `PermissionManager`. Sensitive tools and potentially destructive bash operations are gated, requiring explicit review decisions encoded directly into your workflow execution paths.
