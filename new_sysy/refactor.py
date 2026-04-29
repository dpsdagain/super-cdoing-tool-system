import sys
import re

with open('new_sysy/rag_core.py', 'r') as f:
    content = f.read()

# I want to find where `def build_rag_chain` starts.
idx = content.find("def build_rag_chain(db: Chroma, model: str | None = None):")
if idx == -1:
    print("Cannot find build_rag_chain")
    sys.exit(1)

before = content[:idx]
build_fn = content[idx:]

# The original code has some issues like `MAX_TOKENS` being undefined. We'll leave the import as is.
# Now, let's create the class definition string.
class_def = """class ContextCacheChain:
    \"\"\"
    Unified chain with Agentic Routing, Hybrid Search,
    and Cross-Provider cache awareness.
    \"\"\"
    def __init__(self, db: Chroma, model: str | None, llm, prompt, router, reranker):
        self.db = db
        self.model = model
        self.llm = llm
        self.prompt = prompt
        self.router = router
        self.reranker = reranker
        self.question_answer_chain = prompt | llm
        self._specialist_llm_cache = {}
        self._sentinel_cooldown = {"last_turn": 0}
        self._sentinel_failures = {"count": 0}
        self.MAX_SENTINEL_FAILURES = 3

    def _on_sentinel_done(self, future):
        try:
            future.result()
            self._sentinel_failures["count"] = 0
        except Exception as e:
            self._sentinel_failures["count"] += 1
            logger.error(f"Sentinel failure ({self._sentinel_failures['count']}/{self.MAX_SENTINEL_FAILURES}): {e}")

    def _check_semantic_cache(self, user_input, coll_name, pinned_content, force_retrieval):
        sem_cache = get_semantic_cache()
        if not force_retrieval:
            cached_ans = sem_cache.lookup(user_input, threshold=SEMANTIC_CACHE_THRESHOLD,
                                          collection_scope=coll_name,
                                          pinned_content=pinned_content,
                                          model=self.model)
            if cached_ans:
                return cached_ans, sem_cache
        return None, sem_cache

    def _handle_sentinel(self, should_summarize, inputs, turn_count, full_history):
        background_future = None
        if should_summarize and not inputs.get("sentinel_future_active"):
            self._sentinel_cooldown["last_turn"] = turn_count
            background_future = _background_executor.submit(_background_summarize, list(full_history))
            background_future.add_done_callback(self._on_sentinel_done)
        return background_future

    def _prepare_context(self, final_docs, previous_union, intent):
        stable_hashes = {
            d.metadata.get("content_hash", "")
            for d in previous_union
        } if previous_union and intent == "FOLLOW-UP" else None

        established_docs = []
        new_docs = []
        if stable_hashes:
            for d in final_docs:
                if d.metadata.get("content_hash", "") in stable_hashes:
                    established_docs.append(d)
                else:
                    new_docs.append(d)
        else:
            new_docs = final_docs

        established_docs = _sort_docs_deterministically(established_docs, stable_hashes=None)
        new_docs = _sort_docs_deterministically(new_docs, stable_hashes=None)

        def _format_docs(docs):
            return "\\n\\n".join([f"SOURCE: {d.metadata.get('source')}\\nCONTENT: {d.page_content}" for d in docs]) if docs else ""

        stable_block = _format_docs(established_docs)
        new_block = _format_docs(new_docs)

        stable_context_str = f"<established_context>\\n{stable_block}\\n</established_context>" if stable_block else "None previously established."
        new_context_str = f"<new_discoveries>\\n{new_block}\\n</new_discoveries>" if new_block else "No new discoveries."
        
        return established_docs, new_docs, stable_context_str, new_context_str

    def _execute_llm_stream(self, active_chain, inputs, is_semantic_hit, user_input, coll_name, pinned_content, sem_cache):
        full_answer = []
        for chunk in active_chain.stream(inputs):
            content = chunk.content if hasattr(chunk, "content") else str(chunk)
            full_answer.append(content)
            yield {"answer": content, "raw_chunk": chunk}
            
        full_answer_str = "".join(full_answer)
        if not is_semantic_hit and len(full_answer_str) > 50:
            sem_cache.upsert(user_input, full_answer_str, collection_scope=coll_name,
                             pinned_content=pinned_content, model=self.model)

"""

# We'll use a smart replace approach to modify `_full_context_cache_chain`
# First, extract `def _full_context_cache_chain(inputs: dict):`
match = re.search(r'    def _full_context_cache_chain\(inputs: dict\):\n(.*?)    from langchain_core\.runnables', build_fn, re.DOTALL)
inner_body = match.group(1)

# Modify inner body to fit `__call__`
# It's indented by 8 spaces. We'll unindent by 4 spaces (to class method level).
inner_lines = inner_body.split('\n')
call_body = []
for line in inner_lines:
    if line.startswith('        '):
        new_line = line # Keep the 8 spaces!
    else:
        new_line = line
    # Replace variables with self.*
    new_line = re.sub(r'\bmodel\b', 'self.model', new_line)
    new_line = re.sub(r'\bdb\b', 'self.db', new_line)
    new_line = re.sub(r'\brouter\b', 'self.router', new_line)
    new_line = re.sub(r'\breranker\b', 'self.reranker', new_line)
    new_line = new_line.replace('_sentinel_failures', 'self._sentinel_failures')
    new_line = new_line.replace('_sentinel_cooldown', 'self._sentinel_cooldown')
    new_line = new_line.replace('MAX_SENTINEL_FAILURES', 'self.MAX_SENTINEL_FAILURES')
    new_line = new_line.replace('question_answer_chain', 'self.question_answer_chain')
    new_line = new_line.replace('_specialist_llm_cache', 'self._specialist_llm_cache')
    new_line = new_line.replace('prompt |', 'self.prompt |')
    new_line = new_line.replace('llm', 'self.llm')

    # Fix over-replacements
    new_line = new_line.replace('self.self.', 'self.')
    new_line = new_line.replace('self.model_name', 'model_name')
    new_line = new_line.replace('self.model=', 'model=')
    new_line = new_line.replace('self.model(', 'model(')
    new_line = new_line.replace('self.model_id', 'model_id')
    new_line = new_line.replace('get_self.llm', 'get_llm')
    new_line = new_line.replace('specialist_self.model', 'specialist_model')
    new_line = new_line.replace('base_self.llm', 'base_llm')
    new_line = new_line.replace('active_self.model_id', 'active_model_id')
    new_line = new_line.replace('get_embedding_self.model', 'get_embedding_model')

    call_body.append(new_line)

call_method = "    def __call__(self, inputs: dict):\n" + "\n".join(call_body)

# Replace the specific blocks with method calls
# 1. Semantic cache
call_method = re.sub(
    r'        sem_cache = get_semantic_cache\(\).*?        if not force_retrieval:.*?            cached_ans = sem_cache\.lookup\(.*?                return',
    r'''        cached_ans, sem_cache = self._check_semantic_cache(user_input, coll_name, pinned_content, force_retrieval)
        if cached_ans:
            yield {"answer": cached_ans, "intent": "CACHE_HIT"}
            return''',
    call_method, flags=re.DOTALL
)

# 2. Sentinel trigger
call_method = re.sub(
    r'        background_future = None\n        if should_summarize and not inputs.get\("sentinel_future_active"\):\n.*?background_future\.add_done_callback\(_on_sentinel_done\)',
    r'''        background_future = self._handle_sentinel(should_summarize, inputs, turn_count, full_history)''',
    call_method, flags=re.DOTALL
)
# Note: since the sentinel logic appears twice (once initialized, once triggered), we fix the trigger correctly.
call_method = call_method.replace('background_future.add_done_callback(self._on_sentinel_done)', '')

# Wait, regex replacement for Sentinel trigger
# Let's just do it carefully.
call_method = re.sub(
    r'        # ASYNC SENTINEL TRIGGER — uses full_history.*?background_future\.add_done_callback\(self\._on_sentinel_done\)',
    r'''        # ASYNC SENTINEL TRIGGER
        background_future = self._handle_sentinel(should_summarize, inputs, turn_count, full_history)''',
    call_method, flags=re.DOTALL
)

# 3. Context preparation
call_method = re.sub(
    r'        # Split Context: Prefix cache hits on <established_context>, Relevance hits on <new_discoveries>.*?        inputs\["new_context"\] = f"<new_discoveries>\\n\{new_block\}\\n</new_discoveries>" if new_block else "No new discoveries\."',
    r'''        established_docs, new_docs, stable_block_str, new_block_str = self._prepare_context(final_docs, previous_union, intent)

        inputs["stable_context"] = stable_block_str
        inputs["new_context"] = new_block_str''',
    call_method, flags=re.DOTALL
)

# 4. LLM execution
call_method = re.sub(
    r'        full_answer = "".*?                             pinned_content=pinned_content, model=self\.model\)',
    r'''        yield from self._execute_llm_stream(active_chain, inputs, is_semantic_hit, user_input, coll_name, pinned_content, sem_cache)''',
    call_method, flags=re.DOTALL
)

# Now rebuild the file
full_class = class_def + call_method

# The new `build_rag_chain`
new_build = """def build_rag_chain(db: Chroma, model: str | None = None):
    \"\"\"
    Build a retrieval chain with stable Full-Context Caching (Architecture A).
    \"\"\"
    llm = get_llm(model=model)
    
    is_cc = is_cache_capable(model) and ENABLE_PROMPT_CACHING
    
    max_bp, _ = get_cache_profile(model)

    if is_cc:
        static_system_text = CORE_INSTRUCTIONS
        block_specs = [
            static_system_text,
            "FULL SOURCE CONTEXT (PINNED):\\n{full_source_context}",
            "STABLE RAG CONTEXT (DETERMINISTIC):\\n{stable_context}",
            "CONVERSATION STATE:\\n{sentinel_state}",
            "NEW RAG DISCOVERIES:\\n{new_context}"
        ]
        system_blocks = []
        for idx, text in enumerate(block_specs):
            use_cache_marker = idx < max_bp
            formatted = format_message_content(text, model, use_cache=use_cache_marker)
            if isinstance(formatted, list):
                system_blocks.append(formatted[0])
            else:
                system_blocks.append({"type": "text", "text": formatted})

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_blocks),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
    else:
        system_text = (
            f"{CORE_INSTRUCTIONS}\\n\\n"
            "FULL SOURCE CONTEXT (PINNED):\\n{full_source_context}\\n\\n"
            "STABLE RAG CONTEXT (DETERMINISTIC):\\n{stable_context}\\n\\n"
            "CONVERSATION STATE:\\n{sentinel_state}\\n\\n"
            "NEW RAG DISCOVERIES:\\n{new_context}"
        )
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_text),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])

    router = get_router()
    reranker = get_reranker()

    from langchain_core.runnables import RunnableLambda
    pipeline = ContextCacheChain(db, model, llm, prompt, router, reranker)
    return RunnableLambda(pipeline)
"""

with open('new_sysy/rag_core.py', 'w') as f:
    f.write(before + full_class + "\n" + new_build)

print("Done")
