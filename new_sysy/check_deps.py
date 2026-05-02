import sys
try:
    import rich
    import prompt_toolkit
    import langchain
    import langchain_openai
    import langchain_community
    import langchain_chroma
    import dotenv
    import pydantic
    import chromadb
    import git
    print("SUCCESS: All core dependencies are present.")
except ImportError as e:
    print(f"FAILURE: Missing dependency: {e}")
    sys.exit(1)
except Exception as e:
    print(f"ERROR: {e}")
    sys.exit(1)
