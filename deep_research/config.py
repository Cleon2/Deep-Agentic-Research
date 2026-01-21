import os
from dotenv import load_dotenv

load_dotenv()

# API Keys
NEBIUS_API_KEY = os.getenv("NEBIUS_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2", "false")
LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT", "deep-research")

# Search configs
MAX_SEARCH_RESULTS = 5
SEARCH_TIMEOUT = 30
