# Deep Research Agent

A multi-agent system for conducting comprehensive research on complex queries using LangGraph and Tavily's search API.

## Architecture

The system uses a four-agent workflow orchestrated through LangGraph's state machine:

- **Planner**: Decomposes queries into 3-7 parallel research tasks
- **Executor**: Runs tasks concurrently with intelligent content fetching (snippets first, full extraction when needed)
- **Observer**: Evaluates research completeness, identifies gaps, and generates follow-up tasks
- **Writer**: Synthesizes findings into structured reports with citations

The architecture supports iterative refinement—if gaps are detected, the system generates targeted follow-up tasks for up to 2 iterations.

## Features

- Parallel task execution using asyncio
- Adaptive content fetching to minimize API calls
- Structured outputs with Pydantic schemas
- Citation tracking and confidence scoring
- Stateful context sharing across agents

## Setup

```bash
pip install -r deep_research/requirements.txt
```

Create a `.env` file with your API keys:

```
NEBIUS_API_KEY=your_key_here
TAVILY_API_KEY=your_key_here
```

## Usage

```python
import asyncio
from deep_research.graph import create_research_graph
from deep_research.state import ResearchState

async def main():
    app = create_research_graph()

    initial_state: ResearchState = {
        "query": "Your research question here",
        "research_plan": "",
        "tasks": [],
        "task_results": [],
        "full_context": "",
        "final_output": {},
        "total_tokens_used": 0,
        "execution_time": 0.0,
        "iteration_count": 0,
        "max_iterations": 2,
        "research_complete": False,
        "identified_gaps": [],
        "follow_up_tasks": []
    }

    final_state = await app.ainvoke(initial_state)
    print(final_state["final_output"])

asyncio.run(main())
```

See `deep_research/test_full_graph.py` for a complete example.

## Technologies

- LangGraph for state machine orchestration
- Tavily API for search and content extraction
- Pydantic for structured outputs
- Python asyncio for concurrent execution
