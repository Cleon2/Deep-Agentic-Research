from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from typing import Dict, Any
import time

from deep_research.config import NEBIUS_API_KEY
from deep_research.output_schemas import PlannerOutput
from deep_research.state import ResearchState


async def planner_node(state: ResearchState) -> Dict[str, Any]:
    iteration = state.get("iteration_count", 0)
    is_follow_up = iteration > 0
    
    if is_follow_up:
        print(f"PLANNER (Iteration {iteration + 1}): Planning follow-up research...")
    else:
        print("PLANNER: Analyzing query and creating initial research plan...")
    
    start_time = time.time()
    query = state["query"]
    
    llm = ChatOpenAI(
        model="meta-llama/Llama-3.3-70B-Instruct",
        temperature=0,
        api_key=NEBIUS_API_KEY,
        base_url="https://api.studio.nebius.ai/v1/"
    )
    
    structured_llm = llm.with_structured_output(PlannerOutput)
    
    if is_follow_up:
        # Follow-up planning - refine based on gaps
        gaps = state.get("identified_gaps", [])
        
        system_prompt = """You are an expert research planner handling follow-up research.

You've been given:
1. The original research query
2. Gaps identified in previous research
3. Suggested follow-up areas

Your job:
- Generate 1-3 focused follow-up tasks that address the gaps
- Make tasks specific and targeted
- Avoid duplicating previous research
- Each task should have 2-3 optimized search queries"""

        human_prompt = f"""Original Query: {query}

Previous Research Plan: {state.get('research_plan', '')}

Identified Gaps:
{chr(10).join(f'- {gap}' for gap in gaps)}

Suggested Follow-ups:
{chr(10).join(f'- Task {t["task_id"]}: {t["description"]}' for t in state.get('follow_up_tasks', []))}

Generate 1-3 targeted follow-up tasks to address these gaps."""

    else:
        # Initial planning
        system_prompt = """You are an expert research planner. Your job is to analyze research queries and create comprehensive research plans.

Your planning principles:
1. **Decompose complex queries** into 3-7 focused, parallel research tasks
2. **Each task should be independent** - can be executed without waiting for others
3. **Tasks should complement each other** - together they answer the full query
4. **Avoid redundancy** - don't duplicate research across tasks
5. **Generate 2-3 optimized search queries per task** - make them specific and targeted
6. **Define clear output schemas** - specify what structured data each task should return

Query complexity guidelines:
- Simple factual queries: 1-3 tasks
- Comparison queries: 3-5 tasks (one per entity + synthesis)
- Comprehensive research: 5-7 tasks (multiple dimensions)
- Analysis queries: 4-6 tasks (gather data + analyze from angles)"""

        human_prompt = f"""Analyze this research query and create a comprehensive research plan:

Query: {query}

Generate:
1. A high-level research strategy (2-3 sentences explaining your approach)
2. 3-7 research tasks that will thoroughly answer this query
3. For each task: description, 2-3 search queries, and expected output schema

Be strategic and thorough."""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", human_prompt)
    ])
    
    chain = prompt | structured_llm
    
    try:
        result: PlannerOutput = await chain.ainvoke({"query": query})
        
        # Log the plan
        print(f"Research Strategy:")
        print(f"{result.research_plan}\n")
        
        print(f"📌 Generated {len(result.tasks)} tasks:")
        for task in result.tasks:
            print(f"\n  Task {task['task_id']}:")
            print(f"  - Description: {task['description']}")
            print(f"  - Search Queries: {task['search_queries']}")
        
        execution_time = time.time() - start_time
        
        return {
            "research_plan": result.research_plan if not is_follow_up else state["research_plan"],
            "tasks": [task for task in result.tasks],
            "execution_time": state.get("execution_time", 0) + execution_time,
            "iteration_count": iteration
        }
    
    except Exception as e:
        print(f"\nPlanner Error: {e}")
        raise