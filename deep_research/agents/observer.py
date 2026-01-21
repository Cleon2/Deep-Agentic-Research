from typing import Any

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from deep_research.config import NEBIUS_API_KEY
from deep_research.output_schemas import ResearchEvaluation
from deep_research.state import ResearchState, TaskResult


async def observer_node(state: ResearchState) -> dict[str, Any]:
    print("OBSERVER: Synthesizing research and evaluating completeness...")
    
    full_context = build_full_context(state)
    
    evaluation = await evaluate_research_completeness(
        query=state["query"],
        research_plan=state["research_plan"],
        task_results=state["task_results"],
        iteration_count=state["iteration_count"]
    )
    
    # 3. Decide next steps
    if evaluation["is_complete"]:
        print("Research is comprehensive. Ready to write final report.")
        return {
            "full_context": full_context,
            "research_complete": True,
            "identified_gaps": [],
            "follow_up_tasks": []
        }
    else:
        print(f"Research gaps identified: {evaluation['gaps']}")
        print(f"Generating {len(evaluation['follow_up_tasks'])} follow-up tasks...")
        
        return {
            "full_context": full_context,
            "research_complete": False,
            "identified_gaps": evaluation["gaps"],
            "follow_up_tasks": evaluation["follow_up_tasks"],
            "iteration_count": state["iteration_count"] + 1
        }
    
def build_full_context(state: ResearchState) -> str:
    """Build comprehensive context string"""
    parts = []
    parts.append(f"## Original Query\n{state['query']}\n")
    parts.append(f"## Research Plan\n{state['research_plan']}\n")
    
    parts.append("## Task Execution Results\n")
    for result in state["task_results"]:
        parts.append(f"\n### Task {result['task_id']}\n")
        parts.append(f"**Reasoning**: {result['reasoning']}\n")
        parts.append(f"**Findings**: {result['structured_output']}\n")
        parts.append(f"**Citations**: {result['citations']}\n")
    
    return "\n".join(parts)


async def evaluate_research_completeness(
    query: str,
    research_plan: str,
    task_results: list[TaskResult],
    iteration_count: int
) -> dict[str, Any]:
    # Build summary of current research
    research_summary = "\n\n".join([
        f"Task {r['task_id']}:\n{r['reasoning'][:300]}..."
        for r in task_results
    ])
    
    llm = ChatOpenAI(
        model="meta-llama/Llama-3.3-70B-Instruct",
        temperature=0,
        api_key=NEBIUS_API_KEY,
        base_url="https://api.studio.nebius.ai/v1/"
    )
    
    structured_llm = llm.with_structured_output(ResearchEvaluation)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a research quality evaluator. Assess if the research comprehensively answers the query.

Evaluation criteria:
- Does research cover all aspects of the query?
- Are there obvious gaps or missing perspectives?
- Is information sufficient for a comprehensive answer?
- Would a follow-up question naturally arise?

Be critical but fair. If 80%+ complete, mark as complete."""),
        
        ("human", """Query: {query}

Original Plan: {research_plan}

Research Completed (Iteration {iteration}):
{research_summary}

Evaluate: Is this research complete enough to produce a final report?""")
    ])
    
    chain = prompt | structured_llm
    
    evaluation = await chain.ainvoke({
        "query": query,
        "research_plan": research_plan,
        "iteration": iteration_count + 1,
        "research_summary": research_summary
    })
    
    follow_up_tasks = []
    if not evaluation.is_complete and evaluation.suggested_follow_ups:
        for i, suggestion in enumerate(evaluation.suggested_follow_ups[:3]):  # Max 3 follow-ups
            follow_up_tasks.append({
                "task_id": f"followup_{iteration_count + 1}_{i + 1}",
                "description": suggestion,
                "search_queries": [suggestion],  # Simplified
                "output_schema": {"findings": "string"},
                "status": "pending"
            })
    
    return {
        "is_complete": evaluation.is_complete,
        "confidence": evaluation.confidence,
        "gaps": evaluation.gaps,
        "follow_up_tasks": follow_up_tasks
    }