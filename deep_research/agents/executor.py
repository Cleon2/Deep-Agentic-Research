# agents/executor.py
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from typing import Dict, Any, List
import asyncio
import time

from deep_research.state import ResearchState, TaskResult, ResearchTask
from deep_research.output_schemas import TaskExecutionOutput
from deep_research.tools.search import tavily_search, extract_content
from deep_research.config import NEBIUS_API_KEY


async def task_executor_node(state: ResearchState) -> Dict[str, Any]:
    """
    Executor node: Runs all tasks in parallel using asyncio
    """
    print("EXECUTOR: Running research tasks in parallel...")
    
    start_time = time.time()
    tasks = state["tasks"]
    
    print(f"\nExecuting {len(tasks)} tasks in parallel...\n")
    
    # Run async execution - REMOVE asyncio.run(), just await directly
    task_results = await execute_all_tasks(tasks, state)
    
    execution_time = time.time() - start_time
    print(f"All tasks completed in {execution_time:.2f}s")
    
    return {
        "task_results": task_results,
        "execution_time": state.get("execution_time", 0) + execution_time
    }


async def execute_all_tasks(
    tasks: List[ResearchTask], 
    state: ResearchState
) -> list[TaskResult]:
    """
    Execute all tasks concurrently using asyncio
    """
    task_coroutines = [
        execute_single_task(task, state) 
        for task in tasks
    ]
    
    results = await asyncio.gather(*task_coroutines, return_exceptions=True)
    
    task_results = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            task_id = tasks[i]['task_id']
            print(f"Task {task_id} failed: {result}")
            task_results.append({
                "task_id": task_id,
                "search_results": [],
                "reasoning": f"Task failed: {str(result)}",
                "structured_output": {},
                "citations": []
            })
        else:
            task_results.append(result)
            print(f"Task {result['task_id']} completed")
    
    return task_results


async def execute_single_task(
    task: ResearchTask, 
    state: ResearchState
) -> TaskResult:
    """
    Execute a single research task asynchronously
    """
    print(f"\nTask {task['task_id']}: {task['description']}")
    
    # Step 1: Execute all search queries concurrently
    search_coroutines = [
        tavily_search(query, max_results=5, search_depth="basic")
        for query in task['search_queries']
    ]
    
    search_results_lists = await asyncio.gather(*search_coroutines)
    
    all_search_results = []
    for results in search_results_lists:
        all_search_results.extend(results)
    
    print(f"Found {len(all_search_results)} results")
    
    snippet_sufficient, task_output = await try_reasoning_with_snippets(
        task=task,
        search_results=all_search_results,
        other_task_outputs=get_other_task_outputs(state, task['task_id'])
    )
    
    if not snippet_sufficient:
        print(f"Snippets insufficient, fetching full content...")
        
        top_results = all_search_results[:3]
        content_coroutines = [
            extract_content(result['url']) 
            for result in top_results
        ]
        
        full_contents = await asyncio.gather(*content_coroutines)
        
        for i, content in enumerate(full_contents):
            if content:
                top_results[i]['content'] = content
        
        _, task_output = await try_reasoning_with_snippets(
            task=task,
            search_results=all_search_results,
            other_task_outputs=get_other_task_outputs(state, task['task_id']),
            is_retry=True
        )
    
    citations = list(set([r['url'] for r in all_search_results if r['url']]))
    
    return {
        "task_id": task['task_id'],
        "search_results": all_search_results,
        "reasoning": task_output.reasoning,
        "structured_output": task_output.structured_output,
        "citations": citations[:10],
        "key_insights": task_output.key_insights
    }


async def try_reasoning_with_snippets(
    task: ResearchTask,
    search_results: List[Dict[str, Any]],
    other_task_outputs: List[Dict[str, Any]],
    is_retry: bool = False
) -> tuple[bool, TaskExecutionOutput]:
    """
    Attempt to reason with current search results (async version)
    """
    
    search_context = "\n\n".join([
        f"Source [{i+1}]: {r['title']}\nURL: {r['url']}\nContent: {r['content'][:500]}..."
        for i, r in enumerate(search_results)
    ])
    
    other_context = ""
    if other_task_outputs:
        other_context = "\n\nOther completed research:\n" + "\n".join([
            f"- {out['task_id']}: {out['structured_output']}"
            for out in other_task_outputs
        ])
    
    llm = ChatOpenAI(
        model="meta-llama/Llama-3.3-70B-Instruct",
        temperature=0,
        api_key=NEBIUS_API_KEY,
        base_url="https://api.studio.nebius.ai/v1/"
    )
    
    structured_llm = llm.with_structured_output(TaskExecutionOutput)
    
    content_type = "full content" if is_retry else "snippets"
    snippet_note = "" if is_retry else "If you find the snippets insufficient to confidently answer, indicate this in your reasoning."
    instruction = "Produce your best findings with the available information." if is_retry else "If snippets are insufficient, explain what you need."
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", f"""You are a research task executor. Analyze the search results and produce structured findings.

You are working with search result {content_type}. {snippet_note}

Be thorough, accurate, and cite information appropriately."""),
        
        ("human", """Task Description: {description}

Required Output Schema: {output_schema}

Search Results:
{search_context}

{other_context}

Instructions:
1. Analyze the search results carefully
2. Extract relevant information for each field in the output schema
3. Provide your reasoning process
4. List 3-5 key insights
5. """ + instruction + """

Generate your structured output now.""")
    ])
    
    chain = prompt | structured_llm
    
    try:
        result: TaskExecutionOutput = await chain.ainvoke({
            "description": task['description'],
            "output_schema": task['output_schema'],
            "search_context": search_context,
            "other_context": other_context
        })
        
        insufficient_indicators = [
            "insufficient", "need more", "require full", 
            "snippets are limited", "unclear", "cannot determine"
        ]
        
        is_insufficient = any(
            indicator in result.reasoning.lower() 
            for indicator in insufficient_indicators
        )
        
        if is_retry:
            return True, result
        
        is_sufficient = not is_insufficient
        return is_sufficient, result
    
    except Exception as e:
        print(f"Task execution error: {e}")
        return True, TaskExecutionOutput(
            reasoning=f"Error during execution: {str(e)}",
            structured_output={},
            key_insights=[]
        )


def get_other_task_outputs(state: ResearchState, current_task_id: str) -> List[Dict[str, Any]]:
    """
    Get outputs from other completed tasks (limited view per article)
    """
    other_outputs = []
    for result in state.get("task_results", []):
        if result['task_id'] != current_task_id:
            other_outputs.append({
                "task_id": result['task_id'],
                "structured_output": result['structured_output']
            })
    return other_outputs