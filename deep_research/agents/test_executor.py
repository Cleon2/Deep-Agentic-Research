# test_executor.py
import asyncio
from deep_research.state import ResearchState
from deep_research.agents.planner import planner_node
from deep_research.agents.executor import task_executor_node


async def main():
    # Start with a query
    initial_state: ResearchState = {
        "query": "What are the key differences between Rust and Go for systems programming?",
        "research_plan": "",
        "tasks": [],
        "task_results": [],
        "full_context": "",
        "final_output": {},
        "total_tokens_used": 0,
        "execution_time": 0.0
    }

    # Step 1: Run Planner
    print("STEP 1: PLANNING")
    print("="*80)
    planner_result = await planner_node(initial_state)
    state_after_planning = {**initial_state, **planner_result}

    # Step 2: Run Executor
    print("\n\nSTEP 2: EXECUTION")
    print("="*80)
    executor_result = await task_executor_node(state_after_planning)
    final_state = {**state_after_planning, **executor_result}

    # Print results
    print("\n\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80)

    print(f"\nTotal execution time: {final_state['execution_time']:.2f}s")
    print(f"Tasks completed: {len(final_state['task_results'])}")

    for result in final_state['task_results']:
        print(f"\n--- Task {result['task_id']} ---")
        print(f"Reasoning: {result['reasoning'][:200]}...")
        print(f"Structured Output: {result['structured_output']}")
        print(f"Citations: {len(result['citations'])} sources")
        print(f"Key Insights: {result.get('key_insights', 'N/A')}")


if __name__ == "__main__":
    asyncio.run(main())