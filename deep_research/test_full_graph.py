import asyncio
from deep_research.state import ResearchState
from deep_research.graph import create_research_graph


async def main():
    # Create the graph
    app = create_research_graph()
    
    # Initial state
    initial_state: ResearchState = {
        "query": "Why did Silicon Valley Bank collapse in March 2023? What were the root causes, who was responsible, and what regulations have been proposed since then?",
        "research_plan": "",
        "tasks": [],
        "task_results": [],
        "full_context": "",
        "final_output": {},
        "total_tokens_used": 0,
        "execution_time": 0.0,
        # Iteration controls
        "iteration_count": 0,
        "max_iterations": 2,  # Allow up to 2 iterations
        "research_complete": False,
        "identified_gaps": [],
        "follow_up_tasks": []
    }
    
    print("\n" + "="*80)
    print("🚀 STARTING ITERATIVE DEEP RESEARCH")
    print("="*80)
    print(f"Query: {initial_state['query']}")
    print(f"Max Iterations: {initial_state['max_iterations']}")
    print("="*80)
    
    # Run the graph
    final_state = await app.ainvoke(initial_state)
    
    # Print final results
    print("\n\n" + "="*80)
    print("🎯 FINAL RESULTS")
    print("="*80)
    
    print(f"\nTotal Iterations: {final_state.get('iteration_count', 0) + 1}")
    print(f"Total Execution Time: {final_state.get('execution_time', 0):.2f}s")
    print(f"Total Tasks Executed: {len(final_state.get('task_results', []))}")
    
    final_output = final_state.get("final_output", {})
    if final_output:
        print(f"\n📄 FINAL REPORT:")
        print(f"\nExecutive Summary:")
        print(f"{final_output.get('executive_summary', 'N/A')}")
        
        print(f"\n🔑 Key Insights ({len(final_output.get('key_insights', []))}):")
        for i, insight in enumerate(final_output.get('key_insights', [])[:5], 1):
            print(f"{i}. {insight}")
        
        print(f"\n📊 Methodology:")
        print(f"{final_output.get('methodology', 'N/A')}")
        
        print(f"\n📚 Citations: {len(final_output.get('all_citations', []))} sources")
        print(f"🎯 Confidence Score: {final_output.get('confidence_score', 0):.2f}")


if __name__ == "__main__":
    asyncio.run(main())