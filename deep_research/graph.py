from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from deep_research.agents.executor import task_executor_node
from deep_research.agents.observer import observer_node
from deep_research.agents.planner import planner_node
from deep_research.agents.writer import writer_node
from deep_research.state import ResearchState





def create_research_graph():
    workflow = StateGraph(ResearchState)
    workflow.add_node("planner", planner_node)
    workflow.add_node("executor", task_executor_node)
    workflow.add_node("observer", observer_node)
    workflow.add_node("writer", writer_node)

    workflow.set_entry_point("planner")
    workflow.add_edge("planner", "executor")
    workflow.add_edge("executor", "observer")

    workflow.add_conditional_edges(
        "observer",
        should_continue_research,
        {
            "continue": "planner",
            "write": "writer" 
        }
    )

    workflow.add_edge("writer", END)

    # memory = MemorySaver()
    app = workflow.compile()

    return app

def should_continue_research(state: ResearchState) -> str:
    if state["research_complete"]:
        return "write"
    
    if state["iteration_count"] >= state["max_iterations"]:
        print(f"Max iterations ({state['max_iterations']}) reached. Proceeding to write.")
        return "write"
    
    if state["follow_up_tasks"]:
        print(f"Observer identified gaps. Running {len(state['follow_up_tasks'])} follow-up tasks...")
        return "continue"
    
    return "write"