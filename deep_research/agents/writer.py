from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from typing import Dict, Any

from deep_research.config import NEBIUS_API_KEY
from deep_research.output_schemas import FinalReport
from deep_research.state import ResearchState


async def writer_node(state: ResearchState) -> Dict[str, Any]:
    print("WRITER: Generating final research report...")
    
    llm = ChatOpenAI(
        model="meta-llama/Llama-3.3-70B-Instruct",
        temperature=0.3,
        api_key=NEBIUS_API_KEY,
        base_url="https://api.studio.nebius.ai/v1/"
    )
    
    structured_llm = llm.with_structured_output(FinalReport)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert research writer. Create a comprehensive, well-structured final report.

Your report should:
- Directly answer the original query in the executive summary. This is the actual report itself. Make sure the response is highly in-depth thorough and complete. It is more of a report than an actual summary.
- Present detailed findings in a logical, organized manner.
- Synthesize information from all research tasks
- Highlight key insights and takeaways
- Include methodology transparency
- Provide all citations

Write clearly and professionally. This is the final deliverable."""),
        
        ("human", """Original Query: {query}

Complete Research Context:
{full_context}

Total Iterations: {iterations}

Generate a comprehensive final report that thoroughly answers the query.""")
    ])
    
    chain = prompt | structured_llm
    
    try:
        report: FinalReport = await chain.ainvoke({
            "query": state["query"],
            "full_context": state["full_context"],
            "iterations": state.get("iteration_count", 0) + 1
        })
        
        print("Final report generated!")
        print(f"  Executive Summary: {report.executive_summary[:150]}...")
        print(f"  Key Insights: {len(report.key_insights)} insights")
        print(f"  Citations: {len(report.all_citations)} sources")
        print(f"  Confidence: {report.confidence_score:.2f}")
        
        return {
            "final_output": report.model_dump(),
            "research_complete": True
        }
    
    except Exception as e:
        print(f" Writer error: {e}")
        raise