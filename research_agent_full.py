from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END
from langchain.chat_models import init_chat_model

from utils import get_today_str
from prompts import final_report_generation_prompt
from state_scope import AgentState, AgentInputState
from research_agent_scope import clarify_with_user, write_research_brief
from multi_agent_supervisor import supervisor_agent

#swap in groq, keep max_tokens high so the report doesn't get cut off
writer_model = init_chat_model(model="groq:llama-3.3-70b-versatile", max_tokens=32000)

async def final_report_generation(state: AgentState):
    notes = state.get("notes", [])
    findings = "\n".join(notes)

    final_report_prompt = final_report_generation_prompt.format(
        research_brief=state.get("research_brief", ""),
        findings=findings,
        date=get_today_str()
    )

    final_report = await writer_model.ainvoke([HumanMessage(content=final_report_prompt)])

    return {
        "final_report": final_report.content,
        "messages": ["Here is the final report: " + final_report.content],
    }

deep_researcher_builder = StateGraph(AgentState, input_schema=AgentInputState)

deep_researcher_builder.add_node("clarify_with_user", clarify_with_user)
deep_researcher_builder.add_node("write_research_brief", write_research_brief)
deep_researcher_builder.add_node("supervisor_subgraph", supervisor_agent)
deep_researcher_builder.add_node("final_report_generation", final_report_generation)

deep_researcher_builder.add_edge(START, "clarify_with_user")
deep_researcher_builder.add_edge("write_research_brief", "supervisor_subgraph")
deep_researcher_builder.add_edge("supervisor_subgraph", "final_report_generation")
deep_researcher_builder.add_edge("final_report_generation", END)

#full deployable graph, langgraph.json points here
agent = deep_researcher_builder.compile()