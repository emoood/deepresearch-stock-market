
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END
from langchain.chat_models import init_chat_model

from utils import get_today_str
from prompts import final_report_generation_prompt
from state_scope import AgentState, AgentInputState, MarketResults
from research_agent_scope import clarify_with_user, write_research_brief
from research_agent import researcher_agent
import os
from langchain_ollama import ChatOllama
import json
from langchain_cerebras import ChatCerebras

######## llama instant
#writer_model = init_chat_model(model="groq:llama-3.1-8b-instant", max_tokens=32000)

####### ollama gemma
writer_model = ChatOllama(
    model="gemma4:12b-mlx",
    base_url=os.environ["OLLAMA_BASE_URL"],
    temperature=0.0,
)

######## llama versatile *
#writer_model = init_chat_model(model="groq:llama-3.3-70b-versatile", max_tokens=32000)

######## GLM
# writer_model = ChatCerebras(
#     model="zai-glm-4.7",
#     api_key=os.environ["CEREBRAS_API_KEY"],
# )

######## mistral
#writer_model = init_chat_model(model="mistralai:mistral-large-latest", max_tokens=8192)

######## llama scout
# writer_model = init_chat_model(model="groq:meta-llama/llama-4-scout-17b-16e-instruct", max_tokens=8192)

async def run_researcher(state: AgentState):
    result = await researcher_agent.ainvoke({
        "researcher_messages": [HumanMessage(content=state.get("research_brief", ""))]
    })
    return {"notes": [result.get("compressed_research", "")]}

async def final_report_generation(state: AgentState):
    notes = state.get("notes", [])
    findings = "\n".join(notes)

    final_report_prompt = final_report_generation_prompt.format(
        research_brief=state.get("research_brief", ""),
        findings=findings,
        date=get_today_str()
    )

    #step 1: let the model write the report no structured output 
    draft_response = await writer_model.ainvoke([HumanMessage(content=final_report_prompt)])
    draft_text = draft_response.content

    #step 2: simple extraction call to pull structured fields from the draft
    structured_writer_model = writer_model.with_structured_output(MarketResults, method="function_calling", include_raw=True)

    extraction_prompt = (
        f"Extract the following market analysis into the required structured fields "
        f"(verdict, confidence, summary, tickers, sources). "
        f"Do not add new information, only extract what's present below:\n\n{draft_text}"
    )

    response = await structured_writer_model.ainvoke([HumanMessage(content=extraction_prompt)])

    if response["parsed"] is not None:
        market_results = response["parsed"]
    else:
        raw_text = response["raw"].content.strip()
        raw_text_cleaned = raw_text.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        try:
            market_results = MarketResults(**json.loads(raw_text_cleaned))
        except json.JSONDecodeError:
            market_results = None

    if market_results is None:
        return {
            "final_report": None,
            "messages": ["Could not generate a structured market verdict from this run."],
        }

    summary_text = "\n".join(f"- {bullet}" for bullet in market_results.summary)

    message = (
        f"Verdict: {market_results.verdict} (confidence: {market_results.confidence})\n\n"
        f"Summary:\n{summary_text}\n\n"
        f"Tickers/sectors: {', '.join(market_results.tickers)}\n"
        f"Sources: {', '.join(market_results.sources)}"
    )

    return {
        "final_report": market_results,
        "messages": [message],
    }

deep_researcher_builder = StateGraph(AgentState, input_schema=AgentInputState)

deep_researcher_builder.add_node("clarify_with_user", clarify_with_user)
deep_researcher_builder.add_node("write_research_brief", write_research_brief)
deep_researcher_builder.add_node("run_researcher", run_researcher)
deep_researcher_builder.add_node("final_report_generation", final_report_generation)

deep_researcher_builder.add_edge(START, "clarify_with_user")
deep_researcher_builder.add_edge("write_research_brief", "run_researcher")
deep_researcher_builder.add_edge("run_researcher", "final_report_generation")
deep_researcher_builder.add_edge("final_report_generation", END)

#no-supervisor version, single research agent straight into final node
agent = deep_researcher_builder.compile()