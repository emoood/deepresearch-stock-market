from typing_extensions import Literal
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage, filter_messages
from langchain.chat_models import init_chat_model
from state_research import ResearcherState, ResearcherOutputState
from utils import tavily_search, get_today_str, think_tool
from prompts import research_agent_prompt, compress_research_system_prompt, compress_research_human_message
import os
from langchain_ollama import ChatOllama
from langchain_cerebras import ChatCerebras

tools = [tavily_search, think_tool]
tools_by_name = {tool.name: tool for tool in tools}

# model = init_chat_model(model="groq:llama-3.3-70b-versatile", temperature=0)
# summarization_model = init_chat_model(model="groq:llama-3.3-70b-versatile")
# compress_model = init_chat_model(model="groq:llama-3.3-70b-versatile", max_tokens=16000)

model = ChatOllama(
    model="gemma4:12b-mlx",
    base_url=os.environ["OLLAMA_BASE_URL"],
    temperature=0,
)
summarization_model = ChatOllama(
    model="gemma4:12b-mlx",
    base_url=os.environ["OLLAMA_BASE_URL"],
)
compress_model = ChatOllama(
    model="gemma4:12b-mlx",
    base_url=os.environ["OLLAMA_BASE_URL"],
    max_tokens=32000,
)

# model = init_chat_model(model="google_genai:gemini-3.5-flash", temperature=0)
# summarization_model = init_chat_model(model="google_genai:gemini-3.5-flash")
# compress_model = init_chat_model(model="google_genai:gemini-3.5-flash", max_tokens=8192)

# model = init_chat_model(model="groq:meta-llama/llama-4-scout-17b-16e-instruct", temperature=0)
# summarization_model = init_chat_model(model="groq:meta-llama/llama-4-scout-17b-16e-instruct")
# compress_model = init_chat_model(model="groq:meta-llama/llama-4-scout-17b-16e-instruct", max_tokens=8192)

# model = ChatCerebras(
#     model="zai-glm-4.7",
#     api_key=os.environ["CEREBRAS_API_KEY"],
#     temperature=0,
# )
# summarization_model = ChatCerebras(
#     model="zai-glm-4.7",
#     api_key=os.environ["CEREBRAS_API_KEY"],
# )
# compress_model = ChatCerebras(
#     model="zai-glm-4.7",
#     api_key=os.environ["CEREBRAS_API_KEY"],
#     max_tokens=16000,
# )

# model = init_chat_model(model="mistralai:mistral-large-latest", temperature=0)
# summarization_model = init_chat_model(model="mistralai:mistral-large-latest")
# compress_model = init_chat_model(model="mistralai:mistral-large-latest", max_tokens=16000)

model_with_tools = model.bind_tools(tools)



def llm_call(state: ResearcherState):
    return {
        "researcher_messages": [
            model_with_tools.invoke(
                [SystemMessage(content=f"{research_agent_prompt}\n\nToday's actual date is {get_today_str()}. Trust this date over any internal assumption about the current date.")] + state["researcher_messages"]
            )
        ]
    }

def tool_node(state: ResearcherState):
    tool_calls = state["researcher_messages"][-1].tool_calls

    observations = []
    for tool_call in tool_calls:
        tool = tools_by_name[tool_call["name"]]
        observations.append(tool.invoke(tool_call["args"]))

    tool_outputs = [
        ToolMessage(
            content=observation,
            name=tool_call["name"],
            tool_call_id=tool_call["id"]
        ) for observation, tool_call in zip(observations, tool_calls)
    ]

    return {"researcher_messages": tool_outputs}

def compress_research(state: ResearcherState) -> dict:
    system_message = compress_research_system_prompt.format(date=get_today_str())
    messages = [SystemMessage(content=system_message)] + state.get("researcher_messages", []) + [HumanMessage(content=compress_research_human_message)]
    response = compress_model.invoke(messages)

    #pull raw notes out of the tool and ai messages for later
    raw_notes = [
        str(m.content) for m in filter_messages(
            state["researcher_messages"],
            include_types=["tool", "ai"]
        )
    ]

    return {
        "compressed_research": str(response.content),
        "raw_notes": ["\n".join(raw_notes)]
    }

def should_continue(state: ResearcherState) -> Literal["tool_node", "compress_research"]:
    messages = state["researcher_messages"]
    last_message = messages[-1]

    if last_message.tool_calls:
        return "tool_node"
    return "compress_research"

agent_builder = StateGraph(ResearcherState, output_schema=ResearcherOutputState)

agent_builder.add_node("llm_call", llm_call)
agent_builder.add_node("tool_node", tool_node)
agent_builder.add_node("compress_research", compress_research)

agent_builder.add_edge(START, "llm_call")
agent_builder.add_conditional_edges(
    "llm_call",
    should_continue,
    {
        "tool_node": "tool_node",
        "compress_research": "compress_research",
    },
)
agent_builder.add_edge("tool_node", "llm_call")
agent_builder.add_edge("compress_research", END)

researcher_agent = agent_builder.compile()