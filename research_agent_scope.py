from datetime import datetime
from typing_extensions import Literal
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, AIMessage, get_buffer_string
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command
from prompts import clarify_with_user_instructions, transform_messages_into_research_topic_prompt
from state_scope import AgentState, ClarifyWithUser, ResearchQuestion, AgentInputState
import os
from langchain_ollama import ChatOllama
import json
from langchain_cerebras import ChatCerebras

###### llama instant
#model = init_chat_model(model="groq:llama-3.1-8b-instant", temperature=0.0)

###### ollama gemma
model = ChatOllama(
    model="gemma4:12b-mlx",
    base_url=os.environ["OLLAMA_BASE_URL"],
    temperature=0.0,
)

###### GLM
# model = ChatCerebras(
#     model="zai-glm-4.7",
#     api_key=os.environ["CEREBRAS_API_KEY"],
# )

def get_today_str() -> str:
    return datetime.now().strftime("%a %b %-d, %Y")

def clarify_with_user(state: AgentState) -> Command[Literal["write_research_brief", "__end__"]]:
    structured_output_model = model.with_structured_output(ClarifyWithUser, method="function_calling", include_raw=True)

    response = structured_output_model.invoke([
        HumanMessage(content=clarify_with_user_instructions.format(
            messages=get_buffer_string(messages=state["messages"]),
            date=get_today_str()
        ))
    ])

    if response["parsed"] is not None:
        result = response["parsed"]
    else:
        print(response["raw"].content)
        raw_text = response["raw"].content.strip()
        raw_text = raw_text.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        result = ClarifyWithUser(**json.loads(raw_text))

    #route to end with a question, or move to brief
    if result.need_clarification:
        return Command(
            goto=END,
            update={"messages": [AIMessage(content=result.question)]}
        )
    else:
        return Command(
            goto="write_research_brief",
            update={"messages": [AIMessage(content=result.verification)]}
        )
    


def write_research_brief(state: AgentState):
    structured_output_model = model.with_structured_output(ResearchQuestion, method="function_calling", include_raw=True)

    response = structured_output_model.invoke([
        HumanMessage(content=transform_messages_into_research_topic_prompt.format(
            messages=get_buffer_string(state.get("messages", [])),
            date=get_today_str()
        ))
    ])

    if response["parsed"] is not None:
        result = response["parsed"]
    else:
        raw_text = response["raw"].content.strip()
        raw_text_cleaned = raw_text.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        try:
            result = ResearchQuestion(**json.loads(raw_text_cleaned))
        except json.JSONDecodeError:
            result = ResearchQuestion(research_brief=raw_text)

    return {
        "research_brief": result.research_brief,
        "supervisor_messages": [HumanMessage(content=f"{result.research_brief}.")]
    }

deep_researcher_builder = StateGraph(AgentState, input_schema=AgentInputState)

deep_researcher_builder.add_node("clarify_with_user", clarify_with_user)
deep_researcher_builder.add_node("write_research_brief", write_research_brief)

deep_researcher_builder.add_edge(START, "clarify_with_user")
deep_researcher_builder.add_edge("write_research_brief", END)

#tscoping-only graph, used for testing in isolation
scope_research = deep_researcher_builder.compile()