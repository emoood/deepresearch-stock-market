import operator
from typing_extensions import TypedDict, Annotated, List, Sequence
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class ResearcherState(TypedDict):
    researcher_messages: Annotated[Sequence[BaseMessage], add_messages]
    tool_call_iterations: int
    research_topic: str
    compressed_research: str
    raw_notes: Annotated[List[str], operator.add]

class ResearcherOutputState(TypedDict):
    compressed_research: str
    raw_notes: Annotated[List[str], operator.add]
    researcher_messages: Annotated[Sequence[BaseMessage], add_messages]

#also in state_scope.py. course kept both
class ClarifyWithUser(BaseModel):
    need_clarification: bool = Field(description="Whether the user needs to be asked a clarifying question.")
    question: str = Field(description="A question to ask the user to clarify the report scope")
    verification: str = Field(description="Verify message that we will start research after the user has provided the necessary information.")

class ResearchQuestion(BaseModel):
    research_brief: str = Field(description="A research question that will be used to guide the research.")

#schema for summarizing a single webpage
class Summary(BaseModel):
    summary: str = Field(description="Concise summary of the webpage content")
    key_excerpts: str = Field(description="Important quotes and excerpts from the content")