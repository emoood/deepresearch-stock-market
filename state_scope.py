import operator
from typing_extensions import Optional, Annotated, Sequence
from langchain_core.messages import BaseMessage
from langgraph.graph import MessagesState
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

#input state holds the messages the user sends
class AgentInputState(MessagesState):
    pass

class AgentState(MessagesState):
    research_brief: Optional[str]
    supervisor_messages: Annotated[Sequence[BaseMessage], add_messages]
    raw_notes: Annotated[list[str], operator.add] = []
    notes: Annotated[list[str], operator.add] = []
    final_report: str

#schema for when we need to aska clarifying question
class ClarifyWithUser(BaseModel):
    need_clarification: bool = Field(description="Whether the user needs to be asked a clarifying question.")
    question: str = Field(description="A question to ask the user to clarify the report scope")
    verification: str = Field(description="Verify message that we will start research after the user has provided the necessary information.")

#schema for the research brief we generate once we have enough info
class ResearchQuestion(BaseModel):
    research_brief: str = Field(description="A research question that will be used to guide the research.")
