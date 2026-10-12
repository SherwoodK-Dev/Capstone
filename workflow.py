
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict
try:
    from agents import (
        analyzer_agent,
        investigate_agent,
        planner_agent,
        communicator_agent,
        escalation_agent,
    )
    from state import InquiryState
except ModuleNotFoundError:
    from agents import (
        analyzer_agent,
        investigate_agent,
        planner_agent,
        communicator_agent,
        escalation_agent,
    )
    from state import InquiryState

class State(TypedDict):
    complaint_prompt: str
    complaint_severity_msg: str
    knowledge_base_response: str = ""
    resolution_plan: str = ""
    customer_response: str = ""
    escalation_decision: str = ""

def build_workflow() -> StateGraph:
    workflow = StateGraph(State)
    workflow.add_node("analyzer_agent", analyzer_agent)
    workflow.add_node("investigate_agent", investigate_agent)
    workflow.add_node("planner_agent", planner_agent)
    workflow.add_node("communicator_agent", communicator_agent)
    workflow.add_node("escalation_agent", escalation_agent)
    # Add edges to connect the nodes (steps in order)
    workflow.add_edge(START, "analyzer_agent")
    workflow.add_edge("analyzer_agent", "investigate_agent")
    workflow.add_edge("investigate_agent", "planner_agent")
    workflow.add_edge("planner_agent", "communicator_agent")
    workflow.add_edge("communicator_agent", "escalation_agent")
    workflow.add_edge("escalation_agent", END)
    chain = workflow.compile()
    return chain