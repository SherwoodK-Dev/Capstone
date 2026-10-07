from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from typing import Any

try:
    from complaint_model import query_knowledge_base, predict_priority
    from state import InquiryState
except ModuleNotFoundError:
    from .complaint_model import query_knowledge_base, predict_priority
    from .state import InquiryState


def analyzer_agent(state: InquiryState | Mapping[str, Any]) -> dict:
    """Assesses the complaint severity."""
    result = asdict(state) if is_dataclass(state) else dict(state)
    complaint = result["complaint_prompt"].strip()
    result["complaint_severity"] = predict_priority(complaint)

    return result

def investigate_agent(state: InquiryState | Mapping[str, Any]) -> dict:
    """Investigates the complaint and retrieves relevant information."""
    result = asdict(state) if is_dataclass(state) else dict(state)
    complaint = result["complaint_prompt"].strip()
    result["knowledge_base_response"] = query_knowledge_base(complaint)
    return result

def planner_agent(state: InquiryState | Mapping[str, Any]) -> dict:
    """Plans the resolution for the complaint."""
    result = asdict(state) if is_dataclass(state) else dict(state)
    result["resolution_plan"] = "Plan for resolving the complaint."
    return result

def communicator_agent(state: InquiryState | Mapping[str, Any]) -> dict:
    """Communicates with the customer about the complaint."""
    result = asdict(state) if is_dataclass(state) else dict(state)
    result["customer_response"] = "Response to the customer."
    return result

def excalation_agent(state: InquiryState | Mapping[str, Any]) -> dict:
    """Determines if the complaint needs escalation."""
    result = asdict(state) if is_dataclass(state) else dict(state)
    if result["complaint_severity"] == "High":
        result["escalation_decision"] = "Escalate to supervisor."
    else:
        result["escalation_decision"] = "Handle internally."
    return result
    