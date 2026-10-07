# ---- state.py ----
from dataclasses import dataclass, asdict, field
from typing import List

@dataclass
class InquiryState:
    complaint_prompt: str
    complaint_severity: str
    knowledge_base_response: str = ""
    resolution_plan: str = ""
    customer_response: str = ""
    escalation_decision: str = ""