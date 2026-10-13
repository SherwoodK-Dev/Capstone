from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
import os
from typing import Any
from dotenv import load_dotenv
from pathlib import Path

from openai import OpenAI, AzureOpenAI

try:
    from knowledge_base_functions import get_knowledge_base
    from state import InquiryState
except ModuleNotFoundError:
    from .knowledge_base_functions import get_knowledge_base
    from .state import InquiryState


load_dotenv(Path(__file__).resolve().parent / ".env")

api_key = os.getenv("AZURE_OPENAI_API_KEY")
endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2025-04-14-preview")
deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4.1-mini")

if not api_key or not endpoint or not deployment:
    raise RuntimeError("Missing Azure OpenAI credentials in .env")

if endpoint.rstrip("/").endswith("/openai/v1"):
    client = OpenAI(
    api_key=api_key,
    base_url=endpoint.rstrip("/") + "/",
    )
else:
    client = AzureOpenAI(
    api_key=api_key,
    azure_endpoint=endpoint.rstrip("/"),
    api_version=api_version,
    )
DATA_PATH = Path(__file__).resolve().parent / "complaints_train.csv"
KNOWLEDGE_BASE_PATH = Path(__file__).resolve().parent / "knowledge_base.json"

def analyzer_agent(state: InquiryState | Mapping[str, Any]) -> dict:
    """Assesses the complaint severity."""
    response = client.chat.completions.create(model=deployment,
        messages=[
            {"role": "system", "content": "You want to take input text that is a complaint message and determine it's severity and explain why. The options are 'Low', 'Medium', 'Unsure', 'High' or 'Critical'."},
            {"role": "user", "content": state["complaint_prompt"]},
        ])
    severity_msg = response.choices[0].message.content.strip()
    result = asdict(state) if is_dataclass(state) else dict(state)
    result["complaint_severity_msg"] = severity_msg
    return result

def investigate_agent(state: InquiryState | Mapping[str, Any]) -> dict:
    """Investigates the complaint and retrieves relevant information."""
    result = asdict(state) if is_dataclass(state) else dict(state)
    complaint = result["complaint_prompt"].strip()
    result["knowledge_base_response"] = get_knowledge_base(complaint)
    return result

def planner_agent(state: InquiryState | Mapping[str, Any]) -> dict:
    """Plans the resolution for the complaint."""
    response = client.chat.completions.create(model=deployment,
        messages=[
            {"role": "system", "content": "You want to use the information from the knowledge base to outline steps to resolve the complaint. "
            "Base the answer on the knowledge base and the complaint text."},
            {"role": "user", "content": (
                "Knowledge base:\n"
                + state["knowledge_base_response"]
                + "\n\nComplaint:\n"
                + state["complaint_prompt"]
            )}
        ])
    resolution_plan = response.choices[0].message.content.strip()
    result = asdict(state) if is_dataclass(state) else dict(state)
    result["resolution_plan"] = resolution_plan
    return result

def communicator_agent(state: InquiryState | Mapping[str, Any]) -> dict:
    """Communicates with the customer about the complaint."""
    response = client.chat.completions.create(model=deployment,
        messages=[
            {"role": "system", "content": (
                "You want to create a customer-ready response based on the resolution steps provided. "
                "Start the response with 'Dear Customer,' and end with 'Sincerely, AgriChain'."
            )},
            {"role": "user", "content": (
                "Resolution steps:\n" + state["resolution_plan"]
            )}
        ])
    customer_response = response.choices[0].message.content.strip()
    result = asdict(state) if is_dataclass(state) else dict(state)
    result["customer_response"] = customer_response
    return result

def escalation_agent(state: InquiryState | Mapping[str, Any]) -> dict:
    """Determines if the complaint needs escalation."""
    response = client.chat.completions.create(model=deployment,
        messages=[
            {"role": "system", "content": (
                "Determine whether the complaint requires escalation. "
                "If the severity is 'High', 'Critical', or 'Unsure', respond with "
                "'Needs Escalation'. Otherwise respond with 'No Escalation Necessary'. "
                "Also include a customer-ready explanation of why."
            )},
            {"role": "user", "content": (
                "Complaint severity: "
                + state["complaint_severity_msg"]
                + "\n\nComplaint: "
                + state["complaint_prompt"]
            )}
        ])
    escalation_decision = response.choices[0].message.content.strip()
    result = asdict(state) if is_dataclass(state) else dict(state)
    result["escalation_decision"] = escalation_decision
    return result
    