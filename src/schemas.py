from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Literal


"""
We will define the schema for the banking support cases. 
We will define the resulting and resolution for each step of nodes for our graph. 
The schema will be used to validate the data and ensure that it is in the correct format.
"""

class PolicyDocument(BaseModel):
    """
    A schema for a banking policy document.
    """
    id: str = Field(..., description="Unique identifier for the policy document.")
    title: str = Field(..., description="Title of the policy document.")
    category: Literal[
        "card_dispute",
        "fee_refund",
        "failed_transfer",
        "account_lockout",
        "fraud",
        "loan_application",
        "identity_verification",
        "general"
    ] = Field(..., description="Category of the policy document.")
    content: str = Field(..., description="Content of the policy document.")
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional metadata related to the policy document."
    )
class SearchQuery(BaseModel):
    """
    A schema for a search query.
    """
    query: list[str] = Field(min_length=1, 
                             max_length=5, 
                             description="At most five distinct knowledge base search queries.")
    category: Optional[Literal[
        "card_dispute",
        "fee_refund",
        "failed_transfer",
        "account_lockout",
        "fraud",
        "loan_application",
        "identity_verification",
        "general"
    ]] = Field(
        default=None,
        description="Optional category to filter the search results."
    )
class SearchResult(BaseModel):
    """
    A schema for a search result.
    """
    document_id: str = Field(..., description="Unique identifier for the policy document.")
    title: str = Field(..., description="Title of the policy document.")
    category: str = Field(..., description="Category of the policy document.")
    content: list[str] = Field(..., description="Content of the policy document.")
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional metadata related to the policy document."
    )

class SupportResolution(BaseModel):
    """
    A schema for a support case resolution.
    """
    case_id: str = Field(..., description="Unique identifier for the support case.")
    customer_id: str = Field(..., description="Unique identifier for the customer.")
    issue_type: str = Field(..., description="Type of issue reported in the support case.")
    escalation_needed: bool = Field(..., description="Indicates if the support case needs to be escalated.")
    resolution_steps: List[str] = Field(..., description="Steps taken to resolve the support case.")
    need_human_intervention: bool = Field(..., description="Indicates if human intervention is needed.")
    missing_information: List[str] = Field(..., description="List of information items that are missing.")

class TriageDecision(BaseModel):
    """
    A schema for a triage decision.
    """

    issue_type: Literal[
        "card_dispute",
        "fee_refund",
        "failed_transfer",
        "account_lockout",
        "fraud",
        "loan_application",
        "identity_verification",
        "general"
    ] = Field(..., description="Type of issue reported in the support case.")

    next_steps: Literal[
        "card_agent",
        "loan_agent",
        "account_agent",
        "fraud_agent",
        "clarify",
        "escalate",
        "critics",
        "resolve",
        "humand_approval",

    ] = Field(..., description="Next steps to be taken for the support case.")

    routing_reason: str = Field(..., description="Reason for the routing decision.")

class CardAgentResolution(BaseModel):
    """
    A schema for a card agent resolution.
    """
    case_id: str = Field(..., description="Unique identifier for the support case.")
    customer_id: str = Field(..., description="Unique identifier for the customer.")
    issue_type: Literal["card_dispute"] = Field(..., description="Type of issue reported in the support case.")
    dispute_reason: str = Field(..., description="Reason for the card dispute.")
    tools_used: List[str] = Field(..., description="List of tools used to resolve the card dispute.")
    resolution_steps: List[str] = Field(..., description="Steps taken to resolve the card dispute.")
    resolution_status: Literal["resolved", "escalated", "need_approval"] = Field(
        ...,
        description="Status of the card dispute resolution."
    )
class LoanAgentResolution(BaseModel):
    """
    A schema for a loan agent resolution.
    """
    case_id: str = Field(..., description="Unique identifier for the support case.")
    customer_id: str = Field(..., description="Unique identifier for the customer.")
    issue_type: Literal["loan_application"] = Field(..., description="Type of issue reported in the support case.")
    application_status: Literal["approved", "denied", "pending"] = Field(
        ...,
        description="Status of the loan application."
    )
    tools_used: List[str] = Field(..., description="List of tools used to resolve the loan application issue.")
    resolution_steps: List[str] = Field(..., description="Steps taken to resolve the loan application issue.")
class FraudAgentResolution(BaseModel):
    """
    A schema for a fraud agent resolution.
    """
    case_id: str = Field(..., description="Unique identifier for the support case.")
    customer_id: str = Field(..., description="Unique identifier for the customer.")
    issue_type: Literal["fraud"] = Field(..., description="Type of issue reported in the support case.")
    fraud_type: str = Field(..., description="Type of fraud reported in the support case.")
    tools_used: List[str] = Field(..., description="List of tools used to resolve the fraud issue.")
    resolution_steps: List[str] = Field(..., description="Steps taken to resolve the fraud issue.")
    resolution_status: Literal["resolved", "escalated", "need_approval"] = Field(
        ...,
        description="Status of the fraud resolution."
    )
class AccountAgentResolution(BaseModel):
    """
    A schema for an account agent resolution.
    """
    case_id: str = Field(..., description="Unique identifier for the support case.")
    customer_id: str = Field(..., description="Unique identifier for the customer.")
    issue_type: Literal["account_lockout"] = Field(..., description="Type of issue reported in the support case.")
    lockout_reason: str = Field(..., description="Reason for the account lockout.")
    tools_used: List[str] = Field(..., description="List of tools used to resolve the account lockout issue.")
    resolution_steps: List[str] = Field(..., description="Steps taken to resolve the account lockout issue.")
    resolution_status: Literal["resolved", "escalated", "need_approval"] = Field(
        ...,
        description="Status of the account lockout resolution."
    )

class SupportCaseResolution(BaseModel):
    """
    A schema for a support case resolution.
    """
    case_id: str = Field(..., description="Unique identifier for the support case.")
    customer_id: str = Field(..., description="Unique identifier for the customer.")
    issue_type: str = Field(..., description="Type of issue reported in the support case.")
    tools_used: List[str] = Field(..., description="List of tools used to resolve the support case.")
    resolution_steps: List[str] = Field(..., description="Steps taken to resolve the support case.")
    resolution_status: Literal["resolved", "escalated", "need_approval"] = Field(
        ...,
        description="Status of the support case resolution."
    )

class TicketCreation(BaseModel):
    """
    A schema for a ticket resolution.
    """
    ticket_id: str = Field(..., description="Unique identifier for the ticket.")
    customer_id: str = Field(..., description="Unique identifier for the customer.")
    issue_type: str = Field(..., description="Type of issue reported in the ticket.")
    issue_description: str = Field(..., description="Description of the issue reported in the ticket.")
    resolution_status: Literal["resolved", "escalated", "need_approval"] = Field(
        ...,
        description="Status of the ticket resolution."
    )