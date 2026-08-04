import datetime
from enum import Enum

from pydantic import BaseModel, Field, field_validator
from typing import List,  Literal


"""
We will define the schema for the banking support cases. 
We will define the resulting and resolution for each step of nodes for our graph. 
The schema will be used to validate the data and ensure that it is in the correct format.
"""


class Domain(str, Enum):
    #The triage will select from this domains 
    CARD = "card"
    LOAN = "loan"
    ACCOUNT = "account"
    FRAUD = "fraud"

    # if there is no spcialist domain, we will use this out of scope domain to avoid the model answering 
    # from a non it's knowledge base. 
    # The model will be forced to answer from the knowledge base and not from it's own knowledge.
    OUT_OF_SCOPE = "out_of_scope"


class Verdict(str, Enum):
    # The verdict will be used to determine the next step in the graph. 
    # The model will be forced to answer from the knowledge base and not from it's own knowledge.
    REVISE = "revise"
    ESCALATED = "escalate"
    APPROVE = "approve"


class Outcome(str, Enum):
    # The outcome will be used to determine the next step in the graph. 
    # The model will be forced to answer from the knowledge base and not from it's own knowledge.
    RESOLVED = "resolved"
    PENDING_APPROVAL = "pending_approval"
    ESCALATED = "escalated"
    AWATING_INFO= "awaiting_info"
    CANNOT_ASSIST = "cannot_assist"

ActionType = Literal[
    "refund",
    "reissue_card",
    "reset_account",
    "open_dispute",
    "loan_decision",
    "information_only",
]



class TriageResult(BaseModel):
    domain: Domain = Field(..., description="The domain of the case")
    bank_related: bool = Field(
        default=True,
        description="False when the request has nothing to do with the banking" \
        "domain at all. In this case, the model will be forced to answer from the knowledge base and not from it's own knowledge, "
        "if not it will say this assistant can't help with that.",)
    intent: str = Field(..., description="The intent of the case what the customer is trying to achieve")
    customer_id: str = Field(..., description="The customer id of the case")
    account_id:str|None = Field(None, description="The account id of the case, if applicable")
    card_last_four_digits: str|None = Field(None, description="The last four digits of the card, if applicable")
    amount: float|None = Field(None, description="The amount of the case, if applicable")
    missing_info: List[str] = Field(default_factory=list, 
                                    description="The missing information that the model needs to ask the customer for." \
                                    "Facts only the customer can provide.")
    reasoning: str = Field(..., description="The reasoning of the model for the triage result")


    @field_validator("customer_id", "account_id", mode="before")
    @classmethod
    def _normalize(cls, value:object)->object:
        if isinstance(value, str):
            return value.strip().upper() or None
        return value

# Now let's work on the domain agent base schema. T
# his will be used to validate the data and ensure that it is in the correct format.

class ProposedAction(BaseModel):
    action: ActionType
    description: str = Field(description="The description of the action that the model is proposing to take")
    amount: float|None = Field(default=None, description="The amount of the action that the model is proposing to take, if applicable")
    citation: str|None = Field(default="", 
                               description="The document id from retrieval that the model is citing " \
                               "to support it's action. If the model is not citing any document, it will be an empty string.")

# Here our model will propose an action to take, and it will provide a reasoning for the action.
# it will be passed to the critic agent to validate the action and provide a verdict.
class Draft(BaseModel):
    reply: str = Field(description="The reply of the model to the customer")
    action: ProposedAction
    citations: List[str] = Field(default_factory=list, description="The document id from retrieval that the model is citing "
                               "to support it's reply. If the model is not citing any document, it will be an empty string.")
    notes: str = Field(default="", description="Internal notes for the agent to consider when reviewing the draft. " \
    "This is not visible to the customer.")
    confidence: float = Field(default=0.5, ge=0.0, le=1.0,
                              description="The confidence of the model in the draft. This is a number between 0 and 1.")

# The critic agent will provide a verdict on the draft, and it will provide a rationale for the verdict.
class Critique(BaseModel):
    verdict: Verdict = Field(..., description="The verdict of the critic agent")
    grounded: bool = Field(description="True if the claims trace to retrive sources, " \
    "False if the claims are not grounded in the retrieval")
    problems: List[str] = Field(default_factory=list, description="The problems that the critic agent found in the draft")
    fixes: List[str] = Field(default_factory=list, description="The fixes that the critic agent suggests for the draft")
    escalation_reason: str|None = Field(default=None, description="The reason for escalation if the verdict is escalate")
    rationale: str = Field(default="", description="The rationale of the critic agent for the verdict")

class Source(BaseModel):
    document_id: str = Field(..., description="The document id from retrieval that the model is citing to support it's reply")
    title: str = Field(..., description="The title of the document that the model is citing to support it's reply")
    domain: str = Field(..., description="The domain of the document that the model is citing to support it's reply")
    doc_type: str = Field(..., description="The type of the document that the model is citing to support it's reply")
    content: str = Field(..., description="The content of the document that the model is citing to support it's reply")
    score: float = Field(..., description="The score of the document that the model is citing to support it's reply")


# Here we will define the schema for the tool call. 
# The tool call will be used to call the tool and get the result.
class ToolCall(BaseModel):
    tool: str = Field(..., description="The name of the tool that the model is calling")
    args:dict = Field(..., description="The arguments to the tool that the model is calling")
    ok: bool = Field(..., description="True if the tool call was successful, False otherwise")
    result: str|None = Field(default=None, description="The result of the tool call. This will be filled" \
    " after the tool is called.")
    error: str|None = Field(default=None, description="The error message if the tool call was not successful. " \
    "This will be filled after the tool is called.")
    attempts:int = 1 # It only takes one attempt to call the tool, if it fails, we will not retry.


class TraceEvent(BaseModel):
    node: str = Field(..., description="The step in the graph that the model is currently in")
    role: str = Field(..., description="The action that the model is taking in the current step")
    detail: str = Field(..., description="The reasoning of the model for the action it is taking in the current step")
    at:datetime = Field(default_factory=datetime.datetime.now, description="The timestamp of the event")



class PendingApproval(BaseModel):
    approval_id: str = Field(..., description="The id of the approval request")
    action_type:str
    description: str
    amount:str
    reason:str|None = Field(default=None, description="why a human must decide this.")
    citation:str
    status: Literal["pending", "approved", "rejected"] = "pending"
