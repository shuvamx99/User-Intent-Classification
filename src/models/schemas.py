from pydantic import BaseModel
from typing import Dict, Any, List, Optional, TYPE_CHECKING
from datetime import datetime
from ..models.enums import ConfidenceLevel


class Agent(BaseModel):
    id: str
    name: str
    description: str
    capabilities: List[str]
    metadata: Dict[str, Any] = {}


class ClassificationResult(BaseModel):
    agent: Agent
    confidence_score: float
    confidence_level: ConfidenceLevel
    reasoning: str


class ClassifierConfig(BaseModel):
    openai_api_key: str
    model: str = "gpt-4"
    confidence_threshold: float = 0.7
    high_threshold: float = 0.8
    medium_threshold: float = 0.5
    low_threshold: float = 0.2


if TYPE_CHECKING:
    from typing import Self


class ExecutionResult(BaseModel):
    agent_id: str
    success: bool
    message: str = ""
    errors: List[str] = []
    timestamp: datetime = None

    def __init__(self, **data):
        if "timestamp" not in data or data["timestamp"] is None:
            data["timestamp"] = datetime.now()
        super().__init__(**data)


class UserInteraction(BaseModel):
    interaction_id: str
    user_prompt: str
    prompt_embedding: List[float]
    timestamp: datetime
    selected_agents: List[str]
    llm_confidence_scores: Dict[str, float] = {}
    final_confidence_scores: Dict[str, float] = {}
    user_feedback: Optional["UserFeedback"] = None
    execution_results: List[ExecutionResult] = []  # Store agent execution results
    session_id: Optional[str] = None


class UserFeedback(BaseModel):
    interaction_id: str
    agent_ratings: Dict[str, float]  # agent_id -> rating (1-5)
    overall_satisfaction: float
    feedback_text: Optional[str] = None
    implicit_signals: "ImplicitFeedback"
    timestamp: datetime


class ImplicitFeedback(BaseModel):
    agents_actually_used: List[str]
    time_spent_reviewing_results: float
    follow_up_questions: List[str]
    result_actions_taken: List[str] = []  # exported, shared, etc.


class VectorSearchResult(BaseModel):
    interaction_id: str
    similarity_score: float
    user_interaction: UserInteraction
    feedback_score: float = 0.0


class RerankingConfig(BaseModel):
    llm_weight: float = 0.5
    historical_success_weight: float = 0.5
    similarity_threshold: float = 0.7


# Update forward references - required for string type annotations
UserInteraction.model_rebuild()
UserFeedback.model_rebuild()
ImplicitFeedback.model_rebuild()
ExecutionResult.model_rebuild()
