from pydantic import BaseModel
from typing import Dict, Any, List, Optional
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
    include_conversation_history: bool = False
    include_memory: bool = False