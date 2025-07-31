# src/classifiers/base_classifier.py
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union, Tuple
import json

from ..models.enums import ConfidenceLevel
from ..models.schemas import Agent, ClassificationResult, ClassifierConfig


class BaseClassifier(ABC):
    """
    Abstract base classifier that evaluates a user prompt against a pool of agents
    and selects the best match based on metadata, capabilities, and context.
    """

    def __init__(self, config: ClassifierConfig):
        """
        Initialize the classifier with configuration.

        Args:
            config: Configuration for the classifier
        """
        self.config = config

    def calc_confidence_level(self, score: float) -> ConfidenceLevel:
        """
        Map a confidence score to a ConfidenceLevel enum.

        Args:
            score: Confidence score between 0 and 1

        Returns:
            ConfidenceLevel enum value
        """
        if score >= self.config.high_threshold:
            return ConfidenceLevel.HIGH
        elif score >= self.config.medium_threshold:
            return ConfidenceLevel.MEDIUM
        elif score >= self.config.low_threshold:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.UNSUITABLE

    def create_system_prompt(self, agents: List[Agent]) -> str:
        """
        Create a system prompt for the LLM that includes agent metadata.

        Args:
            agents: List of available agents

        Returns:
            System prompt as a string
        """
        prompt = "You are a classifier that matches user requests to the most appropriate agent.\n\n"
        prompt += "Available agents:\n"

        for idx, agent in enumerate(agents, 1):
            prompt += f"{idx}. {agent.name}: {agent.description}\n"
            prompt += f"   Capabilities: {', '.join(agent.capabilities)}\n"
            if agent.metadata:
                prompt += f"   Additional info: {json.dumps(agent.metadata, indent=2)}\n"
            prompt += "\n"

        prompt += "Your task is to evaluate which agent best matches the user's request.\n"
        prompt += "For each agent, provide:\n"
        prompt += "1. A confidence score (0.0 to 1.0) indicating how well the agent matches\n"
        prompt += "2. A confidence level (HIGH, MEDIUM, LOW, UNSUITABLE)\n"
        prompt += "3. A brief reasoning explaining why\n\n"
        prompt += "Format your response as JSON with this structure for each agent:\n"
        prompt += '{\n  "matches": [\n    {\n      "agent_id": "agent_id",\n'
        prompt += '      "confidence_score": 0.95,\n      "confidence_level": "HIGH",\n'
        prompt += '      "reasoning": "This agent is the best match because..."\n    }\n  ]\n}'

        return prompt

    @abstractmethod
    def get_llm_response(self, prompt: str, agents: List[Agent],
                         conversation_history: Optional[List[Dict[str, str]]] = None,
                         memory: Optional[Dict[str, Any]] = None) -> Any:
        """
        Get a response from the LLM for classification.

        Args:
            prompt: User prompt to classify
            agents: List of available agents
            conversation_history: Optional conversation history
            memory: Optional memory/context

        Returns:
            Raw LLM response
        """
        pass

    @abstractmethod
    def parse_llm_response(self, response: Any, agents: List[Agent]) -> List[ClassificationResult]:
        """
        Parse the LLM response into ClassificationResult objects.

        Args:
            response: Raw LLM response
            agents: List of available agents

        Returns:
            List of ClassificationResult objects
        """
        pass

    def classify(self, prompt: str, agents: List[Agent],
                 conversation_history: Optional[List[Dict[str, str]]] = None,
                 memory: Optional[Dict[str, Any]] = None) -> List[ClassificationResult]:
        """
        Classify a user prompt against a pool of agents.

        Args:
            prompt: User prompt to classify
            agents: List of available agents
            conversation_history: Optional conversation history
            memory: Optional memory/context

        Returns:
            List of ClassificationResult objects sorted by confidence score (descending)
        """
        # Get response from LLM
        response = self.get_llm_response(
            prompt=prompt,
            agents=agents,
            conversation_history=conversation_history if self.config.include_conversation_history else None,
            memory=memory if self.config.include_memory else None
        )

        # Parse response into ClassificationResult objects
        results = self.parse_llm_response(response, agents)

        # Sort results by confidence score (descending)
        results.sort(key=lambda x: x.confidence_score, reverse=True)

        return results

    def get_best_agent(self, prompt: str, agents: List[Agent],
                       conversation_history: Optional[List[Dict[str, str]]] = None,
                       memory: Optional[Dict[str, Any]] = None) -> Tuple[Agent, ClassificationResult]:
        """
        Get the best matching agent for a user prompt.

        Args:
            prompt: User prompt to classify
            agents: List of available agents
            conversation_history: Optional conversation history
            memory: Optional memory/context

        Returns:
            Tuple of (best matching agent, classification result)
        """
        results = self.classify(prompt, agents, conversation_history, memory)
        if not results:
            raise ValueError("No classification results returned")

        best_result = results[0]
        return best_result.agent, best_result