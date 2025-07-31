from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from pydantic import BaseModel
import pandas as pd
from ..models.schemas import Agent


class AgentOutput(BaseModel):
    """
    Base model for agent outputs.
    """
    agent_id: str
    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None
    errors: List[str] = []


class BaseAgent(ABC):
    """Base class for all agents."""

    @property
    @abstractmethod
    def id(self) -> str:
        """Get the agent's ID."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Get the agent's name."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Get the agent's description."""
        pass

    @property
    @abstractmethod
    def capabilities(self) -> List[str]:
        """Get the agent's capabilities."""
        pass

    @property
    @abstractmethod
    def metadata(self) -> Agent:
        """Get the agent's metadata as an Agent model."""
        pass


    @abstractmethod
    async def execute(self, input_data: pd.DataFrame,
                context: Optional[Dict[str, Any]] = None) -> AgentOutput:
        """
        Execute the agent's task.

        Args:
            input_data: Pandas DataFrame containing input data for the agent
            context: Optional context/memory for the agent

        Returns:
            Dictionary with the agent's results
        """
        pass

