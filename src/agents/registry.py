from typing import List, Type
from ..agents.base_agent import BaseAgent
from ..agents.geo_mismatch_agent import GeoMismatchAgent
from ..agents.ip_address_agent import IPAddressAgent
from ..agents.fast_completion_agent import FastCompletionAgent
from ..models.schemas import Agent

class AgentRegistry:
    """Manages a dynamic pool of agents."""

    _agents: List[Type[BaseAgent]] = [
        GeoMismatchAgent,
        IPAddressAgent,
        FastCompletionAgent
    ]

    @classmethod
    def get_agents(cls) -> List[BaseAgent]:
        """Returns instances of all registered agents."""
        return [agent() for agent in cls._agents]

    @classmethod
    def register_agent(cls, agent_class: Type[BaseAgent]) -> None:
        """Registers a new agent class."""
        if agent_class not in cls._agents:
            cls._agents.append(agent_class)

    @classmethod
    def get_agent_by_id(cls, agent_id: str) -> BaseAgent:
        """Retrieves an agent instance by ID."""
        for agent_class in cls._agents:
            agent = agent_class()
            if agent.id == agent_id:
                return agent
        raise ValueError(f"No agent found with ID: {agent_id}")
