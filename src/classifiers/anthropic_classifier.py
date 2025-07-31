# src/classifiers/anthropic_classifier.py
import json
from typing import List, Dict, Any, Optional
import anthropic

from .base_classifier import BaseClassifier
from ..models.enums import ConfidenceLevel
from ..models.schemas import Agent, ClassificationResult, ClassifierConfig


class AnthropicClassifier(BaseClassifier):
    """Classifier implementation using Anthropic's Claude API."""

    def __init__(self, config: ClassifierConfig):
        """
        Initialize the Anthropic classifier.

        Args:
            config: Configuration for the classifier
        """
        super().__init__(config)
        self.client = anthropic.Anthropic(api_key=config.api_key)

    def get_llm_response(self, prompt: str, agents: List[Agent],
                         conversation_history: Optional[List[Dict[str, str]]] = None,
                         memory: Optional[Dict[str, Any]] = None) -> str:
        """
        Get a response from Claude for classification.

        Args:
            prompt: User prompt to classify
            agents: List of available agents
            conversation_history: Optional conversation history
            memory: Optional memory/context

        Returns:
            Claude's response as a string
        """
        system_prompt = self.create_system_prompt(agents)

        # Prepare conversation context
        context = ""
        if conversation_history and self.config.include_conversation_history:
            context += "Previous conversation:\n"
            for msg in conversation_history:
                role = msg.get("role", "unknown")
                content = msg.get("content", "")
                context += f"{role.capitalize()}: {content}\n"
            context += "\n"

        if memory and self.config.include_memory:
            context += "Memory/context:\n"
            context += json.dumps(memory, indent=2)
            context += "\n\n"

        # Prepare the user message
        user_message = f"{context}User request: {prompt}"

        # Query Claude
        response = self.client.messages.create(
            model=self.config.model,
            system=system_prompt,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            messages=[
                {"role": "user", "content": user_message}
            ]
        )

        return response.content[0].text

    def parse_llm_response(self, response: str, agents: List[Agent]) -> List[ClassificationResult]:
        """
        Parse Claude's response into ClassificationResult objects.

        Args:
            response: Claude's response as a string
            agents: List of available agents

        Returns:
            List of ClassificationResult objects
        """
        # Create agent id to agent mapping for easier lookup
        agent_map = {agent.id: agent for agent in agents}

        # Extract JSON from the response
        try:
            # Find JSON block in the response
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            if json_start == -1 or json_end == 0:
                raise ValueError("No JSON found in response")

            json_str = response[json_start:json_end]
            data = json.loads(json_str)

            # Extract matches
            matches = data.get("matches", [])
            if not matches:
                raise ValueError("No matches found in response")

            results = []
            for match in matches:
                agent_id = match.get("agent_id")
                if agent_id not in agent_map:
                    continue  # Skip unknown agents

                agent = agent_map[agent_id]
                confidence_score = float(match.get("confidence_score", 0))
                confidence_level_str = match.get("confidence_level", "UNSUITABLE")

                # Map string to enum if needed
                try:
                    confidence_level = ConfidenceLevel(confidence_level_str)
                except ValueError:
                    # Use calculated confidence level as fallback
                    confidence_level = self.calc_confidence_level(confidence_score)

                reasoning = match.get("reasoning", "No reasoning provided")

                # Create result
                result = ClassificationResult(
                    agent=agent,
                    confidence_score=confidence_score,
                    confidence_level=confidence_level,
                    reasoning=reasoning
                )
                results.append(result)

            return results

        except (json.JSONDecodeError, ValueError, KeyError) as e:
            # Fallback to a more lenient parsing method
            results = []
            for agent in agents:
                # Try to extract agent-specific information from the response
                agent_section = response.lower().find(agent.name.lower())
                if agent_section == -1:
                    # Agent not mentioned, assign low confidence
                    results.append(ClassificationResult(
                        agent=agent,
                        confidence_score=0.1,
                        confidence_level=ConfidenceLevel.UNSUITABLE,
                        reasoning=f"Agent not mentioned in response. Error: {str(e)}"
                    ))
                    continue

                # Simple heuristic: How prominently is this agent mentioned?
                # Earlier mentions get higher scores
                position_score = 1.0 - (agent_section / len(response))  # 0-1 based on position
                position_score = max(0.1, min(0.9, position_score))  # Clamp between 0.1 and 0.9

                confidence_level = self.calc_confidence_level(position_score)

                results.append(ClassificationResult(
                    agent=agent,
                    confidence_score=position_score,
                    confidence_level=confidence_level,
                    reasoning=f"Fallback scoring: agent found at position {agent_section} in response"
                ))

            return results