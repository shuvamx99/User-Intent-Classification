import json
from typing import List, Optional, Dict, Any
from openai import AsyncOpenAI
from ..classifiers.base_classifier import BaseClassifier
from ..models.schemas import Agent, ClassificationResult, ClassifierConfig
from ..models.enums import ConfidenceLevel

class OpenAIClassifier(BaseClassifier):
    """Classifier using OpenAI's LLM for agent selection."""

    def __init__(self, config: ClassifierConfig):
        super().__init__(config)
        self.client = AsyncOpenAI(api_key=config.openai_api_key)
        self.model = config.model

    async def classify(self, prompt: str, agents: List[Agent]) -> List[ClassificationResult]:
        try:
            response = await self.get_llm_response(prompt, agents)
            results = self.parse_llm_response(response, agents)
            # Filter results by confidence threshold
            return [r for r in results if r.confidence_score >= self.config.confidence_threshold]
        except Exception as e:
            raise ValueError(f"Classification failed: {str(e)}")

    async def get_llm_response(self, prompt: str, agents: List[Agent],
                               conversation_history: Optional[List[Dict[str, str]]] = None,
                               memory: Optional[Dict[str, Any]] = None) -> str:
        system_prompt = self.create_system_prompt(agents)
        user_prompt = (
            f"Prompt: {prompt}\n\n"
            "Identify all relevant agents for the prompt. For each agent, return a JSON object with "
            "'agent_id', 'confidence_score' (0-1), 'confidence_level' (HIGH, MEDIUM, LOW, UNSUITABLE), "
            "and 'reasoning'. Include all agents that could reasonably address any part of the prompt."
        )

        # Include conversation history if enabled
        messages = [{"role": "system", "content": system_prompt}]
        if conversation_history and self.config.include_conversation_history:
            messages.extend(conversation_history)
        messages.append({"role": "user", "content": user_prompt})

        # Include memory in the system prompt if enabled
        if memory and self.config.include_memory:
            system_prompt += f"\n\nMemory Context: {json.dumps(memory, indent=2)}"
            messages[0]["content"] = system_prompt

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=1000,
                temperature=0.3
            )
            return response.choices[0].message.content
        except Exception as e:
            raise ValueError(f"Failed to get LLM response: {str(e)}")

    def parse_llm_response(self, response: str, agents: List[Agent]) -> List[ClassificationResult]:
        try:
            parsed = json.loads(response)
            if "matches" not in parsed:
                raise ValueError("Invalid LLM response format: 'matches' key missing")

            results = []
            for match in parsed["matches"]:
                agent_id = match.get("agent_id")
                confidence_score = match.get("confidence_score")
                confidence_level = ConfidenceLevel(match.get("confidence_level"))
                reasoning = match.get("reasoning")

                # Find the corresponding agent
                agent = next((a for a in agents if a.id == agent_id), None)
                if not agent:
                    continue

                # Validate confidence score
                if not isinstance(confidence_score, (int, float)) or not 0 <= confidence_score <= 1:
                    raise ValueError(f"Invalid confidence score for agent {agent_id}: {confidence_score}")

                results.append(ClassificationResult(
                    agent=agent,
                    confidence_score=confidence_score,
                    confidence_level=confidence_level,
                    reasoning=reasoning
                ))

            return sorted(results, key=lambda x: x.confidence_score, reverse=True)
        except Exception as e:
            raise ValueError(f"Failed to parse LLM response: {str(e)}")

    def create_system_prompt(self, agents: List[Agent]) -> str:
        agent_info = "\n".join(
            [f"ID: {a.id}, Name: {a.name}, Description: {a.description}, Capabilities: {a.capabilities}"
             for a in agents]
        )
        return (
            "You are an agent selection system. Given a user prompt and a list of agents with their metadata, "
            "identify all agents relevant to the prompt. Assign a confidence score (0-1) to each agent based on "
            "relevance, provide reasoning, and classify the confidence level as HIGH, MEDIUM, LOW, or UNSUITABLE. "
            "Return a JSON object with a 'matches' key containing a list of objects, each with 'agent_id', "
            "'confidence_score', 'confidence_level', and 'reasoning'. Include multiple agents if the prompt "
            "implies multiple tasks.\n\n"
            f"Agents:\n{agent_info}"
        )