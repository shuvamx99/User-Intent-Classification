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

    async def classify(
        self,
        prompt: str,
        agents: List[Agent],
        conversation_history: Optional[List[Dict[str, str]]] = None,
        memory: Optional[Dict[str, Any]] = None,
    ) -> List[ClassificationResult]:
        try:
            response = await self.get_llm_response(
                prompt, agents, conversation_history, memory
            )
            results = self.parse_llm_response(response, agents)
            # Filter results by confidence threshold
            return [
                r
                for r in results
                if r.confidence_score >= self.config.confidence_threshold
            ]
        except Exception as e:
            raise ValueError(f"Classification failed: {str(e)}")

    async def get_llm_response(
        self,
        prompt: str,
        agents: List[Agent],
        conversation_history: Optional[List[Dict[str, str]]] = None,
        memory: Optional[Dict[str, Any]] = None,
    ) -> str:
        system_prompt = self.create_system_prompt(agents)
        user_prompt = (
            f"Prompt: {prompt}\n\n"
            "Analyze this prompt and identify all relevant agents. Return ONLY a valid JSON object with the exact format shown in the system prompt. "
            "Include all agents that could reasonably address any part of the prompt. "
            "Do not include any text outside the JSON object."
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
                model=self.model, messages=messages, max_tokens=1000, temperature=0.3
            )
            return response.choices[0].message.content
        except Exception as e:
            raise ValueError(f"Failed to get LLM response: {str(e)}")

    def parse_llm_response(
        self, response: str, agents: List[Agent]
    ) -> List[ClassificationResult]:
        try:
            # Check if response is empty or whitespace
            if not response or not response.strip():
                print("Warning: Empty LLM response received")
                raise ValueError("Empty LLM response received")

            # Try to parse as JSON
            try:
                parsed = json.loads(response)
            except json.JSONDecodeError as e:
                # Log the raw response for debugging
                print(f"Raw LLM response: {repr(response)}")
                raise ValueError(f"Failed to parse LLM response as JSON: {str(e)}")

            if "matches" not in parsed:
                # Try to extract JSON from the response if it's wrapped in text
                print(f"Response missing 'matches' key. Full response: {parsed}")
                raise ValueError("Invalid LLM response format: 'matches' key missing")

            results = []
            for match in parsed["matches"]:
                agent_id = match.get("agent_id")
                confidence_score = match.get("confidence_score")
                confidence_level_str = match.get("confidence_level", "MEDIUM")
                reasoning = match.get("reasoning", "No reasoning provided")

                # Find the corresponding agent
                agent = next((a for a in agents if a.id == agent_id), None)
                if not agent:
                    print(f"Warning: Unknown agent_id '{agent_id}' in response")
                    continue  # Skip unknown agents

                # Validate confidence score
                if (
                    not isinstance(confidence_score, (int, float))
                    or not 0 <= confidence_score <= 1
                ):
                    confidence_score = 0.5  # Default to medium confidence
                    print(
                        f"Warning: Invalid confidence_score for {agent_id}, using default 0.5"
                    )

                # Validate confidence level
                try:
                    confidence_level = ConfidenceLevel(confidence_level_str)
                except ValueError:
                    # Map confidence score to level as fallback
                    if confidence_score >= 0.8:
                        confidence_level = ConfidenceLevel.HIGH
                    elif confidence_score >= 0.5:
                        confidence_level = ConfidenceLevel.MEDIUM
                    elif confidence_score >= 0.2:
                        confidence_level = ConfidenceLevel.LOW
                    else:
                        confidence_level = ConfidenceLevel.UNSUITABLE
                    print(
                        f"Warning: Invalid confidence_level '{confidence_level_str}' for {agent_id}, using calculated level"
                    )

                # Create result
                result = ClassificationResult(
                    agent=agent,
                    confidence_score=float(confidence_score),
                    confidence_level=confidence_level,
                    reasoning=reasoning,
                )
                results.append(result)

            if not results:
                print("Warning: No valid agents found in LLM response, using fallback")
                # Fallback: assign medium confidence to all agents
                for agent in agents:
                    results.append(
                        ClassificationResult(
                            agent=agent,
                            confidence_score=0.5,
                            confidence_level=ConfidenceLevel.MEDIUM,
                            reasoning="Fallback assignment due to parsing error",
                        )
                    )

            return sorted(results, key=lambda x: x.confidence_score, reverse=True)

        except Exception as e:
            print(f"Error parsing LLM response: {str(e)}")
            print(f"Raw response: {repr(response)}")
            # Fallback: return all agents with medium confidence
            print("Using fallback: assigning medium confidence to all agents")
            return [
                ClassificationResult(
                    agent=agent,
                    confidence_score=0.5,
                    confidence_level=ConfidenceLevel.MEDIUM,
                    reasoning=f"Fallback due to parsing error: {str(e)}",
                )
                for agent in agents
            ]

    def create_system_prompt(self, agents: List[Agent]) -> str:
        agent_info = "\n".join(
            [
                f"ID: {a.id}, Name: {a.name}, Description: {a.description}, Capabilities: {a.capabilities}"
                for a in agents
            ]
        )
        return (
            "You are an agent selection system. Given a user prompt and a list of agents with their metadata, "
            "identify all agents relevant to the prompt. Assign a confidence score (0-1) to each agent based on "
            "relevance, provide reasoning, and classify the confidence level as HIGH, MEDIUM, LOW, or UNSUITABLE. "
            "Return ONLY a valid JSON object with this exact format:\n"
            "{\n"
            '  "matches": [\n'
            "    {\n"
            '      "agent_id": "agent_id_here",\n'
            '      "confidence_score": 0.8,\n'
            '      "confidence_level": "HIGH",\n'
            '      "reasoning": "Explanation here"\n'
            "    }\n"
            "  ]\n"
            "}\n\n"
            "Important: Return ONLY the JSON object, no additional text or explanations. "
            "Include multiple agents if the prompt implies multiple tasks.\n\n"
            f"Available Agents:\n{agent_info}"
        )
