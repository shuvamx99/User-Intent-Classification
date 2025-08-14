"""Re-ranking service for improving agent selection using historical performance data."""

from typing import List, Dict
from ..models.schemas import (
    ClassificationResult,
    VectorSearchResult,
    RerankingConfig,
)
from ..models.enums import ConfidenceLevel
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class AgentReRankingService:
    """Service for re-ranking agents based on historical performance and user feedback."""

    def __init__(self, config: RerankingConfig = None):
        """
        Initialize the re-ranking service.

        Args:
            config: Configuration for re-ranking weights and thresholds
        """
        self.config = config or RerankingConfig()
        logger.info(f"Initialized AgentReRankingService with config: {self.config}")

    async def rerank_agents(
        self,
        llm_results: List[ClassificationResult],
        similar_interactions: List[VectorSearchResult],
    ) -> List[ClassificationResult]:
        """
        Re-rank agents based on historical performance and user feedback.

        Args:
            llm_results: Initial classification results from LLM
            similar_interactions: Similar historical interactions from vector search

        Returns:
            Re-ranked list of classification results
        """
        try:
            logger.info(
                f"Re-ranking {len(llm_results)} agents using {len(similar_interactions)} historical interactions"
            )

            reranked_results = []

            for result in llm_results:
                agent_id = result.agent.id
                llm_score = result.confidence_score

                # Calculate historical performance metrics
                historical_success = self._calculate_historical_success_rate(
                    agent_id, similar_interactions
                )

                # Calculate final score using simplified weighted combination
                final_score = (
                    self.config.llm_weight * llm_score
                    + self.config.historical_success_weight * historical_success
                )

                # Ensure score is between 0 and 1
                final_score = max(0.0, min(final_score, 1.0))

                # Create enhanced reasoning
                enhanced_reasoning = (
                    f"{result.reasoning} [Historical data: "
                    f"success_rate={historical_success:.2f}, "
                    f"adjusted_confidence={final_score:.2f}]"
                )

                # Create new result with updated score
                reranked_result = ClassificationResult(
                    agent=result.agent,
                    confidence_score=final_score,
                    confidence_level=self._calculate_confidence_level(final_score),
                    reasoning=enhanced_reasoning,
                )

                reranked_results.append(reranked_result)

                logger.debug(
                    f"Agent {agent_id}: LLM={llm_score:.3f} -> Final={final_score:.3f} "
                    f"(historical_success={historical_success:.3f})"
                )

            # Sort by final confidence score
            reranked_results.sort(key=lambda x: x.confidence_score, reverse=True)

            logger.info(
                f"Re-ranking completed. Top agent: {reranked_results[0].agent.id if reranked_results else 'None'}"
            )
            return reranked_results

        except Exception as e:
            logger.error(f"Failed to re-rank agents: {str(e)}")
            # Return original results if re-ranking fails
            return llm_results

    def _calculate_historical_success_rate(
        self, agent_id: str, interactions: List[VectorSearchResult]
    ) -> float:
        """
        Calculate success rate for an agent from historical interactions.

        Args:
            agent_id: ID of the agent to calculate success rate for
            interactions: List of historical interactions

        Returns:
            Success rate between 0.0 and 1.0
        """
        try:
            relevant_interactions = [
                interaction
                for interaction in interactions
                if agent_id in interaction.user_interaction.selected_agents
            ]

            if not relevant_interactions:
                return 0.5  # Neutral score if no history

            success_count = 0
            total_count = 0

            for interaction in relevant_interactions:
                for result in interaction.user_interaction.execution_results:
                    if result.agent_id == agent_id:
                        total_count += 1
                        if result.success:
                            success_count += 1

            if total_count == 0:
                return 0.5  # Neutral score if no execution results

            success_rate = success_count / total_count
            logger.debug(
                f"Agent {agent_id} historical success rate: {success_rate:.3f} ({success_count}/{total_count})"
            )
            return success_rate

        except Exception as e:
            logger.warning(f"Error calculating success rate for {agent_id}: {str(e)}")
            return 0.5

    def _calculate_confidence_level(self, score: float) -> ConfidenceLevel:
        """
        Convert numerical score to confidence level enum.

        Args:
            score: Confidence score between 0.0 and 1.0

        Returns:
            ConfidenceLevel enum value
        """
        if score >= 0.8:
            return ConfidenceLevel.HIGH
        elif score >= 0.5:
            return ConfidenceLevel.MEDIUM
        elif score >= 0.2:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.UNSUITABLE

    def update_config(self, new_config: RerankingConfig):
        """
        Update the re-ranking configuration.

        Args:
            new_config: New configuration to use
        """
        self.config = new_config
        logger.info(f"Updated re-ranking config: {self.config}")

    def get_agent_performance_summary(
        self, agent_id: str, interactions: List[VectorSearchResult]
    ) -> Dict[str, float]:
        """
        Get a performance summary for a specific agent.

        Args:
            agent_id: ID of the agent
            interactions: List of historical interactions

        Returns:
            Dictionary with performance metrics
        """
        try:
            return {
                "success_rate": self._calculate_historical_success_rate(
                    agent_id, interactions
                ),
                "total_interactions": len(
                    [
                        i
                        for i in interactions
                        if agent_id in i.user_interaction.selected_agents
                    ]
                ),
            }
        except Exception as e:
            logger.error(f"Error getting performance summary for {agent_id}: {str(e)}")
            return {
                "success_rate": 0.5,
                "total_interactions": 0,
            }

    def calculate_similarity_boost(
        self, interactions: List[VectorSearchResult]
    ) -> float:
        """
        Calculate a boost factor based on similarity scores of retrieved interactions.

        Args:
            interactions: List of similar interactions

        Returns:
            Boost factor between 0.0 and 1.0
        """
        try:
            if not interactions:
                return 0.0

            # Use average similarity score as boost factor
            similarity_scores = [
                interaction.similarity_score for interaction in interactions
            ]
            avg_similarity = sum(similarity_scores) / len(similarity_scores)

            # Apply threshold to ensure only highly similar interactions contribute
            if avg_similarity > self.config.similarity_threshold:
                boost = (avg_similarity - self.config.similarity_threshold) / (
                    1.0 - self.config.similarity_threshold
                )
                return min(boost, 0.2)  # Cap boost at 0.2

            return 0.0

        except Exception as e:
            logger.warning(f"Error calculating similarity boost: {str(e)}")
            return 0.0
