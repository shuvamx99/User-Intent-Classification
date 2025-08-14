"""Vector-enhanced classifier that combines LLM classification with historical feedback."""

import logging
from typing import List, Optional, Dict, Any
from datetime import datetime

from .base_classifier import BaseClassifier
from .openai_classifier import OpenAIClassifier
from ..models.schemas import (
    Agent,
    ClassificationResult,
    ClassifierConfig,
    UserInteraction,
    VectorSearchResult,
)
from ..services.vector_service import MilvusVectorService
from ..services.embedding_service import EmbeddingService
from ..services.reranking_service import AgentReRankingService, RerankingConfig

logger = logging.getLogger(__name__)


class VectorEnhancedClassifier(BaseClassifier):
    """
    Enhanced classifier that combines LLM classification with vector similarity search
    and historical feedback-based reranking.
    """

    def __init__(
        self,
        config: ClassifierConfig,
        base_classifier: BaseClassifier,
        vector_service: MilvusVectorService,
        embedding_service: EmbeddingService,
        reranking_service: Optional[AgentReRankingService] = None,
    ):
        """
        Initialize the vector-enhanced classifier.

        Args:
            config: Classifier configuration
            base_classifier: Base LLM classifier (e.g., OpenAIClassifier)
            vector_service: Vector database service for similarity search
            embedding_service: Service for generating embeddings
            reranking_service: Service for reranking based on historical feedback
        """
        super().__init__(config)
        self.base_classifier = base_classifier
        self.vector_service = vector_service
        self.embedding_service = embedding_service
        self.reranking_service = reranking_service or AgentReRankingService(
            RerankingConfig()
        )

        logger.info(
            "Initialized VectorEnhancedClassifier with feedback-based reranking"
        )

    async def classify(
        self,
        prompt: str,
        agents: List[Agent],
        conversation_history: Optional[List[Dict[str, str]]] = None,
        memory: Optional[Dict[str, Any]] = None,
    ) -> List[ClassificationResult]:
        """
        Enhanced classification that uses LLM + vector search + feedback reranking.

        Args:
            prompt: User prompt to classify
            agents: List of available agents
            conversation_history: Optional conversation history
            memory: Optional memory/context

        Returns:
            Re-ranked list of classification results based on historical feedback
        """
        try:
            logger.info(
                f"Starting enhanced classification for prompt: '{prompt[:50]}...'"
            )

            # Step 1: Get initial LLM classification
            logger.debug("Step 1: Getting initial LLM classification")
            llm_results = await self.base_classifier.classify(
                prompt, agents, conversation_history, memory
            )

            if not llm_results:
                logger.warning("No LLM results returned")
                return []

            logger.info(f"LLM selected {len(llm_results)} agents")

            # Step 2: Perform vector similarity search for historical interactions
            logger.debug("Step 2: Performing vector similarity search")
            similar_interactions = await self._get_similar_interactions(prompt)

            logger.info(
                f"Found {len(similar_interactions)} similar historical interactions"
            )

            # Step 3: Re-rank results using historical feedback
            if similar_interactions:
                logger.debug("Step 3: Re-ranking with historical feedback")
                enhanced_results = await self.reranking_service.rerank_agents(
                    llm_results, similar_interactions
                )
                logger.info("Successfully re-ranked agents using historical feedback")
                return enhanced_results
            else:
                logger.info("No historical data available, using LLM results only")
                return llm_results

        except Exception as e:
            logger.error(f"Error in enhanced classification: {str(e)}")
            # Fallback to base classifier
            logger.warning("Falling back to base LLM classifier")
            return await self.base_classifier.classify(
                prompt, agents, conversation_history, memory
            )

    async def _get_similar_interactions(
        self, prompt: str, top_k: int = 10
    ) -> List[VectorSearchResult]:
        """
        Get similar historical interactions using vector similarity search.

        Args:
            prompt: User prompt to find similar interactions for
            top_k: Number of top similar interactions to retrieve

        Returns:
            List of similar historical interactions with their similarity scores
        """
        try:
            # Generate embedding for the current prompt
            prompt_embedding = await self.embedding_service.embed_text(prompt)

            # Perform similarity search in vector database
            similar_interactions = await self.vector_service.similarity_search(
                prompt_embedding, top_k=top_k
            )

            logger.debug(
                f"Vector search returned {len(similar_interactions)} similar interactions"
            )

            # Log similarity scores for debugging
            for i, interaction in enumerate(similar_interactions[:3]):  # Log top 3
                logger.debug(
                    f"Similar interaction {i+1}: "
                    f"similarity={interaction.similarity_score:.3f}, "
                    f"prompt='{interaction.user_interaction.user_prompt[:30]}...'"
                )

            return similar_interactions

        except Exception as e:
            logger.error(f"Error in vector similarity search: {str(e)}")
            return []

    async def get_llm_response(
        self,
        prompt: str,
        agents: List[Agent],
        conversation_history: Optional[List[Dict[str, str]]] = None,
        memory: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Delegate to base classifier."""
        return await self.base_classifier.get_llm_response(
            prompt, agents, conversation_history, memory
        )

    def parse_llm_response(
        self, response: Any, agents: List[Agent]
    ) -> List[ClassificationResult]:
        """Delegate to base classifier."""
        return self.base_classifier.parse_llm_response(response, agents)

    async def store_interaction_feedback(
        self,
        interaction_id: str,
        user_prompt: str,
        selected_agents: List[str],
        llm_confidence_scores: Dict[str, float],
        final_confidence_scores: Dict[str, float],
    ):
        """
        Store interaction data for future feedback-based improvements.

        Args:
            interaction_id: Unique interaction identifier
            user_prompt: Original user prompt
            selected_agents: List of selected agent IDs
            llm_confidence_scores: Original LLM confidence scores
            final_confidence_scores: Final confidence scores after reranking
        """
        try:
            prompt_embedding = await self.embedding_service.embed_text(user_prompt)

            interaction = UserInteraction(
                interaction_id=interaction_id,
                user_prompt=user_prompt,
                prompt_embedding=prompt_embedding,
                timestamp=datetime.now(),
                selected_agents=selected_agents,
                llm_confidence_scores=llm_confidence_scores,
                final_confidence_scores=final_confidence_scores,
            )

            await self.vector_service.store_interaction(interaction)
            logger.info(
                f"Stored interaction {interaction_id} for future feedback learning"
            )

        except Exception as e:
            logger.error(f"Error storing interaction feedback: {str(e)}")

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the enhanced classifier."""
        try:
            vector_stats = self.vector_service.get_collection_stats()
            return {
                "classifier_type": "VectorEnhancedClassifier",
                "base_classifier": type(self.base_classifier).__name__,
                "vector_database": vector_stats,
                "reranking_config": {
                    "llm_weight": self.reranking_service.config.llm_weight,
                    "historical_success_weight": self.reranking_service.config.historical_success_weight,
                    "similarity_threshold": self.reranking_service.config.similarity_threshold,
                },
            }
        except Exception as e:
            logger.error(f"Error getting classifier stats: {str(e)}")
            return {"error": str(e)}
