"""Embedding service for generating text embeddings using OpenAI."""

from openai import AsyncOpenAI
from typing import List
import logging

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Service for generating text embeddings using OpenAI's embedding models."""

    def __init__(self, api_key: str, model: str = "text-embedding-ada-002"):
        """
        Initialize the embedding service.

        Args:
            api_key: OpenAI API key
            model: Embedding model to use
        """
        self.client = AsyncOpenAI(api_key=api_key)
        self.model = model
        logger.info(f"Initialized EmbeddingService with model: {model}")

    async def embed_text(self, text: str) -> List[float]:
        """
        Generate embeddings for text using OpenAI.

        Args:
            text: Text to embed

        Returns:
            List of float values representing the embedding
        """
        try:
            # Clean and truncate text if necessary
            cleaned_text = text.strip()
            if len(cleaned_text) == 0:
                raise ValueError("Cannot embed empty text")

            # OpenAI has a token limit, truncate if too long
            if len(cleaned_text) > 8000:  # Conservative limit
                cleaned_text = cleaned_text[:8000]
                logger.warning(f"Text truncated to 8000 characters for embedding")

            response = await self.client.embeddings.create(
                input=cleaned_text, model=self.model
            )

            embedding = response.data[0].embedding
            logger.debug(
                f"Generated embedding of dimension {len(embedding)} for text of length {len(cleaned_text)}"
            )
            return embedding

        except Exception as e:
            logger.error(f"Failed to generate embedding: {str(e)}")
            raise

    async def embed_agent_metadata(
        self, agent_name: str, description: str, capabilities: List[str]
    ) -> List[float]:
        """
        Create embeddings for agent metadata.

        Args:
            agent_name: Name of the agent
            description: Description of the agent
            capabilities: List of agent capabilities

        Returns:
            List of float values representing the agent embedding
        """
        try:
            # Construct comprehensive agent description
            capabilities_text = (
                ", ".join(capabilities) if capabilities else "No specific capabilities"
            )
            agent_text = f"Agent: {agent_name}. Description: {description}. Capabilities: {capabilities_text}"

            return await self.embed_text(agent_text)

        except Exception as e:
            logger.error(
                f"Failed to generate agent embedding for {agent_name}: {str(e)}"
            )
            raise

    async def embed_prompt_with_context(
        self, prompt: str, context: str = ""
    ) -> List[float]:
        """
        Generate embeddings for a prompt with optional context.

        Args:
            prompt: User prompt
            context: Additional context (e.g., conversation history)

        Returns:
            List of float values representing the embedding
        """
        try:
            if context:
                combined_text = f"Context: {context}\n\nUser Query: {prompt}"
            else:
                combined_text = prompt

            return await self.embed_text(combined_text)

        except Exception as e:
            logger.error(f"Failed to generate prompt embedding: {str(e)}")
            raise





    async def batch_embed(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts in batch.

        Args:
            texts: List of texts to embed

        Returns:
            List of embeddings corresponding to input texts
        """
        try:
            if not texts:
                return []

            # Clean texts
            cleaned_texts = [text.strip() for text in texts if text.strip()]
            if not cleaned_texts:
                raise ValueError("No valid texts to embed")

            # Truncate texts if necessary
            truncated_texts = []
            for text in cleaned_texts:
                if len(text) > 8000:
                    truncated_texts.append(text[:8000])
                    logger.warning(
                        f"Text truncated to 8000 characters for batch embedding"
                    )
                else:
                    truncated_texts.append(text)

            response = await self.client.embeddings.create(
                input=truncated_texts, model=self.model
            )

            embeddings = [data.embedding for data in response.data]
            logger.info(f"Generated {len(embeddings)} embeddings in batch")
            return embeddings

        except Exception as e:
            logger.error(f"Failed to generate batch embeddings: {str(e)}")
            raise

    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of embeddings produced by the current model.

        Returns:
            Embedding dimension
        """
        # Known dimensions for OpenAI models
        model_dimensions = {
            "text-embedding-ada-002": 1536,
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
        }

        return model_dimensions.get(self.model, 1536)  # Default to ada-002 dimension
