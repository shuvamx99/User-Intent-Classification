"""Vector database service using Milvus for storing and retrieving user interactions."""

from pymilvus import (
    Collection,
    CollectionSchema,
    FieldSchema,
    DataType,
    connections,
    utility,
)
from typing import List, Dict, Any, Optional
import numpy as np
from ..models.schemas import UserInteraction, VectorSearchResult
import json
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class MilvusVectorService:
    """Service for managing vector embeddings and similarity search using Milvus."""

    def __init__(self, host: str = "localhost", port: str = "19530"):
        """
        Initialize the Milvus vector service.

        Args:
            host: Milvus server host
            port: Milvus server port
        """
        self.host = host
        self.port = port
        self.collection_name = "user_interactions"
        self.embedding_dim = 1536  # OpenAI embedding dimension
        self.collection = None
        self.connect()
        self.setup_collection()

    def connect(self):
        """Connect to Milvus server."""
        try:
            connections.connect("default", host=self.host, port=self.port)
            logger.info(f"Connected to Milvus at {self.host}:{self.port}")
        except Exception as e:
            logger.error(f"Failed to connect to Milvus: {str(e)}")
            raise

    def health_check(self) -> bool:
        """Check if Milvus connection is healthy."""
        try:
            # Try to list collections as a health check
            utility.list_collections()
            return True
        except Exception as e:
            logger.error(f"Milvus health check failed: {str(e)}")
            return False

    def setup_collection(self):
        """Set up the collection schema and create if it doesn't exist."""
        try:
            # Define schema
            fields = [
                FieldSchema(
                    name="interaction_id",
                    dtype=DataType.VARCHAR,
                    max_length=100,
                    is_primary=True,
                ),
                FieldSchema(
                    name="prompt_embedding",
                    dtype=DataType.FLOAT_VECTOR,
                    dim=self.embedding_dim,
                ),
                FieldSchema(
                    name="user_prompt", dtype=DataType.VARCHAR, max_length=1000
                ),
                FieldSchema(
                    name="selected_agents", dtype=DataType.VARCHAR, max_length=500
                ),
                FieldSchema(name="timestamp", dtype=DataType.INT64),
                FieldSchema(name="session_id", dtype=DataType.VARCHAR, max_length=100),
                FieldSchema(name="feedback_score", dtype=DataType.FLOAT),
                FieldSchema(
                    name="interaction_data", dtype=DataType.VARCHAR, max_length=10000
                ),  # Essential interaction data as JSON
            ]

            schema = CollectionSchema(
                fields, "User interaction embeddings for agent classification"
            )

            # Create collection if it doesn't exist
            if not self.collection_exists():
                self.collection = Collection(self.collection_name, schema)
                self.create_index()
                logger.info(f"Created new collection: {self.collection_name}")
            else:
                self.collection = Collection(self.collection_name)
                logger.info(f"Connected to existing collection: {self.collection_name}")

        except Exception as e:
            logger.error(f"Failed to setup collection: {str(e)}")
            raise

    def collection_exists(self) -> bool:
        """Check if the collection exists."""
        try:
            return utility.has_collection(self.collection_name)
        except Exception as e:
            logger.error(f"Error checking collection existence: {str(e)}")
            return False

    def create_index(self):
        """Create index for vector similarity search."""
        try:
            index_params = {
                "metric_type": "IP",  # Inner Product for cosine similarity
                "index_type": "IVF_FLAT",
                "params": {"nlist": 128},
            }
            self.collection.create_index("prompt_embedding", index_params)
            logger.info("Created index for prompt_embedding field")
        except Exception as e:
            logger.error(f"Failed to create index: {str(e)}")
            raise

    async def store_interaction(self, interaction: UserInteraction):
        """
        Store a user interaction in the vector database.

        Args:
            interaction: UserInteraction object to store
        """
        try:
            # Ensure embedding dimension is correct
            if len(interaction.prompt_embedding) != self.embedding_dim:
                raise ValueError(
                    f"Embedding dimension mismatch. Expected {self.embedding_dim}, got {len(interaction.prompt_embedding)}"
                )

            # Store only essential data instead of full interaction JSON
            essential_data = {
                "interaction_id": interaction.interaction_id,
                "user_prompt": interaction.user_prompt,
                "selected_agents": interaction.selected_agents,
                "llm_confidence_scores": interaction.llm_confidence_scores,
                "final_confidence_scores": interaction.final_confidence_scores,
                "timestamp": interaction.timestamp.isoformat(),
                "session_id": interaction.session_id,
            }

            data = [
                [interaction.interaction_id],
                [interaction.prompt_embedding],
                [interaction.user_prompt[:1000]],  # Truncate if too long
                [json.dumps(interaction.selected_agents)],
                [int(interaction.timestamp.timestamp())],
                [interaction.session_id or ""],
                [0.0],  # Initial feedback score
                [json.dumps(essential_data)],  # Store essential data as JSON
            ]

            self.collection.insert(data)
            self.collection.flush()
            logger.info(f"Stored interaction: {interaction.interaction_id}")

        except Exception as e:
            logger.error(
                f"Failed to store interaction {interaction.interaction_id}: {str(e)}"
            )
            raise

    async def similarity_search(
        self, prompt_embedding: List[float], top_k: int = 10
    ) -> List[VectorSearchResult]:
        """
        Search for similar interactions based on prompt embedding.

        Args:
            prompt_embedding: Embedding vector of the query prompt
            top_k: Number of similar results to return

        Returns:
            List of VectorSearchResult objects
        """
        try:
            # Ensure embedding dimension is correct
            if len(prompt_embedding) != self.embedding_dim:
                raise ValueError(
                    f"Embedding dimension mismatch. Expected {self.embedding_dim}, got {len(prompt_embedding)}"
                )

            self.collection.load()

            search_params = {"metric_type": "IP", "params": {"nprobe": 10}}
            results = self.collection.search(
                [prompt_embedding],
                "prompt_embedding",
                search_params,
                limit=top_k,
                output_fields=[
                    "interaction_id",
                    "user_prompt",
                    "interaction_data",
                    "feedback_score",
                ],
            )

            search_results = []
            for hits in results:
                for hit in hits:
                    try:
                        essential_data = json.loads(hit.entity.get("interaction_data"))

                        # Reconstruct UserInteraction from essential data
                        # We don't have the embedding in the essential data, so we'll create a placeholder
                        # Parse execution results if available
                        execution_results = []
                        if "execution_results" in essential_data:
                            from ..models.schemas import ExecutionResult

                            for result_data in essential_data["execution_results"]:
                                execution_results.append(
                                    ExecutionResult(
                                        agent_id=result_data["agent_id"],
                                        success=result_data["success"],
                                        message=result_data.get("message", ""),
                                        errors=result_data.get("errors", []),
                                        timestamp=datetime.fromisoformat(
                                            result_data.get(
                                                "timestamp", datetime.now().isoformat()
                                            )
                                        ),
                                    )
                                )

                        interaction = UserInteraction(
                            interaction_id=essential_data["interaction_id"],
                            user_prompt=essential_data["user_prompt"],
                            prompt_embedding=[],  # Empty placeholder since we don't need it for reranking
                            timestamp=datetime.fromisoformat(
                                essential_data["timestamp"]
                            ),
                            selected_agents=essential_data["selected_agents"],
                            llm_confidence_scores=essential_data.get(
                                "llm_confidence_scores", {}
                            ),
                            final_confidence_scores=essential_data.get(
                                "final_confidence_scores", {}
                            ),
                            execution_results=execution_results,
                            session_id=essential_data.get("session_id"),
                        )

                        search_results.append(
                            VectorSearchResult(
                                interaction_id=hit.entity.get("interaction_id"),
                                similarity_score=hit.score,
                                user_interaction=interaction,
                                feedback_score=hit.entity.get(
                                    "feedback_score", 0.0
                                ),  # Include feedback score
                            )
                        )
                    except Exception as e:
                        logger.warning(
                            f"Failed to parse interaction data for {hit.entity.get('interaction_id')}: {str(e)}"
                        )
                        continue

            logger.info(f"Found {len(search_results)} similar interactions")
            return search_results

        except Exception as e:
            logger.error(f"Failed to perform similarity search: {str(e)}")
            return []

    async def update_feedback(self, interaction_id: str, feedback_score: float):
        """
        Update feedback score for an interaction.

        Args:
            interaction_id: ID of the interaction to update
            feedback_score: New feedback score (0.0 to 1.0)
        """
        try:
            # For simplicity, we'll use delete and re-insert approach
            # In production, consider using update operations when available
            expr = f'interaction_id == "{interaction_id}"'

            # First, get the existing data
            self.collection.load()
            results = self.collection.query(expr=expr, output_fields=["*"])

            if results:
                # Delete existing record
                self.collection.delete(expr)

                # Update feedback score and re-insert
                record = results[0]
                record["feedback_score"] = feedback_score

                # Re-insert updated record
                data = [
                    [record["interaction_id"]],
                    [record["prompt_embedding"]],
                    [record["user_prompt"]],
                    [record["selected_agents"]],
                    [record["timestamp"]],
                    [record["session_id"]],
                    [feedback_score],
                    [record["interaction_data"]],
                ]

                self.collection.insert(data)
                self.collection.flush()
                logger.info(f"Updated feedback for interaction: {interaction_id}")
            else:
                logger.warning(f"Interaction not found for update: {interaction_id}")

        except Exception as e:
            logger.error(f"Failed to update feedback for {interaction_id}: {str(e)}")
            raise

    async def update_execution_results(
        self, interaction_id: str, execution_results: List[Dict[str, Any]]
    ):
        """
        Update execution results for an interaction.

        Args:
            interaction_id: ID of the interaction to update
            execution_results: List of execution results with agent_id, success, message, etc.
        """
        try:
            # For simplicity, we'll use delete and re-insert approach
            expr = f'interaction_id == "{interaction_id}"'

            # First, get the existing data
            self.collection.load()
            results = self.collection.query(expr=expr, output_fields=["*"])

            if results:
                # Delete existing record
                self.collection.delete(expr)

                # Update execution results in the interaction data
                record = results[0]
                interaction_data = json.loads(record["interaction_data"])

                # Update execution results
                interaction_data["execution_results"] = execution_results

                # Re-insert updated record
                data = [
                    [record["interaction_id"]],
                    [record["prompt_embedding"]],
                    [record["user_prompt"]],
                    [record["selected_agents"]],
                    [record["timestamp"]],
                    [record["session_id"]],
                    [record["feedback_score"]],
                    [json.dumps(interaction_data)],
                ]

                self.collection.insert(data)
                self.collection.flush()
                logger.info(
                    f"Updated execution results for interaction: {interaction_id}"
                )
            else:
                logger.warning(
                    f"Interaction not found for execution results update: {interaction_id}"
                )

        except Exception as e:
            logger.error(
                f"Failed to update execution results for {interaction_id}: {str(e)}"
            )
            raise

    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection."""
        try:
            self.collection.load()
            stats = {
                "name": self.collection_name,
                "num_entities": self.collection.num_entities,
                "description": self.collection.description,
            }
            return stats
        except Exception as e:
            logger.error(f"Failed to get collection stats: {str(e)}")
            return {}

    def close_connection(self):
        """Close the connection to Milvus."""
        try:
            connections.disconnect("default")
            logger.info("Disconnected from Milvus")
        except Exception as e:
            logger.warning(f"Error disconnecting from Milvus: {str(e)}")
