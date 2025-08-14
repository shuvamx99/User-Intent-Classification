import asyncio
import pandas as pd
import os
import uuid
from datetime import datetime
from src.classifiers.openai_classifier import OpenAIClassifier
from src.classifiers.vector_enhanced_classifier import VectorEnhancedClassifier
from src.models.schemas import (
    ClassifierConfig,
    UserInteraction,
    UserFeedback,
    ImplicitFeedback,
)
from src.services.reranking_service import AgentReRankingService, RerankingConfig
from src.agents.registry import AgentRegistry
from src.agents.geo_mismatch_agent import GeoMismatchAgent
from src.agents.ip_address_agent import IPAddressAgent
from src.agents.fast_completion_agent import FastCompletionAgent
from src.services.vector_service import MilvusVectorService
from src.services.embedding_service import EmbeddingService


async def collect_user_feedback(
    interaction_id: str, executed_agents: list, user_prompt: str
) -> UserFeedback:
    """Collect feedback from the user about the agent selection and results."""
    print("\n" + "=" * 50)
    print("FEEDBACK COLLECTION")
    print("=" * 50)
    print("Please provide feedback to help improve the system!")
    print(f"\nYour request was: '{user_prompt}'")
    print(
        f"Agents selected: {', '.join([agent['agent_name'] for agent in executed_agents])}"
    )

    # First, ask about agent selection appropriateness
    print("\n" + "-" * 50)
    print("AGENT SELECTION FEEDBACK")
    print("-" * 50)

    agent_selection_rating = None
    while True:
        try:
            selection_prompt = (
                "\nHow appropriate was the agent selection for your request? "
                "(1=completely wrong agents, 3=okay but missing some, 5=perfect selection): "
            )
            selection_rating = input(selection_prompt)
            selection_rating = float(selection_rating)
            if 1 <= selection_rating <= 5:
                agent_selection_rating = selection_rating
                break
            else:
                print("Please enter a rating between 1 and 5.")
        except ValueError:
            print("Please enter a valid number between 1 and 5.")

    # Ask if any agents were missing
    missing_agents_feedback = None
    if agent_selection_rating < 4:
        missing_agents_feedback = input(
            "\nWhich agents do you think should have been selected that weren't? "
            "(e.g., 'IP Address Agent', 'User Behavior Agent', etc.) or 'none': "
        ).strip()
        if missing_agents_feedback.lower() in ["none", ""]:
            missing_agents_feedback = None

    # Ask if any agents were unnecessary
    unnecessary_agents_feedback = None
    if agent_selection_rating < 4:
        unnecessary_agents_feedback = input(
            "\nWere any of the selected agents unnecessary for your request? "
            "If yes, specify which ones or 'none': "
        ).strip()
        if unnecessary_agents_feedback.lower() in ["none", ""]:
            unnecessary_agents_feedback = None

    # Collect individual agent performance ratings
    print("\n" + "-" * 50)
    print("AGENT PERFORMANCE FEEDBACK")
    print("-" * 50)

    agent_ratings = {}
    for agent_info in executed_agents:
        agent_name = agent_info["agent_name"]
        agent_id = agent_info["agent_id"]

        while True:
            try:
                rating = input(
                    f"\nRate '{agent_name}' performance (1=poor results, 5=excellent results): "
                )
                rating = float(rating)
                if 1 <= rating <= 5:
                    agent_ratings[agent_id] = rating
                    break
                else:
                    print("Please enter a rating between 1 and 5.")
            except ValueError:
                print("Please enter a valid number between 1 and 5.")

    # Collect overall satisfaction
    print("\n" + "-" * 50)
    print("OVERALL FEEDBACK")
    print("-" * 50)

    while True:
        try:
            overall = input(
                "\nOverall satisfaction with the complete system response (1-5): "
            )
            overall = float(overall)
            if 1 <= overall <= 5:
                break
            else:
                print("Please enter a rating between 1 and 5.")
        except ValueError:
            print("Please enter a valid number between 1 and 5.")

    # Collect detailed feedback text
    feedback_parts = []
    if missing_agents_feedback:
        feedback_parts.append(f"Missing agents: {missing_agents_feedback}")
    if unnecessary_agents_feedback:
        feedback_parts.append(f"Unnecessary agents: {unnecessary_agents_feedback}")

    additional_feedback = input(
        "\nAny additional comments about the system? (optional): "
    ).strip()
    if additional_feedback:
        feedback_parts.append(f"Additional: {additional_feedback}")

    feedback_text = " | ".join(feedback_parts) if feedback_parts else None

    # Create enhanced implicit feedback
    implicit_feedback = ImplicitFeedback(
        agents_actually_used=[agent["agent_id"] for agent in executed_agents],
        time_spent_reviewing_results=0.0,  # Could be calculated if needed
        follow_up_questions=[],
        result_actions_taken=[],
    )

    # Store agent selection rating in the agent_ratings with a special key
    enhanced_agent_ratings = agent_ratings.copy()
    enhanced_agent_ratings["__agent_selection_appropriateness__"] = (
        agent_selection_rating
    )

    return UserFeedback(
        interaction_id=interaction_id,
        agent_ratings=enhanced_agent_ratings,
        overall_satisfaction=overall,
        feedback_text=feedback_text,
        implicit_signals=implicit_feedback,
        timestamp=datetime.now(),
    )


async def main():
    # Initialize registry and classifier
    registry = AgentRegistry()
    registry.register_agent(GeoMismatchAgent)
    registry.register_agent(IPAddressAgent)
    registry.register_agent(FastCompletionAgent)
    api_key = os.getenv("API_KEY")
    if not api_key:
        raise ValueError("API_KEY environment variable not set")
    config = ClassifierConfig(
        openai_api_key=api_key,
        model="gpt-4",
        confidence_threshold=0.7
    )
    # Initialize services for storing interactions and feedback
    vector_service = None
    embedding_service = None
    reranking_service = None
    classifier = None

    try:
        # Get Milvus connection details from environment variables
        milvus_host = "milvus-standalone"
        milvus_port = "19530"

        print(f"Attempting to connect to Milvus at {milvus_host}:{milvus_port}...")

        # Initialize vector and embedding services
        vector_service = MilvusVectorService(host=milvus_host, port=milvus_port)
        embedding_service = EmbeddingService(api_key)

        # Initialize reranking service with simplified weights
        reranking_config = RerankingConfig(
            llm_weight=0.5,  # 50% weight for LLM confidence
            historical_success_weight=0.5,  # 50% weight for historical success
            similarity_threshold=0.7,  # Minimum similarity for relevance
        )
        reranking_service = AgentReRankingService(reranking_config)

        # Initialize base LLM classifier
        base_classifier = OpenAIClassifier(config)

        # Create vector-enhanced classifier that uses feedback
        classifier = VectorEnhancedClassifier(
            config=config,
            base_classifier=base_classifier,
            vector_service=vector_service,
            embedding_service=embedding_service,
            reranking_service=reranking_service,
        )

        print(
            f"Successfully initialized Vector-Enhanced Classifier with feedback learning!"
        )
        print(f"Milvus connection: {milvus_host}:{milvus_port}")
        print(
            f"Reranking weights: LLM={reranking_config.llm_weight}, Historical={reranking_config.historical_success_weight}"
        )

    except Exception as e:
        print(f"Warning: Could not initialize vector-enhanced classifier: {e}")
        print("Falling back to basic LLM classifier without feedback learning.")
        print("To enable vector-enhanced features, ensure:")
        print("1. Docker containers are running: docker-compose up -d")
        print("2. Milvus is healthy: docker-compose ps")
        print("3. Environment variables are set correctly")

        # Fallback to basic classifier
        classifier = OpenAIClassifier(config)
        vector_service = None
        embedding_service = None
        reranking_service = None

    data = pd.read_csv("data/generated_data.csv")

    print("Welcome to the Fraud Detection System!")
    while True:
        # Get user input
        prompt = input(
            "\nWhat do you want to do? (e.g., 'perform geo mismatch and ip address analysis') or 'exit' to quit: "
        )
        if prompt.lower() == "exit":
            print("Goodbye!")
            break

        # Generate interaction ID
        interaction_id = str(uuid.uuid4())
        start_time = datetime.now()

        # Classify prompt to select agents
        try:
            agents = registry.get_agents()
            if not agents:
                print("No agents registered. Please try again.")
                continue

            print(f"Classifying prompt: '{prompt}'")
            try:
                results = await classifier.classify(
                    prompt, [agent.metadata for agent in agents]
                )
            except Exception as e:
                print(f"Error in enhanced classification: {str(e)}")
                print("Falling back to base LLM classifier")
                # Try with base classifier if available
                if hasattr(classifier, "base_classifier"):
                    try:
                        results = await classifier.base_classifier.classify(
                            prompt, [agent.metadata for agent in agents]
                        )
                    except Exception as e2:
                        print(f"Base classifier also failed: {str(e2)}")
                        print("Please try a different prompt or check your API key.")
                        continue
                else:
                    print("Please try a different prompt or check your API key.")
                    continue

            if not results:
                print("No suitable agents found. Please try again.")
                continue

            # Store initial interaction using enhanced classifier if available
            if (
                hasattr(classifier, "store_interaction_feedback")
                and vector_service
                and embedding_service
            ):
                try:
                    # Extract confidence scores for storage
                    llm_scores = {
                        result.agent.id: result.confidence_score for result in results
                    }
                    final_scores = {
                        result.agent.id: result.confidence_score for result in results
                    }

                    await classifier.store_interaction_feedback(
                        interaction_id=interaction_id,
                        user_prompt=prompt,
                        selected_agents=[result.agent.id for result in results],
                        llm_confidence_scores=llm_scores,
                        final_confidence_scores=final_scores,
                    )
                    print(
                        f"Stored interaction {interaction_id[:8]}... for feedback learning"
                    )
                except Exception as e:
                    print(f"Warning: Could not store interaction: {e}")

            # Execute all selected agents
            print("\nSelected Agents:")
            for result in results:
                agent = registry.get_agent_by_id(result.agent.id)
                print(
                    f"- {agent.name} (Confidence: {result.confidence_score:.2f}, Reasoning: {result.reasoning})"
                )

            # Show feedback enhancement status
            if hasattr(classifier, "reranking_service") and vector_service:
                print("Selections enhanced with historical feedback learning")

            combined_results = []
            executed_agents = []
            for result in results:
                agent = registry.get_agent_by_id(result.agent.id)
                output = await agent.execute(data)
                combined_results.append(
                    {"agent_name": agent.name, "agent_id": agent.id, "output": output}
                )
                executed_agents.append(
                    {
                        "agent_name": agent.name,
                        "agent_id": agent.id,
                        "success": output.success,
                    }
                )

            # Store execution results if vector service is available
            if vector_service:
                try:
                    execution_results = [
                        {
                            "agent_id": agent["agent_id"],
                            "success": agent["success"],
                            "message": combined_results[i]["output"].message,
                            "errors": combined_results[i]["output"].errors,
                            "timestamp": datetime.now().isoformat(),
                        }
                        for i, agent in enumerate(executed_agents)
                    ]
                    await vector_service.update_execution_results(
                        interaction_id, execution_results
                    )
                    print(
                        f"Stored execution results for interaction {interaction_id[:8]}..."
                    )
                except Exception as e:
                    print(f"Warning: Could not store execution results: {e}")

            # Display combined results
            print("\nResults:")
            for res in combined_results:
                output = res["output"]
                print(f"\nAgent: {res['agent_name']}")
                if output.success:
                    print(f"Message: {output.message}")
                    if output.data:
                        print(
                            f"Fraudulent Users: {output.data.get('fraudulent_users', [])}"
                        )
                        print(
                            f"Total Users Analyzed: {output.data.get('total_users_analyzed', 0)}"
                        )
                        print(
                            f"Fraudulent User Count: {output.data.get('fraudulent_user_count', 0)}"
                        )
                else:
                    print(f"Error: {output.message}")
                    print(f"Errors: {output.errors}")

            # Ask if user wants to provide feedback
            feedback_choice = (
                input(
                    "\nWould you like to provide feedback to improve the system? (y/n, default=y): "
                )
                .strip()
                .lower()
            )
            if feedback_choice in ["", "y", "yes"]:
                try:
                    feedback = await collect_user_feedback(
                        interaction_id, executed_agents, prompt
                    )

                    # Store feedback in vector database if services are available
                    if vector_service:
                        # Calculate overall feedback score (average of agent ratings)
                        if feedback.agent_ratings:
                            # Normalize to 0-1 scale: (average_rating - 1) / 4
                            avg_rating = sum(feedback.agent_ratings.values()) / len(
                                feedback.agent_ratings
                            )
                            feedback_score = (avg_rating - 1) / 4
                            await vector_service.update_feedback(
                                interaction_id, feedback_score
                            )
                            print(
                                f"\nThank you for your feedback! It will help improve future recommendations."
                            )
                        else:
                            print(f"\nThank you for your feedback!")
                    else:
                        print(
                            f"\nThank you for your feedback! (Note: Feedback storage is unavailable)"
                        )

                except KeyboardInterrupt:
                    print("\nFeedback collection skipped.")
                except Exception as e:
                    print(f"\nError collecting feedback: {e}")
            else:
                print("\nFeedback skipped. You can provide feedback next time!")

        except Exception as e:
            print(f"\nAn error occurred: {str(e)}")


if __name__ == "__main__":
    asyncio.run(main())
