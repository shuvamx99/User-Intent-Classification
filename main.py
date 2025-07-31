import asyncio
import pandas as pd
import os
from datetime import datetime
from src.classifiers.openai_classifier import OpenAIClassifier
from src.models.schemas import ClassifierConfig
from src.agents.registry import AgentRegistry
from src.agents.geo_mismatch_agent import GeoMismatchAgent
from src.agents.ip_address_agent import IPAddressAgent
from src.agents.fast_completion_agent import FastCompletionAgent


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
        confidence_threshold=0.7,
        include_conversation_history=False,
        include_memory=False
    )
    classifier = OpenAIClassifier(config)

    data = pd.read_csv("data/generated_data.csv")


    print("Welcome to the Fraud Detection System!")
    while True:
        # Get user input
        prompt = input("\nWhat do you want to do? (e.g., 'perform geo mismatch and ip address analysis') or 'exit' to quit: ")
        if prompt.lower() == "exit":
            print("Goodbye!")
            break

        # Classify prompt to select agents
        try:
            agents = registry.get_agents()
            if not agents:
                print("No agents registered. Please try again.")
                continue
            results = await classifier.classify(prompt, [agent.metadata for agent in agents])
            if not results:
                print("No suitable agents found. Please try again.")
                continue

            # Execute all selected agents
            print("\nSelected Agents:")
            for result in results:
                agent = registry.get_agent_by_id(result.agent.id)
                print(f"- {agent.name} (Confidence: {result.confidence_score:.2f}, Reasoning: {result.reasoning})")

            combined_results = []
            for result in results:
                agent = registry.get_agent_by_id(result.agent.id)
                output = await agent.execute(data)
                combined_results.append({
                    "agent_name": agent.name,
                    "output": output
                })

            # Display combined results
            print("\nResults:")
            for res in combined_results:
                output = res["output"]
                print(f"\nAgent: {res['agent_name']}")
                if output.success:
                    print(f"Message: {output.message}")
                    if output.data:
                        print(f"Fraudulent Users: {output.data.get('fraudulent_users', [])}")
                        print(f"Total Users Analyzed: {output.data.get('total_users_analyzed', 0)}")
                        print(f"Fraudulent User Count: {output.data.get('fraudulent_user_count', 0)}")
                else:
                    print(f"Error: {output.message}")
                    print(f"Errors: {output.errors}")

        except Exception as e:
            print(f"\nAn error occurred: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main())
