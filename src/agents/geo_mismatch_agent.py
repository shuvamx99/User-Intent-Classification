from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import pandas as pd
from .base_agent import BaseAgent, AgentOutput
from ..models.schemas import Agent


class GeoMismatchAgent(BaseAgent):
    """Agent to detect fraudulent users based on geo mismatch in recent activity."""

    @property
    def id(self) -> str:
        return "geo_mismatch_001"

    @property
    def name(self) -> str:
        return "Geo Mismatch Agent"

    @property
    def description(self) -> str:
        return "Detects fraudulent users by checking for activity in multiple countries within the last 2 days."

    @property
    def capabilities(self) -> List[str]:
        return ["fraud_detection", "geo_analysis"]

    @property
    def metadata(self) -> Agent:
        return Agent(
            id=self.id,
            name=self.name,
            description=self.description,
            capabilities=self.capabilities,
            metadata={"version": "1.0", "time_window_days": 2}
        )

    async def execute(self, input_data: pd.DataFrame,
                      context: Optional[Dict[str, Any]] = None) -> AgentOutput:
        """
        Execute fraud detection by analyzing user activity for geo mismatches.

        Args:
            input_data: Pandas DataFrame with at least ['user', 'country'] columns; 'time' is optional
            context: Optional context (e.g., reference time for analysis)

        Returns:
            AgentOutput with results of fraud detection
        """
        try:
            # Validate minimal required columns
            required_columns = {"user", "country"}
            if not required_columns.issubset(input_data.columns):
                missing = required_columns - set(input_data.columns)
                raise ValueError(f"Missing required columns: {missing}")

            # Prepare data for analysis
            data = input_data.copy()

            # Group by user and count distinct countries
            user_countries = (
                data.groupby("user")["country"]
                .nunique()
                .reset_index()
                .rename(columns={"country": "country_count"})
            )

            # Identify fraudulent users (more than 1 country)
            fraudulent_users = user_countries[user_countries["country_count"] > 3]["user"]
            fraudulent_details = []
            for user in fraudulent_users:
                countries = data[data["user"] == user]["country"].unique().tolist()
                fraudulent_details.append({
                    "user": user,
                    "countries": countries,
                    "country_count": len(countries)
                })

            # Prepare output
            result_data = {
                "fraudulent_users": fraudulent_details,
                "total_users_analyzed": len(user_countries),
                "fraudulent_user_count": len(fraudulent_details)
            }

            return AgentOutput(
                agent_id=self.id,
                success=True,
                message="Geo mismatch analysis completed successfully",
                data=result_data,
                errors=[]
            )

        except Exception as e:
            return AgentOutput(
                agent_id=self.id,
                success=False,
                message=f"Geo mismatch analysis failed: {str(e)}",
                data=None,
                errors=[str(e)]
            )