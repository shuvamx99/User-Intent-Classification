from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from pydantic import BaseModel
import pandas as pd
from .base_agent import BaseAgent, AgentOutput
from ..models.schemas import Agent


class FastCompletionAgent(BaseAgent):
    """Agent to detect fraudulent users based on rapid task completion."""

    @property
    def id(self) -> str:
        return "fast_completion_001"

    @property
    def name(self) -> str:
        return "Fast Completion Agent"

    @property
    def description(self) -> str:
        return "Detects fraudulent users by identifying rapid task completions within the last 2 days."

    @property
    def capabilities(self) -> List[str]:
        return ["fraud_detection", "time_analysis"]

    @property
    def metadata(self) -> Agent:
        return Agent(
            id=self.id,
            name=self.name,
            description=self.description,
            capabilities=self.capabilities,
            metadata={"version": "1.0", "time_window_days": 2, "threshold_hours": 1}
        )

    async def execute(self, input_data: pd.DataFrame,
                context: Optional[Dict[str, Any]] = None) -> AgentOutput:
        """
        Execute fraud detection by analyzing user activity for rapid task completions.

        Args:
            input_data: Pandas DataFrame with columns ['user', 'country', 'ip_address', 'time']
            context: Optional context (e.g., reference time for analysis)

        Returns:
            AgentOutput with results of fraud detection
        """
        try:
            # Validate DataFrame schema
            required_columns = {"user", "country", "ip_address", "time"}
            if not required_columns.issubset(input_data.columns):
                raise ValueError(f"DataFrame must contain columns: {required_columns}")

            # Filter for recent activity
            data = input_data.copy()
            data["time"] = pd.to_datetime(data["time"])

            # Sort by user and time to calculate time differences
            data = data.sort_values(["user", "time"])
            data["time_diff"] = data.groupby("user")["time"].diff().dt.total_seconds() / 3600

            # Identify rapid completions (time difference < 1 hour)
            rapid_completions = data[data["time_diff"] < 1]
            fraudulent_users = rapid_completions["user"].unique()

            fraudulent_details = []
            for user in fraudulent_users:
                user_data = rapid_completions[rapid_completions["user"] == user]
                timestamps = user_data["time"].tolist()
                fraudulent_details.append({
                    "user": user,
                    "rapid_completion_count": len(user_data),
                    "timestamps": timestamps
                })

            # Prepare output
            result_data = {
                "fraudulent_users": fraudulent_details,
                "total_users_analyzed": len(data["user"].unique()),
                "fraudulent_user_count": len(fraudulent_details)
            }

            return AgentOutput(
                agent_id=self.id,
                success=True,
                message="Fast completion analysis completed successfully",
                data=result_data,
                errors=[]
            )

        except Exception as e:
            return AgentOutput(
                agent_id=self.id,
                success=False,
                message=f"Fast completion analysis failed: {str(e)}",
                data=None,
                errors=[str(e)]
            )