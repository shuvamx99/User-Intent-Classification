from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from pydantic import BaseModel
import pandas as pd
from .base_agent import BaseAgent, AgentOutput
from ..models.schemas import Agent


class IPAddressAgent(BaseAgent):
    """Agent to detect fraudulent users based on multiple IP addresses in recent activity."""

    @property
    def id(self) -> str:
        return "ip_address_001"

    @property
    def name(self) -> str:
        return "IP Address Agent"

    @property
    def description(self) -> str:
        return "Detects fraudulent users by checking for multiple IP addresses within the last 2 days."

    @property
    def capabilities(self) -> List[str]:
        return ["fraud_detection", "ip_analysis"]

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
        Execute fraud detection by analyzing user activity for multiple IP addresses.

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

            data = input_data.copy()

            # Group by user and count distinct IP addresses
            user_ips = (
                data.groupby("user")["ip_address"]
                .nunique()
                .reset_index()
                .rename(columns={"ip_address": "ip_count"})
            )

            # Identify fraudulent users (more than 1 IP address)
            fraudulent_users = user_ips[user_ips["ip_count"] > 3]["user"]
            fraudulent_details = []
            for user in fraudulent_users:
                ip_addresses = data[data["user"] == user]["ip_address"].unique().tolist()
                fraudulent_details.append({
                    "user": user,
                    "ip_addresses": ip_addresses,
                    "ip_count": len(ip_addresses)
                })

            # Prepare output
            result_data = {
                "fraudulent_users": fraudulent_details,
                "total_users_analyzed": len(user_ips),
                "fraudulent_user_count": len(fraudulent_details)
            }

            return AgentOutput(
                agent_id=self.id,
                success=True,
                message="IP address analysis completed successfully",
                data=result_data,
                errors=[]
            )

        except Exception as e:
            return AgentOutput(
                agent_id=self.id,
                success=False,
                message=f"IP address analysis failed: {str(e)}",
                data=None,
                errors=[str(e)]
            )
