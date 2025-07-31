import json
from typing import Any, Dict

def parse_json_output(output: str) -> Dict[str, Any]:
    """
    Parse JSON output from LLM response.

    Args:
        output: Raw string output

    Returns:
        Parsed JSON dictionary
    """
    try:
        return json.loads(output)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON output: {str(e)}")