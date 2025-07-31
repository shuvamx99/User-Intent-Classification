from typing import List, Dict, Any, Optional

def parse_conversation_history(history: Optional[List[Dict[str, str]]]) -> str:
    """
    Parse conversation history into a string.

    Args:
        history: List of conversation messages

    Returns:
        Formatted string of conversation history
    """
    if not history:
        return ""
    return "\n".join(f"{msg['role']}: {msg['content']}" for msg in history)