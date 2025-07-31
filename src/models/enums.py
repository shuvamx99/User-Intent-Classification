from enum import Enum, auto


class ConfidenceLevel(str, Enum):
    """
    Enum of classifer confidence Levels
    """
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    UNSUITABLE = "UNSUITABLE"

