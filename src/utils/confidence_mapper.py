from src.models.enums import ConfidenceLevel

def map_confidence_score(score: float, high_threshold: float = 0.8,
                         medium_threshold: float = 0.5, low_threshold: float = 0.2) -> ConfidenceLevel:
    """
    Map a confidence score to a ConfidenceLevel.

    Args:
        score: Confidence score (0 to 1)
        high_threshold: Threshold for HIGH confidence
        medium_threshold: Threshold for MEDIUM confidence
        low_threshold: Threshold for LOW confidence

    Returns:
        ConfidenceLevel enum
    """
    if score >= high_threshold:
        return ConfidenceLevel.HIGH
    elif score >= medium_threshold:
        return ConfidenceLevel.MEDIUM
    elif score >= low_threshold:
        return ConfidenceLevel.LOW
    else:
        return ConfidenceLevel.UNSUITABLE