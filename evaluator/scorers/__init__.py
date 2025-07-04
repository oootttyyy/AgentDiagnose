from .base import BaseScorer, LLMScorer, ScorerResult
from .navigation_path_scorer import NavigationPathScorer
from .objective_quality import ObjectiveQualityScorer
from .reasoning_quality import ReasoningQualityScorer


def get_available_scorers():
    return [
        "navigation_path",
        "objective_quality",
        "reasoning_quality",
    ]


def create_scorer(name: str, **kwargs):
    scorers = {
        "navigation_path": NavigationPathScorer,
        "objective_quality": ObjectiveQualityScorer,
        "reasoning_quality": ReasoningQualityScorer,
    }
    
    if name.lower() not in scorers:
        raise ValueError(f"Unknown scorer: {name}")
    
    return scorers[name.lower()](**kwargs) 