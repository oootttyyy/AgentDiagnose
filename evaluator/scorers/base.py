import json
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

import tiktoken
from pydantic import BaseModel, Field

from evaluator.trajectory import Trajectory


class ScorerResult(BaseModel):    
    score: float
    name: str
    description: str
    details: Dict[str, Any]
    confidence: float
    weight: float = 1.0
    
    @property
    def weighted_score(self) -> float:
        return self.score * self.weight
    
    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()


class BaseScorer(ABC):
    def __init__(self, 
                 weight: float = 1.0,
                 name: Optional[str] = None,
                 description: Optional[str] = None) -> None:
        self.weight = weight
        self._name = name or self.__class__.__name__
        self._description = description or ""
    
    @property
    def name(self) -> str:
        return self._name
    
    @property
    def description(self) -> str:
        return self._description
    
    @abstractmethod
    def score(self, trajectory: Trajectory) -> ScorerResult:
        pass
    
    def __call__(self, trajectory: Trajectory) -> ScorerResult:
        result = self.score(trajectory)
        result.weight = self.weight
        result.name = self.name
        result.description = self.description
        return result


class LLMScorer(BaseScorer):
    def __init__(self,
                 weight: float = 1.0,
                 name: Optional[str] = None,
                 description: Optional[str] = None,
                 model: str = "gpt-4o",
                 max_tokens: int = 2048,
                 temperature: float = 0.0) -> None:
        super().__init__(weight, name, description)
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
    
    @abstractmethod
    def generate_prompt(self, trajectory: Trajectory) -> Union[str, List[Dict[str, str]]]:
        pass
    
    @abstractmethod
    def parse_response(self, response: str) -> ScorerResult:
        pass
        
    def count_tokens(self, prompt: Union[str, List[Dict[str, str]]]) -> int:
        encoder = tiktoken.get_encoding("cl100k_base")
        
        if isinstance(prompt, str):
            return len(encoder.encode(prompt))
        else:
            token_count = 0
            for message in prompt:
                if 'content' in message:
                    token_count += len(encoder.encode(message['content']))
            return token_count
    
    def dry_run(self, trajectory: Trajectory) -> ScorerResult:
        prompt = self.generate_prompt(trajectory)        
        token_count = self.count_tokens(prompt)
        return ScorerResult(
            score=0.0,
            name=self.name,
            description=self.description,
            details={"dry_run": True, "token_count": token_count, "prompt": prompt},
            confidence=0.0,
            weight=self.weight
        )
       