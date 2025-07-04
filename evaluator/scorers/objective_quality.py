import json
import os
import re
from collections import defaultdict
from typing import Dict, List, Union, Any, Optional

from dotenv import load_dotenv
from litellm import completion
from openai import OpenAI
from pydantic import BaseModel

from evaluator.scorers.base import LLMScorer, ScorerResult
from evaluator.trajectory import Trajectory
from .prompts.prompt import OBJECTIVE_QUALITY_SYSTEM_PROMPT, OBJECTIVE_QUALITY_USER_PROMPT

load_dotenv()


class ObjectiveQualityScore(BaseModel):
    actionability: float
    
    @property
    def overall(self) -> float:
        return self.actionability


class ObjectiveQualityScorer(LLMScorer):
    def __init__(self,
                 weight: float = 1.0,
                 name: str = "ObjectiveQuality",
                 description: str = "ObjectiveQuality",
                 model: str = "gemini-2.5-pro-exp-03-25",
                 max_tokens: int = 2048,
                 temperature: float = 0.0,
                 **kwargs) -> None:
        super().__init__(weight, name, description, model, max_tokens, temperature)
    
    def generate_prompt(self, trajectory: Trajectory) -> List[Dict[str, str]]:
        prompt = [
            {"role": "system", "content": OBJECTIVE_QUALITY_SYSTEM_PROMPT},
            {"role": "user", "content": OBJECTIVE_QUALITY_USER_PROMPT.format(
                objective=trajectory.objective
            )}
        ]
        
        return prompt
    
    def parse_response(self, response: str, trajectory: Trajectory = None) -> ScorerResult:
        try:
            json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                json_str = response
                
            evaluation = json.loads(json_str)
            
            score = evaluation["score"] / 4.0
            
            detailed_score = ObjectiveQualityScore(
                actionability=score
            )
            
            details = {
                "objective_quality": {
                    "actionability": score,
                    "raw_score": evaluation["score"]
                },
                "justification": evaluation["justification"]
            }
            
            if trajectory is not None:
                details["objective_text"] = trajectory.objective
            
            return ScorerResult(
                score=score,
                name=self.name,
                description=self.description,
                details=details,
                confidence=0.9
            )
        except Exception as e:
            return ScorerResult(
                score=0.0,
                name=self.name,
                description=self.description,
                details={"error": f"Failed to parse LLM response: {str(e)}", "raw_response": response},
                confidence=0.0
            )
    
    def score(self, trajectory: Trajectory) -> ScorerResult:
        prompt = self.generate_prompt(trajectory)
        max_retries = 2
        retry_count = 0
        errors = []
        token_usage = defaultdict(int)
        
        api_key = os.getenv('LLM_API_KEY')
        
        while retry_count <= max_retries:
            response = completion(
                api_key=api_key,
                model=f"{self.model}",
                messages=prompt,
                safety_settings=[
                    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
                ],
            )
            
            try:
                response_text = response.choices[0].message.content
                
                if hasattr(response, 'usage'):
                    token_usage['prompt_tokens'] += response.usage.prompt_tokens
                    token_usage['completion_tokens'] += response.usage.completion_tokens
                    token_usage['total_tokens'] += response.usage.total_tokens
                    print(f"Token usage: {token_usage['prompt_tokens']} prompt + {token_usage['completion_tokens']} completion = {token_usage['total_tokens']} total")
            except Exception as e:
                print(f"Error extracting response data: {e}")
                retry_count += 1
                continue
            
            result = self.parse_response(response_text, trajectory)
            
            if "error" not in result.details:
                if token_usage:
                    result.details['token_usage'] = token_usage
                
                return result
            else:
                retry_count += 1
                continue
                        
        return ScorerResult(
            score=0.0,
            name=self.name,
            description=self.description,
            details={
                "error": f"Failed after {retry_count} attempts",
                "objective_text": trajectory.objective
            },
            confidence=0.0
        )