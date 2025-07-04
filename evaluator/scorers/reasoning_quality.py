import json
import os
import re
from typing import Dict, List, Union, Any, Optional

from dotenv import load_dotenv
from litellm import completion
from pydantic import BaseModel

from evaluator.scorers.base import LLMScorer, ScorerResult
from evaluator.trajectory import Trajectory
from .prompts.prompt import REASONING_QUALITY_SYSTEM_PROMPT, REASONING_QUALITY_USER_PROMPT

load_dotenv()


class ReasoningQualityScore(BaseModel):
    strategic_backtracking: Optional[float]
    task_decomposition: float
    observation_reading: float
    self_verification: float
    
    @property
    def overall(self) -> float:
        scores = [self.task_decomposition, self.observation_reading, self.self_verification]
        if self.strategic_backtracking is not None:
            scores.append(self.strategic_backtracking)
        
        return sum(scores) / len(scores)


class ReasoningQualityScorer(LLMScorer):
    def __init__(self,
                 weight: float = 1.0,
                 name: str = "ReasoningQuality",
                 description: str = "Evaluates the quality of the agent's reasoning process",
                 model: str = "gemini-1.5-pro",
                 max_tokens: int = 2048,
                 temperature: float = 0.0,
                 **kwargs) -> None:
        super().__init__(weight, name, description, model, max_tokens, temperature)

    def generate_prompt(self, trajectory: Trajectory) -> List[Dict[str, str]]:
        prompt = [
            {"role": "system", "content": REASONING_QUALITY_SYSTEM_PROMPT},
            {"role": "user", "content": REASONING_QUALITY_USER_PROMPT.format(
                objective=trajectory.objective,
                steps=self._format_steps(trajectory)
            )}
        ]
        
        return prompt
    
    def _format_steps(self, trajectory: Trajectory) -> str:
        step_text = ""
        
        if not hasattr(trajectory, 'actions') or not trajectory.actions:
            return "No steps recorded."
        
        for i, action in enumerate(trajectory.actions):
            step_text += f"\n--- Step {i+1} ---\n"
            
            if hasattr(action, 'action') and action.action:
                step_text += f"Action: {action.action}\n"
            
            if hasattr(action, 'reasoning') and action.reasoning:
                reasoning_text = action.reasoning
                step_text += f"Reasoning: {reasoning_text}\n"
        
        return step_text
    
    def parse_response(self, response: str) -> ScorerResult:
        try:
            json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                json_str = response
                
            evaluation = json.loads(json_str)
            
            backtrack_score = evaluation["backtrack_and_explore"]["score"]
            if backtrack_score == "N/A":
                backtrack_and_explore = None
            else:
                backtrack_and_explore = float(backtrack_score) / 4.0
            
            task_decomposition = evaluation["task_decomposition"]["score"] / 4.0
            observation_reading = evaluation["observation_reading"]["score"] / 4.0
            self_verification = evaluation["self_verification"]["score"] / 4.0
            
            if backtrack_and_explore is not None:
                overall_score = (
                    backtrack_and_explore + 
                    task_decomposition + 
                    observation_reading + 
                    self_verification
                ) / 4.0
            else:
                overall_score = (
                    task_decomposition + 
                    observation_reading + 
                    self_verification
                ) / 3.0
            
            detailed_score = ReasoningQualityScore(
                strategic_backtracking=backtrack_and_explore,
                task_decomposition=task_decomposition,
                observation_reading=observation_reading,
                self_verification=self_verification
            )
            
            return ScorerResult(
                score=overall_score,
                name=self.name,
                description=self.description,
                details={
                    "reasoning_quality": {
                        "strategic_backtracking": backtrack_and_explore,
                        "task_decomposition": task_decomposition,
                        "observation_reading": observation_reading,
                        "self_verification": self_verification,
                        "raw_scores": {
                            "strategic_backtracking": evaluation["backtrack_and_explore"]["score"],
                            "task_decomposition": evaluation["task_decomposition"]["score"],
                            "observation_reading": evaluation["observation_reading"]["score"],
                            "self_verification": evaluation["self_verification"]["score"]
                        }
                    },
                    "justifications": {
                        "strategic_backtracking": evaluation["backtrack_and_explore"]["justification"],
                        "task_decomposition": evaluation["task_decomposition"]["justification"],
                        "observation_reading": evaluation["observation_reading"]["justification"],
                        "self_verification": evaluation["self_verification"]["justification"]
                    }
                },
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
        
        api_key = os.getenv('LLM_API_KEY')

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
            return self.parse_response(response_text)
        except Exception as e:
            return ScorerResult(
                score=0.0,
                name=self.name,
                description=self.description,
                details={"error": f"Failed to get LLM response: {str(e)}"},
                confidence=0.0
            )