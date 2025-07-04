import json
import os
from typing import Dict, List, Optional, Union, Any, Type

from pydantic import BaseModel, Field

from evaluator.scorers.base import BaseScorer, ScorerResult
from evaluator.trajectory import Trajectory


class EvaluationResult(BaseModel):
    trajectory_id: str = Field(..., description="ID of the evaluated trajectory")
    overall_score: float = Field(..., description="Overall aggregated score (0.0 to 1.0)")
    weighted_average: float = Field(..., description="Weighted average of all scores")
    detailed_scores: List[ScorerResult] = Field(
        ..., description="Detailed scores from individual scorers"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional evaluation metadata"
    )
    
    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()
    
    def to_json(self, file_path: str = None) -> Union[str, None]:
        json_data = json.dumps(self.to_dict(), indent=2)
        if file_path:
            with open(file_path, "w") as f:
                f.write(json_data)
            return None
        return json_data
    
    def print_summary(self) -> None:
        print(f"Evaluation of trajectory: {self.trajectory_id}")
        print(f"Overall score: {self.overall_score:.2f}")
        print(f"Weighted average: {self.weighted_average:.2f}")
        print("\nDetailed scores:")
        
        for score in self.detailed_scores:
            print(f"  {score.name}: {score.score:.2f} (weight: {score.weight:.2f})")
        
        print("\nMetadata:")
        for key, value in self.metadata.items():
            print(f"  {key}: {value}")


class TrajectoryJudge:
    def __init__(self, 
                scorers: List[BaseScorer],
                name: str = "TrajectoryJudge",
                aggregation_method: str = "weighted_average") -> None:
        self.scorers = scorers
        self.name = name
        self.aggregation_method = aggregation_method
        
        if not scorers:
            raise ValueError("At least one scorer must be provided")
        
        if aggregation_method not in ("weighted_average", "min", "max"):
            raise ValueError(
                f"Invalid aggregation method: {aggregation_method}. "
                "Must be one of 'weighted_average', 'min', 'max'"
            )
    
    def evaluate(self, 
                trajectory: Union[str, Trajectory],
                metadata: Dict[str, Any] = None) -> EvaluationResult:
        if isinstance(trajectory, str):
            if os.path.exists(trajectory):
                trajectory = Trajectory.from_file(trajectory)
            else:
                raise ValueError(f"Trajectory file not found: {trajectory}")
        
        detailed_scores = []
        for scorer in self.scorers:
            result = scorer(trajectory)
            detailed_scores.append(result)
        
        if self.aggregation_method == "weighted_average":
            total_weight = sum(s.weight for s in detailed_scores)
            if total_weight == 0:
                weighted_average = 0.0
            else:
                weighted_average = sum(s.weighted_score for s in detailed_scores) / total_weight
            overall_score = weighted_average
        elif self.aggregation_method == "min":
            weighted_average = sum(s.weighted_score for s in detailed_scores) / sum(s.weight for s in detailed_scores)
            overall_score = min(s.score for s in detailed_scores)
        elif self.aggregation_method == "max":
            weighted_average = sum(s.weighted_score for s in detailed_scores) / sum(s.weight for s in detailed_scores)
            overall_score = max(s.score for s in detailed_scores)
        
        return EvaluationResult(
            trajectory_id=trajectory.id,
            overall_score=overall_score,
            weighted_average=weighted_average,
            detailed_scores=detailed_scores,
            metadata=metadata or {}
        )
    
    def evaluate_batch(self, 
                     trajectories: List[Union[str, Trajectory]],
                     output_dir: Optional[str] = None,
                     metadata: Dict[str, Any] = None) -> List[EvaluationResult]:
        results = []
        for traj in trajectories:
            result = self.evaluate(traj, metadata)
            results.append(result)
            
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                traj_id = result.trajectory_id
                result.to_json(os.path.join(output_dir, f"{traj_id}_result.json"))
        
        return results
    
    def add_scorer(self, scorer: BaseScorer) -> None:
        self.scorers.append(scorer)
    
    def get_scorer_by_name(self, name: str) -> Optional[BaseScorer]:
        for scorer in self.scorers:
            if scorer.name == name:
                return scorer
        return None
    
    @classmethod
    def from_config(cls, config_path: str) -> "TrajectoryJudge":
        with open(config_path, "r") as f:
            config = json.load(f)
        
        scorers = []
        for scorer_config in config.get("scorers", []):
            module_path = scorer_config["module"]
            class_name = scorer_config["class"]
            
            module = __import__(module_path, fromlist=[class_name])
            scorer_class = getattr(module, class_name)
            
            scorer_args = scorer_config.get("args", {})
            scorer = scorer_class(**scorer_args)
            scorers.append(scorer)
        
        return cls(
            scorers=scorers,
            name=config.get("name", "TrajectoryJudge"),
            aggregation_method=config.get("aggregation_method", "weighted_average")
        )