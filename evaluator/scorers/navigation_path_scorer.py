import json
import os
import re
import urllib.parse
from collections import defaultdict
from typing import Dict, List, Any, Union, Tuple

from litellm import completion

from evaluator.scorers.base import BaseScorer, ScorerResult
from evaluator.trajectory import Trajectory, Action


class NavigationPathScorer(BaseScorer):
    print_details = True
    
    def __init__(self,
                 weight: float = 1.0,
                 name: str = "NavigationPath",
                 description: str = "Evaluates the navigation path based on length, backtracking, domain transitions, and page revisits",
                 model: str = "gemini-1.5-pro",
                 max_tokens: int = 2048,
                 temperature: float = 0.0,
                 **kwargs) -> None:
        super().__init__(weight=weight, name=name, description=description)
    
    def score(self, trajectory: Trajectory) -> ScorerResult:
        actions = trajectory.actions
        
        if not actions:
            return ScorerResult(
                score=0.0,
                name=self.name,
                description=self.description,
                details={"error": "No actions found in trajectory"},
                confidence=0.0,
                weight=self.weight
            )
        
        navigation_metrics = self._analyze_navigation_path(actions)
        
        return ScorerResult(
            score=0.0,
            name=self.name,
            description=self.description,
            details={
                "metrics": navigation_metrics,
                "overall_length": navigation_metrics["overall_length"],
                "backtrack_count": navigation_metrics["backtrack_count"],
                "domain_transitions": navigation_metrics["domain_transitions"],
                "unique_urls_visited": navigation_metrics["unique_urls_visited"],
                "urls_in_order": navigation_metrics["urls_in_order"],
                "actions_per_url_segment": navigation_metrics["actions_per_url_segment"],
                "actions_per_url": navigation_metrics["actions_per_url"]
            },
            confidence=0.8,
            weight=self.weight
        )
    
    def _analyze_navigation_path(self, actions: List[Action]) -> Dict[str, Any]:
        metrics = {
            "overall_length": 0,
            "backtrack_count": 0,
            "domain_transitions": [],
            "unique_urls_visited": 0,
            "urls_in_order": [],
            "actions_per_url_segment": [],
            "actions_per_url": defaultdict(int)
        }
        
        previous_url = None
        visited_urls = set()
        url_history = []
        current_segment_actions = []

        if self.print_details:
            print("\n" + "="*80)
            print("Navigation Path Analysis")
            print("="*80)
        
        for i, action in enumerate(actions):
            url = self._extract_url(action)
            
            if not url:
                continue
                
            metrics["overall_length"] += 1
            url_history.append(url)
            
            if url != previous_url:
                if previous_url is not None and current_segment_actions:
                    metrics["actions_per_url_segment"].append(current_segment_actions.copy())
                
                metrics["urls_in_order"].append(url)
                current_segment_actions = []
                
                if url in visited_urls:
                    metrics["backtrack_count"] += 1
            
            action_str = str(getattr(action, 'action', ''))
            current_segment_actions.append(action_str)
            
            visited_urls.add(url)
            
            metrics["actions_per_url"][url] += 1
            
            domain = self._extract_domain(url)
            
            if previous_url and domain:
                prev_domain = self._extract_domain(previous_url)
                if prev_domain != domain:
                    metrics["domain_transitions"].append((prev_domain, domain))
            
            if self.print_details:
                print(f"\nStep {i+1}:")
                print(f"URL: {url}")
                if url != previous_url and url in visited_urls and previous_url is not None:
                    print("Backtrack: Yes (returning to previously visited URL)")
                else:
                    print("Backtrack: No")
                print(f"Actions on this URL so far: {metrics['actions_per_url'][url]}")
                print("-"*80)
            
            previous_url = url
        
        if previous_url is not None and current_segment_actions:
            metrics["actions_per_url_segment"].append(current_segment_actions)
        
        metrics["unique_urls_visited"] = len(visited_urls)
        
        metrics["actions_per_url"] = dict(metrics["actions_per_url"])
        
        return metrics
    
    def _extract_url(self, action: Action) -> str:
        if hasattr(action, 'url'):
            return action.url
        
        if hasattr(action, 'action') and isinstance(action.action, dict):
            return action.action.get('url', '')
        
        return ""
    
    def _extract_domain(self, url: str) -> str:
        try:
            parsed = urllib.parse.urlparse(url)
            return parsed.netloc
        except:
            return "" 