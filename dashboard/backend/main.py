"""
FastAPI backend for the trajectory dashboard.
Refactored from web_dashboard.py to separate backend from frontend.
"""

import os
import json
import webbrowser
import threading
import time
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import pandas as pd

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    import plotly.offline as pyo
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    import numpy as np
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import pickle
    PICKLE_AVAILABLE = True
except ImportError:
    PICKLE_AVAILABLE = False

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from evaluator.scorers.base import ScorerResult


class FilterRequest(BaseModel):
    criteria: str  # Human-readable description of filter criteria
    filtered_trajectories: List[str]  # List of trajectory IDs that pass this filter
    source: str = "unknown"  # Which component/tab created this filter
    metadata: Optional[Dict[str, Any]] = None  # Additional metadata about the filter


class DashboardAPI:
    """FastAPI backend for trajectory dashboard."""
    
    def __init__(self):
        if not PLOTLY_AVAILABLE:
            raise ImportError("Plotly is required for interactive charts. Install with: pip install plotly")
            
        self.app = FastAPI(title="Trajectory Inspector API", version="1.0.0")
        self.results_data = {}
        self.trajectories_data = {}
        self.original_trajectories = {}  # Store original trajectory objects
        self.active_filters = {}  # Store active plot filters
        self._setup_middleware()
        self._setup_routes()
    
    def _setup_middleware(self):
        """Setup CORS middleware for frontend communication."""
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # Allow all origins in development
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    
    def _setup_routes(self):
        """Setup API routes for the dashboard."""
        
        @self.app.get("/api/results")
        async def api_results():
            """API endpoint for results data."""
            return self.results_data
        
        @self.app.get("/api/trajectories")
        async def api_trajectories(
            page: int = None,
            page_size: int = None,
            include_actions: bool = False,
            summary_only: bool = True
        ):
            """API endpoint for trajectory data with optional pagination."""
            if not self.trajectories_data:
                return {
                    'trajectories': {},
                    'total': 0
                }
            
            # Get trajectory IDs (apply filters if any)
            if self.active_filters:
                filtered_ids = self._get_filtered_trajectory_ids()
                traj_ids = filtered_ids
            else:
                traj_ids = list(self.trajectories_data.keys())
            
            total = len(traj_ids)
            
            # Only apply pagination if both page and page_size are provided
            if page is not None and page_size is not None:
                total_pages = (total + page_size - 1) // page_size
                start_idx = (page - 1) * page_size
                end_idx = start_idx + page_size
                page_traj_ids = traj_ids[start_idx:end_idx]
                
                pagination_info = {
                    'page': page,
                    'page_size': page_size,
                    'total_pages': total_pages,
                    'has_next': page < total_pages,
                    'has_prev': page > 1
                }
            else:
                # Return all trajectories without pagination
                page_traj_ids = traj_ids
                pagination_info = {}
            
            # Build response data
            trajectories = {}
            for traj_id in page_traj_ids:
                if traj_id in self.trajectories_data:
                    traj_data = self.trajectories_data[traj_id].copy()
                    
                    if summary_only:
                        # Remove large fields for summary view
                        if 'actions' in traj_data:
                            # Keep only action count and basic info
                            action_count = len(traj_data['actions'])
                            traj_data['num_actions'] = action_count
                            
                            # Keep a preview of first action for display
                            if action_count > 0:
                                first_action = traj_data['actions'][0].copy()
                                # Remove large fields from preview
                                first_action.pop('observation', None)
                                first_action.pop('image_encoding', None)
                                first_action.pop('reasoning', None)
                                traj_data['first_action_preview'] = first_action
                            
                            # Remove full actions array to save memory
                            if not include_actions:
                                del traj_data['actions']
                    
                    trajectories[traj_id] = traj_data
            
            response = {
                'trajectories': trajectories,
                'total': total
            }
            
            # Add pagination info only if pagination was requested
            response.update(pagination_info)
            
            return response
        
        @self.app.get("/api/trajectories/rewards")
        async def api_trajectory_rewards():
            """API endpoint for trajectory reward data only."""
            rewards = {}
            
            # Debug information
            print(f"DEBUG: Rewards endpoint called")
            print(f"DEBUG: original_trajectories keys: {list(self.original_trajectories.keys()) if self.original_trajectories else 'None'}")
            print(f"DEBUG: trajectories_data keys: {list(self.trajectories_data.keys()) if self.trajectories_data else 'None'}")
            
            # First try to get rewards from original trajectories
            if self.original_trajectories:
                print(f"DEBUG: Using original_trajectories ({len(self.original_trajectories)} trajectories)")
                for traj_id, trajectory in self.original_trajectories.items():
                    if isinstance(trajectory, dict):
                        reward = trajectory.get('reward', -1)
                        print(f"DEBUG: Trajectory {traj_id} (dict): reward = {reward}")
                        rewards[traj_id] = reward
                    else:
                        # Handle trajectory objects
                        reward = getattr(trajectory, 'reward', -1)
                        print(f"DEBUG: Trajectory {traj_id} (object): reward = {reward}")
                        rewards[traj_id] = reward
            elif self.trajectories_data:
                print(f"DEBUG: Using trajectories_data ({len(self.trajectories_data)} trajectories)")
                # Fallback to processed trajectory data
                for traj_id, trajectory_data in self.trajectories_data.items():
                    if isinstance(trajectory_data, dict):
                        reward = trajectory_data.get('reward', -1)
                        print(f"DEBUG: Trajectory {traj_id} (dict): reward = {reward}")
                        rewards[traj_id] = reward
                    else:
                        reward = getattr(trajectory_data, 'reward', -1)
                        print(f"DEBUG: Trajectory {traj_id} (object): reward = {reward}")
                        rewards[traj_id] = reward
            else:
                print("DEBUG: No trajectory data available")
            
            print(f"DEBUG: Final rewards: {rewards}")
            return rewards
        
        @self.app.get("/api/trajectories/{traj_id}")
        async def api_trajectory_detail(traj_id: str, include_actions: bool = True):
            """Get detailed information about a specific trajectory."""
            if traj_id not in self.trajectories_data:
                raise HTTPException(status_code=404, detail="Trajectory not found")
            
            traj_data = self.trajectories_data[traj_id].copy()
            
            if not include_actions and 'actions' in traj_data:
                # Keep action count but remove full actions
                traj_data['num_actions'] = len(traj_data['actions'])
                del traj_data['actions']
            elif include_actions and 'actions' in traj_data:
                # Keep original actions with full data including screenshots for now
                # This maintains backward compatibility
                actions = []
                for action in traj_data['actions']:
                    action_data = action.copy()
                    # Keep all original fields including image_encoding
                    actions.append(action_data)
                traj_data['actions'] = actions
            
            return traj_data
        
        @self.app.get("/api/trajectories/{traj_id}/actions")
        async def api_trajectory_actions(traj_id: str, include_screenshots: bool = False):
            """Get all actions for a specific trajectory."""
            if traj_id not in self.trajectories_data:
                raise HTTPException(status_code=404, detail="Trajectory not found")
            
            traj_data = self.trajectories_data[traj_id]
            if 'actions' not in traj_data:
                return {'actions': []}
            
            actions = []
            for i, action in enumerate(traj_data['actions']):
                action_data = action.copy()
                
                # Remove screenshot by default to save bandwidth
                if not include_screenshots:
                    has_screenshot = 'image_encoding' in action_data and action_data['image_encoding']
                    action_data.pop('image_encoding', None)
                    action_data['has_screenshot'] = has_screenshot
                
                actions.append(action_data)
            
            return {'actions': actions}
        
        @self.app.get("/api/trajectories/{traj_id}/actions/{action_index}")
        async def api_trajectory_action_detail(traj_id: str, action_index: int):
            """Get detailed information about a specific action including full observation."""
            if traj_id not in self.trajectories_data:
                raise HTTPException(status_code=404, detail="Trajectory not found")
            
            traj_data = self.trajectories_data[traj_id]
            if 'actions' not in traj_data or action_index >= len(traj_data['actions']):
                raise HTTPException(status_code=404, detail="Action not found")
            
            action = traj_data['actions'][action_index].copy()
            # Remove screenshot from this endpoint (use separate screenshot endpoint)
            has_screenshot = 'image_encoding' in action and action['image_encoding']
            action.pop('image_encoding', None)
            action['has_screenshot'] = has_screenshot
            
            return action
        
        @self.app.get("/api/trajectories/{traj_id}/actions/{action_index}/screenshot")
        async def api_trajectory_screenshot(traj_id: str, action_index: int):
            """Get screenshot for a specific action."""
            if traj_id not in self.trajectories_data:
                raise HTTPException(status_code=404, detail="Trajectory not found")
            
            traj_data = self.trajectories_data[traj_id]
            if 'actions' not in traj_data or action_index >= len(traj_data['actions']):
                raise HTTPException(status_code=404, detail="Action not found")
            
            action = traj_data['actions'][action_index]
            if 'image_encoding' not in action or not action['image_encoding']:
                raise HTTPException(status_code=404, detail="Screenshot not available")
            
            return {
                'image_encoding': action['image_encoding'],
                'step': action.get('step', action_index + 1)
            }
        
        @self.app.get("/api/plot/{plot_type}")
        async def api_plot(plot_type: str):
            """Generate and return plot data."""
            if plot_type == 'overview':
                return self._generate_overview_plot()
            elif plot_type == 'reasoning':
                return self._generate_reasoning_quality_plot()
            elif plot_type == 'objective':
                return self._generate_objective_quality_plot()
            else:
                raise HTTPException(status_code=400, detail='Unknown plot type')
        
        @self.app.get("/api/plot/scorer/{scorer_name}")
        async def api_scorer_plot(scorer_name: str):
            """Generate and return plot data for a specific scorer."""
            try:
                # Route to specific plot functions for known scorers
                if 'reasoning' in scorer_name.lower():
                    return self._generate_reasoning_quality_plot()
                elif 'objective' in scorer_name.lower():
                    return self._generate_objective_quality_plot()
                
            except Exception as e:
                raise HTTPException(status_code=500, detail=f'Failed to generate plot for {scorer_name}: {str(e)}')
        
        @self.app.get("/api/filters")
        async def api_get_filters():
            """Get all active filters."""
            return {
                'filters': list(self.active_filters.values()),
                'count': len(self.active_filters)
            }
        
        @self.app.post("/api/filters")
        async def api_add_filter(filter_request: FilterRequest):
            """Add a new filter."""
            # Generate filter ID
            filter_id = f"filter_{len(self.active_filters) + 1}"
            
            # Create filter
            filter_data = {
                'filter_id': filter_id,
                'criteria': filter_request.criteria,
                'filtered_trajectories': filter_request.filtered_trajectories,
                'source': filter_request.source,
                'metadata': filter_request.metadata or {},
                'trajectory_count': len(filter_request.filtered_trajectories)
            }
            
            self.active_filters[filter_id] = filter_data
            
            return {
                'success': True,
                'filter_id': filter_id,
                'filter': filter_data,
                'total_filters': len(self.active_filters)
            }
        
        @self.app.delete("/api/filters/{filter_id}")
        async def api_remove_filter(filter_id: str):
            """Remove a filter by ID."""
            if filter_id not in self.active_filters:
                raise HTTPException(status_code=404, detail='Filter not found')
            
            removed_filter = self.active_filters.pop(filter_id)
            
            return {
                'success': True,
                'removed_filter': removed_filter,
                'total_filters': len(self.active_filters)
            }
        
        @self.app.delete("/api/filters")
        async def api_clear_filters():
            """Clear all filters."""
            self.active_filters.clear()
            
            return {
                'success': True,
                'total_filters': 0
            }
        
        @self.app.get("/api/embeddings/data")
        async def api_embeddings_data():
            """Get embedding data for visualization."""
            return self._get_embedding_data()
        
        @self.app.get("/api/embeddings/tsne")
        async def api_embeddings_tsne(
            data_type: str = "verbs",  # verbs, nouns, or both
            aggregation: str = "steps",  # steps or trajectories
            perplexity: int = 30,
            n_components: int = 2
        ):
            """Generate t-SNE visualization data for embeddings."""
            return self._generate_tsne_plot(data_type, aggregation, perplexity, n_components)
        
        @self.app.post("/api/embeddings/filter")
        async def api_embedding_filter(request: dict):
            """Filter trajectories based on selected embedding points."""
            selected_points = request.get('selectedPoints', [])
            plot_data = request.get('plotData', {})
            aggregation = request.get('aggregation', 'trajectories')
            
            if not selected_points or not plot_data:
                return {"filtered_trajectories": []}
            
            # Get trajectory IDs for selected points
            filtered_trajectories = []
            
            texts = plot_data.get('texts', [])
            metadata = plot_data.get('metadata', [])
            
            for point_idx in selected_points:
                if 0 <= point_idx < len(metadata):
                    # Use metadata to get trajectory file name
                    meta = metadata[point_idx]
                    
                    if isinstance(meta, dict) and 'trajectory_file' in meta:
                        traj_id = meta['trajectory_file']
                        
                        # Clean trajectory ID - remove .json extension if present
                        if traj_id.endswith('.json'):
                            traj_id = traj_id[:-5]
                        
                        if traj_id not in filtered_trajectories:
                            filtered_trajectories.append(traj_id)
                elif 0 <= point_idx < len(texts):
                    # Fallback: extract from text
                    text = texts[point_idx]
                    
                    if aggregation == 'trajectories':
                        # Format: "Trajectory: {traj_id}"
                        if isinstance(text, str) and text.startswith("Trajectory: "):
                            traj_id = text.replace("Trajectory: ", "").strip()
                            
                            # Clean trajectory ID - remove .json extension if present
                            if traj_id.endswith('.json'):
                                traj_id = traj_id[:-5]
                            
                            if traj_id not in filtered_trajectories:
                                filtered_trajectories.append(traj_id)
                    else:
                        # For step-level aggregation, try to extract trajectory ID from text
                        if isinstance(text, str):
                            import re
                            patterns = [
                                r'trajectory[_\s]*(\d+)',  # trajectory_123 or trajectory 123
                                r'traj[_\s]*(\d+)',       # traj_123 or traj 123
                                r'(\d+)\.json',           # 123.json
                                r'(\d+)',                 # any number
                            ]
                            
                            for pattern in patterns:
                                match = re.search(pattern, text, re.IGNORECASE)
                                if match:
                                    traj_id = match.group(1) if len(match.groups()) > 0 else match.group(0)
                                    
                                    # Clean and add
                                    if traj_id.endswith('.json'):
                                        traj_id = traj_id[:-5]
                                    
                                    if traj_id not in filtered_trajectories:
                                        filtered_trajectories.append(traj_id)
                                    break
            
            # Check what trajectory IDs are available in the system
            available_traj_ids = list(self.results_data.get('trajectories', {}).keys()) if self.results_data else []
            
            # Find exact matches
            matching_trajectories = [tid for tid in filtered_trajectories if tid in available_traj_ids]
            
            # If no exact matches, try fuzzy matching
            if not matching_trajectories and filtered_trajectories and available_traj_ids:
                fuzzy_matches = []
                
                for ftraj in filtered_trajectories:
                    for avail_traj in available_traj_ids:
                        # Try substring matching
                        if str(ftraj) in str(avail_traj) or str(avail_traj) in str(ftraj):
                            if avail_traj not in fuzzy_matches:
                                fuzzy_matches.append(avail_traj)
                            break
                        # Try numeric matching (extract numbers)
                        import re
                        ftraj_nums = re.findall(r'\d+', str(ftraj))
                        avail_nums = re.findall(r'\d+', str(avail_traj))
                        if ftraj_nums and avail_nums and ftraj_nums[0] == avail_nums[0]:
                            if avail_traj not in fuzzy_matches:
                                fuzzy_matches.append(avail_traj)
                            break
                
                if fuzzy_matches:
                    matching_trajectories = fuzzy_matches
            
            # Create filter using the new add_filter API
            data_type = request.get('dataType', 'both')
            
            # Create shorter, more informative criteria
            type_short = {"verbs": "V", "nouns": "N", "both": "V+N"}[data_type]
            agg_short = {"steps": "Steps", "trajectories": "Trajs"}[aggregation]
            criteria = f"Embedding {type_short} ({agg_short}): {len(matching_trajectories)}"
            
            filter_request = FilterRequest(
                criteria=criteria,
                filtered_trajectories=matching_trajectories,
                source="embedding_visualization",
                metadata={
                    "data_type": data_type,
                    "aggregation": aggregation,
                    "selected_points": selected_points,
                    "original_filtered_count": len(filtered_trajectories),
                    "matched_count": len(matching_trajectories)
                }
            )
            
            # Add the filter to the system
            response = await api_add_filter(filter_request)
            
            return {
                "filtered_trajectories": matching_trajectories,
                "filter_id": response.get("filter_id"),
                "total_filters": response.get("total_filters"),
                "debug": {
                    "original_count": len(filtered_trajectories),
                    "matched_count": len(matching_trajectories),
                    "available_count": len(available_traj_ids)
                }
            }
        
        @self.app.get("/api/reasoning/tagcloud")
        async def api_reasoning_tagcloud(ngram_size: int = 1):
            """Get reasoning tag cloud data from pre-generated JSON files."""
            return self._get_reasoning_tagcloud_data(ngram_size)
        
        @self.app.get("/api/action-phrases/tagcloud")
        async def api_action_phrases_tagcloud(phrase_type: str = "pairs"):
            """Load action phrases tag cloud data from pre-generated JSON files."""
            return self._get_action_phrases_tagcloud_data(phrase_type)
    
    def mount_static_files(self, static_dir: Path):
        """Mount static files for production deployment."""
        if static_dir.exists():
            # Mount static assets first
            assets_dir = static_dir / "_app"
            if assets_dir.exists():
                self.app.mount("/_app", StaticFiles(directory=assets_dir), name="assets")
            
            # Mount the main static files with HTML fallback for SPA
            self.app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")
    
    def load_results(self, results: Dict[str, Dict[str, ScorerResult]], trajectories: Optional[Dict] = None):
        """Load evaluation results into the dashboard."""
        self.results_data = self._process_results(results)
        if trajectories:
            self.original_trajectories = trajectories  # Store original trajectories
            self.trajectories_data = self._process_trajectories(trajectories)
    
    def _process_results(self, results: Dict[str, Dict[str, ScorerResult]]) -> Dict:
        """Process raw results into dashboard format."""
        processed = {
            'summary': {
                'total_trajectories': len(results),
                'scorers_used': set(),
                'avg_scores': {},
                'score_ranges': {}
            },
            'trajectories': {}
        }
        
        all_scores = {}
        
        for traj_id, scorer_results in results.items():
            processed['trajectories'][traj_id] = {
                'id': traj_id,
                'scores': {},
                'sub_scores': {}
            }
            
            for scorer_name, result in scorer_results.items():
                processed['summary']['scorers_used'].add(scorer_name)
                
                # Store complete ScorerResult object (as dict) to preserve details
                processed['trajectories'][traj_id]['scores'][scorer_name] = {
                    'score': result.score,
                    'name': result.name,
                    'description': result.description,
                    'details': result.details if hasattr(result, 'details') else {},
                    'confidence': result.confidence if hasattr(result, 'confidence') else 0.0,
                    'weight': result.weight if hasattr(result, 'weight') else 1.0
                }
                
                # Store sub-scores if available
                if hasattr(result, 'sub_scores') and result.sub_scores:
                    processed['trajectories'][traj_id]['sub_scores'][scorer_name] = result.sub_scores
                
                # Track all scores for summary statistics
                if scorer_name not in all_scores:
                    all_scores[scorer_name] = []
                all_scores[scorer_name].append(result.score)
        
        # Convert set to list for JSON serialization
        processed['summary']['scorers_used'] = list(processed['summary']['scorers_used'])
        
        # Calculate summary statistics
        for scorer_name, scores in all_scores.items():
            processed['summary']['avg_scores'][scorer_name] = sum(scores) / len(scores)
            processed['summary']['score_ranges'][scorer_name] = {
                'min': min(scores),
                'max': max(scores)
            }
        
        return processed
    
    def _process_trajectories(self, trajectories: Dict) -> Dict:
        """Process trajectory objects for display."""
        processed = {}
        
        for traj_id, trajectory in trajectories.items():
            processed[traj_id] = {
                'id': traj_id,
                'objective': getattr(trajectory, 'objective', 'No objective available'),
                'reward': getattr(trajectory, 'reward', -1),  # Add reward field
                'actions': []
            }
            
            # Process actions if available
            if hasattr(trajectory, 'actions'):
                for i, action in enumerate(trajectory.actions):
                    action_data = {
                        'step': i + 1,
                        'action_type': getattr(action, 'action_type', 'unknown'),
                        'observation': getattr(action, 'observation', 'No observation available'),
                        'reasoning': getattr(action, 'reasoning', ''),
                        'action': getattr(action, 'action', ''),
                        'target_element': getattr(action, 'target_element', None),
                        'label_line': getattr(action, 'label_line', None),
                    }
                    
                    # Handle image encoding if available
                    if hasattr(action, 'image_encoding'):
                        action_data['image_encoding'] = action.image_encoding
                    
                    processed[traj_id]['actions'].append(action_data)
        
        return processed
    
    def _generate_overview_plot(self) -> Dict:
        """Generate overview plot data."""
        if not self.results_data or 'trajectories' not in self.results_data:
            return {'error': 'No results data available'}
        
        # Extract data for plotting
        trajectory_ids = []
        scores_by_scorer = {}
        
        for traj_id, traj_data in self.results_data['trajectories'].items():
            trajectory_ids.append(traj_id)
            
            for scorer_name, score_obj in traj_data['scores'].items():
                if scorer_name not in scores_by_scorer:
                    scores_by_scorer[scorer_name] = []
                # Extract numeric score from the score object
                score_value = score_obj['score'] if isinstance(score_obj, dict) else score_obj
                scores_by_scorer[scorer_name].append(score_value)
        
        # Create plot data
        plot_data = {
            'data': [],
            'layout': {
                'title': 'Trajectory Scores Overview',
                'xaxis': {'title': 'Trajectory ID'},
                'yaxis': {'title': 'Score'},
                'hovermode': 'closest'
            }
        }
        
        for scorer_name, scores in scores_by_scorer.items():
            plot_data['data'].append({
                'x': trajectory_ids,
                'y': scores,
                'type': 'scatter',
                'mode': 'markers+lines',
                'name': scorer_name,
                'hovertemplate': f'{scorer_name}: %{{y}}<br>Trajectory: %{{x}}<extra></extra>'
            })
        
        return plot_data
    
    def _generate_reasoning_quality_plot(self) -> Dict:
        """Generate reasoning quality plot data showing distributions for each sub-score."""
        if not self.results_data or 'trajectories' not in self.results_data:
            return {'error': 'No results data available'}

        # Apply filters to the full trajectory data structure first
        filtered_trajectory_data = self._apply_filters_to_data(
            self.results_data['trajectories']
        )

        # Extract reasoning quality data from filtered trajectories
        reasoning_data = {}
        scorer_found = None
        
        # First try exact match for ReasoningQuality
        for traj_id, traj_data in filtered_trajectory_data.items():
            if 'ReasoningQuality' in traj_data['scores']:
                reasoning_data[traj_id] = traj_data['scores']['ReasoningQuality']
                scorer_found = 'ReasoningQuality'
        
        # If no exact match, try to find any scorer with "reasoning" in the name
        if not reasoning_data:
            for traj_id, traj_data in filtered_trajectory_data.items():
                for scorer_name in traj_data['scores'].keys():
                    if 'reasoning' in scorer_name.lower():
                        reasoning_data[traj_id] = traj_data['scores'][scorer_name]
                        scorer_found = scorer_name
                        break

        if not reasoning_data:
            # Debug information - show what scorers are actually available
            available_scorers = set()
            for traj_data in self.results_data['trajectories'].values():
                available_scorers.update(traj_data['scores'].keys())
            
            return {'error': f'No reasoning quality scores available. Available scorers: {list(available_scorers)}'}

        # Extract sub-scores from details["reasoning_quality"] - ONLY real scores
        sub_scores = {}
        for traj_id, score_data in reasoning_data.items():
            if 'details' in score_data and score_data['details']:
                details = score_data['details']
                # Look for reasoning_quality nested structure
                if 'reasoning_quality' in details and isinstance(details['reasoning_quality'], dict):
                    reasoning_quality_scores = details['reasoning_quality']
                    for sub_score_name, sub_score_value in reasoning_quality_scores.items():
                        if isinstance(sub_score_value, (int, float)):
                            if sub_score_name not in sub_scores:
                                sub_scores[sub_score_name] = []
                            sub_scores[sub_score_name].append(sub_score_value)

        if not sub_scores:
            # Try to debug what's actually in the details
            sample_details = {}
            for traj_id, score_data in list(reasoning_data.items())[:1]:  # Take first sample
                if 'details' in score_data:
                    sample_details = score_data['details']
                    break
            
            return {'error': f'No sub-scores found in reasoning quality details. Sample details structure: {list(sample_details.keys()) if sample_details else "No details found"}'}

        # Create subplots using Plotly data structure
        num_subscores = len(sub_scores)
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#98D8C8', '#F7DC6F']
        
        # Create data traces for each subplot
        data_traces = []
        subplot_titles = []
        
        for idx, (sub_score_name, scores) in enumerate(sub_scores.items()):
            color = colors[idx % len(colors)]
            
            # Create histogram trace
            trace = {
                'x': scores,
                'type': 'histogram',
                'nbinsx': 4,  # Changed from 15 to 4 to match score range
                'name': sub_score_name.replace("_", " ").title(),
                'marker': {'color': color, 'opacity': 0.7},
                'xaxis': f'x{idx+1}' if idx > 0 else 'x',
                'yaxis': f'y{idx+1}' if idx > 0 else 'y',
                'showlegend': False,
                'xbins': {
                    'start': 0.125,
                    'end': 1.125,
                    'size': 0.25
                }
            }
            data_traces.append(trace)
            subplot_titles.append(f'{sub_score_name.replace("_", " ").title()} Distribution')
        
        # Create layout with subplots
        layout = {
            'title': {
                'text': f'Reasoning Quality Sub-Score Distributions (Using: {scorer_found})',
                'x': 0.5
            },
            'height': max(400, num_subscores * 300),
            'showlegend': False,
            'annotations': []
        }
        
        # Add subplot layout configuration and titles
        for idx in range(num_subscores):
            y_position = 1 - (idx / num_subscores)
            y_bottom = max(0, y_position - (1 / num_subscores) + 0.05)
            y_top = min(1, y_position - 0.05)
            
            # X-axis
            xaxis_key = f'xaxis{idx+1}' if idx > 0 else 'xaxis'
            layout[xaxis_key] = {
                'title': 'Score',
                'domain': [0.1, 0.9],
                'anchor': f'y{idx+1}' if idx > 0 else 'y',
                'tickmode': 'array',
                'tickvals': [0.25, 0.5, 0.75, 1.0],
                'ticktext': ['0.25', '0.5', '0.75', '1.0'],
                'range': [0.1, 1.15]
            }
            
            # Y-axis - just "Count"
            yaxis_key = f'yaxis{idx+1}' if idx > 0 else 'yaxis'
            layout[yaxis_key] = {
                'title': 'Count',
                'domain': [y_bottom, y_top],
                'anchor': f'x{idx+1}' if idx > 0 else 'x'
            }
            
            # Add subplot title as annotation above the subplot
            sub_score_name = list(sub_scores.keys())[idx]
            layout['annotations'].append({
                'text': f'<b>{sub_score_name.replace("_", " ").title()}</b>',
                'x': 0.5,  # Center horizontally
                'y': y_top + 0.03,  # Position above the subplot
                'xref': 'paper',
                'yref': 'paper',
                'xanchor': 'center',
                'yanchor': 'bottom',
                'showarrow': False,
                'font': {'size': 14, 'color': '#2c3e50'}
            })
        
        # Add filter info if applicable
        total_trajectories = len(self.results_data['trajectories'])
        filtered_trajectories = len(filtered_trajectory_data)
        if filtered_trajectories < total_trajectories:
            filter_count = len(self.active_filters)
            layout['title']['text'] += f'<br>Filtered: {filtered_trajectories}/{total_trajectories} trajectories ({filter_count} filter{"s" if filter_count != 1 else ""})'
        
        plot_data = {
            'data': data_traces,
            'layout': layout,
            'filter_info': {
                'total_trajectories': total_trajectories,
                'filtered_trajectories': filtered_trajectories,
                'active_filters': list(self.active_filters.values())
            },
            'subplot_info': {
                'available_sub_scores': list(sub_scores.keys()),
                'subplot_count': num_subscores,
                'scorer_name': scorer_found
            }
        }
        
        return plot_data
    
    def _apply_filters_to_data(self, data: Dict) -> Dict:
        """Apply active filters to trajectory data using filtered trajectory lists."""
        if not self.active_filters:
            return data
        
        # Get intersection of all filtered trajectories from active filters
        filtered_trajectory_sets = []
        
        for filter_id, filter_data in self.active_filters.items():
            filtered_trajectories = set(filter_data['filtered_trajectories'])
            filtered_trajectory_sets.append(filtered_trajectories)
        
        # Find intersection of all filters (trajectories that pass ALL filters)
        if filtered_trajectory_sets:
            final_filtered_trajectories = filtered_trajectory_sets[0]
            for trajectory_set in filtered_trajectory_sets[1:]:
                final_filtered_trajectories = final_filtered_trajectories.intersection(trajectory_set)
        else:
            final_filtered_trajectories = set(data.keys())
        
        # Return only trajectories that are in the intersection
        filtered_data = {}
        for traj_id in final_filtered_trajectories:
            if traj_id in data:
                filtered_data[traj_id] = data[traj_id]
        
        return filtered_data
    
    def _generate_objective_quality_plot(self) -> Dict:
        """Generate objective quality plot data."""
        if not self.results_data or 'trajectories' not in self.results_data:
            return {'error': 'No results data available'}

        # Apply filters to the full trajectory data structure first
        filtered_trajectory_data = self._apply_filters_to_data(
            self.results_data['trajectories']
        )

        # Extract objective scores from filtered trajectories
        objective_scores = []
        trajectory_ids = []
        scorer_found = None
        
        # First try exact match for ObjectiveQuality
        for traj_id, traj_data in filtered_trajectory_data.items():
            if 'ObjectiveQuality' in traj_data['scores']:
                score_value = traj_data['scores']['ObjectiveQuality']['score'] if isinstance(traj_data['scores']['ObjectiveQuality'], dict) else traj_data['scores']['ObjectiveQuality']
                objective_scores.append(score_value)
                trajectory_ids.append(traj_id)
                scorer_found = 'ObjectiveQuality'
        
        # If no exact match, try to find any scorer with "objective" in the name
        if not objective_scores:
            for traj_id, traj_data in filtered_trajectory_data.items():
                for scorer_name, score_obj in traj_data['scores'].items():
                    if 'objective' in scorer_name.lower():
                        # Extract numeric score from the score object
                        score_value = score_obj['score'] if isinstance(score_obj, dict) else score_obj
                        objective_scores.append(score_value)
                        trajectory_ids.append(traj_id)
                        scorer_found = scorer_name
                        break

        if not objective_scores:
            # Debug information - show what scorers are actually available
            available_scorers = set()
            for traj_data in self.results_data['trajectories'].values():
                available_scorers.update(traj_data['scores'].keys())
            
            return {'error': f'No objective quality scores available. Available scorers: {list(available_scorers)}'}

        # Create histogram
        plot_data = {
            'data': [{
                'x': objective_scores,
                'type': 'histogram',
                'name': 'Objective Quality',
                'nbinsx': 4,  # 4 bins for the 4 score values
                'hovertemplate': 'Score Range: %{x}<br>Count: %{y}<extra></extra>',
                'xbins': {
                    'start': 0.125,
                    'end': 1.125,
                    'size': 0.25
                }
            }],
            'layout': {
                'title': {
                    'text': f'Objective Quality Distribution (Using: {scorer_found})',
                    'x': 0.5
                },
                'xaxis': {
                    'title': 'Objective Score',
                    'tickmode': 'array',
                    'tickvals': [0.25, 0.5, 0.75, 1.0],
                    'ticktext': ['0.25', '0.5', '0.75', '1.0'],
                    'range': [0.1, 1.15]
                },
                'yaxis': {'title': 'Count'},
                'bargap': 0.1,
                'height': 400
            }
        }

        # Add filter info if applicable
        total_trajectories = len(self.results_data['trajectories'])
        filtered_trajectories = len(filtered_trajectory_data)
        if filtered_trajectories < total_trajectories:
            filter_count = len(self.active_filters)
            plot_data['layout']['title']['text'] += f'<br>Filtered: {filtered_trajectories}/{total_trajectories} trajectories ({filter_count} filter{"s" if filter_count != 1 else ""})'
        
        # Add filter and plot metadata to the response
        plot_data['filter_info'] = {
            'total_trajectories': total_trajectories,
            'filtered_trajectories': filtered_trajectories,
            'active_filters': list(self.active_filters.values())
        }
        
        plot_data['plot_info'] = {
            'scorer_name': scorer_found,
            'plot_type': 'objective_quality',
            'supports_zoom_filter': True
        }

        return plot_data
    
    def _get_filtered_trajectory_ids(self) -> List[str]:
        """Get list of trajectory IDs that pass current filters."""
        if not self.active_filters or not self.results_data:
            return list(self.results_data.get('trajectories', {}).keys()) if self.results_data else []
        
        # Apply filters to trajectory data using the generalized method
        filtered_trajectories = self._apply_filters_to_data(
            self.results_data.get('trajectories', {})
        )
        
        return list(filtered_trajectories.keys())

    def _get_reasoning_tagcloud_data(self, ngram_size: int) -> Dict:
        """Load reasoning tag cloud data from pre-generated JSON files and filter based on active filters."""
        # Validate ngram_size
        if ngram_size not in [1, 2, 3, 4]:
            return {'error': f'Invalid ngram_size: {ngram_size}. Must be 1, 2, 3, or 4.'}
        
        try:
            # Use project root directory (2 levels up from dashboard/backend/main.py)
            project_root = Path(__file__).parent.parent.parent
            tag_cloud_dir = project_root / "tag_cloud"
            tag_cloud_file = tag_cloud_dir / f"reasoning_tag_cloud_{ngram_size}gram.json"
            
            if not tag_cloud_file.exists():
                return {'error': f'Tag cloud file not found: {tag_cloud_file}'}
            
            # Load the pre-generated tag cloud data
            with open(tag_cloud_file, 'r', encoding='utf-8') as f:
                tag_cloud_data = json.load(f)
        
        except FileNotFoundError as e:
            return {'error': f'Tag cloud file not found: {str(e)}'}
        except json.JSONDecodeError as e:
            return {'error': f'Error parsing tag cloud JSON: {str(e)}'}
        except Exception as e:
            return {'error': f'Error loading reasoning tag cloud: {str(e)}'}
        
        # Get currently filtered trajectory IDs
        filtered_trajectory_ids = set(self._get_filtered_trajectory_ids())
        
        # If no filters are active, return the original data
        if not self.active_filters:
            return {
                'status': 'success',
                'tags': tag_cloud_data.get('tags', []),
                'statistics': tag_cloud_data.get('statistics', {}),
                'filter_info': {
                    'total_trajectories': len(filtered_trajectory_ids),
                    'filtered_trajectories': len(filtered_trajectory_ids),
                    'active_filters': 0
                },
                'ngram_size': ngram_size
            }
        
        # Filter and reweight tags based on active filters
        filtered_tags = []
        original_tags = tag_cloud_data.get('tags', [])
        
        for tag in original_tags:
            # Get trajectories for this tag that are in the filtered set
            tag_trajectories = set(tag.get('trajectories_using', []))
            filtered_tag_trajectories = tag_trajectories.intersection(filtered_trajectory_ids)
            
            # Skip tags that don't appear in any filtered trajectories
            if not filtered_tag_trajectories:
                continue
            
            # Calculate new metrics based on filtered trajectories
            new_raw_frequency = len(filtered_tag_trajectories)
            new_document_frequency = len(filtered_tag_trajectories)
            total_filtered_trajectories = len(filtered_trajectory_ids)
            
            # Recalculate weight proportionally
            original_doc_freq = tag.get('doc_frequency', 1)
            weight_ratio = new_document_frequency / max(original_doc_freq, 1)
            new_weight = tag.get('weight', 0) * weight_ratio
            
            # Create filtered tag entry (using correct field names)
            filtered_tag = {
                'text': tag.get('text', ''),
                'weight': new_weight,
                'raw_frequency': new_raw_frequency,
                'doc_frequency': new_document_frequency,
                'tfidf_score': new_weight,  # Use weight as TF-IDF approximation
                'trajectories': list(filtered_tag_trajectories)
            }
            
            filtered_tags.append(filtered_tag)
        
        # Sort tags by weight (descending)
        filtered_tags.sort(key=lambda x: x.get('weight', 0), reverse=True)
        
        # Update statistics
        original_stats = tag_cloud_data.get('statistics', {})
        filtered_stats = {
            'total_reasoning_texts': len(filtered_trajectory_ids),
            'unique_terms_found': len(filtered_tags),
            'total_trajectories_processed': len(filtered_trajectory_ids),
            'ngram_size': ngram_size,
            'frequency_threshold': original_stats.get('frequency_threshold', 3),
            'original_total_terms': original_stats.get('unique_terms_found', len(original_tags)),
            'filtering_applied': True
        }
        
        return {
            'status': 'success',
            'tags': filtered_tags,
            'statistics': filtered_stats,
            'filter_info': {
                'total_trajectories': original_stats.get('total_trajectories_processed', len(original_tags)),
                'filtered_trajectories': len(filtered_trajectory_ids),
                'active_filters': len(self.active_filters)
            },
            'ngram_size': ngram_size
        }

    def _get_action_phrases_tagcloud_data(self, phrase_type: str = "pairs") -> Dict:
        """Load action phrases tag cloud data from pre-generated JSON files."""
        # Validate phrase_type
        if phrase_type not in ["verbs", "nouns", "pairs"]:
            return {'error': f'Invalid phrase_type: {phrase_type}. Must be verbs, nouns, or pairs.'}
        
        try:
            # Use project root directory (2 levels up from dashboard/backend/main.py)
            project_root = Path(__file__).parent.parent.parent
            tag_cloud_dir = project_root / "tag_cloud"
            
            # Load the specific phrase type file
            tag_cloud_file = tag_cloud_dir / f"action_phrases_tag_cloud_{phrase_type}.json"
            
            if not tag_cloud_file.exists():
                return {'error': f'Action phrases tag cloud file not found: {tag_cloud_file}'}
            
            # Load tag cloud data
            with open(tag_cloud_file, 'r', encoding='utf-8') as f:
                tag_cloud_data = json.load(f)
            
            # Get currently filtered trajectory IDs  
            filtered_trajectory_ids = set(self._get_filtered_trajectory_ids())
            
            # If no filters are active, return the original data
            if not self.active_filters:
                return {
                    'status': 'success',
                    'tags': tag_cloud_data.get('tags', []),
                    'statistics': tag_cloud_data.get('processing_stats', {}),
                    'filter_info': {
                        'total_trajectories': tag_cloud_data.get('generation_settings', {}).get('total_trajectories', 0),
                        'filtered_trajectories': tag_cloud_data.get('generation_settings', {}).get('total_trajectories', 0),
                        'active_filters': 0
                    },
                    'phrase_type': phrase_type
                }
            
            # Filter and reweight tags based on active filters
            filtered_tags = []
            original_tags = tag_cloud_data.get('tags', [])
            
            for tag in original_tags:
                # Get trajectories for this tag that are in the filtered set
                tag_trajectories = set(tag.get('trajectories', []))
                filtered_tag_trajectories = tag_trajectories.intersection(filtered_trajectory_ids)
                
                # Skip tags that don't appear in any filtered trajectories
                if not filtered_tag_trajectories:
                    continue
                
                # Calculate new metrics based on filtered trajectories
                new_raw_frequency = len(filtered_tag_trajectories)
                new_document_frequency = len(filtered_tag_trajectories)
                
                # Recalculate weight proportionally
                original_doc_freq = tag.get('doc_frequency', 1)
                weight_ratio = new_document_frequency / max(original_doc_freq, 1)
                new_weight = tag.get('weight', 0) * weight_ratio
                
                # Create filtered tag entry
                filtered_tag = {
                    'text': tag.get('text', ''),
                    'weight': new_weight,
                    'raw_frequency': new_raw_frequency,
                    'doc_frequency': new_document_frequency,
                    'tfidf_score': new_weight,
                    'trajectories': list(filtered_tag_trajectories)
                }
                
                filtered_tags.append(filtered_tag)
            
            # Sort tags by weight (descending)
            filtered_tags.sort(key=lambda x: x.get('weight', 0), reverse=True)
            
            # Update statistics
            original_stats = tag_cloud_data.get('processing_stats', {})
            filtered_stats = {
                'total_action_phrases': original_stats.get('total_action_phrases', 0),
                'unique_terms_found': len(filtered_tags),
                'total_trajectories_processed': len(filtered_trajectory_ids),
                'phrase_type': phrase_type,
                'frequency_threshold': tag_cloud_data.get('generation_settings', {}).get('min_frequency', 2),
                'original_total_terms': original_stats.get('unique_terms_found', len(original_tags)),
                'filtering_applied': True
            }
            
            return {
                'status': 'success',
                'tags': filtered_tags,
                'statistics': filtered_stats,
                'filter_info': {
                    'total_trajectories': tag_cloud_data.get('generation_settings', {}).get('total_trajectories', 0),
                    'filtered_trajectories': len(filtered_trajectory_ids),
                    'active_filters': len(self.active_filters)
                },
                'phrase_type': phrase_type
            }
            
        except FileNotFoundError as e:
            return {'error': f'Tag cloud file not found: {str(e)}'}
        except json.JSONDecodeError as e:
            return {'error': f'Error parsing tag cloud JSON: {str(e)}'}
        except Exception as e:
            return {'error': f'Error loading action phrases tag cloud: {str(e)}'}
    
    def _get_embedding_data(self) -> Dict:
        """Load and return embedding data from disk."""
        if not PICKLE_AVAILABLE or not SKLEARN_AVAILABLE:
            return {'error': 'Required packages (pickle, sklearn) not available'}
        
        # Use project root directory (2 levels up from dashboard/backend/main.py)
        project_root = Path(__file__).parent.parent.parent
        embedding_dir = project_root / "embeddings_output"
        
        if not embedding_dir.exists():
            return {'error': f'Embedding directory not found: {embedding_dir}'}
        
        try:
            # Load embedding data
            with open(embedding_dir / "embeddings_data.pkl", "rb") as f:
                embedding_data = pickle.load(f)
            
            # Load embedding arrays
            verb_embeddings = None
            noun_embeddings = None
            pair_embeddings = None
            
            if (embedding_dir / "verb_embeddings.npy").exists():
                verb_embeddings = np.load(embedding_dir / "verb_embeddings.npy")
            if (embedding_dir / "noun_embeddings.npy").exists():
                noun_embeddings = np.load(embedding_dir / "noun_embeddings.npy")
            if (embedding_dir / "pair_embeddings.npy").exists():
                pair_embeddings = np.load(embedding_dir / "pair_embeddings.npy")
            
            return {
                'status': 'success',
                'data': {
                    'model_name': embedding_data.get('model_name', 'Unknown'),
                    'embedding_dim': embedding_data.get('embedding_dim', 0),
                    'verb_texts': embedding_data.get('verb_texts', []),
                    'noun_texts': embedding_data.get('noun_texts', []),
                    'pair_texts': embedding_data.get('pair_texts', []),
                    'verb_metadata': embedding_data.get('verb_metadata', []),
                    'noun_metadata': embedding_data.get('noun_metadata', []),
                    'pair_metadata': embedding_data.get('pair_metadata', []),
                    'counts': {
                        'verbs': len(embedding_data.get('verb_texts', [])),
                        'nouns': len(embedding_data.get('noun_texts', [])),
                        'pairs': len(embedding_data.get('pair_texts', []))
                    }
                },
                'embeddings': {
                    'verbs': verb_embeddings.tolist() if verb_embeddings is not None else None,
                    'nouns': noun_embeddings.tolist() if noun_embeddings is not None else None,
                    'pairs': pair_embeddings.tolist() if pair_embeddings is not None else None
                }
            }
            
        except Exception as e:
            return {'error': f'Error loading embedding data: {str(e)}'}
    
    def _generate_tsne_plot(self, data_type: str, aggregation: str, perplexity: int, n_components: int) -> Dict:
        """Generate t-SNE plot data for embeddings."""
        if not SKLEARN_AVAILABLE:
            return {'error': 'scikit-learn not available for t-SNE'}
        
        embedding_data = self._get_embedding_data()
        if 'error' in embedding_data:
            return embedding_data
        
        try:
            data = embedding_data['data']
            emb_arrays = embedding_data['embeddings']
            
            # Prepare embeddings, texts, and metadata based on data type
            embeddings = []
            texts = []
            metadata = []
            colors = []
            
            if data_type == 'verbs':
                if not emb_arrays['verbs']:
                    return {'error': 'No verb embeddings available'}
                embeddings = emb_arrays['verbs']
                texts = data['verb_texts']
                metadata = data['verb_metadata']
                colors = ['verb'] * len(texts)
                
            elif data_type == 'nouns':
                if not emb_arrays['nouns']:
                    return {'error': 'No noun embeddings available'}
                embeddings = emb_arrays['nouns']
                texts = data['noun_texts']
                metadata = data['noun_metadata']
                colors = ['noun'] * len(texts)
                
            elif data_type == 'pairs':
                if not emb_arrays.get('pairs'):
                    return {'error': 'No pair embeddings available'}
                embeddings = emb_arrays['pairs']
                texts = data['pair_texts']
                metadata = data['pair_metadata']
                colors = ['pair'] * len(texts)
                
            elif data_type == 'both':
                # Create combined verb+noun embeddings by element-wise addition
                if not emb_arrays['verbs'] or not emb_arrays['nouns']:
                    return {'error': 'Both verb and noun embeddings required for combined mode'}
                
                verb_embs = np.array(emb_arrays['verbs'])
                noun_embs = np.array(emb_arrays['nouns'])
                verb_texts = data['verb_texts']
                noun_texts = data['noun_texts']
                verb_metadata = data['verb_metadata']
                noun_metadata = data['noun_metadata']
                
                # Match verbs and nouns by trajectory and action step
                for i, verb_meta in enumerate(verb_metadata):
                    # Find corresponding noun from same trajectory and step
                    for j, noun_meta in enumerate(noun_metadata):
                        if (verb_meta['trajectory_file'] == noun_meta['trajectory_file'] and 
                            verb_meta.get('action_index') == noun_meta.get('action_index')):
                            # Combine embeddings by addition: verb + noun
                            combined_embedding = verb_embs[i] + noun_embs[j]
                            embeddings.append(combined_embedding)
                            texts.append(f"{verb_texts[i]} + {noun_texts[j]}")
                            metadata.append(verb_meta)  # Use verb metadata as base
                            colors.append('both')
                            break
                
                if not embeddings:
                    return {'error': 'No matching verb-noun pairs found for combination'}
            
            else:
                return {'error': f'Unknown data_type: {data_type}'}
            
            embeddings = np.array(embeddings)
            
            # Handle trajectory aggregation by averaging embeddings per trajectory
            if aggregation == 'trajectories':
                traj_data = {}
                
                # Group embeddings by trajectory
                for emb, text, meta, color in zip(embeddings, texts, metadata, colors):
                    traj_id = meta['trajectory_file']
                    if traj_id not in traj_data:
                        traj_data[traj_id] = {
                            'embeddings': [],
                            'metadata': meta,
                            'color': color
                        }
                    traj_data[traj_id]['embeddings'].append(emb)
                
                # Create averaged embeddings per trajectory
                embeddings = []
                texts = []
                metadata = []
                colors = []
                
                for traj_id, traj_info in traj_data.items():
                    avg_embedding = np.mean(traj_info['embeddings'], axis=0)
                    embeddings.append(avg_embedding)
                    texts.append(f"Trajectory: {traj_id}")
                    metadata.append(traj_info['metadata'])
                    colors.append(traj_info['color'])
                
                embeddings = np.array(embeddings)
            
            # Apply t-SNE dimensionality reduction
            tsne = TSNE(
                n_components=n_components, 
                perplexity=min(perplexity, len(embeddings)-1), 
                random_state=42
            )
            tsne_result = tsne.fit_transform(embeddings)
            
            # Return plot data
            return {
                'status': 'success',
                'plot_data': {
                    'x': tsne_result[:, 0].tolist(),
                    'y': tsne_result[:, 1].tolist(),
                    'texts': texts,
                    'colors': colors,
                    'metadata': metadata,
                    'aggregation': aggregation,
                    'data_type': data_type,
                    'total_points': len(embeddings)
                },
                'config': {
                    'perplexity': perplexity,
                    'n_components': n_components,
                    'data_type': data_type,
                    'aggregation': aggregation
                }
            }
            
        except Exception as e:
            return {'error': f'Error generating t-SNE plot: {str(e)}'}


# Create the dashboard API instance
dashboard_api = DashboardAPI()

# Export the FastAPI app for uvicorn
app = dashboard_api.app

# Load data from environment variable if provided
data_file = os.environ.get('DASHBOARD_DATA_FILE')
if data_file:
    print(f"📊 Loading data from {data_file}")
    try:
        with open(data_file, 'rb') as f:
            data = pickle.load(f)
            dashboard_api.load_results(data['results'], data['trajectories'])
            print(f"✅ Loaded {len(data['results'])} results and {len(data['trajectories'])} trajectories into backend")
        
        # Clean up the temporary file
        try:
            os.unlink(data_file)
            print(f"🗑️  Cleaned up data file {data_file}")
        except:
            pass
            
    except Exception as e:
        print(f"❌ Error loading data from file: {e}")
        import traceback
        traceback.print_exc()


def get_app():
    """Get the FastAPI app instance."""
    return dashboard_api.app


def load_results(results: Dict[str, Dict[str, ScorerResult]], trajectories: Optional[Dict] = None):
    """Load results into the dashboard API."""
    dashboard_api.load_results(results, trajectories) 