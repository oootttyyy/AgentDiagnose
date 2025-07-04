import argparse
import concurrent.futures
import json
import os
import tiktoken
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import pandas as pd
from tqdm import tqdm

from evaluator.scorers import (
    get_available_scorers,
    create_scorer,
    ScorerResult,
    BaseScorer
)
from evaluator.scorers.navigation_path_scorer import NavigationPathScorer
from evaluator.scorers.objective_quality import ObjectiveQualityScorer
from evaluator.scorers.reasoning_quality import ReasoningQualityScorer
from evaluator.trajectory import Trajectory
from dashboard.web_dashboard import create_web_dashboard


def load_trajectory(json_file: str) -> Optional[Trajectory]:
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        trajectory = Trajectory.from_json(data)
        return trajectory
    except Exception as e:
        print(f"Error loading trajectory from {json_file}: {e}")
        return None


def load_trajectories(input_dir: str) -> Dict[str, Trajectory]:
    trajectories = {}
    input_path = Path(input_dir)
    json_files = list(input_path.glob("*.json"))
    
    for json_file in tqdm(json_files, desc="Loading trajectories"):
        traj_id = json_file.stem.replace("trajectory_", "")
        trajectory = load_trajectory(str(json_file))
        if trajectory:
            trajectories[traj_id] = trajectory
    
    return trajectories


def evaluate_trajectory(trajectory: Trajectory, scorers: List[BaseScorer], dry_run: bool = False) -> Dict[str, ScorerResult]:
    results = {}
    
    for scorer in scorers:
        if dry_run:
            if hasattr(scorer, 'dry_run'):
                result = scorer.dry_run(trajectory)
                results[scorer.name] = result
            else:
                results[scorer.name] = ScorerResult(
                    score=0.0,
                    name=scorer.name,
                    description=scorer.description,
                    details={"dry_run": True, "error": "Scorer doesn't implement dry_run method"},
                    confidence=0.0,
                    weight=scorer.weight
                )
        else:
            result = scorer(trajectory)
            results[scorer.name] = result
    
    return results


def evaluate_trajectories(trajectories: Dict[str, Trajectory], 
                         scorer_names: List[str],
                         dry_run: bool = False,
                         parallelize: bool = True,
                         max_workers: int = 4,
                         **scorer_kwargs) -> Dict[str, Dict[str, ScorerResult]]:
    
    if not scorer_names or scorer_names == ['default']:
        scorer_names = ["reasoning_quality", "objective_quality", "navigation_path"]

    models = {
        "reasoning_quality": "gemini-2.5-pro-exp-03-25",
        "objective_quality": "gemini-2.0-flash",
        "navigation_path": "no_model_needed",
    }
    
    scorers = []
    for name in scorer_names:
        scorer_kwargs['model'] = models[name]
        scorer = get_scorer(name, **scorer_kwargs)
        scorers.append(scorer)
    
    output_json = scorer_kwargs.get('output_json', None)
    existing_results = {}
    trajectories_to_evaluate = dict(trajectories)
    
    if output_json and not dry_run and os.path.exists(output_json):
        print(f"Found existing output file: {output_json}")
        with open(output_json, 'r', encoding='utf-8') as f:
            serialized_results = json.load(f)
        
        for traj_id, scorer_dict in serialized_results.items():
            if traj_id in trajectories:
                del trajectories_to_evaluate[traj_id]
                existing_results[traj_id] = {}
                for scorer_name, result_dict in scorer_dict.items():
                    existing_results[traj_id][scorer_name] = ScorerResult(
                        score=result_dict['score'],
                        name=result_dict['name'],
                        description=result_dict['description'],
                        details=result_dict['details'],
                        confidence=result_dict['confidence'],
                        weight=result_dict['weight']
                    )
        
        print(f"Loaded {len(existing_results)} existing results. {len(trajectories_to_evaluate)} trajectories remain.")
    
    results = dict(existing_results)
    total_input_tokens = 0
    total_output_tokens = 0
    token_counts = {}
    save_interval = 50
    last_save_count = len(existing_results)
    
    if output_json and not dry_run:
        print(f"Will save intermediate results every {save_interval} trajectories")
    
    if dry_run:
        print("DRY RUN: Only counting tokens without making API calls")
    
    def save_intermediate_results(current_results):
        nonlocal last_save_count
        
        if not output_json or dry_run:
            return
            
        serializable_results = {}
        for traj_id, scorer_results in current_results.items():
            serializable_results[traj_id] = {
                name: result.to_dict() for name, result in scorer_results.items()
            }
        
        temp_file = f"{output_json}.temp"
        try:
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(serializable_results, f, indent=2)
                
            if os.path.exists(output_json):
                os.remove(output_json)
            os.rename(temp_file, output_json)
            
            print(f"Intermediate save: {len(current_results)} trajectories evaluated")
            last_save_count = len(current_results)
        except Exception as e:
            print(f"Warning: Failed to save intermediate results: {e}")
    
    def should_save_intermediate(results_count):
        return results_count % save_interval == 0 and results_count > last_save_count
    
    def process_parallel_result(traj_id, result):
        nonlocal results, last_save_count, total_input_tokens, total_output_tokens, token_counts
        
        results[traj_id] = result
        
        for scorer_name, scorer_result in result.items():
            if 'token_usage' in scorer_result.details:
                token_usage = scorer_result.details['token_usage']
                input_tokens = token_usage.get('input_tokens', 0)
                output_tokens = token_usage.get('output_tokens', 0)
                
                total_input_tokens += input_tokens
                total_output_tokens += output_tokens
                
                if scorer_name not in token_counts:
                    token_counts[scorer_name] = {'input': 0, 'output': 0}
                token_counts[scorer_name]['input'] += input_tokens
                token_counts[scorer_name]['output'] += output_tokens
        
        if output_json and not dry_run:
            if should_save_intermediate(len(results)):
                save_intermediate_results(results)
    
    if not trajectories_to_evaluate:
        print("All trajectories have already been evaluated.")
        
        for traj_id, scorer_results in existing_results.items():
            for scorer_name, result in scorer_results.items():
                if 'token_usage' in result.details:
                    token_usage = result.details['token_usage']
                    input_tokens = token_usage.get('input_tokens', 0)
                    output_tokens = token_usage.get('output_tokens', 0)
                    
                    total_input_tokens += input_tokens
                    total_output_tokens += output_tokens
                    
                    if scorer_name not in token_counts:
                        token_counts[scorer_name] = {'input': 0, 'output': 0}
                    token_counts[scorer_name]['input'] += input_tokens
                    token_counts[scorer_name]['output'] += output_tokens
        
        if not dry_run and (total_input_tokens > 0 or total_output_tokens > 0):
            print(f"TOKEN USAGE SUMMARY:")
            print(f"Total input tokens: {total_input_tokens}")
            print(f"Total output tokens: {total_output_tokens}")
            print(f"Total tokens: {total_input_tokens + total_output_tokens}")
            
            for scorer_name, counts in token_counts.items():
                print(f"  {scorer_name}: {counts['input']} input, {counts['output']} output")
        
        return results
    
    if parallelize and len(trajectories_to_evaluate) > 1 and not dry_run:
        print(f"Evaluating {len(trajectories_to_evaluate)} trajectories in parallel with {max_workers} workers")
        
        args_list = [(traj_id, traj, scorers, dry_run) for traj_id, traj in trajectories_to_evaluate.items()]
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(evaluate_trajectory_parallel, args): args[0] for args in args_list}
            
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Evaluating trajectories"):
                traj_id = futures[future]
                try:
                    _, result = future.result()
                    process_parallel_result(traj_id, result)
                except Exception as e:
                    print(f"Error processing trajectory {traj_id}: {e}")
    else:
        for traj_id, trajectory in tqdm(trajectories_to_evaluate.items(), desc="Evaluating trajectories"):
            results[traj_id] = evaluate_trajectory(trajectory, scorers, dry_run)
            
            if output_json and not dry_run:
                if should_save_intermediate(len(results)):
                    save_intermediate_results(results)
    
    if output_json and not dry_run:
        save_intermediate_results(results)
    
    if dry_run:
        total_tokens = 0
        token_counts = {}
        
        for traj_id, scorer_results in results.items():
            for scorer_name, result in scorer_results.items():
                token_count = 0
                if 'token_count' in result.details:
                    token_count = result.details['token_count']
                    total_tokens += token_count
                    
                    if scorer_name not in token_counts:
                        token_counts[scorer_name] = 0
                    token_counts[scorer_name] += token_count
        
        print(f"DRY RUN SUMMARY:")
        print(f"Total tokens for all trajectories: {total_tokens}")
        for scorer_name, count in token_counts.items():
            print(f"  {scorer_name}: {count} tokens")
    else:
        if total_input_tokens > 0 or total_output_tokens > 0:
            print(f"TOKEN USAGE SUMMARY:")
            print(f"Total input tokens: {total_input_tokens}")
            print(f"Total output tokens: {total_output_tokens}")
            print(f"Total tokens: {total_input_tokens + total_output_tokens}")
            
            for scorer_name, counts in token_counts.items():
                print(f"  {scorer_name}: {counts['input']} input, {counts['output']} output")
    
    return results


def evaluate_trajectory_parallel(args):
    traj_id, trajectory, scorers, dry_run = args
    result = evaluate_trajectory(trajectory, scorers, dry_run)
    return traj_id, result


def aggregate_results(evaluation_results: Dict[str, Dict[str, ScorerResult]]) -> pd.DataFrame:
    data = []
    
    for traj_id, scorer_results in evaluation_results.items():
        row = {'trajectory_id': traj_id}
        
        total_score = 0.0
        total_weight = 0.0
        
        for name, result in scorer_results.items():
            row[f'{name}_score'] = result.score
            row[f'{name}_confidence'] = result.confidence
            row[f'{name}_weight'] = result.weight
            
            total_score += result.weighted_score
            total_weight += result.weight
        
        if total_weight > 0:
            row['overall_score'] = total_score / total_weight
        else:
            row['overall_score'] = 0.0
        
        data.append(row)
    
    return pd.DataFrame(data)

def save_detailed_results(results: Dict[str, Dict[str, ScorerResult]], output_file: str) -> None:
    serializable_results = {}
    for traj_id, scorer_results in results.items():
        serializable_results[traj_id] = {
            name: result.to_dict() for name, result in scorer_results.items()
        }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"Detailed results saved to {output_file}")


def get_scorer(scorer_name, **kwargs):
    scorers = {
        "reasoning_quality": ReasoningQualityScorer,
        "objective_quality": ObjectiveQualityScorer,
        "navigation_path": NavigationPathScorer,
    }
    
    if scorer_name.lower() not in scorers:
        raise ValueError(f"Unknown scorer: {scorer_name}")
    
    return scorers[scorer_name.lower()](**kwargs)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate NNetNav trajectories")
    parser.add_argument("--input", "-i", required=True, help="Directory containing trajectory JSON files")
    parser.add_argument("--output-json", "-j", help="Output file for detailed evaluation results (JSON)")
    parser.add_argument("--scorers", "-s", nargs="+", default=["default"], 
                        help="Scorers to use.")
    parser.add_argument("--temperature", "-t", type=float, default=0.0, help="Temperature for LLM sampling")
    parser.add_argument("--save-details", "-d", action="store_true", help="Save detailed results to JSON")
    parser.add_argument("--dry-run", action="store_true", help="Count tokens without making API calls")
    parser.add_argument("--no-parallel", action="store_true", help="Disable parallel processing")
    parser.add_argument("--max-workers", type=int, default=4, help="Maximum number of parallel workers")
    parser.add_argument("--limit", type=int, help="Limit the number of trajectories to evaluate")
    parser.add_argument("--web-dashboard", action="store_true", help="Launch interactive web dashboard")
    parser.add_argument("--dashboard-port", type=int, default=5000, help="Port for web dashboard")
    parser.add_argument("--dashboard-host", default="127.0.0.1", help="Host for web dashboard")
    parser.add_argument("--no-auto-browser", action="store_true", help="Don't automatically open browser")
    parser.add_argument("--tunnel-token", help="Cloudflare tunnel token")
    parser.add_argument("--enable-tunnel", action="store_true", help="Enable Cloudflare tunnel")
    return parser.parse_args()


def main():
    args = parse_args()
    
    print(f"Evaluating trajectories in {args.input}")
    if args.web_dashboard:
        print(f"Web dashboard will be launched at http://{args.dashboard_host}:{args.dashboard_port}")
    
    trajectories = load_trajectories(args.input)
    
    if args.limit and args.limit > 0 and args.limit < len(trajectories):
        print(f"Limiting to {args.limit} trajectories")
        traj_ids = list(trajectories.keys())[:args.limit]
        trajectories = {traj_id: trajectories[traj_id] for traj_id in traj_ids}
    
    results = evaluate_trajectories(
        trajectories, 
        args.scorers,
        dry_run=args.dry_run,
        parallelize=not args.no_parallel,
        max_workers=args.max_workers,
        temperature=args.temperature,
        output_json=args.output_json
    )
        
    if args.web_dashboard:
        print("Launching web dashboard...")
        create_web_dashboard(
            results=results,
            trajectories=trajectories,
            port=args.dashboard_port,
            host=args.dashboard_host,
            open_browser=not args.no_auto_browser,
            enable_tunnel=args.enable_tunnel,
            tunnel_token=args.tunnel_token
        )


if __name__ == "__main__":
    main()



