import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from evaluator.trajectory import Trajectory
from visualization.reasoning_tag_cloud import create_reasoning_tag_cloud
from visualization.action_phrases_tag_cloud import create_action_phrases_tag_cloud


def load_trajectories_from_directory(directory_path: str) -> List[Trajectory]:
    trajectories = []
    directory = Path(directory_path)
    
    if not directory.exists():
        print(f"Error: Directory {directory_path} does not exist")
        return trajectories
    
    json_files = list(directory.glob("*.json"))
    print(f"Found {len(json_files)} JSON files in {directory_path}")
    
    for json_file in sorted(json_files):
        with open(json_file, 'r', encoding='utf-8') as f:
            trajectory_data = json.load(f)
            
        trajectory = Trajectory.from_json(trajectory_data)
        trajectories.append(trajectory)
        print(f"Loaded: {json_file.name}")
    
    print(f"Successfully loaded {len(trajectories)} trajectories")
    return trajectories


def generate_reasoning_tag_cloud(trajectories: List[Trajectory], 
                                ngram_range: Tuple[int, int] = (1, 4),
                                min_frequency: int = 2,
                                max_frequency_rate: float = 0.8,
                                max_tags: int = 200) -> Dict[str, Any]:
    print(f"Generating reasoning tag cloud with n-gram range {ngram_range}...")
    
    tag_data = create_reasoning_tag_cloud(
        trajectories=trajectories,
        ngram_range=ngram_range,
        min_frequency=min_frequency,
        max_frequency_rate=max_frequency_rate,
        max_tags=max_tags
    )
    
    tag_data['generation_settings'] = {
        'tag_source': 'reasoning',
        'ngram_range': ngram_range,
        'min_frequency': min_frequency,
        'max_frequency_rate': max_frequency_rate,
        'max_tags': max_tags,
        'total_trajectories': len(trajectories)
    }
    
    return tag_data


def generate_action_phrases_tag_cloud(trajectories: List[Trajectory], 
                                     phrase_type: str = "pairs",
                                     min_frequency: int = 2,
                                     max_frequency_rate: float = 0.8,
                                     max_tags: int = 200) -> Dict[str, Any]:
    print(f"Generating action phrases tag cloud for {phrase_type}...")
    
    tag_data = create_action_phrases_tag_cloud(
        trajectories=trajectories,
        min_frequency=min_frequency,
        max_frequency_rate=max_frequency_rate,
        max_tags=max_tags,
        phrase_type=phrase_type
    )
    
    tag_data['generation_settings'] = {
        'tag_source': 'action_phrases',
        'phrase_type': phrase_type,
        'min_frequency': min_frequency,
        'max_frequency_rate': max_frequency_rate,
        'max_tags': max_tags,
        'total_trajectories': len(trajectories)
    }
    
    return tag_data


def generate_tag_cloud(input_directory: str,
                      output_directory: str,
                      tag_source: str = "reasoning",
                      ngram_range: Tuple[int, int] = (1, 4),
                      min_frequency: int = 2,
                      max_frequency_rate: float = 0.8,
                      max_tags: int = 200) -> str:
    trajectories = load_trajectories_from_directory(input_directory)
    
    if not trajectories:
        raise ValueError("No trajectories loaded")
    
    if tag_source == "reasoning":
        tag_data = generate_reasoning_tag_cloud(
            trajectories=trajectories,
            ngram_range=ngram_range,
            min_frequency=min_frequency,
            max_frequency_rate=max_frequency_rate,
            max_tags=max_tags
        )
    elif tag_source == "action_phrases":
        phrase_type_map = {1: "verbs", 2: "nouns", 3: "pairs"}
        phrase_type = phrase_type_map.get(ngram_range[0], "pairs")
        
        tag_data = generate_action_phrases_tag_cloud(
            trajectories=trajectories,
            phrase_type=phrase_type,
            min_frequency=min_frequency,
            max_frequency_rate=max_frequency_rate,
            max_tags=max_tags
        )
    else:
        raise ValueError(f"Unsupported tag source: {tag_source}")
    
    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if tag_source == "reasoning":
        ngram_suffix = f"{ngram_range[0]}gram" if ngram_range[0] == ngram_range[1] else f"{ngram_range[0]}-{ngram_range[1]}gram"
        filename = f"{tag_source}_tag_cloud_{ngram_suffix}.json"
    elif tag_source == "action_phrases":
        phrase_type_map = {1: "verbs", 2: "nouns", 3: "pairs"}
        phrase_type = phrase_type_map.get(ngram_range[0], "pairs")
        filename = f"{tag_source}_tag_cloud_{phrase_type}.json"
    
    output_file = output_path / filename
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(tag_data, f, indent=2, ensure_ascii=False)
    
    print(f"Saved tag cloud data to: {output_file}")
    
    tags = tag_data.get('tags', [])
    stats = tag_data.get('processing_stats', {})
    settings = tag_data.get('generation_settings', {})
    
    print(f"\nTag Cloud Generation Summary:")
    print(f"  Source: {settings.get('tag_source', 'unknown')}")
    
    if tag_source == "reasoning":
        print(f"  N-gram range: {settings.get('ngram_range', 'unknown')}")
        print(f"  Generated {len(tags)} tags from {stats.get('total_reasoning_texts', 0)} texts")
    elif tag_source == "action_phrases":
        print(f"  Phrase type: {settings.get('phrase_type', 'unknown')}")
        action_stats = tag_data.get('statistics', {})
        print(f"  Generated {len(tags)} tags from {action_stats.get('total_action_phrases', 0)} action phrases")
    
    print(f"  Total trajectories: {settings.get('total_trajectories', 0)}")
    if tags:
        print(f"  Top tag: '{tags[0]['text']}' (weight: {tags[0]['weight']:.4f})")
    
    return str(output_file)


def main():
    parser = argparse.ArgumentParser(description='Generate tag cloud data from trajectory sources')
    
    parser.add_argument('--input-dir', '-i', 
                       default='trajectory_webarena_run_good_data_v3_3_labeled',
                       help='Input directory containing trajectory JSON files')
    
    parser.add_argument('--output-dir', '-o',
                       default='tag_cloud',
                       help='Output directory for tag cloud data')
    
    parser.add_argument('--tag-source', '-s',
                       choices=['reasoning', 'action_phrases'],
                       default='reasoning',
                       help='Source of text for tag generation')
    
    parser.add_argument('--ngram-range', '-n',
                       type=str,
                       default='1-4',
                       help='N-gram range (e.g., "1-4" or "2-2") for reasoning. For action_phrases: 1=verbs, 2=nouns, 3=pairs')
    
    parser.add_argument('--min-frequency', '-f',
                       type=int,
                       default=2,
                       help='Minimum frequency threshold')
    
    parser.add_argument('--max-frequency-rate', '-r',
                       type=float,
                       default=0.8,
                       help='Maximum frequency rate threshold')
    
    parser.add_argument('--max-tags', '-t',
                       type=int,
                       default=200,
                       help='Maximum number of tags to generate')
    
    parser.add_argument('--all-ngrams', '-a',
                       action='store_true',
                       help='Generate tag clouds for all n-gram types (reasoning: 1gram, 2gram, 3gram, 4gram; action_phrases: verbs, nouns, pairs)')
    
    args = parser.parse_args()
    
    if '-' in args.ngram_range:
        start, end = map(int, args.ngram_range.split('-'))
        ngram_range = (start, end)
    else:
        n = int(args.ngram_range)
        ngram_range = (n, n)
    
    print("Tag Cloud Generator")
    print("=" * 40)
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Tag source: {args.tag_source}")
    print(f"N-gram range: {ngram_range}")
    print(f"Min frequency: {args.min_frequency}")
    print(f"Max frequency rate: {args.max_frequency_rate}")
    print(f"Max tags: {args.max_tags}")
    
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory '{args.input_dir}' not found")
        return
    
    if args.all_ngrams:
        if args.tag_source == "reasoning":
            ngram_configs = [
                (1, 1, "unigrams"),
                (2, 2, "bigrams"), 
                (3, 3, "trigrams"),
                (4, 4, "4grams")
            ]
        elif args.tag_source == "action_phrases":
            ngram_configs = [
                (1, 1, "verbs"),
                (2, 2, "nouns"), 
                (3, 3, "pairs")
            ]
        
        print(f"\nGenerating tag clouds for all {args.tag_source} types...")
        for start, end, name in ngram_configs:
            print(f"\n--- Processing {name} ---")
            generate_tag_cloud(
                input_directory=args.input_dir,
                output_directory=args.output_dir,
                tag_source=args.tag_source,
                ngram_range=(start, end),
                min_frequency=args.min_frequency,
                max_frequency_rate=args.max_frequency_rate,
                max_tags=args.max_tags
            )
    else:
        print(f"\nGenerating tag cloud...")
        generate_tag_cloud(
            input_directory=args.input_dir,
            output_directory=args.output_dir,
            tag_source=args.tag_source,
            ngram_range=ngram_range,
            min_frequency=args.min_frequency,
            max_frequency_rate=args.max_frequency_rate,
            max_tags=args.max_tags
        )
    
    print(f"\nTag cloud generation completed!")


if __name__ == "__main__":
    main()