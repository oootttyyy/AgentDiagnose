import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

from tqdm import tqdm

from evaluator.trajectory import Trajectory

# All Part-of-Speech Tags:
ALL_POS_TAGS = set()


def process_trajectory(filename, data_dir, output_dir, print_stats=False):
    global ALL_POS_TAGS
    
    try:
        file_path = os.path.join(data_dir, filename)
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        trajectory = Trajectory.from_json(data)
        
        trajectory.label_verb_noun_pairs(print_stats=print_stats)
        
        verb_counter = Counter()
        noun_counter = Counter()
        pair_counter = Counter()
        
        for action in trajectory.actions:
            if action.output_root_verb:
                verb_counter[action.output_root_verb] += 1
            
            if action.output_root_noun:
                noun_counter[action.output_root_noun] += 1
            
            for verb, noun in action.output_verb_noun_pairs:
                pair_counter[(verb, noun)] += 1
        
        try:
            from evaluator.trajectory import get_nlp_model
            nlp = get_nlp_model()
            
            for action in trajectory.actions:
                text = ""
                if action.reasoning:
                    text = action.reasoning
                
                if text:
                    doc = nlp(text)
                    
                    for token in doc:
                        ALL_POS_TAGS.add(token.pos_)
        except Exception as e:
            if print_stats:
                print(f"Error collecting POS tags: {e}")
        
        output_path = os.path.join(output_dir, filename)
        with open(output_path, 'w') as f:
            json.dump(trajectory.to_json(), f, indent=2)
        
        return verb_counter, noun_counter, pair_counter
    
    except Exception as e:
        if print_stats:
            print(f"Error processing {filename}: {e}")
        return Counter(), Counter(), Counter()


def process_all_trajectories(data_dir, output_dir, print_stats=False):
    os.makedirs(output_dir, exist_ok=True)
    
    json_files = [f for f in os.listdir(data_dir) if f.endswith('.json')]
    
    if not json_files:
        print(f"No JSON files found in {data_dir}")
        return
    
    print(f"Found {len(json_files)} trajectory files")
    
    total_verb_counter = Counter()
    total_noun_counter = Counter()
    total_pair_counter = Counter()
    
    for filename in tqdm(json_files, desc="Processing trajectories"):
        verb_counter, noun_counter, pair_counter = process_trajectory(
            filename, data_dir, output_dir, print_stats
        )
        
        total_verb_counter.update(verb_counter)
        total_noun_counter.update(noun_counter)
        total_pair_counter.update(pair_counter)
    
    print(f"\nProcessing complete!")
    print(f"Total unique verbs: {len(total_verb_counter)}")
    print(f"Total unique nouns: {len(total_noun_counter)}")
    print(f"Total unique verb-noun pairs: {len(total_pair_counter)}")
    
    print(f"\nTop 10 verbs:")
    for verb, count in total_verb_counter.most_common(10):
        print(f"  {verb}: {count}")
    
    print(f"\nTop 10 nouns:")
    for noun, count in total_noun_counter.most_common(10):
        print(f"  {noun}: {count}")
    
    print(f"\nTop 10 verb-noun pairs:")
    for (verb, noun), count in total_pair_counter.most_common(10):
        print(f"  {verb} {noun}: {count}")
    
    print(f"\nAll Part-of-Speech Tags encountered: {sorted(ALL_POS_TAGS)}")
    
    stats_file = os.path.join(output_dir, "verb_noun_stats.json")
    with open(stats_file, 'w') as f:
        json.dump({
            "total_verbs": len(total_verb_counter),
            "total_nouns": len(total_noun_counter),
            "total_pairs": len(total_pair_counter),
            "top_verbs": dict(total_verb_counter.most_common(20)),
            "top_nouns": dict(total_noun_counter.most_common(20)),
            "top_pairs": {f"{verb} {noun}": count for (verb, noun), count in total_pair_counter.most_common(20)},
            "all_pos_tags": sorted(ALL_POS_TAGS)
        }, f, indent=2)
    
    print(f"\nStatistics saved to {stats_file}")


def main():
    parser = argparse.ArgumentParser(description="Process trajectories and label verb-noun pairs")
    parser.add_argument("--input-dir", "-i",
                        help="Directory containing trajectory JSON files")
    parser.add_argument("--output-dir", "-o",
                        help="Directory to save labeled trajectories")
    parser.add_argument("--print-stats", "-p", 
                        action="store_true",
                        help="Print verbose processing statistics")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory {args.input_dir} does not exist")
        return
    
    print(f"Processing trajectories from {args.input_dir}")
    print(f"Saving labeled trajectories to {args.output_dir}")
    
    process_all_trajectories(args.input_dir, args.output_dir, args.print_stats)


if __name__ == "__main__":
    sys.exit(main()) 

# ==================================================
# All Part-of-Speech Tags:
# ==================================================
# ADJ
# ADP
# ADV
# AUX
# CCONJ
# DET
# INTJ
# NOUN
# NUM
# PART
# PRON
# PROPN
# PUNCT
# SCONJ
# SPACE
# SYM
# VERB
# X
