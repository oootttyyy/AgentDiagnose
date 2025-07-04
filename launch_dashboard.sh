#!/bin/bash
set -e  
INPUT_DIR="examples/sample_trajectories"
LABELED_DIR="examples/sample_trajectories_labeled"
TAG_CLOUD_DIR="tag_cloud"
EMBEDDINGS_DIR="embeddings_output"
RESULTS_FILE="examples/results.json"
DASHBOARD_HOST="0.0.0.0"
DASHBOARD_PORT="8080"

python generate_verb_nouns.py --input-dir "$INPUT_DIR" --output-dir "$LABELED_DIR"
python generate_tag_cloud.py --input-dir "$LABELED_DIR" --output-dir "$TAG_CLOUD_DIR" --tag-source reasoning --all-ngrams
python generate_tag_cloud.py --input-dir "$LABELED_DIR" --output-dir "$TAG_CLOUD_DIR" --tag-source action_phrases --all-ngrams
python generate_embeddings.py --input-dir "$LABELED_DIR" --output-dir "$EMBEDDINGS_DIR"

python evaluate_trajectories.py \
    --input "$INPUT_DIR" \
    --scorers reasoning_quality objective_quality \
    --output-json "$RESULTS_FILE" \
    --web-dashboard \
    --dashboard-host "$DASHBOARD_HOST" \
    --dashboard-port "$DASHBOARD_PORT" \
    --no-auto-browser

echo "Dashboard running at http://$DASHBOARD_HOST:$DASHBOARD_PORT" 