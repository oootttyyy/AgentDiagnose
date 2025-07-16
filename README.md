# AgentDiagnose: An Open Toolkit for Diagnosing LLM Agent Trajectories

## Usage

AgentDiagnose provides two main ways to analyze agent trajectories:

### Option 1: All-In-One Dashboard

The easiest way to get started is to use the launch dashboard script, which runs the complete pipeline automatically:

```bash
./launch_dashboard.sh
```

This script will:
1. Extract verb-noun pairs from trajectories
2. Generate tag clouds for reasoning and action phrases
3. Generate embeddings on action phrases
4. Evaluate trajectories with quality scorers
5. Launch the interactive web dashboard

### Option 2: Run Evaluator With Custom Options

For more control over the evaluation of specific agent qualities, use the evaluator directly:

```bash
# Evaluate with specific scorers
python evaluate_trajectories.py --input examples/sample_trajectories --scorers reasoning_quality objective_quality --output-json results.json

# Dry run to estimate token usage and costs
python evaluate_trajectories.py --input examples/sample_trajectories --scorers reasoning_quality objective_quality --output-json results.json --dry-run
```

## Available Scorers

- `reasoning_quality`: Evaluates the quality of the agent's reasoning
- `objective_quality`: Assesses the task's objective quality
- `navigation_path`: Analyzes navigation paths within trajectories

## Configuration

Set your LLM API key in the environment:
```bash
export LLM_API_KEY="your-api-key-here"
```

## Web Dashboard

The web dashboard provides an interactive interface for:
- Visualizing trajectory evaluator's results
- Exploring reasoning patterns through tag clouds
- Analyzing action phrase distributions
- Examining embedding-based trajectory clustering

Access the dashboard at the URL displayed when launching (default`http://localhost:8080`).