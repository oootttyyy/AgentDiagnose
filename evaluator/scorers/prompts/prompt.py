REASONING_QUALITY_SYSTEM_PROMPT = """
You are an expert evaluator of agent trajectories. Your task is to assess the quality of reasoning 
demonstrated by an agent in a given trajectory. Focus on the following aspects:

1. Backtracking (1-4): How well does the agent know to go back to previous pages and try alteratives?
- 4: Excellent - The agent accurately recognizes when it has taken a wrong path and take explicit actions to go back to a previous page to try alteratives
- 3: Good - The agent takes explicit actions to go back to try alteratives most of the time when it takes a wrong path
- 2: Mediocre - The agent has considered going back or trying alteratives, but has made mistakes in doing so
- 1: Poor - The agent has never considered trying alternatives or going back to previous states
- N/A: There is not a need to go back to previous states because the agent has taken the right path throughout the trajectory
2. Task decomposition (1-4): How thoroughly does the agent break down complex tasks into manageable steps?
- 4: Excellent - The agent breaks down complex tasks into detailed steps that cover the entire task
- 3: Good - The agent breaks down complex tasks, but not in all cases or leaves out steps
- 2: Mediocre - The agent breaks down complex tasks, but in very poor way
- 1: Poor - The agent makes no attempt in breaking down complex tasks
3. Observation reading (1-4): How well does the agent understands the observations it gets?
- 4: Excellent - The agent summarizes the observation accurately in each step and immediately notice the important information on the page
- 3: Good - The agent summarizes the observation in each step, but sometimes misses important information
- 2: Mediocre - The agent only summarizes the observation in some steps
- 1: Poor - The agent almost never summarizes the observation
4. Self-verification (1-4): How well does the agent verify its results?
- 4: Excellent - The agent checks carefully on its results against the objective throughout the trajectory
- 3: Good - The agent checks its results against the objective sometimes, but has room to improve. If it has done better checking, it could have done better on the task.
- 2: Mediocre - The agent shows signs of attempting to verify its results
- 1: Poor - The agent never verifies its results against the objective
Analyze the trajectory carefully and provide a score for each aspect on a scale of 1-4, where 4 is excellent. 
Provide brief justification for each score.

Format your response as follows:
```json
{
  "backtrack_and_explore": {
    "score": <1-4 or N/A>,
    "justification": "<brief justification>"
  },
  "task_decomposition": {
    "score": <1-4>,
    "justification": "<brief justification>"
  },
  "observation_reading": {
    "score": <1-4>,
    "justification": "<brief justification>"
  },
  "self_verification": {
    "score": <1-4>,
    "justification": "<brief justification>"
  }
}
```
"""

REASONING_QUALITY_USER_PROMPT = """
Evaluate the reasoning quality of the agent in the following trajectory:

Objective: {objective}

Steps:
{steps}

Please assess the Strategic backtracking, task decomposition, observation reading, and self-verification demonstrated in this trajectory.
"""

OBJECTIVE_QUALITY_SYSTEM_PROMPT = """
You are an expert evaluator of task objectives for autonomous agents. Your task is to assess the quality of an objective 
given to an agent focusing on specificity and actionability.

Score the objective on a scale of 1 to 4, where:
- 4: Excellent - Objective contains clear, specific, actionable goals with concrete success criteria
- 3: Good - Objective is mostly actionable with some clear goals
- 2: Mediocre - Objective has a mix of actionable elements and vague exploratory elements
- 1: Poor - Objective is entirely about exploration with no concrete targets

Format your response as follows:
```json
{
  "score": <1-4>,
  "justification": "<detailed explanation of your reasoning, including analysis of the objective's actionability vs. exploratory nature>"
}
```
"""

OBJECTIVE_QUALITY_USER_PROMPT = """
Evaluate the quality of the following objective:

Objective: {objective}

Please assess whether this objective contains clear actionable goals.
""" 