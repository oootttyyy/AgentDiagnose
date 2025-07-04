import ast
import base64
import json
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict, Any

import spacy
import tiktoken
from tqdm import tqdm

_NLP_MODEL = None


def get_nlp_model():
    import benepar
    global _NLP_MODEL
    if _NLP_MODEL is None:
        _NLP_MODEL = spacy.load('en_core_web_md')
        if spacy.__version__.startswith('2'):
            _NLP_MODEL.add_pipe(benepar.BeneparComponent("benepar_en3"))
        else:
            _NLP_MODEL.add_pipe("benepar", config={"model": "benepar_en3"})
    return _NLP_MODEL


@dataclass
class Action:
    action: str
    reasoning: str
    action_type: str
    observation: str
    url: str
    label_line: int = -1
    total_line: int = 0
    label_pos: float = -1
    nth_step: int = 0
    output: str = ""
    output_root_verb: str = ""
    output_root_noun: str = ""
    output_verb_noun_pairs: List[str] = None
    image_encoding: Optional[str] = None
    
    def __post_init__(self):
        if self.output_verb_noun_pairs is None:
            self.output_verb_noun_pairs = []
    
    def __str__(self) -> str:
        return f"Action: {self.action}\nReasoning: {self.reasoning}"
    
    def to_dict(self) -> dict:
        return {
            "action": self.action,
            "reasoning": self.reasoning,
            "action_type": self.action_type,
            "observation": self.observation,
            "url": self.url,
            "label_line": self.label_line,
            "total_line": self.total_line,
            "label_pos": self.label_pos,
            "nth_step": self.nth_step,
            "output": self.output,
            "output_root_verb": self.output_root_verb,
            "output_root_noun": self.output_root_noun,
            "output_verb_noun_pairs": self.output_verb_noun_pairs,
            "image_encoding": self.image_encoding
        }


class Trajectory:
    def __init__(self, objective: str, actions: List[Action], task_website: str, reward: int = -1):
        self.objective = objective
        self.actions = actions
        self.task_website = task_website
        self.reward = reward
    
    @classmethod
    def from_browsergym_pickle(cls, exp_dir):
        from browsergym.experiments.loop import ExpResult
        
        exp_dir = Path(exp_dir)
        exp_result = ExpResult(exp_dir)
        actions = []
        objective = ""
        task_website = ""
        
        total_steps = exp_result.summary_info.get('n_steps', 0)
        
        for i in range(total_steps + 1):
            try:
                step_info = exp_result.get_step_info(i)
                
                observation = ""
                if step_info.obs and 'axtree_txt' in step_info.obs:
                    observation = step_info.obs['axtree_txt']
                
                image_encoding = None
                screenshot_path = exp_dir / f"screenshot_step_{i}.png"
                if screenshot_path.exists():
                    try:
                        with open(screenshot_path, 'rb') as img_file:
                            image_encoding = base64.b64encode(img_file.read()).decode('utf-8')
                    except Exception as e:
                        print(f"Error loading screenshot for step {i}: {e}")
                        image_encoding = None
                
                website = ""
                if step_info.obs and 'url' in step_info.obs:
                    website = step_info.obs['url'].replace('https://', '')
                
                if i == 0 and step_info.obs and 'goal' in step_info.obs:
                    objective = step_info.obs['goal']
                    task_website = website
                
                action_str = step_info.action if step_info.action else ""
                reasoning = step_info.agent_info.get('think', '') if step_info.agent_info else ""
                
                action_type = ""
                if action_str:
                    if 'click(' in action_str:
                        action_type = 'click'
                    elif 'type(' in action_str:
                        action_type = 'type'
                    elif 'fill(' in action_str:
                        action_type = 'fill'
                
                label_line = -1
                total_line = len(observation.split('\n'))
                label_pos = -1
                
                if action_str:
                    bid_match = re.search(r"['\"](\d+)['\"]", action_str)
                    if bid_match:
                        bid = bid_match.group(1)
                        for j, line in enumerate(observation.split('\n')):
                            if f'[{bid}]' in line:
                                label_line = j
                                label_pos = j / total_line if total_line > 0 else 0
                                break
                
                output = f"{reasoning}\n\nAction: {action_str}" if reasoning or action_str else ''
                
                action = Action(
                    action=action_str,
                    reasoning=reasoning,
                    action_type=action_type,
                    observation=observation,
                    url=website,
                    label_line=label_line,
                    total_line=total_line,
                    label_pos=label_pos,
                    nth_step=i + 1,
                    output=output,
                    output_root_verb="",
                    output_root_noun="",
                    output_verb_noun_pairs=[],
                    image_encoding=image_encoding
                )
                
                actions.append(action)
                
            except Exception as e:
                print(f"Error loading step {i}: {e}")
                break
        
        return cls(objective, actions, task_website)

    def to_json(self):
        return {
            "objective": self.objective,
            "actions": [action.to_dict() for action in self.actions],
            "task_website": self.task_website,
            "reward": self.reward
        }

    @classmethod
    def from_json(cls, json_data):
        actions = []
        for action_data in json_data.get("actions", []):
            action = Action(
                action=action_data.get("action", ""),
                reasoning=action_data.get("reasoning", ""),
                action_type=action_data.get("action_type", ""),
                observation=action_data.get("observation", ""),
                url=action_data.get("url", ""),
                label_line=action_data.get("label_line", -1),
                total_line=action_data.get("total_line", 0),
                label_pos=action_data.get("label_pos", -1),
                nth_step=action_data.get("nth_step", 0),
                output=action_data.get("output", ""),
                output_root_verb=action_data.get("output_root_verb", ""),
                output_root_noun=action_data.get("output_root_noun", ""),
                output_verb_noun_pairs=action_data.get("output_verb_noun_pairs", []),
                image_encoding=action_data.get("image_encoding", None)
            )
            actions.append(action)
        
        return cls(
            objective=json_data.get("objective", ""),
            actions=actions,
            task_website=json_data.get("task_website", ""),
            reward=json_data.get("reward", -1)
        )

    @classmethod
    def from_cuga_record(cls, file_path: str) -> 'Trajectory':
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            objective = data.get('intent', '')
            actions = []
            
            success = data.get('success', False)
            reward = 1 if success else 0
            
            steps = data.get('steps', [])
            
            i = 0
            while i < len(steps):
                current_step = steps[i]
                
                if current_step.get('name') == 'PlannerAgent':
                    next_step = None
                    is_final_answer = False
                    
                    if i + 1 < len(steps):
                        next_step = steps[i + 1]
                        if next_step.get('name') == 'ActionAgent':
                            is_final_answer = False
                            i += 2
                        elif next_step.get('name') == 'FinalAnswerAgent':
                            is_final_answer = True
                            i += 2
                        else:
                            i += 1
                            continue
                    else:
                        i += 1
                        continue
                    
                    reasoning = ""
                    plan_text = current_step.get('plan', '')
                    if plan_text:
                        try:
                            plan_data = json.loads(plan_text)
                            reasoning = plan_data.get('instruction', '')
                        except (json.JSONDecodeError, TypeError):
                            reasoning = plan_text
                    
                    if is_final_answer:
                        action_formatted = "Final Answer"
                        action_type = "final_answer"
                        
                        final_answer_data = next_step.get('data', '')
                        if final_answer_data:
                            try:
                                data_json = json.loads(final_answer_data)
                                final_answer = data_json.get('final_answer', '')
                                if final_answer:
                                    action_formatted = f"Final Answer: {final_answer}"
                            except (json.JSONDecodeError, TypeError):
                                if final_answer_data.strip():
                                    action_formatted = f"Final Answer: {final_answer_data}"
                    else:
                        action_formatted = next_step.get('action_formatted', '') if next_step else ''
                        action_type = next_step.get('action_type', '') if next_step else ''
                    
                    observation = current_step.get('observation_before', '')
                    
                    image_encoding = None
                    image_data = current_step.get('image_before', '')
                    if image_data and image_data.strip():
                        if image_data.startswith('data:image/png;base64,'):
                            image_data = image_data.replace('data:image/png;base64,', '')
                        
                        try:
                            missing_padding = len(image_data) % 4
                            if missing_padding:
                                image_data += '=' * (4 - missing_padding)
                            
                            base64.b64decode(image_data[:100])
                            image_encoding = image_data
                        except Exception as e:
                            print(f"Warning: Invalid base64 image data in step, skipping image: {e}")
                    
                    url = ""
                    url = current_step.get('current_url', '')
                    
                    if not url and not is_final_answer and next_step:
                        url = next_step.get('current_url', '')
                    
                    label_line = -1
                    label_pos = -1
                    total_line = len(observation.split('\n')) if observation else 0
                    
                    if not is_final_answer and action_formatted and next_step:
                        action_args = next_step.get('action_args', {})
                        if isinstance(action_args, dict) and 'bid' in action_args:
                            bid = str(action_args['bid'])
                            observation_lines = observation.split('\n')
                            for line_idx, line in enumerate(observation_lines):
                                if f'[{bid}]' in line:
                                    label_line = line_idx
                                    label_pos = line_idx / total_line if total_line > 0 else 0
                                    break
                    
                    step_number = len(actions) + 1
                    output = f"{reasoning}\n\nAction: {action_formatted}" if reasoning or action_formatted else ''
                    
                    action = Action(
                        action=action_formatted,
                        action_type=action_type,
                        reasoning=reasoning,
                        observation=observation,
                        url=url,
                        label_line=label_line,
                        total_line=total_line,
                        label_pos=label_pos,
                        nth_step=step_number,
                        output=output,
                        output_root_verb="",
                        output_root_noun="",
                        output_verb_noun_pairs=[],
                        image_encoding=image_encoding
                    )
                    actions.append(action)
                    
                else:
                    i += 1
            
            return cls(objective=objective, actions=actions, task_website="", reward=reward)
            
        except Exception as e:
            raise ValueError(f"Error parsing CUGA file {file_path}: {e}")

    @classmethod
    def from_synatra_training_file(cls, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        trajectory_groups = {}
        
        for entry in tqdm(data, desc="Processing Synatra entries"):
            prompt = entry.get('prompt', '')
            response = entry.get('response', '')
            
            objective_match = re.search(r'objective = "(.*?)"', prompt)
            if not objective_match:
                continue
                
            objective = objective_match.group(1)
            
            website_match = re.search(r'website = "(.*?)"', prompt)
            website = website_match.group(1) if website_match else ""
            
            observation_match = re.search(r'observation = """(.*?)"""', prompt, re.DOTALL)
            observation = observation_match.group(1) if observation_match else ""
            
            observation = re.sub(r'\| Tab \d+: Playwright Trace Viewer \|', ' ', observation)
            observation = re.sub(r'Tab \d+: Playwright Trace Viewer \|', ' ', observation)
            observation = re.sub(r'\| Tab \d+: Playwright Trace Viewer', ' ', observation)
            
            past_actions = []
            actions_match = re.search(r'def solve\(\):(.*)', prompt, re.DOTALL)
            if actions_match:
                action_lines = actions_match.group(1).strip().split('\n')
                for line in action_lines:
                    line = line.strip()
                    if line.startswith('#'):
                        continue
                    if line and 'click(' in line or 'type(' in line or 'goto(' in line:
                        past_actions.append(line)
            
            current_action = ''
            reasoning = ''
            step_number = 1
            
            if response:
                step_match = re.search(r'#\s*step\s+(\d+):', response, re.IGNORECASE)
                if step_match:
                    step_number = int(step_match.group(1))
                
                if '#' in response:
                    parts = response.split('#', 1)
                    comment_part = parts[1].strip()
                    
                    reasoning_match = re.search(r'.*?:(.*?)(?=\n|$)', comment_part)
                    if reasoning_match:
                        reasoning = reasoning_match.group(1).strip()
                    
                    action_match = re.search(r'\n(.*?(?:click|type|goto|hover|key_press|go_back|go_forward|new_tab|close_tab|switch_tab|stop)\(.*?\))', comment_part)
                    if action_match:
                        current_action = action_match.group(1).strip()
                    else:
                        action_match = re.search(r'(?:click|type|goto|hover|key_press|go_back|go_forward|new_tab|close_tab|switch_tab|stop)\(.*?\)', comment_part)
                        if action_match:
                            current_action = action_match.group(0).strip()
                else:
                    action_match = re.search(r'(?:click|type|goto|hover|key_press|go_back|go_forward|new_tab|close_tab|switch_tab|stop)\(.*?\)', response)
                    if action_match:
                        current_action = action_match.group(0).strip()
            
            action_type = ""
            if current_action:
                for act_type in ['click', 'type', 'goto', 'hover', 'key_press', 'go_back', 
                                'go_forward', 'new_tab', 'close_tab', 'switch_tab', 'stop']:
                    if current_action.startswith(act_type):
                        action_type = act_type
                        break
            
            if objective not in trajectory_groups:
                trajectory_groups[objective] = {
                    'objective': objective,
                    'website': website,
                    'actions': [],
                    'max_step': 0
                }
            
            action = Action(
                action=current_action,
                reasoning=reasoning,
                action_type=action_type,
                observation=observation,
                url=website,
                label_line=-1,
                total_line=len(observation.split('\n')),
                label_pos=-1,
                nth_step=step_number,
                output=f"{reasoning}\n\nAction: {current_action}" if reasoning or current_action else '',
                output_root_verb="",
                output_root_noun="",
                output_verb_noun_pairs=[],
                image_encoding=None
            )
            
            trajectory_groups[objective]['actions'].append(action)
            trajectory_groups[objective]['max_step'] = max(
                trajectory_groups[objective]['max_step'],
                step_number
            )
        
        for obj, group_data in trajectory_groups.items():
            sorted_actions = sorted(group_data['actions'], key=lambda a: a.nth_step)
            trajectory_groups[obj]['actions'] = sorted_actions
        
        trajectories = []
        for group_data in trajectory_groups.values():
            trajectory = cls(
                objective=group_data['objective'],
                actions=group_data['actions'],
                task_website=group_data['website']
            )
            trajectories.append(trajectory)
        
        print(f"Created {len(trajectories)} trajectories from {len(data)} training entries")
        return trajectories

    @classmethod
    def from_agentTrek_training_file(cls, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        trajectories = []
        current_trajectory = None
        current_step_numbers = []
        
        for conversation in tqdm(data, desc="Processing AgentTrek entries"):
            system_msg = ""
            user_msg = ""
            assistant_msg = ""
            
            if 'messages' not in conversation:
                continue
                
            for message in conversation['messages']:
                if not isinstance(message, dict) or "role" not in message or "content" not in message:
                    continue
                    
                if message["role"] == "system":
                    system_msg = message["content"]
                elif message["role"] == "user":
                    user_msg = message["content"]
                elif message["role"] == "assistant":
                    assistant_msg = message["content"]
            
            if not user_msg or not assistant_msg:
                continue
                
            goal_match = re.search(r'## Goal:\s*(.*?)(?=\n\n|\n#)', user_msg, re.DOTALL)
            if not goal_match:
                continue
                
            objective = goal_match.group(1).strip()
            
            observation_match = re.search(r'## AXTree:\s*(.*?)(?=\n\n#|\n\n##|\n\n\n#|\n\n$)', user_msg, re.DOTALL)
            current_observation = observation_match.group(1).strip() if observation_match else ""
            
            website = ""
            website_match = re.search(r'RootWebArea [\'"]([^\'"]*)[\'"]\,', current_observation)
            if website_match:
                website = website_match.group(1)
            
            history_match = re.search(r'# History of interaction with the task:(.*?)(?=# Action space:|\Z)', user_msg, re.DOTALL)
            history_steps = []
            
            if history_match:
                history_text = history_match.group(1).strip()
                step_sections = re.findall(r'## step (\d+)', history_text)
                if step_sections:
                    history_steps = [int(step) for step in step_sections]
            
            current_step = max(history_steps) + 1 if history_steps else 0
            
            is_new_trajectory = False
            
            if not current_trajectory:
                is_new_trajectory = True
            elif not history_steps and current_step_numbers:
                is_new_trajectory = True
            elif history_steps and set(history_steps) != set(current_step_numbers):
                is_new_trajectory = True
            
            if is_new_trajectory:
                if current_trajectory and current_trajectory['actions']:
                    actions = []
                    for action_data in current_trajectory['actions']:
                        action = Action(
                            action=action_data['action'],
                            reasoning=action_data['reasoning'],
                            action_type=action_data['action_type'],
                            observation=action_data['observation'],
                            url=action_data['url'],
                            label_line=-1,
                            total_line=len(action_data['observation'].split('\n')) if action_data['observation'] else 0,
                            label_pos=-1,
                            nth_step=action_data['nth_step'],
                            output=f"{action_data['reasoning']}\n\nAction: {action_data['action']}" if action_data['reasoning'] or action_data['action'] else '',
                            output_root_verb="",
                            output_root_noun="",
                            output_verb_noun_pairs=[],
                            image_encoding=None
                        )
                        actions.append(action)
                    
                    trajectory = cls(
                        objective=current_trajectory['objective'],
                        actions=actions,
                        task_website=current_trajectory['website']
                    )
                    trajectories.append(trajectory)
                
                current_trajectory = {
                    'objective': objective,
                    'website': website,
                    'actions': []
                }
                current_step_numbers = history_steps.copy()
            
            current_step_numbers.append(current_step)
            
            action_match = re.search(r'<action>\s*(.*?)\s*</action>', assistant_msg, re.DOTALL)
            think_match = re.search(r'<think>\s*(.*?)\s*</think>', assistant_msg, re.DOTALL)
            memory_match = re.search(r'<memory>\s*(.*?)\s*</memory>', assistant_msg, re.DOTALL)
            
            current_action = action_match.group(1).strip() if action_match else ""
            reasoning = think_match.group(1).strip() if think_match else ""
            memory = memory_match.group(1).strip() if memory_match else ""
            
            if not current_action:
                continue
                
            action_type = ""
            for act_type in ['click', 'fill', 'goto', 'hover', 'press', 'noop', 
                           'scroll', 'select_option', 'focus', 'clear', 'drag_and_drop',
                           'go_back', 'go_forward', 'upload_file', 'send_msg_to_user']:
                if current_action.startswith(act_type + "("):
                    action_type = act_type
                    break
            
            url = website
            if action_type == "goto" and "http" in current_action:
                url_match = re.search(r"goto\([\'\"](https?://[^\'\"]+)[\'\"]", current_action)
                if url_match:
                    url = url_match.group(1)
                    url = url.replace("https://", "").replace("http://", "")
            
            combined_reasoning = ""
            if reasoning:
                combined_reasoning += reasoning
            if memory:
                if combined_reasoning:
                    combined_reasoning += "\n\n"
                combined_reasoning += f"Memory: {memory}"
            
            current_trajectory['actions'].append({
                'action': current_action,
                'reasoning': combined_reasoning,
                'action_type': action_type,
                'observation': current_observation,
                'url': url,
                'nth_step': current_step
            })
        
        if current_trajectory and current_trajectory['actions']:
            actions = []
            for action_data in current_trajectory['actions']:
                action = Action(
                    action=action_data['action'],
                    reasoning=action_data['reasoning'],
                    action_type=action_data['action_type'],
                    observation=action_data['observation'],
                    url=action_data['url'],
                    label_line=-1,
                    total_line=len(action_data['observation'].split('\n')) if action_data['observation'] else 0,
                    label_pos=-1,
                    nth_step=action_data['nth_step'],
                    output=f"{action_data['reasoning']}\n\nAction: {action_data['action']}" if action_data['reasoning'] or action_data['action'] else '',
                    output_root_verb="",
                    output_root_noun="",
                    output_verb_noun_pairs=[],
                    image_encoding=None
                )
                actions.append(action)
            
            trajectory = cls(
                objective=current_trajectory['objective'],
                actions=actions,
                task_website=current_trajectory['website']
            )
            trajectories.append(trajectory)
        
        print(f"Created {len(trajectories)} trajectories from {len(data)} AgentTrek entries")
        return trajectories

    def label_verb_noun_pairs(self, print_stats=False):
        data = [[action] for action in self.actions]
        labeled_data = label_verb_noun(data, print_stats=print_stats)
        
        for i, action_list in enumerate(labeled_data):
            if i < len(self.actions):
                self.actions[i].output_root_verb = action_list[0].output_root_verb
                self.actions[i].output_root_noun = action_list[0].output_root_noun
                self.actions[i].output_verb_noun_pairs = action_list[0].output_verb_noun_pairs
        
        return self


def find_largest_step(directory):
    steps = []
    for filename in os.listdir(directory):
        if filename.startswith('agent_input_step_') and filename.endswith('.txt'):
            step_num = int(filename.replace('agent_input_step_', '').replace('.txt', ''))
            steps.append(step_num)
    return max(steps) + 1 if steps else 0


def label_verb_noun(data, plot_name="", print_stats=False):
    nlp = get_nlp_model()
    
    for i in range(len(data)):
        for j in range(len(data[i])):
            if data[i][j].reasoning:
                text = data[i][j].reasoning
            else:
                if print_stats:
                    print("No text to process for this action")
                continue
            
            text = re.sub(r'"[^"]*?"', 'x', text.strip().replace('\\',''))
            
            if print_stats:
                print(f"\nAction {i+1}.{j+1} - Processing text")
            
            verb, noun = '', ''
            verb_noun_pairs = []
            
            try:
                sentences = text.split('.')[:5]
                
                for s_idx, sentence in enumerate(sentences):
                    if not sentence.strip():
                        continue
                        
                    clean_sentence = sentence.strip() + '.'
                    
                    if print_stats:
                        print(f"Processing sentence {s_idx+1}: {clean_sentence}")
                    
                    doc = nlp(clean_sentence)
                    
                    for token in doc:
                        if token.pos_ == "VERB":
                            for child in token.children:
                                if child.dep_ == "dobj" and child.pos_ == "NOUN":
                                    pair = (token.lemma_, child.lemma_)
                                    verb_noun_pairs.append(pair)
                                    if print_stats:
                                        print(f"Found pair: {pair}")
                
                if verb_noun_pairs:
                    verb, noun = verb_noun_pairs[0]
                    if print_stats:
                        print(f"Root verb: {verb}, Root noun: {noun}")
                        print(f"All pairs: {verb_noun_pairs}")
                elif print_stats:
                    print("No verb-noun pairs found")
            except Exception as e:
                if print_stats:
                    print(f"Error processing text: {e}")
            
            data[i][j].output_root_verb = verb
            data[i][j].output_root_noun = noun
            data[i][j].output_verb_noun_pairs = verb_noun_pairs
    
    return data


def extract_bid(code):
    if not code:
        return None
    match = re.search(r"\[(\d+)\]", code)
    return match.group(1) if match else None


def extract_action_type(code):
    if not code:
        return ""
    
    action_types = {
        "click": r"^click\s*\[",
        "type": r"^type\s*\[",
        "fill": r"^fill\s*\[",
        "hover": r"^hover\s*\[",
        "press": r"^press\s*\[",
        "scroll": r"^scroll\s*\[",
        "new_tab": r"^new_tab",
        "tab_focus": r"^tab_focus",
        "close_tab": r"^close_tab",
        "goto": r"^goto\s*\[",
        "go_back": r"^go_back",
        "go_forward": r"^go_forward",
        "stop": r"^stop\s*\["
    }
    
    for action_type, pattern in action_types.items():
        if re.search(pattern, code):
            return action_type
            
    return ""
