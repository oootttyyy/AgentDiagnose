import json
import math
import re
from collections import Counter, defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from evaluator.trajectory import Trajectory


class ActionPhrasesProcessor:
    def __init__(self, 
                 min_frequency: int = 2,
                 max_frequency_rate: float = 0.8,
                 max_tags: int = 100,
                 phrase_type: str = "pairs"):
        self.min_frequency = min_frequency
        self.max_frequency_rate = max_frequency_rate
        self.max_tags = max_tags
        self.phrase_type = phrase_type
    
    def extract_action_phrases(self, trajectories: List[Trajectory]) -> Dict[str, List[str]]:
        action_phrases = {}
        
        for i, trajectory in enumerate(trajectories):
            trajectory_id = f"{i:04d}"
            phrases = []
            
            if hasattr(trajectory, 'actions') and trajectory.actions:
                for action in trajectory.actions:
                    if self.phrase_type == "verbs":
                        if hasattr(action, 'output_root_verb') and action.output_root_verb:
                            verb = action.output_root_verb.strip().lower()
                            if verb and len(verb) > 1:
                                phrases.append(verb)
                    
                    elif self.phrase_type == "nouns":
                        if hasattr(action, 'output_root_noun') and action.output_root_noun:
                            noun = action.output_root_noun.strip().lower()
                            if noun and len(noun) > 1:
                                phrases.append(noun)
                    
                    elif self.phrase_type == "pairs":
                        if hasattr(action, 'output_verb_noun_pairs') and action.output_verb_noun_pairs:
                            for pair in action.output_verb_noun_pairs:
                                if isinstance(pair, (list, tuple)) and len(pair) == 2:
                                    verb, noun = pair
                                    if verb and noun:
                                        verb = verb.strip().lower()
                                        noun = noun.strip().lower()
                                        if len(verb) > 1 and len(noun) > 1:
                                            phrase = f"{verb} {noun}"
                                            phrases.append(phrase)
            
            if phrases:
                action_phrases[trajectory_id] = phrases
                
        return action_phrases
    
    def calculate_frequencies(self, trajectory_phrases: Dict[str, List[str]]) -> Tuple[Dict[str, int], Dict[str, int], Dict[str, float]]:
        all_phrases = []
        doc_phrase_sets = []
        
        for trajectory_id, phrases in trajectory_phrases.items():
            all_phrases.extend(phrases)
            doc_phrase_sets.append(set(phrases))
        
        term_frequencies = Counter(all_phrases)
        
        doc_frequencies = {}
        for phrase in term_frequencies:
            doc_freq = sum(1 for doc_phrases in doc_phrase_sets if phrase in doc_phrases)
            doc_frequencies[phrase] = doc_freq
        
        total_docs = len(trajectory_phrases)
        max_doc_freq = int(total_docs * self.max_frequency_rate)
        
        filtered_terms = {}
        for phrase, doc_freq in doc_frequencies.items():
            if doc_freq >= self.min_frequency and doc_freq <= max_doc_freq:
                filtered_terms[phrase] = doc_freq
        
        tfidf_scores = {}
        for phrase in filtered_terms:
            tf = term_frequencies[phrase] / len(all_phrases) if all_phrases else 0
            idf = math.log(total_docs / doc_frequencies[phrase]) if doc_frequencies[phrase] > 0 else 0
            tfidf_scores[phrase] = tf * idf
        
        filtered_term_frequencies = {phrase: term_frequencies[phrase] for phrase in filtered_terms}
        filtered_doc_frequencies = {phrase: doc_frequencies[phrase] for phrase in filtered_terms}
        
        return filtered_term_frequencies, filtered_doc_frequencies, tfidf_scores


class ActionPhrasesTagCloudGenerator:
    def __init__(self, processor: Optional[ActionPhrasesProcessor] = None):
        self.processor = processor or ActionPhrasesProcessor()
        self.processing_stats = {}
    
    def process_trajectories(self, trajectories: List[Trajectory]) -> Dict[str, Any]:
        trajectory_phrases = self.processor.extract_action_phrases(trajectories)
        
        if not trajectory_phrases:
            return {
                'tags': [],
                'statistics': {'total_action_phrases': 0, 'unique_terms_found': 0},
                'metadata': {'processing_date': str(datetime.now())}
            }
        
        term_frequencies, doc_frequencies, tfidf_scores = self.processor.calculate_frequencies(trajectory_phrases)
        
        tags = []
        for phrase in sorted(tfidf_scores.keys(), key=lambda x: tfidf_scores.get(x, 0), reverse=True):
            trajectories_using = []
            for trajectory_id, phrases in trajectory_phrases.items():
                if phrase in phrases:
                    trajectories_using.append(trajectory_id)
            
            tag_data = {
                'text': phrase,
                'weight': tfidf_scores.get(phrase, 0.0),
                'raw_frequency': term_frequencies[phrase],
                'doc_frequency': doc_frequencies[phrase],
                'tfidf_score': tfidf_scores.get(phrase, 0.0),
                'trajectories_using': trajectories_using
            }
            tags.append(tag_data)
        
        if len(tags) > self.processor.max_tags:
            tags = tags[:self.processor.max_tags]
        
        total_phrases = sum(len(phrases) for phrases in trajectory_phrases.values())
        statistics = {
            'total_action_phrases': total_phrases,
            'unique_terms_found': len(term_frequencies),
            'total_trajectories_processed': len(trajectory_phrases),
            'frequency_threshold': self.processor.min_frequency,
            'phrase_type': self.processor.phrase_type
        }
        
        metadata = {
            'phrase_type': self.processor.phrase_type,
            'min_frequency': self.processor.min_frequency,
            'max_frequency_rate': self.processor.max_frequency_rate,
            'processing_date': str(datetime.now()),
            'total_trajectories': len(trajectories)
        }
        
        return {
            'tags': tags,
            'statistics': statistics,
            'metadata': metadata
        }
    
    def export_tag_data(self, tag_data: Dict[str, Any], format_type: str = 'json') -> str:
        if format_type == 'json':
            return json.dumps(tag_data, indent=2)
        
        elif format_type == 'd3':
            d3_data = {
                'words': [
                    {
                        'text': tag['text'],
                        'size': max(10, int(tag['weight'] * 100)),
                        'frequency': tag['raw_frequency']
                    }
                    for tag in tag_data['tags']
                ]
            }
            return json.dumps(d3_data, indent=2)
        
        elif format_type == 'wordcloud2':
            wordcloud2_data = [
                [tag['text'], tag['weight'] * 100]
                for tag in tag_data['tags']
            ]
            return json.dumps(wordcloud2_data, indent=2)
        
        else:
            raise ValueError(f"Unsupported format type: {format_type}")
    
    def apply_filters(self, tag_data: Dict[str, Any], 
                     min_weight: float = 0.0,
                     min_frequency: int = 1,
                     trajectory_filter: Optional[List[str]] = None) -> Dict[str, Any]:
        filtered_tags = []
        
        for tag in tag_data['tags']:
            if tag['weight'] < min_weight or tag['raw_frequency'] < min_frequency:
                continue
            
            if trajectory_filter is not None:
                if not any(traj_id in trajectory_filter for traj_id in tag['trajectories_using']):
                    continue
            
            filtered_tags.append(tag)
        
        filtered_data = tag_data.copy()
        filtered_data['tags'] = filtered_tags
        filtered_data['metadata']['filters_applied'] = {
            'min_weight': min_weight,
            'min_frequency': min_frequency,
            'trajectory_filter': trajectory_filter is not None
        }
        
        return filtered_data


def create_action_phrases_tag_cloud(trajectories: List[Trajectory], 
                                  min_frequency: int = 2,
                                  max_frequency_rate: float = 0.8,
                                  max_tags: int = 100,
                                  phrase_type: str = "pairs") -> Dict[str, Any]:
    processor = ActionPhrasesProcessor(
        min_frequency=min_frequency,
        max_frequency_rate=max_frequency_rate,
        max_tags=max_tags,
        phrase_type=phrase_type
    )
    
    generator = ActionPhrasesTagCloudGenerator(processor)
    return generator.process_trajectories(trajectories) 