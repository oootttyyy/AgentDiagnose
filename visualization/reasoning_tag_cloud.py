import json
import math
import re
from collections import Counter, defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from evaluator.trajectory import Trajectory


class ReasoningTextProcessor:
    def __init__(self, 
                 min_frequency: int = 3,
                 max_frequency_rate: float = 0.8,
                 max_tags: int = 100,
                 ngram_range: Tuple[int, int] = (1, 4)):
        self.min_frequency = min_frequency
        self.max_frequency_rate = max_frequency_rate
        self.max_tags = max_tags
        self.ngram_range = ngram_range
        
        self.stop_words = {
            'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
            'by', 'from', 'up', 'about', 'into', 'through', 'during', 'before',
            'after', 'above', 'below', 'between', 'among', 'around', 'is', 'are',
            'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do',
            'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might',
            'must', 'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he',
            'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them', 'my',
            'your', 'his', 'our', 'their', 'a', 'an', 'as', 'if', 'then',
            'than', 'so', 'very', 'just', 'now', 'here', 'there', 'when',
            'where', 'why', 'how', 'what', 'which', 'who', 'whom', 'whose',
            'unless', 'until', 'while', 'during', 'action', 'step',
            'next', 'perform', 'seems', 'appears', 'looks', 'given', 'based'
        }
        
        self.domain_stop_words = {
            'trajectory', 'agent', 'perform', 'action', 'step', 'next', 'current',
            'state', 'page', 'website', 'browser', 'element', 'observation', 'lti', 'cmu', 'cs', 'edu', 'http', 'metis',
        }
        
        self.all_stop_words = self.stop_words | self.domain_stop_words
    
    def extract_reasoning_texts(self, trajectories: List[Trajectory]) -> Dict[str, List[str]]:
        reasoning_texts = {}
        
        for i, trajectory in enumerate(trajectories):
            trajectory_id = f"{i:04d}"
            texts = []
            
            if hasattr(trajectory, 'actions') and trajectory.actions:
                for action in trajectory.actions:
                    if hasattr(action, 'reasoning') and action.reasoning:
                        cleaned_text = self._clean_text(action.reasoning)
                        if cleaned_text:
                            texts.append(cleaned_text)
            
            if texts:
                reasoning_texts[trajectory_id] = texts
                
        return reasoning_texts
    
    def _clean_text(self, text: str) -> str:
        if not text:
            return ""
        
        text = text.lower()
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s\-_]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def generate_ngrams(self, tokens: List[str], n: int) -> List[str]:
        if len(tokens) < n:
            return []
        
        ngrams = []
        for i in range(len(tokens) - n + 1):
            ngram = ' '.join(tokens[i:i + n])
            ngrams.append(ngram)
        
        return ngrams
    
    def tokenize_and_extract_ngrams(self, text: str) -> List[str]:
        if not text:
            return []
        
        tokens = text.split()
        tokens = [token for token in tokens if token not in ['lti', 'cmu', 'cs', 'edu', 'http', 'metis'] and len(token) > 1]
        
        filtered_tokens = [token for token in tokens if token not in self.all_stop_words and len(token) > 1]
        
        all_ngrams = []
        
        for n in range(self.ngram_range[0], self.ngram_range[1] + 1):
            if n == 1:
                all_ngrams.extend(filtered_tokens)
            else:
                ngrams = self.generate_ngrams(tokens, n)
                for ngram in ngrams:
                    ngram_tokens = ngram.split()
                    if any(token not in self.all_stop_words for token in ngram_tokens):
                        all_ngrams.append(ngram)
        
        return all_ngrams
    
    def calculate_frequencies(self, texts: List[str], ngram_size: int) -> Tuple[Dict[str, int], Dict[str, int], Dict[str, float]]:
        all_ngrams = []
        doc_term_sets = []
        
        for text in texts:
            ngrams = self.generate_ngrams(text, ngram_size)
            all_ngrams.extend(ngrams)
            doc_term_sets.append(set(ngrams))
        
        term_frequencies = Counter(all_ngrams)
        
        doc_frequencies = {}
        for term in term_frequencies:
            doc_freq = sum(1 for doc_terms in doc_term_sets if term in doc_terms)
            doc_frequencies[term] = doc_freq
        
        total_docs = len(texts)
        max_doc_freq = int(total_docs * self.max_frequency_rate)
        
        filtered_terms = {}
        for term, doc_freq in doc_frequencies.items():
            if doc_freq >= self.min_frequency and doc_freq <= max_doc_freq:
                filtered_terms[term] = doc_freq
        
        tfidf_scores = {}
        for term in filtered_terms:
            tf = term_frequencies[term] / len(all_ngrams) if all_ngrams else 0
            idf = math.log(total_docs / doc_frequencies[term]) if doc_frequencies[term] > 0 else 0
            tfidf_scores[term] = tf * idf
        
        filtered_term_frequencies = {term: term_frequencies[term] for term in filtered_terms}
        filtered_doc_frequencies = {term: doc_frequencies[term] for term in filtered_terms}
        
        return filtered_term_frequencies, filtered_doc_frequencies, tfidf_scores


class ReasoningTagCloudGenerator:
    def __init__(self, processor: Optional[ReasoningTextProcessor] = None):
        self.processor = processor or ReasoningTextProcessor()
        self.processing_stats = {}
    
    def process_trajectories(self, trajectories: List[Trajectory]) -> Dict[str, Any]:
        trajectory_reasoning_texts = self.processor.extract_reasoning_texts(trajectories)
        
        all_reasoning_texts = []
        trajectory_map = []
        
        for trajectory_id, reasoning_texts in trajectory_reasoning_texts.items():
            combined_text = ' '.join(reasoning_texts)
            cleaned_text = self.processor._clean_text(combined_text)
            
            if cleaned_text:
                all_reasoning_texts.append(cleaned_text)
                trajectory_map.append(trajectory_id)
        
        if not all_reasoning_texts:
            return {
                'tags': [],
                'statistics': {'total_reasoning_texts': 0, 'unique_terms_found': 0},
                'metadata': {'processing_date': str(datetime.now())}
            }
        
        all_ngrams = []
        doc_term_sets = []
        
        for text in all_reasoning_texts:
            ngrams = self.processor.tokenize_and_extract_ngrams(text)
            all_ngrams.extend(ngrams)
            doc_term_sets.append(set(ngrams))
        
        term_frequencies = Counter(all_ngrams)
        
        doc_frequencies = {}
        for term in term_frequencies:
            doc_freq = sum(1 for doc_terms in doc_term_sets if term in doc_terms)
            doc_frequencies[term] = doc_freq
        
        total_docs = len(all_reasoning_texts)
        max_doc_freq = int(total_docs * self.processor.max_frequency_rate)
        
        filtered_terms = {}
        for term, doc_freq in doc_frequencies.items():
            if doc_freq >= self.processor.min_frequency and doc_freq <= max_doc_freq:
                filtered_terms[term] = doc_freq
        
        tfidf_scores = {}
        for term in filtered_terms:
            tf = term_frequencies[term] / len(all_ngrams) if all_ngrams else 0
            idf = math.log(total_docs / doc_frequencies[term]) if doc_frequencies[term] > 0 else 0
            tfidf_scores[term] = tf * idf
        
        tags = []
        for term in sorted(filtered_terms.keys(), key=lambda x: tfidf_scores.get(x, 0), reverse=True):
            trajectories_using = []
            for i, text in enumerate(all_reasoning_texts):
                text_ngrams = self.processor.tokenize_and_extract_ngrams(text)
                if term in text_ngrams:
                    if trajectory_map[i] not in trajectories_using:
                        trajectories_using.append(trajectory_map[i])
            
            tag_data = {
                'text': term,
                'weight': tfidf_scores.get(term, 0.0),
                'raw_frequency': term_frequencies[term],
                'doc_frequency': doc_frequencies[term],
                'tfidf_score': tfidf_scores.get(term, 0.0),
                'trajectories_using': trajectories_using
            }
            tags.append(tag_data)
        
        if len(tags) > self.processor.max_tags:
            tags = tags[:self.processor.max_tags]
        
        statistics = {
            'total_reasoning_texts': len(all_reasoning_texts),
            'unique_terms_found': len(term_frequencies),
            'total_trajectories_processed': len(trajectory_reasoning_texts),
            'frequency_threshold': self.processor.min_frequency
        }
        
        metadata = {
            'ngram_range': self.processor.ngram_range,
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


def create_reasoning_tag_cloud(trajectories: List[Trajectory], 
                             min_frequency: int = 3,
                             max_frequency_rate: float = 0.8,
                             max_tags: int = 100,
                             ngram_range: Tuple[int, int] = (1, 4)) -> Dict[str, Any]:
    processor = ReasoningTextProcessor(
        min_frequency=min_frequency,
        max_frequency_rate=max_frequency_rate,
        max_tags=max_tags,
        ngram_range=ngram_range
    )
    
    generator = ReasoningTagCloudGenerator(processor)
    return generator.process_trajectories(trajectories) 