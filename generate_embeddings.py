import argparse
import json
import os
import pickle
import sys
from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from evaluator.trajectory import Trajectory


class VerbNounEmbeddingStore:
    def __init__(self, model_name: str = "Qwen/Qwen3-Embedding-0.6B", faiss_index_type: str = "flat",
                 model_kwargs: Optional[Dict[str, Any]] = None, tokenizer_kwargs: Optional[Dict[str, Any]] = None,
                 prompt_name: str = "query"):
        self.model_name = model_name
        self.faiss_index_type = faiss_index_type
        self.model_kwargs = model_kwargs or {"device_map": "auto"}
        self.tokenizer_kwargs = tokenizer_kwargs or {"padding_side": "left"}
        self.prompt_name = prompt_name
        self.model = None
        self.embedding_dim = None
        
        self.verb_embeddings = []
        self.noun_embeddings = []
        self.pair_embeddings = []
        
        self.verb_texts = []
        self.noun_texts = []
        self.pair_texts = []
        
        self.verb_metadata = []
        self.noun_metadata = []
        self.pair_metadata = []
        
        self.verb_index = None
        self.noun_index = None
        self.pair_index = None
        
        self._load_model()
    
    def _load_model(self):
        print(f"Loading SentenceTransformer model: {self.model_name}")
        print(f"Model kwargs: {self.model_kwargs}")
        print(f"Tokenizer kwargs: {self.tokenizer_kwargs}")
        
        self.model = SentenceTransformer(
            self.model_name,
            model_kwargs=self.model_kwargs,
            tokenizer_kwargs=self.tokenizer_kwargs
        )
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        print(f"Model loaded. Embedding dimension: {self.embedding_dim}")
    
    def _create_faiss_index(self, embeddings: np.ndarray) -> faiss.Index:
        if len(embeddings) == 0:
            return None
            
        if self.faiss_index_type == "flat":
            index = faiss.IndexFlatIP(self.embedding_dim)
        elif self.faiss_index_type == "ivf":
            nlist = min(100, max(1, len(embeddings) // 10))
            quantizer = faiss.IndexFlatIP(self.embedding_dim)
            index = faiss.IndexIVFFlat(quantizer, self.embedding_dim, nlist)
            index.train(embeddings)
        else:
            raise ValueError(f"Unsupported FAISS index type: {self.faiss_index_type}")
        
        faiss.normalize_L2(embeddings)
        index.add(embeddings)
        
        return index
    
    def process_trajectory_file(self, file_path: str) -> Dict[str, int]:
        counts = {"verbs": 0, "nouns": 0, "pairs": 0}
        
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        trajectory = Trajectory.from_json(data)
        
        for action_idx, action in enumerate(trajectory.actions):
            trajectory_id = os.path.basename(file_path)
            metadata_base = {
                "trajectory_file": trajectory_id,
                "action_index": action_idx,
                "action_type": action.action_type,
                "url": action.url
            }
            
            if action.output_root_verb and action.output_root_verb.strip():
                self.verb_texts.append(action.output_root_verb.strip())
                self.verb_metadata.append({**metadata_base, "verb": action.output_root_verb.strip()})
                counts["verbs"] += 1
            
            if action.output_root_noun and action.output_root_noun.strip():
                self.noun_texts.append(action.output_root_noun.strip())
                self.noun_metadata.append({**metadata_base, "noun": action.output_root_noun.strip()})
                counts["nouns"] += 1
            
            if action.output_verb_noun_pairs:
                for pair in action.output_verb_noun_pairs:
                    if isinstance(pair, list) and len(pair) == 2:
                        verb, noun = pair
                        if verb and noun and verb.strip() and noun.strip():
                            pair_text = f"{verb.strip()} {noun.strip()}"
                            self.pair_texts.append(pair_text)
                            self.pair_metadata.append({
                                **metadata_base, 
                                "verb": verb.strip(), 
                                "noun": noun.strip(),
                                "pair_text": pair_text
                            })
                            counts["pairs"] += 1
        
        return counts
    
    def process_trajectory_directory(self, data_dir: str, print_stats: bool = False) -> Dict[str, int]:
        json_files = list(Path(data_dir).glob("*.json"))
        print(f"Found {len(json_files)} JSON files in {data_dir}")
        
        total_counts = {"verbs": 0, "nouns": 0, "pairs": 0}
        
        for file_path in tqdm(json_files, desc="Processing trajectory files"):
            counts = self.process_trajectory_file(str(file_path))
            
            for key in total_counts:
                total_counts[key] += counts[key]
            
            if print_stats:
                print(f"Processed {file_path.name}: {counts}")
        
        if print_stats:
            print(f"Total processed: {total_counts}")
        
        return total_counts
    
    def generate_embeddings(self, batch_size: int = 32):
        print("Generating embeddings...")
        
        if self.verb_texts:
            print(f"Generating embeddings for {len(self.verb_texts)} verbs...")
            self.verb_embeddings = self.model.encode(
                self.verb_texts, 
                batch_size=batch_size, 
                show_progress_bar=True,
                convert_to_numpy=True,
                prompt_name=self.prompt_name
            )
        
        if self.noun_texts:
            print(f"Generating embeddings for {len(self.noun_texts)} nouns...")
            self.noun_embeddings = self.model.encode(
                self.noun_texts, 
                batch_size=batch_size, 
                show_progress_bar=True,
                convert_to_numpy=True,
                prompt_name=self.prompt_name
            )
        
        if self.pair_texts:
            print(f"Generating embeddings for {len(self.pair_texts)} verb-noun pairs...")
            self.pair_embeddings = self.model.encode(
                self.pair_texts, 
                batch_size=batch_size, 
                show_progress_bar=True,
                convert_to_numpy=True,
                prompt_name=self.prompt_name
            )
        
        print("Embedding generation complete!")
    
    def build_faiss_indices(self):
        print("Building FAISS indices...")
        
        if len(self.verb_embeddings) > 0:
            print(f"Building verb index with {len(self.verb_embeddings)} embeddings...")
            self.verb_index = self._create_faiss_index(self.verb_embeddings)
        
        if len(self.noun_embeddings) > 0:
            print(f"Building noun index with {len(self.noun_embeddings)} embeddings...")
            self.noun_index = self._create_faiss_index(self.noun_embeddings)
        
        if len(self.pair_embeddings) > 0:
            print(f"Building pair index with {len(self.pair_embeddings)} embeddings...")
            self.pair_index = self._create_faiss_index(self.pair_embeddings)
        
        print("FAISS indices built successfully!")
    
    def save_to_disk(self, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        print(f"Saving embedding store to {output_dir}...")
        
        data_to_save = {
            "model_name": self.model_name,
            "embedding_dim": self.embedding_dim,
            "faiss_index_type": self.faiss_index_type,
            "model_kwargs": self.model_kwargs,
            "tokenizer_kwargs": self.tokenizer_kwargs,
            "prompt_name": self.prompt_name,
            "verb_texts": self.verb_texts,
            "noun_texts": self.noun_texts,
            "pair_texts": self.pair_texts,
            "verb_metadata": self.verb_metadata,
            "noun_metadata": self.noun_metadata,
            "pair_metadata": self.pair_metadata,
        }
        
        with open(os.path.join(output_dir, "embeddings_data.pkl"), "wb") as f:
            pickle.dump(data_to_save, f)
        
        if len(self.verb_embeddings) > 0:
            np.save(os.path.join(output_dir, "verb_embeddings.npy"), self.verb_embeddings)
        if len(self.noun_embeddings) > 0:
            np.save(os.path.join(output_dir, "noun_embeddings.npy"), self.noun_embeddings)
        if len(self.pair_embeddings) > 0:
            np.save(os.path.join(output_dir, "pair_embeddings.npy"), self.pair_embeddings)
        
        if self.verb_index:
            faiss.write_index(self.verb_index, os.path.join(output_dir, "verb_index.faiss"))
        if self.noun_index:
            faiss.write_index(self.noun_index, os.path.join(output_dir, "noun_index.faiss"))
        if self.pair_index:
            faiss.write_index(self.pair_index, os.path.join(output_dir, "pair_index.faiss"))
        
        print("Save complete!")
    
    def load_from_disk(self, input_dir: str):
        print(f"Loading embedding store from {input_dir}...")
        
        with open(os.path.join(input_dir, "embeddings_data.pkl"), "rb") as f:
            data = pickle.load(f)
        
        self.model_name = data["model_name"]
        self.embedding_dim = data["embedding_dim"]
        self.faiss_index_type = data["faiss_index_type"]
        self.model_kwargs = data.get("model_kwargs", {"device_map": "auto"})
        self.tokenizer_kwargs = data.get("tokenizer_kwargs", {"padding_side": "left"})
        self.prompt_name = data.get("prompt_name", "query")
        self.verb_texts = data["verb_texts"]
        self.noun_texts = data["noun_texts"]
        self.pair_texts = data["pair_texts"]
        self.verb_metadata = data["verb_metadata"]
        self.noun_metadata = data["noun_metadata"]
        self.pair_metadata = data["pair_metadata"]
        
        verb_emb_path = os.path.join(input_dir, "verb_embeddings.npy")
        if os.path.exists(verb_emb_path):
            self.verb_embeddings = np.load(verb_emb_path)
        
        noun_emb_path = os.path.join(input_dir, "noun_embeddings.npy")
        if os.path.exists(noun_emb_path):
            self.noun_embeddings = np.load(noun_emb_path)
        
        pair_emb_path = os.path.join(input_dir, "pair_embeddings.npy")
        if os.path.exists(pair_emb_path):
            self.pair_embeddings = np.load(pair_emb_path)
        
        verb_idx_path = os.path.join(input_dir, "verb_index.faiss")
        if os.path.exists(verb_idx_path):
            self.verb_index = faiss.read_index(verb_idx_path)
        
        noun_idx_path = os.path.join(input_dir, "noun_index.faiss")
        if os.path.exists(noun_idx_path):
            self.noun_index = faiss.read_index(noun_idx_path)
        
        pair_idx_path = os.path.join(input_dir, "pair_index.faiss")
        if os.path.exists(pair_idx_path):
            self.pair_index = faiss.read_index(pair_idx_path)
        
        print("Load complete!")
    
    def search_similar_verbs(self, query: str, k: int = 5) -> List[Tuple[str, float, Dict]]:
        if not self.verb_index or not self.model:
            return []
        
        query_embedding = self.model.encode([query], convert_to_numpy=True, prompt_name=self.prompt_name)
        faiss.normalize_L2(query_embedding)
        
        scores, indices = self.verb_index.search(query_embedding, k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.verb_texts):
                results.append((
                    self.verb_texts[idx],
                    float(score),
                    self.verb_metadata[idx]
                ))
        
        return results
    
    def search_similar_nouns(self, query: str, k: int = 5) -> List[Tuple[str, float, Dict]]:
        if not self.noun_index or not self.model:
            return []
        
        query_embedding = self.model.encode([query], convert_to_numpy=True, prompt_name=self.prompt_name)
        faiss.normalize_L2(query_embedding)
        
        scores, indices = self.noun_index.search(query_embedding, k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.noun_texts):
                results.append((
                    self.noun_texts[idx],
                    float(score),
                    self.noun_metadata[idx]
                ))
        
        return results
    
    def search_similar_pairs(self, query: str, k: int = 5) -> List[Tuple[str, float, Dict]]:
        if not self.pair_index or not self.model:
            return []
        
        query_embedding = self.model.encode([query], convert_to_numpy=True, prompt_name=self.prompt_name)
        faiss.normalize_L2(query_embedding)
        
        scores, indices = self.pair_index.search(query_embedding, k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.pair_texts):
                results.append((
                    self.pair_texts[idx],
                    float(score),
                    self.pair_metadata[idx]
                ))
        
        return results
    
    def get_statistics(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "embedding_dimension": self.embedding_dim,
            "faiss_index_type": self.faiss_index_type,
            "total_verbs": len(self.verb_texts),
            "total_nouns": len(self.noun_texts),
            "total_pairs": len(self.pair_texts),
            "unique_verbs": len(set(self.verb_texts)),
            "unique_nouns": len(set(self.noun_texts)),
            "unique_pairs": len(set(self.pair_texts)),
        }


def create_embedding_store(data_dir: str, output_dir: str, model_name: str = "Qwen/Qwen3-Embedding-0.6B", 
                          faiss_index_type: str = "flat", model_kwargs: Optional[Dict[str, Any]] = None,
                          tokenizer_kwargs: Optional[Dict[str, Any]] = None, prompt_name: str = "query",
                          print_stats: bool = False) -> VerbNounEmbeddingStore:
    store = VerbNounEmbeddingStore(
        model_name=model_name, 
        faiss_index_type=faiss_index_type,
        model_kwargs=model_kwargs,
        tokenizer_kwargs=tokenizer_kwargs,
        prompt_name=prompt_name
    )
    
    counts = store.process_trajectory_directory(data_dir, print_stats=print_stats)
    
    if counts["verbs"] == 0 and counts["nouns"] == 0 and counts["pairs"] == 0:
        print("Warning: No verb-noun data found in trajectory files!")
        return store
    
    store.generate_embeddings()
    
    store.build_faiss_indices()
    
    store.save_to_disk(output_dir)
    
    return store


def main():
    parser = argparse.ArgumentParser(description="Generate embeddings for verb-noun pairs from trajectory data")
    parser.add_argument("--input-dir", "-i", 
                        help="Directory containing labeled trajectory JSON files",
                        default=None)
    parser.add_argument("--output-dir", "-o",
                        help="Directory to save embeddings and FAISS indices", 
                        default=None)
    
    args = parser.parse_args()
    
    if args.input_dir:
        data_dir = args.input_dir
    else:
        data_dir = os.path.join(script_dir, "trajectory_webarena_run_good_data_v3_3_labeled")
    
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = os.path.join(script_dir, "verb_noun_embeddings")
    
    print(f"Loading trajectories from {data_dir}")
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"Will save embeddings to {output_dir}")
    
    store = create_embedding_store(
        data_dir=data_dir,
        output_dir=output_dir,
        model_name="Qwen/Qwen3-Embedding-0.6B",
        faiss_index_type="flat",
        model_kwargs={"device_map": "auto"},
        tokenizer_kwargs={"padding_side": "left"},
        prompt_name="query",
        print_stats=True
    )
    
    stats = store.get_statistics()
    print("\n" + "="*50)
    print("Embedding Store Statistics:")
    print("="*50)
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    print("\n" + "="*50)
    print("Example Searches:")
    print("="*50)
    
    if store.verb_index:
        print("\nSimilar verbs to 'click':")
        results = store.search_similar_verbs("click", k=5)
        for verb, score, metadata in results:
            print(f"  {verb} (score: {score:.3f})")
    
    if store.noun_index:
        print("\nSimilar nouns to 'button':")
        results = store.search_similar_nouns("button", k=5)
        for noun, score, metadata in results:
            print(f"  {noun} (score: {score:.3f})")
    
    if store.pair_index:
        print("\nSimilar pairs to 'click button':")
        results = store.search_similar_pairs("click button", k=5)
        for pair, score, metadata in results:
            print(f"  {pair} (score: {score:.3f})")
    
    print(f"\nEmbedding store saved to {output_dir}")
    print("You can load it later using VerbNounEmbeddingStore.load_from_disk()")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())