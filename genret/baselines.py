import re
import math
import json
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
from collections import Counter
from sentence_transformers import SentenceTransformer

from genret import config
from genret.embed import load_embeddings, load_chunks_jsonl

def tokenize_bm25(text: str) -> List[str]:
    """Simple tokenization: lowercase, alphanumeric tokens."""
    return re.findall(r"\w+", text.lower())

class BM25Okapi:
    """Pure Python, zero-dependency implementation of standard BM25Okapi."""
    def __init__(self, corpus: List[List[str]], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus_size = len(corpus)
        self.doc_lens = [len(doc) for doc in corpus]
        self.avgdl = sum(self.doc_lens) / max(1, self.corpus_size)
        self.doc_freqs = []
        self.idf = {}

        df = Counter()
        for doc in corpus:
            frequencies = Counter(doc)
            self.doc_freqs.append(frequencies)
            for word in frequencies.keys():
                df[word] += 1

        for word, freq in df.items():
            # Standard Lucene/BM25 IDF formula
            self.idf[word] = math.log(1 + (self.corpus_size - freq + 0.5) / (freq + 0.5))

    def get_scores(self, query: List[str]) -> np.ndarray:
        scores = np.zeros(self.corpus_size, dtype=np.float32)
        for q_word in query:
            if q_word not in self.idf:
                continue
            q_idf = self.idf[q_word]
            for doc_idx in range(self.corpus_size):
                freq = self.doc_freqs[doc_idx].get(q_word, 0)
                if freq > 0:
                    numerator = freq * (self.k1 + 1)
                    denominator = freq + self.k1 * (1 - self.b + self.b * (self.doc_lens[doc_idx] / self.avgdl))
                    scores[doc_idx] += q_idf * (numerator / denominator)
        return scores

class BM25Retriever:
    """Lexical baseline using pure-python BM25Okapi."""
    def __init__(self, chunks_path: Path = config.CHUNKS_PATH):
        self.chunks = load_chunks_jsonl(chunks_path)
        self.corpus_tokens = [tokenize_bm25(c["text"]) for c in self.chunks]
        self.bm25 = BM25Okapi(self.corpus_tokens)

    def get_all_scores(self, query: str) -> Dict[str, float]:
        """Compute normalized [0, 1] BM25 prior scores across all corpus documents for PAG."""
        query_tokens = tokenize_bm25(query)
        scores = self.bm25.get_scores(query_tokens)
        max_s = float(np.max(scores)) if len(scores) > 0 and np.max(scores) > 0 else 1.0
        return {self.chunks[i]["chunk_id"]: float(scores[i]) / max_s for i in range(len(self.chunks))}

    def score_candidates(self, query: str, candidate_chunks: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        In-candidate lexical scoring (O(k) complexity).
        Computes BM25 term scores strictly for the candidate chunks retrieved by AutoSLM.
        """
        query_tokens = tokenize_bm25(query)
        chunk_id_to_idx = {c["chunk_id"]: i for i, c in enumerate(self.chunks)}
        scores = {}
        for c in candidate_chunks:
            cid = c["chunk_id"]
            if cid in chunk_id_to_idx:
                idx = chunk_id_to_idx[cid]
                doc_len = self.bm25.doc_lens[idx]
                doc_freq = self.bm25.doc_freqs[idx]
                score = 0.0
                for q_word in query_tokens:
                    if q_word in self.bm25.idf:
                        q_idf = self.bm25.idf[q_word]
                        freq = doc_freq.get(q_word, 0)
                        if freq > 0:
                            num = freq * (self.bm25.k1 + 1)
                            den = freq + self.bm25.k1 * (1 - self.bm25.b + self.bm25.b * (doc_len / self.bm25.avgdl))
                            score += q_idf * (num / den)
                scores[cid] = float(score)
            else:
                scores[cid] = 0.0
        return scores

    def retrieve(self, query: str, k: int = config.TOP_K) -> List[Dict[str, Any]]:
        query_tokens = tokenize_bm25(query)
        scores = self.bm25.get_scores(query_tokens)
        top_k_indices = np.argsort(-scores)[:k]

        results = []
        for rank, idx in enumerate(top_k_indices, start=1):
            chunk = self.chunks[idx]
            results.append({
                "rank": rank,
                "chunk_id": chunk["chunk_id"],
                "text": chunk["text"],
                "score": float(scores[idx])
            })
        return results

class DenseRetriever:
    """Dense semantic baseline using frozen BGE encoder and exact cosine similarity."""
    def __init__(
        self,
        chunks_path: Path = config.CHUNKS_PATH,
        emb_path: Path = config.EMB_PATH,
        meta_path: Path = config.EMB_META_PATH
    ):
        self.chunks = load_chunks_jsonl(chunks_path)
        self.embeddings, self.meta = load_embeddings(emb_path, meta_path)
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.encoder = SentenceTransformer(self.meta["encoder"], device=device)
        self.query_prefix = self.meta.get("query_prefix", "Represent this sentence for searching relevant passages: ")

    def retrieve(self, query: str, k: int = config.TOP_K) -> List[Dict[str, Any]]:
        # BGE requires task prefix on query side
        prefixed_query = self.query_prefix + query if self.query_prefix else query
        query_emb = self.encoder.encode(
            [prefixed_query],
            normalize_embeddings=self.meta.get("normalized", True),
            convert_to_numpy=True
        ).astype(np.float32)[0]

        # Cosine similarity is dot product when normalized
        scores = np.dot(self.embeddings, query_emb)
        top_k_indices = np.argsort(-scores)[:k]

        results = []
        for rank, idx in enumerate(top_k_indices, start=1):
            chunk = self.chunks[idx]
            results.append({
                "rank": rank,
                "chunk_id": chunk["chunk_id"],
                "text": chunk["text"],
                "score": float(scores[idx])
            })
        return results
