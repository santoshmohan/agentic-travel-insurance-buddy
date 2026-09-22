import logging
from typing import List, Dict, Any
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

logger = logging.getLogger(__name__)

class RetrievalMetrics:
    """Metrics for evaluating retrieval quality."""
    
    @staticmethod
    def mean_reciprocal_rank(retrieved_docs: List[str], 
                             ground_truth: List[str]) -> float:
        """
        MRR: Position of first relevant document.
        Higher is better (1.0 = perfect).
        """
        for i, doc in enumerate(retrieved_docs, 1):
            if doc in ground_truth:
                return 1.0 / i
        return 0.0
    
    @staticmethod
    def ndcg_at_k(retrieved_docs: List[str], 
                  ground_truth: List[str], 
                  k: int = 5) -> float:
        """
        NDCG@K: Normalized Discounted Cumulative Gain.
        Measures ranking quality. Range: 0-1.
        """
        # DCG
        dcg = 0.0
        for i, doc in enumerate(retrieved_docs[:k], 1):
            relevance = 1.0 if doc in ground_truth else 0.0
            dcg += relevance / np.log2(i + 1)
        
        # Ideal DCG
        idcg = 0.0
        for i in range(1, min(len(ground_truth), k) + 1):
            idcg += 1.0 / np.log2(i + 1)
        
        return dcg / idcg if idcg > 0 else 0.0
    
    @staticmethod
    def precision_at_k(retrieved_docs: List[str], 
                       ground_truth: List[str], 
                       k: int = 5) -> float:
        """
        Precision@K: % of top-k results that are relevant.
        """
        relevant_count = sum(1 for doc in retrieved_docs[:k] if doc in ground_truth)
        return relevant_count / k if k > 0 else 0.0
    
    @staticmethod
    def recall_at_k(retrieved_docs: List[str], 
                    ground_truth: List[str], 
                    k: int = 5) -> float:
        """
        Recall@K: % of ground truth docs that appear in top-k.
        """
        relevant_count = sum(1 for doc in retrieved_docs[:k] if doc in ground_truth)
        return relevant_count / len(ground_truth) if len(ground_truth) > 0 else 0.0
    
    @staticmethod
    def map_at_k(retrieved_docs_list: List[List[str]], 
                 ground_truth_list: List[List[str]], 
                 k: int = 5) -> float:
        """
        MAP@K: Mean Average Precision across multiple queries.
        """
        aps = []
        for retrieved, ground_truth in zip(retrieved_docs_list, ground_truth_list):
            ap = 0.0
            relevant_count = 0
            for i, doc in enumerate(retrieved[:k], 1):
                if doc in ground_truth:
                    relevant_count += 1
                    ap += relevant_count / i
            ap = ap / len(ground_truth) if len(ground_truth) > 0 else 0.0
            aps.append(ap)
        return np.mean(aps) if aps else 0.0


class RAGMetrics:
    """Metrics for evaluating end-to-end RAG quality."""
    
    @staticmethod
    def context_relevance(context: str, 
                         question: str, 
                         ground_truth_chunks: List[str]) -> float:
        """
        How much of the context comes from ground truth chunks.
        Higher = more relevant context selected.
        """
        # Simple: check if ground truth chunks appear in context
        matches = sum(1 for chunk_id in ground_truth_chunks if chunk_id in context)
        return matches / len(ground_truth_chunks) if ground_truth_chunks else 0.0
    
    @staticmethod
    def answer_relevance(answer: str, 
                        question: str) -> Dict[str, float]:
        """
        Basic relevance checks (can be extended with LLM eval).
        """
        # Check if answer contains key terms from question
        question_terms = set(question.lower().split())
        answer_terms = set(answer.lower().split())
        overlap = len(question_terms & answer_terms) / len(question_terms)
        
        return {
            "term_overlap": min(overlap, 1.0),
            "answer_length": len(answer) / 100,  # normalized
            "has_context": 1.0 if len(answer) > 50 else 0.5
        }
    
    @staticmethod
    def faithfulness(answer: str, 
                    context: str) -> float:
        """
        Check if answer is grounded in context (simple heuristic).
        """
        # Count how many sentences in answer reference context
        answer_sents = answer.split(".")
        context_words = set(context.lower().split())
        
        grounded = 0
        for sent in answer_sents:
            sent_words = set(sent.lower().split())
            if not sent_words:
                continue
            if len(sent_words & context_words) / len(sent_words) > 0.3:
                grounded += 1
        
        return grounded / len(answer_sents) if answer_sents else 0.0


class EvalResults:
    """Store and aggregate evaluation results."""
    
    def __init__(self):
        self.results = []
    
    def add(self, query_id: str, metric_name: str, value: float, config: Dict[str, Any]):
        self.results.append({
            "query_id": query_id,
            "metric": metric_name,
            "value": value,
            "config": config
        })
    
    def summary(self) -> Dict[str, float]:
        """Aggregate results by metric."""
        summary = {}
        for result in self.results:
            metric = result["metric"]
            if metric not in summary:
                summary[metric] = []
            summary[metric].append(result["value"])
        
        return {metric: np.mean(values) for metric, values in summary.items()}
    
    def to_dict(self) -> List[Dict]:
        return self.results