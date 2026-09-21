from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

@dataclass
class TextChunk:
    chunk_id: int
    text: str
    token_count: int
    start_char: int
    end_char: int

@dataclass
class SummarizationResult:
    summary_text: str
    chunk_summaries: List[str]
    input_token_count: int
    summary_token_count: int
    compression_ratio: float
    model_name: str
    is_fallback: bool = False

@dataclass
class SentimentResult:
    label: str  # POSITIVE, NEGATIVE, NEUTRAL
    score: float
    scores_breakdown: Dict[str, float]
    model_name: str
    is_fallback: bool = False

@dataclass
class EvaluationMetrics:
    rouge1_f1: float
    rouge1_precision: float
    rouge1_recall: float
    rouge2_f1: float
    rouge2_precision: float
    rouge2_recall: float
    rougel_f1: float
    rougel_precision: float
    rougel_recall: float
    bertscore_f1: Optional[float]
    compression_ratio: float
    latency_seconds: float

@dataclass
class PipelineResult:
    input_text: str
    summary: SummarizationResult
    sentiment: SentimentResult
    evaluation: Optional[EvaluationMetrics] = None
