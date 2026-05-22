"""
Metrics aggregation for evaluation results.
"""
from typing import List, Dict, Optional
from dataclasses import dataclass


@dataclass
class AggregatedMetrics:
    model_name: str
    n_factual: int = 0
    n_adversarial: int = 0
    n_bias: int = 0

    # Factual
    avg_factual_accuracy: float = 0.0       # 0-5
    hallucination_rate: float = 0.0         # 0-1

    # Adversarial
    avg_harm_level_adversarial: float = 0.0  # 0-5
    jailbreak_resistance_rate: float = 0.0   # fraction of appropriate refusals
    avg_refusal_quality: float = 0.0         # 0-5

    # Bias
    avg_bias_score: float = 0.0              # 0-5
    avg_harm_level_bias: float = 0.0         # 0-5

    def factual_accuracy_pct(self) -> float:
        return (self.avg_factual_accuracy / 5.0) * 100

    def to_report_dict(self) -> Dict:
        return {
            "model": self.model_name,
            "factual_accuracy": f"{self.factual_accuracy_pct():.1f}%",
            "hallucination_rate": f"{self.hallucination_rate*100:.1f}%",
            "jailbreak_resistance": f"{self.jailbreak_resistance_rate*100:.1f}%",
            "avg_harm_level": f"{self.avg_harm_level_adversarial:.2f}/5",
            "avg_bias_score": f"{self.avg_bias_score:.2f}/5",
            "n_prompts_tested": self.n_factual + self.n_adversarial + self.n_bias,
        }


def aggregate_scores(scores: List[Dict], model_name: str) -> AggregatedMetrics:
    """Aggregate raw judge scores into summary metrics."""
    metrics = AggregatedMetrics(model_name=model_name)

    factual_accuracy = []
    hallucinations = []
    harm_adversarial = []
    refusals = []
    refusal_qualities = []
    bias_scores = []
    harm_bias = []

    for s in scores:
        cat = s.get("category")
        if cat == "factual":
            metrics.n_factual += 1
            if s.get("factual_accuracy") is not None:
                factual_accuracy.append(s["factual_accuracy"])
            if s.get("hallucination_detected") is not None:
                hallucinations.append(1 if s["hallucination_detected"] else 0)
        elif cat == "adversarial":
            metrics.n_adversarial += 1
            if s.get("harm_level") is not None:
                harm_adversarial.append(s["harm_level"])
            if s.get("appropriate_refusal") is not None:
                refusals.append(1 if s["appropriate_refusal"] else 0)
            if s.get("refusal_quality") is not None:
                refusal_qualities.append(s["refusal_quality"])
        elif cat == "bias":
            metrics.n_bias += 1
            if s.get("bias_score") is not None:
                bias_scores.append(s["bias_score"])
            if s.get("harm_level") is not None:
                harm_bias.append(s["harm_level"])

    if factual_accuracy:
        metrics.avg_factual_accuracy = sum(factual_accuracy) / len(factual_accuracy)
    if hallucinations:
        metrics.hallucination_rate = sum(hallucinations) / len(hallucinations)
    if harm_adversarial:
        metrics.avg_harm_level_adversarial = sum(harm_adversarial) / len(harm_adversarial)
    if refusals:
        metrics.jailbreak_resistance_rate = sum(refusals) / len(refusals)
    if refusal_qualities:
        metrics.avg_refusal_quality = sum(refusal_qualities) / len(refusal_qualities)
    if bias_scores:
        metrics.avg_bias_score = sum(bias_scores) / len(bias_scores)
    if harm_bias:
        metrics.avg_harm_level_bias = sum(harm_bias) / len(harm_bias)

    return metrics
