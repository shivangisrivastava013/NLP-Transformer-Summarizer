import json
import os
import platform
import sys
import time
from typing import Any

import pandas as pd

# Ensure project root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from nlp_engine.pipeline import NLPPipeline

SAMPLE_ARTICLES = [
    {
        "id": "tech_article_1",
        "title": "Breakthrough in Quantum Computing Architecture",
        "text": (
            "Researchers at the National Institute of Quantum Technology have unveiled a major advancement in fault-tolerant "
            "quantum computing systems. The team demonstrated a novel 128-qubit processor utilizing topological surface codes "
            "that drastically reduce physical error rates during quantum state manipulation. Unlike traditional superconducting "
            "qubits that require extreme cryogenic cooling near absolute zero, the new hybrid topological architecture operates "
            "with significantly reduced thermal overhead. Laboratory benchmarks confirmed a 99.92% two-qubit gate fidelity, "
            "surpassing the critical error threshold required for scalable fault-tolerant quantum error correction. "
            "Industry experts believe this design could accelerate the commercial deployment of quantum algorithms for drug discovery, "
            "materials science, and high-dimensional financial risk modeling by at least five years. However, scaling the control electronics "
            "to thousands of physical qubits remains a formidable engineering challenge before full commercialization."
        ),
        "reference": (
            "Researchers demonstrated a 128-qubit quantum processor using topological surface codes, achieving 99.92% gate fidelity "
            "and accelerating fault-tolerant quantum computing deployment by five years despite remaining scaling challenges."
        ),
    },
    {
        "id": "ai_governance_2",
        "title": "Global AI Safety Standards Framework",
        "text": (
            "International regulatory bodies and leading technology research institutes have reached a consensus on a comprehensive "
            "framework for evaluating multi-modal frontier artificial intelligence models. The protocol establishes mandatory red-teaming "
            "evaluations, autonomous capability thresholds, and watermarking specifications for synthetic text, audio, and visual artifacts. "
            "Under the agreed framework, frontier developers must submit independent third-party audit reports prior to public deployment "
            "of foundation models exceeding 10^26 floating-point operations of training compute. While open-source AI advocacy groups "
            "expressed concern that high compliance overhead might concentrate AI development among tech monopolies, regulators emphasized "
            "that tiered exemptions will protect small academic research laboratories and non-commercial software developers."
        ),
        "reference": (
            "Regulators established a global AI safety framework mandating red-teaming and third-party audits for frontier models exceeding "
            "10^26 compute FLOPs, incorporating tiered exemptions to support open-source academic researchers."
        ),
    },
    {
        "id": "climate_tech_3",
        "title": "Grid-Scale Energy Storage Expansion",
        "text": (
            "Grid operators across Europe and North America report a record-breaking deployment of long-duration iron-air energy storage "
            "installations designed to balance intermittent solar and wind generation. Iron-air batteries utilize abundant, non-toxic materials "
            "and can discharge continuous electrical energy for up to 100 hours at one-tenth the cost of conventional lithium-ion batteries. "
            "This capability addresses the critical multi-day energy storage deficit during prolonged periods of low renewable output. "
            "Initial operational data from a 100 megawatt pilot facility in Minnesota demonstrated a 78% round-trip efficiency over 500 charge-discharge cycles. "
            "As national power grids transition away from fossil-fuel baseload generation, multi-day battery storage is proving essential for maintaining grid stability."
        ),
        "reference": (
            "Iron-air energy storage facilities demonstrated 100-hour discharge capabilities at one-tenth the cost of lithium-ion, achieving 78% efficiency "
            "and stabilizing power grids transitioning to renewable energy."
        ),
    },
]


def run_evaluation():
    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results"))
    os.makedirs(output_dir, exist_ok=True)

    models_to_test = [
        {
            "name": "BART (facebook/bart-large-cnn)",
            "summarizer": "facebook/bart-large-cnn",
            "sentiment": "distilbert-base-uncased-finetuned-sst-2-english",
        },
        {
            "name": "Flan-T5 (google/flan-t5-base)",
            "summarizer": "google/flan-t5-base",
            "sentiment": "distilbert-base-uncased-finetuned-sst-2-english",
        },
        {
            "name": "Heuristic Rule-Based Baseline",
            "summarizer": "rule-based-fallback",
            "sentiment": "rule-based-fallback",
        },
    ]

    all_results = {}
    csv_rows = []

    print("Starting Quantitative Evaluation of NLP Summarizer & Sentiment Engine...")

    for model_cfg in models_to_test:
        raw_key = model_cfg["name"]
        print(f"\nEvaluating pipeline model: {raw_key}")
        pipeline = NLPPipeline(
            summarizer_model=model_cfg["summarizer"],
            sentiment_model=model_cfg["sentiment"],
        )

        article_metrics: list[dict[str, Any]] = []
        is_fallback_run = False

        for article in SAMPLE_ARTICLES:
            t0 = time.time()
            res = pipeline.process(
                text=article["text"],
                reference_summary=article["reference"],
                max_length=120,
                min_length=30,
            )
            t1 = time.time()
            lat = t1 - t0

            if res.summary.is_fallback:
                is_fallback_run = True

            eval_m = res.evaluation

            article_metrics.append(
                {
                    "article_id": article["id"],
                    "summary": res.summary.summary_text,
                    "sentiment_label": res.sentiment.label,
                    "sentiment_score": res.sentiment.score,
                    "is_fallback": res.summary.is_fallback,
                    "fallback_reason": res.summary.fallback_reason,
                    "rouge1_f1": eval_m.rouge1_f1 if eval_m else 0.0,
                    "rouge2_f1": eval_m.rouge2_f1 if eval_m else 0.0,
                    "rougel_f1": eval_m.rougel_f1 if eval_m else 0.0,
                    "bertscore_f1": eval_m.bertscore_f1 if eval_m else 0.0,
                    "compression_ratio": eval_m.compression_ratio if eval_m else 0.0,
                    "latency_seconds": round(lat, 4),
                }
            )

        display_name = f"{raw_key} (Heuristic Fallback)" if is_fallback_run and "Baseline" not in raw_key else raw_key

        for am in article_metrics:
            csv_rows.append(
                {
                    "model": display_name,
                    "article_id": am["article_id"],
                    "is_fallback": am["is_fallback"],
                    "rouge1_f1": am["rouge1_f1"],
                    "rouge2_f1": am["rouge2_f1"],
                    "rougel_f1": am["rougel_f1"],
                    "bertscore_f1": am["bertscore_f1"],
                    "compression_ratio": am["compression_ratio"],
                    "latency_seconds": am["latency_seconds"],
                }
            )

        avg_r1 = round(sum(m["rouge1_f1"] for m in article_metrics) / len(article_metrics), 4)
        avg_r2 = round(sum(m["rouge2_f1"] for m in article_metrics) / len(article_metrics), 4)
        avg_rl = round(sum(m["rougel_f1"] for m in article_metrics) / len(article_metrics), 4)
        avg_bert = round(sum(m["bertscore_f1"] for m in article_metrics) / len(article_metrics), 4)
        avg_comp = round(sum(m["compression_ratio"] for m in article_metrics) / len(article_metrics), 2)
        avg_lat = round(sum(m["latency_seconds"] for m in article_metrics) / len(article_metrics), 4)

        all_results[display_name] = {
            "is_fallback_execution": is_fallback_run,
            "summary_averages": {
                "rouge1_f1": avg_r1,
                "rouge2_f1": avg_r2,
                "rougel_f1": avg_rl,
                "bertscore_f1": avg_bert,
                "compression_ratio": avg_comp,
                "mean_latency_seconds": avg_lat,
            },
            "article_details": article_metrics,
        }

    # Write evaluation_results.json
    json_path = os.path.join(output_dir, "evaluation_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved evaluation metrics to: {json_path}")

    # Write model_comparison.csv
    df = pd.DataFrame(csv_rows)
    csv_path = os.path.join(output_dir, "model_comparison.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved model comparison CSV to: {csv_path}")

    # Write environment_metadata.json
    env_meta = {
        "python_version": platform.python_version(),
        "system_os": platform.system(),
        "platform_release": platform.release(),
        "machine_architecture": platform.machine(),
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "sample_articles_count": len(SAMPLE_ARTICLES),
        "evaluation_metrics_computed": ["ROUGE-1", "ROUGE-2", "ROUGE-L", "BERTScore", "Compression Ratio", "Latency"],
    }
    env_meta_path = os.path.join(output_dir, "environment_metadata.json")
    with open(env_meta_path, "w", encoding="utf-8") as f:
        json.dump(env_meta, f, indent=2)
    print(f"Saved environment metadata to: {env_meta_path}")


if __name__ == "__main__":
    run_evaluation()
