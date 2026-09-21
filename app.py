import os
import sys

import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from nlp_engine.chunking import TokenAwareChunker
from nlp_engine.pipeline import NLPPipeline

st.set_page_config(
    page_title="NLP Transformer Summarizer & Sentiment Workbench",
    page_icon="📝",
    layout="wide",
)

st.title("📝 NLP Transformer Summarizer & Sentiment Workbench")
st.markdown(
    "Long-document summarization engine featuring **Token-Aware Sliding-Window Chunking**, "
    "**Hierarchical BART / Flan-T5 Architecture**, **Threshold-Based Neutral Sentiment Classification**, "
    "and **Quantitative ROUGE & BERTScore Evaluation**."
)

# Sidebar controls
st.sidebar.header("⚙️ Pipeline Configuration")

model_option = st.sidebar.selectbox(
    "Summarization Model",
    ["BART (facebook/bart-large-cnn)", "Flan-T5 (google/flan-t5-base)", "Heuristic Fallback Baseline"],
)

model_map = {
    "BART (facebook/bart-large-cnn)": "facebook/bart-large-cnn",
    "Flan-T5 (google/flan-t5-base)": "google/flan-t5-base",
    "Heuristic Fallback Baseline": "rule-based-fallback",
}

chunk_size = st.sidebar.slider("Max Chunk Tokens", min_value=128, max_value=1024, value=512, step=64)
overlap_size = st.sidebar.slider("Chunk Overlap Tokens", min_value=16, max_value=128, value=64, step=16)
max_len = st.sidebar.slider("Max Summary Length (words)", min_value=30, max_value=300, value=120, step=10)
min_len = st.sidebar.slider("Min Summary Length (words)", min_value=10, max_value=100, value=25, step=5)

# Input area
st.subheader("📄 Document Input")
input_method = st.radio("Choose Input Method:", ["Text Input", "Upload Text File"], horizontal=True)

input_text = ""
if input_method == "Text Input":
    default_sample = (
        "Artificial Intelligence and machine learning engineering have transitioned from isolated experimental research "
        "into foundational enterprise software infrastructure. Modern industrial applications demand robust, token-aware "
        "architectures capable of processing long-form documents without context truncation or silent memory overflow. "
        "By integrating sliding window chunking with pre-trained transformer models such as BART and Flan-T5, system "
        "architects achieve high compression ratios while preserving critical factual entities and semantic nuance. "
        "Furthermore, combining abstractive text summarization with calibrated three-class sentiment analysis and "
        "reproducible ROUGE evaluation pipelines provides verifiable benchmarks for deploying production-grade "
        "Generative AI solutions across finance, healthcare, and software engineering domains."
    )
    input_text = st.text_area("Enter Article or Document Text:", value=default_sample, height=200)
else:
    uploaded_file = st.file_uploader("Upload a .txt or .md document", type=["txt", "md"])
    if uploaded_file is not None:
        input_text = uploaded_file.read().decode("utf-8")
        st.success(f"Loaded file '{uploaded_file.name}' ({len(input_text)} characters)")

ref_summary = st.text_input("Optional Reference Summary (for ROUGE / BERTScore evaluation):", value="")

if st.button("🚀 Run Pipeline", type="primary"):
    if not input_text.strip():
        st.warning("Please enter text or upload a file.")
    else:
        with st.spinner("Processing document through NLP Pipeline..."):
            pipeline = NLPPipeline(summarizer_model=model_map[model_option])
            res = pipeline.process(
                text=input_text,
                reference_summary=ref_summary,
                max_length=max_len,
                min_length=min_len,
                max_chunk_tokens=chunk_size,
                overlap_tokens=overlap_size,
            )

        if res.summary.is_fallback:
            st.warning(f"⚠️ **Summarizer Running in Fallback Mode**: {res.summary.fallback_reason}")

        if res.sentiment.is_fallback:
            st.info(f"ℹ️ **Sentiment Analyzer Running in Heuristic Mode**: {res.sentiment.fallback_reason}")

        tab1, tab2, tab3, tab4 = st.tabs(
            [
                "📊 Summary & Sentiment",
                "🧩 Token Chunk Inspector",
                "📈 Evaluation Workbench",
                "📁 Committed Benchmarks",
            ]
        )

        with tab1:
            col1, col2, col3 = st.columns(3)
            col1.metric("Input Tokens", res.summary.input_token_count)
            col2.metric("Summary Tokens", res.summary.summary_token_count)
            col3.metric("Compression Ratio", f"{res.summary.compression_ratio}%")

            st.markdown("### 📝 Generated Summary")
            st.info(res.summary.summary_text)

            st.markdown("### 🎭 Sentiment Classification")
            score_disp = f"{res.sentiment.score}" if res.sentiment.score is not None else "N/A (Heuristic)"
            heur_note = " (Heuristic ratio, not probability)" if res.sentiment.is_heuristic else ""
            st.markdown(f"**Predicted Label**: `{res.sentiment.label}` | **Confidence**: `{score_disp}`{heur_note}")

            if res.sentiment.scores_breakdown:
                st.write("Score Breakdown:", res.sentiment.scores_breakdown)

        with tab2:
            st.markdown("### 🧩 Token-Aware Sliding Window Chunks")
            chunker = TokenAwareChunker(max_tokens=chunk_size, overlap_tokens=overlap_size)
            chunks = chunker.chunk_text(input_text)
            st.write(
                f"Total Chunks Generated (Max `{chunk_size}` tokens, Overlap `{overlap_size}` tokens): **{len(chunks)}**"
            )

            for i, c in enumerate(chunks):
                with st.expander(f"Chunk {i+1} ({c.token_count} tokens, chars {c.start_char}-{c.end_char})"):
                    st.write(c.text)
                    if i < len(res.summary.chunk_summaries):
                        st.caption(f"**Chunk Summary**: {res.summary.chunk_summaries[i]}")

        with tab3:
            st.markdown("### 📈 Quantitative Evaluation")
            if res.evaluation:
                eval_cols = st.columns(4)
                eval_cols[0].metric("ROUGE-1 F1", res.evaluation.rouge1_f1)
                eval_cols[1].metric("ROUGE-2 F1", res.evaluation.rouge2_f1)
                eval_cols[2].metric("ROUGE-L F1", res.evaluation.rougel_f1)
                eval_cols[3].metric("BERTScore F1", res.evaluation.bertscore_f1)
                st.caption(f"Evaluation Latency: {res.evaluation.latency_seconds} seconds")
            else:
                st.info("Provide a reference summary above to calculate ROUGE and BERTScore metrics.")

        with tab4:
            st.markdown("### 📁 Committed Repository Results")
            results_csv = os.path.abspath(os.path.join(os.path.dirname(__file__), "results", "model_comparison.csv"))
            if os.path.exists(results_csv):
                df_res = pd.read_csv(results_csv)
                st.dataframe(df_res, use_container_width=True)
            else:
                st.info("Run `python scripts/evaluate_summarization.py` to generate committed comparison results.")
