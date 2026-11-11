import streamlit as st
import torch
import pandas as pd
import numpy as np
import os
import sys

sys.path.append(os.path.abspath("src"))

from geofinbert.inference import GeoFinBERTInferencePipeline

st.set_page_config(
    page_title="GeoFinBERT: Geopolitical & Energy News Sentiment",
    page_icon="📈",
    layout="wide"
)

st.title("📈 GeoFinBERT: Geopolitical & Energy Market Sentiment Analyzer")
st.markdown("""
*An academic machine-learning model adapting **ProsusAI/FinBERT** to crude oil, gold bullion, and geopolitical conflict news (2000–2026).*
""")

# Sidebar settings
st.sidebar.header("Model Configuration")
checkpoint_dir = st.sidebar.text_input(
    "Model Checkpoint Path",
    value="models/checkpoints/GeoFinBERT-smoke" if os.path.exists("models/checkpoints/GeoFinBERT-smoke") else "models/checkpoints/GeoFinBERT"
)

device_choice = st.sidebar.selectbox("Compute Device", ["auto", "cpu", "cuda"])

@st.cache_resource
def load_pipeline(path: str, device: str):
    if not os.path.exists(path):
        return None
    return GeoFinBERTInferencePipeline(model_dir_or_name=path, device=device)

pipeline = load_pipeline(checkpoint_dir, device_choice)

if pipeline is None:
    st.warning(f"Checkpoint not found at `{checkpoint_dir}`. Run training or smoke test first.")
else:
    st.success(f"Model loaded successfully from `{checkpoint_dir}` onto `{pipeline.device}`")

st.subheader("Analyze Headline Sentiment")

sample_examples = [
    "Crude oil prices surge 4.2% as Strait of Hormuz naval confrontations disrupt Persian Gulf tanker transit.",
    "Gold bullion falls sharply following unexpected hawkish rate remarks from Federal Reserve.",
    "OPEC+ ministers agree to keep crude oil output quotas unchanged for the next quarter.",
    "US Treasury announces secondary sanctions targeting foreign entities facilitating Iranian petroleum exports.",
    "Global central banks accelerate gold reserve acquisitions to hedge against currency depreciation."
]

selected_example = st.selectbox("Or choose a pre-loaded geopolitical headline:", ["-- Select an example --"] + sample_examples)

user_input = st.text_area(
    "Enter a news headline or article snippet:",
    value="" if selected_example == "-- Select an example --" else selected_example,
    height=100
)

if st.button("Analyze Sentiment", type="primary"):
    if not user_input.strip():
        st.error("Please enter a headline to analyze.")
    elif pipeline is None:
        st.error("Model pipeline is not loaded.")
    else:
        with st.spinner("Analyzing sentiment..."):
            result = pipeline.predict(user_input.strip())
            
        pred = result["prediction"].upper()
        conf = result["confidence"]
        probs = result["probabilities"]
        
        col1, col2, col3 = st.columns([1, 1, 2])
        
        with col1:
            if pred == "POSITIVE":
                st.metric("Predicted Sentiment", "🟢 POSITIVE", delta=f"{conf:.1%} confidence")
            elif pred == "NEGATIVE":
                st.metric("Predicted Sentiment", "🔴 NEGATIVE", delta=f"{conf:.1%} confidence", delta_color="inverse")
            else:
                st.metric("Predicted Sentiment", "⚪ NEUTRAL", delta=f"{conf:.1%} confidence", delta_color="off")
                
        with col2:
            st.markdown("### Probability Distribution")
            st.progress(probs["positive"], text=f"Positive: {probs['positive']:.1%}")
            st.progress(probs["negative"], text=f"Negative: {probs['negative']:.1%}")
            st.progress(probs["neutral"], text=f"Neutral: {probs['neutral']:.1%}")
            
        with col3:
            st.markdown("### Macro & Commodity Impact Analysis")
            text_lower = user_input.lower()
            if any(k in text_lower for k in ["crude", "oil", "brent", "wti", "opec", "refinery"]):
                st.info("🛢️ **Energy Domain**: Direct price sensitivity to physical supply disruptions, OPEC quotas, and strategic petroleum reserves.")
            if any(k in text_lower for k in ["gold", "bullion", "xau", "precious metal"]):
                st.info("🪙 **Bullion Domain**: Safe-haven dynamic responsive to real yields, sovereign reserve diversification, and inflation expectations.")
            if any(k in text_lower for k in ["iran", "hormuz", "sanction", "middle east", "tehran", "red sea"]):
                st.warning("⚓ **Geopolitical Conflict**: High maritime transit risk, insurance premium surges, and diplomatic friction.")

st.markdown("---")
st.markdown("Academic Research Paper: **GeoFinBERT** | [GitHub Repository](https://github.com/angelina-2206/GeoFinBERT)")
