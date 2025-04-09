import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import time
import numpy as np

# Caching the model loading to improve performance
@st.cache_resource
def load_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return tokenizer, model

# Available models dictionary (BERT and FNet style)
available_models = {
    "BERT": "M-FAC/bert-mini-finetuned-sst2",
    "FNet": "finiteautomata/bertweet-base-sentiment-analysis"
}

# Streamlit App Layout
st.title("🎬 Movie Review Analysis")
st.markdown("Enter your text review below, select a model, and get the predicted sentiment.")

text = st.text_area("📥 Enter your review here:", height=150)
model_choice = st.selectbox("🤖 Select Model", list(available_models.keys()))

if st.button("Analyze Sentiment"):
    if text.strip() == "":
        st.warning("⚠️ Please enter some text to analyze.")
    else:
        model_name = available_models[model_choice]
        tokenizer, model = load_model(model_name)

        # Inference
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)

        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        pred_class = torch.argmax(probs).item()
        confidence = probs[0][pred_class].item()

        # Use Good/Bad instead of Positive/Negative
        labels = ["Bad", "Good"]
        sentiment = labels[pred_class if pred_class < 2 else 1]  # Default to "Good" if out of range

        # Emoji Mapping
        emojis = {
            "Bad": "👎",
            "Good": "👍"
        }

        # Sentiment color coding
        sentiment_color = {
            "Bad": "#ff4b4b",   # red
            "Good": "#2ecc71"   # green
        }

        # Output Display with color styling
        st.subheader("🧠 Sentiment Result")
        st.markdown(
            f"""<div style='padding:10px; border-radius:10px; background-color:#f9f9f9;'>
                <h4 style='color:{sentiment_color[sentiment]};'>
                    Sentiment: {sentiment} {emojis.get(sentiment, '')}
                </h4>
                <p style='color:#3366cc; font-size:16px;'>
                    Confidence: {confidence * 100:.2f}%
                </p>
            </div>""",
            unsafe_allow_html=True
        )
