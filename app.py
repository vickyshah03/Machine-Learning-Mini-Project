import re
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Multi-Label Topic Classifier", page_icon="🏷️", layout="centered")
st.title("🏷️ Multi-Label Text & News Topic Classifier")
st.write("Trained on **249,887 articles** across **23 topic categories** using **3 Machine Learning models**.")

vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
models = joblib.load("models/all_models.pkl")
topic_cols = joblib.load("models/topic_cols.pkl")

def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

user_text = st.text_area(
    "Paste Article or News Snippet:",
    height=150,
    placeholder="Example: Apple unveiled its latest smartphone featuring advanced AI chips and improved battery efficiency..."
)

if st.button("Predict Topic Tags", type="primary"):
    if not user_text.strip():
        st.warning("Please enter some text first!")
    else:
        vec = vectorizer.transform([clean_text(user_text)])
        st.subheader("1. Predicted Topics Across 3 ML Models")
        cols = st.columns(3)
        for idx, (name, model) in enumerate(models.items()):
            with cols[idx]:
                st.info(f"**{name}**")
                if hasattr(model, "predict_proba"):
                    probs = model.predict_proba(vec)[0]
                    tags = [f"{topic_cols[i].upper()} ({probs[i]*100:.0f}%)" for i in range(len(topic_cols)) if probs[i] >= 0.30]
                    if not tags:
                        best_i = probs.argmax()
                        tags = [f"{topic_cols[best_i].upper()} ({probs[best_i]*100:.0f}%)"]
                else:
                    scores = model.decision_function(vec)[0]
                    tags = [topic_cols[i].upper() for i in range(len(topic_cols)) if scores[i] > 0]
                    if not tags:
                        tags = [topic_cols[scores.argmax()].upper()]
                for tag in tags:
                    st.success(f"✅ {tag}")

        st.subheader("2. Top 8 Topic Confidence Scores (Logistic Regression)")
        lr_probs = models["Logistic Regression"].predict_proba(vec)[0]
        prob_df = pd.DataFrame({
            "Topic": [t.capitalize() for t in topic_cols],
            "Confidence (%)": [round(p * 100, 2) for p in lr_probs]
        }).sort_values(by="Confidence (%)", ascending=False).head(8)
        st.bar_chart(prob_df.set_index("Topic"))