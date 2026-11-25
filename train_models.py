import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score, hamming_loss

os.makedirs("results", exist_ok=True)
os.makedirs("models", exist_ok=True)

print("1. Loading 250k dataset 'topic_dataset.csv'...")
df = pd.read_csv("topic_dataset.csv")
topic_cols = [col for col in df.columns if col != 'text']
print(f"Loaded {len(df):,} rows across {len(topic_cols)} topic categories.")

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

print("2. Cleaning text...")
df['clean_text'] = df['text'].apply(clean_text)
df = df[df['clean_text'].str.len() > 15].reset_index(drop=True)
df['word_count'] = df['clean_text'].apply(lambda x: len(x.split()))
print(f"Clean rows ready for training: {len(df):,}")

print("3. Generating EDA charts in 'results/'...")
topic_counts = df[topic_cols].sum().sort_values(ascending=False)

plt.figure(figsize=(11, 5))
sns.barplot(x=topic_counts.index, y=topic_counts.values, hue=topic_counts.index, palette="viridis", legend=False)
plt.title("Topic Frequency Distribution Across 250k Dataset")
plt.xlabel("Topic Category")
plt.ylabel("Number of Articles")
plt.xticks(rotation=35, ha='right')
plt.tight_layout()
plt.savefig("results/topic_distribution.png")
plt.close()

plt.figure(figsize=(8, 4))
sns.histplot(df['word_count'], bins=40, kde=True, color='teal')
plt.title("Article Word Count Distribution")
plt.xlabel("Word Count")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("results/word_count_distribution.png")
plt.close()

X_train, X_test, Y_train, Y_test = train_test_split(
    df['clean_text'], df[topic_cols].values.astype(np.int8), test_size=0.2, random_state=42
)

print("4. Extracting TF-IDF features (35,000 n-grams, float32 sparse matrix)...")
vectorizer = TfidfVectorizer(
    stop_words='english',
    max_features=35000,
    ngram_range=(1, 2),
    sublinear_tf=True,
    dtype=np.float32
)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

models = {
    "Multinomial Naive Bayes": OneVsRestClassifier(MultinomialNB(alpha=0.1), n_jobs=-1),
    "Logistic Regression": OneVsRestClassifier(LogisticRegression(solver='saga', max_iter=500, C=2.0), n_jobs=-1),
    "Support Vector Machine (LinearSVC)": OneVsRestClassifier(LinearSVC(random_state=42, C=0.5), n_jobs=-1)
}

results = []
print("\n5. Training & Comparing 3 Multi-Label ML Models on 250k Dataset:")
print("-" * 78)
for name, model in models.items():
    model.fit(X_train_tfidf, Y_train)
    Y_pred = model.predict(X_test_tfidf)
    
    micro_f1 = f1_score(Y_test, Y_pred, average='micro', zero_division=0)
    macro_f1 = f1_score(Y_test, Y_pred, average='macro', zero_division=0)
    h_loss = hamming_loss(Y_test, Y_pred)
    
    results.append({
        "Algorithm": name,
        "Micro F1-Score (%)": round(micro_f1 * 100, 2),
        "Macro F1-Score (%)": round(macro_f1 * 100, 2),
        "Hamming Loss": round(h_loss, 4)
    })
    print(f"{name:35s} | Micro-F1: {micro_f1*100:.2f}% | Macro-F1: {macro_f1*100:.2f}% | Hamming Loss: {h_loss:.4f}")

results_df = pd.DataFrame(results)
results_df.to_csv("results/model_comparison.csv", index=False)

plt.figure(figsize=(8, 5))
sns.barplot(data=results_df, x="Algorithm", y="Micro F1-Score (%)", hue="Algorithm", palette="magma", legend=False)
plt.title("Multi-Label Model Comparison on 250k Dataset (Micro F1-Score %)")
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig("results/model_comparison.png")
plt.close()

joblib.dump(vectorizer, "models/tfidf_vectorizer.pkl")
joblib.dump(models, "models/all_models.pkl")
joblib.dump(topic_cols, "models/topic_cols.pkl")
print("-" * 78)
print("SUCCESS! 250k models and EDA charts saved.")