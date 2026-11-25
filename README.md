# 🏷️ Multi-Label Text & News Topic Classifier

An end-to-end Machine Learning system that automatically classifies news articles and text snippets across **23 overlapping topic categories** using multi-label classification (`OneVsRestClassifier`), TF-IDF feature engineering, and an interactive **Streamlit** web prototype.

> **Academic Context:** Developed as a Machine Learning Mini-Project under the guidance of **Prof. Showmik Bhowmik**, Department of Computer Science & Engineering, **Ghani Khan Choudhury Institute of Engineering and Technology (GKCIET)** (B.Tech CSE 2024–2028).

---

## 1. Problem Statement & Objectives

Digital newsrooms and content aggregators process thousands of articles daily that rarely fit into a single rigid category (e.g., an article on central bank interest rates spans both `economy` and `business`, while an international summit spans `politics` and `news`).

**Key Objectives:**
- Build a scalable text preprocessing and multi-label feature extraction pipeline using sublinear TF-IDF (`float32` sparse matrices).
- Perform Exploratory Data Analysis (EDA) to inspect topic imbalance and article length distributions.
- Train, tune, and compare **three supervised Machine Learning algorithms** (`Multinomial Naive Bayes`, `Logistic Regression`, and `Support Vector Machine`) wrapped in a One-vs-Rest multi-label architecture.
- Evaluate models using multi-label metrics (**Micro F1-Score**, **Macro F1-Score**, and **Hamming Loss**) and deploy an interactive **Streamlit** prototype.

---

## 2. Dataset & Multi-Source Data Fusion

To balance broad 23-topic coverage with real-world journalistic phrasing, the training corpus was constructed by streaming and merging data from Hugging Face:

1. **Primary Multi-Label Corpus (`jonaskoenig/topic_classification`):** Streamed **249,887 clean labeled rows** across 23 categories (`news`, `politics`, `sports`, `economy`, `business`, `technology`, `science`, `health`, `movies`, `music`, `cars`, `travel`, etc.).
2. **Domain Adaptation (`fancyzhx/ag_news` & `IINOVAII/news-classification`):** Augmented with real global and Indian news articles mapped into the unified 23-topic schema to improve recognition of modern news entities.

*Note: Raw CSV files and serialized `.pkl` artifacts are excluded from Git tracking via `.gitignore` due to GitHub's 100 MB file limit; running `prepare_data.ipynb` followed by `train_models.py` reproduces the full pipeline locally.*

---

## 3. Project Structure

```text
ML-Mini-Project/
├── prepare_data.ipynb              # Streams & merges multi-label dataset from Hugging Face
├── train_models.py                 # Text cleaning, EDA plots, TF-IDF & 3-model training
├── app.py                          # Interactive Streamlit web application
├── requirements.txt                # Python dependencies
├── .gitignore                      # Excludes large dataset/model binaries
├── results/
│   ├── topic_distribution.png      # EDA: Topic frequency bar chart
│   ├── word_count_distribution.png # EDA: Article word count histogram
│   ├── model_comparison.csv        # Quantitative benchmark table
│   └── model_comparison.png        # Visual comparison of Micro F1-Scores
└── README.md
```

---

## 4. Methodology & Model Comparison

- **Preprocessing:** Lowercasing, URL/noise removal via regex, length filtering, and stop-word removal.
- **Vectorization:** `TfidfVectorizer(max_features=35000, ngram_range=(1, 2), sublinear_tf=True, dtype=np.float32)`.
- **Multi-Label Strategy:** `OneVsRestClassifier` training 23 binary classifiers per algorithm.

### Experimental Results (249,887 Clean Rows, 80/20 Train-Test Split)

| Algorithm | Micro F1-Score (%) | Macro F1-Score (%) | Hamming Loss (Lower = Better) |
| :--- | :---: | :---: | :---: |
| **Multinomial Naive Bayes** (`alpha=0.1`) | 38.82% | 24.28% | 0.0691 |
| **Logistic Regression** (`saga`, `C=2.0`) | 46.00% | 34.53% | 0.0650 |
| **Support Vector Machine (`LinearSVC`)** (`C=0.5`) | **47.57%** | **38.38%** | **0.0649** |

### Key Observations & Error Analysis

- **Best Performer:** **LinearSVC** achieved the highest Micro-F1 (**47.57%**) and lowest Hamming Loss (**0.0649**, corresponding to ~93.5% exact label-level accuracy across 23 labels), outperforming Multinomial Naive Bayes by **+8.75% Micro-F1** due to linear margin separation on high-dimensional bigram TF-IDF features.
- **Scale Impact:** Scaling from an initial 10k subset to ~250k rows boosted Logistic Regression Micro-F1 from 35.67% to **46.00%** (+10.33%).
- **Domain Shift Observation:** On short, single-sentence headlines with out-of-vocabulary proper nouns, bag-of-words TF-IDF can occasionally over-weight generic statistical words (e.g., mapping numerical figures to `science`). Multi-sentence articles and domain-augmented news samples yield high-confidence, accurate multi-label tags.

---

## 5. How to Run Locally

```bash
# 1. Clone the repository
git clone https://github.com/vickyshah03/multi-label-topic-classifier.git
cd multi-label-topic-classifier

# 2. Install dependencies
pip install -r requirements.txt

# 3. Stream dataset & train models (generates models/ and results/)
python prepare_data.ipynb
python train_models.py

# 4. Launch the Streamlit web app
streamlit run app.py
```

---

## 6. Acknowledgements

- **Dataset:** [`jonaskoenig/topic_classification`](https://huggingface.co/datasets/jonaskoenig/topic_classification), [`fancyzhx/ag_news`](https://huggingface.co/datasets/fancyzhx/ag_news), and [`IINOVAII/news-classification`](https://huggingface.co/datasets/IINOVAII/news-classification) via Hugging Face Datasets.
- **Libraries:** `scikit-learn`, `pandas`, `numpy`, `matplotlib`, `seaborn`, and `streamlit`.