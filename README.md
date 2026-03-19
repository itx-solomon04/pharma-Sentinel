# 💊 Pharma Sentinel — Patient Intelligence Platform

> An end-to-end pharmacovigilance NLP pipeline that analyzes patient-reported drug experiences from online reviews — detecting adverse events, classifying severity, and understanding sentiment across multiple drugs.

🔗 **Live Demo:** [pharma-sentinal-and-adverse-effect.streamlit.app](https://pharma-sentinal-and-adverse-effect.streamlit.app/)
📁 **Dataset:** UCI Drug Review Dataset (215,063 reviews)

---

## 📌 Project Overview

Pharma Sentinel mines unstructured patient review data to surface clinically meaningful insights — comparing how patients experience different drugs across efficacy, side effects, cost, and overall sentiment. Built as a mini-project for pharmacovigilance research using real-world NLP and ML techniques.

**Drugs analyzed:** Ozempic (Semaglutide) vs Metformin
**Reviews processed:** 822 patient reviews
**Overall positive sentiment:** 47.2%

---

## 🚀 Features

- **Adverse Drug Event (ADE) Detection** — rule-based NER with MedDRA System Organ Class mapping
- **3-Tier Severity Classification** — Mild / Moderate / Severe using linguistic intensity analysis
- **Aspect-Based Sentiment Analysis** — separately scores Efficacy, Side Effects, Cost, and Experience using VADER
- **Topic Modelling** — BERTopic to discover recurring patient discussion themes
- **Live Review Analyzer** — type any patient review and get instant ML analysis in real time
- **CSV Upload Mode** — upload any drug review dataset to run the full pipeline
- **Demo Mode** — preloaded Ozempic vs Metformin comparison with 822 reviews

---

## 🧠 Key Findings

| Metric | Ozempic | Metformin |
|--------|---------|-----------|
| Positive Sentiment | 47.4% | 48.8% |
| Top ADE | Nausea (103 mentions) | Diarrhea (79 mentions) |
| Top Severity | Severe | Moderate |
| Avg Rating | 5.9 / 10 | 7.0 / 10 |
| Main Topic | Dramatic short-term outcomes | Long-term condition management |

---

## 🏗️ Architecture

```
Raw Reviews → Preprocessing → ADE Detection → Sentiment Analysis → Topic Modelling → Dashboard
     ↓              ↓               ↓                  ↓                  ↓              ↓
 UCI Dataset    Clean text      MedDRA mapping      VADER + Aspect     BERTopic      Streamlit
 215K reviews   HTML decode     Severity scoring    Radar chart        Clustering    Plotly charts
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Data Collection | UCI Drug Review Dataset, Kaggle |
| Preprocessing | Python, Pandas, NLTK, Regex |
| ADE Detection | Rule-based NER, MedDRA Ontology |
| Sentiment Analysis | VADER, Aspect-Based Scoring |
| Topic Modelling | BERTopic, scikit-learn |
| Dashboard | Streamlit, Plotly, Matplotlib |
| Deployment | Streamlit Cloud, GitHub |

---

## 📁 Project Structure

```
pharma-sentiment-tracker/
├── src/
│   ├── collect.py        # Data collection and synthetic generation
│   ├── preprocess.py     # Text cleaning and normalization
│   ├── ade.py            # ADE detection, MedDRA mapping, severity classification
│   ├── sentiment.py      # VADER and aspect-based sentiment analysis
│   ├── topics.py         # BERTopic topic modelling
│   └── genai.py          # GenAI summary generation (GPT-4 ready)
├── dashboard/
│   └── app.py            # Streamlit dashboard — 5 pages, 3 input modes
├── data/
│   ├── raw/              # Raw collected data (gitignored)
│   └── processed/        # Cleaned ML-ready CSV files
├── requirements.txt      # Python dependencies
└── README.md
```

---

## ⚙️ Setup & Run Locally

```bash
# Clone the repo
git clone https://github.com/itx-solomon04/pharma-Sentinel.git
cd pharma-Sentinel

# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the dashboard
streamlit run dashboard/app.py
```

---

## 🔬 Pipeline Details

### 1. Data Collection (`src/collect.py`)
Loads the UCI Drug Review dataset (215,063 records) and filters for target drugs. Generates synthetic reviews for drugs with limited dataset coverage using documented clinical outcomes.

### 2. Preprocessing (`src/preprocess.py`)
Cleans raw text by decoding HTML entities, lowercasing, removing URLs and special characters, and filtering stop words — while preserving medical negations like "not", "no", "never" which are critical for accurate sentiment scoring.

### 3. ADE Detection (`src/ade.py`)
Uses rule-based NER with a curated medical keyword dictionary mapped to MedDRA System Organ Classes. Classifies each detected ADE as Mild, Moderate, or Severe using a two-layer approach: linguistic intensity keywords + VADER compound score.

### 4. Sentiment Analysis (`src/sentiment.py`)
Runs VADER for overall sentiment scoring, then performs aspect-based analysis to separately score patient feelings about Efficacy, Side Effects, Cost, and Experience. Both drugs output consistent normalized scores (-1 to +1) for direct comparison.

### 5. Topic Modelling (`src/topics.py`)
Applies BERTopic to discover recurring discussion themes without predefined categories. Key finding: Ozempic patients discuss dramatic short-term physical outcomes while Metformin patients discuss long-term chronic condition management.

### 6. Dashboard (`dashboard/app.py`)
5-page interactive Streamlit app with Modern Slate professional theme. Supports 3 input modes: Demo (preloaded data), CSV Upload (custom datasets), and Live Review Analyzer (real-time single-review ML inference).

---

## 📊 Sample Output — Live Review Analyzer

**Input:** *"I lost 20 pounds on Ozempic but the nausea is absolutely unbearable every morning"*

```
Sentiment:    NEGATIVE (-0.62)
ADEs found:   2
  → NAUSEA     [SEVERE]   Gastrointestinal disorders
  → WEIGHT LOSS [MILD]    Metabolism disorders
MedDRA cats:  2 organ systems
```

---

## 🔮 Future Improvements

- Replace synthetic Ozempic data with real Reddit API data (scraper built, pending API approval)
- Fine-tune BioBERT on medical NER for higher ADE detection accuracy
- Add time-series sentiment tracking to monitor drug perception over time
- Expand to 6+ drugs across multiple therapeutic categories
- Integrate GPT-4 for automated pharmacovigilance report generation

---

## 👤 Author

Built by **Elankavi solomon** as part of a pharmacovigilance NLP research project.
📧 studiossander@gmail.com
🔗 [GitHub](https://github.com/itx-solomon04)

---

*This project is for educational and research purposes. Not intended for clinical decision making.*
