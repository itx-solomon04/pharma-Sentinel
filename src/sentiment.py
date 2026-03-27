"""
sentiment.py — Contextual NLP Profiler
Measures global VADER compound mapping against targeted keyword-aspect sequences.
"""
import numpy as np
import os
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()

ASPECTS = {
    "efficacy": ["works", "effective", "helped", "improved", "better", "control", "lower", "reduced", "lost", "loss"],
    "side_effects": ["nausea", "vomiting", "diarrhea", "tired", "fatigue", "hair", "stomach", "pain", "dizzy", "headache"],
    "cost": ["expensive", "cost", "insurance", "afford", "price", "cheap", "covered"],
    "experience": ["easy", "injection", "convenient", "doctor", "prescription", "dose", "weekly"]
}

def get_vader_sentiment(text):
    if not isinstance(text, str): return 0.0, "neutral"
    compound = analyzer.polarity_scores(text)['compound']
    return compound, "positive" if compound >= 0.05 else "negative" if compound <= -0.05 else "neutral"

def get_aspect_sentiment(text):
    if not isinstance(text, str): return {aspect: np.nan for aspect in ASPECTS.keys()}
    aspect_scores = {aspect: [] for aspect in ASPECTS.keys()}
    sentences = [s.strip().lower() for s in text.replace('!', '.').replace('?', '.').split('.') if s.strip()]
    
    for sentence in sentences:
        for aspect, keywords in ASPECTS.items():
            if any(keyword in sentence for keyword in keywords):
                aspect_scores[aspect].append(analyzer.polarity_scores(sentence)['compound'])
                
    return {aspect: float(np.mean(scores)) if scores else np.nan for aspect, scores in aspect_scores.items()}

def analyze_dataframe(df):
    df = df.copy()
    vader_scores, vader_labels = [], []
    eff_scores, se_scores, cost_scores, exp_scores = [], [], [], []
    overall_aspect_scores = []
    
    for idx, row in df.iterrows():
        text = str(row.get('clean_text', row.get('text', '')))
        v_score, v_label = get_vader_sentiment(text)
        aspect_results = get_aspect_sentiment(text)
        
        vader_scores.append(v_score)
        vader_labels.append(v_label)
        eff_scores.append(aspect_results['efficacy'])
        se_scores.append(aspect_results['side_effects'])
        cost_scores.append(aspect_results['cost'])
        exp_scores.append(aspect_results['experience'])
        
        valid_scores = [s for s in aspect_results.values() if pd.notna(s)]
        overall_aspect_scores.append(float(np.mean(valid_scores)) if valid_scores else np.nan)
            
    df['vader_score'], df['vader_label'] = vader_scores, vader_labels
    df['efficacy_sentiment'], df['side_effect_sentiment'] = eff_scores, se_scores
    df['cost_sentiment'], df['experience_sentiment'] = cost_scores, exp_scores
    df['overall_sentiment'] = overall_aspect_scores
    return df

def print_distributions(df, name):
    dist = df['vader_label'].value_counts(normalize=True) * 100
    print(f"\n--- {name} Sentiment ---")
    print(f"Positive: {dist.get('positive', 0):.1f}% | Negative: {dist.get('negative', 0):.1f}% | Neutral: {dist.get('neutral', 0):.1f}%")

def main():
    drugs = ['ozempic', 'metformin', 'ibuprofen', 'sertraline', 'lisinopril', 'jardiance']
    for drug in drugs:
        in_file = f'data/processed/{drug}_ades.csv'
        out_file = f'data/processed/{drug}_sentiment.csv'
        if not os.path.exists(in_file): continue
            
        df = pd.read_csv(in_file)
        sent_df = analyze_dataframe(df)
        sent_df.to_csv(out_file, index=False)
        print_distributions(sent_df, drug.capitalize())

if __name__ == "__main__":
    main()
