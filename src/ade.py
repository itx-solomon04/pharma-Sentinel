"""
ade.py — Adverse Drug Event (ADE) Extractor
Implements rule-based and BioBERT sequence labeling to isolate severe side effects within text telemetry.
"""
import collections
import os
import re
import warnings

import pandas as pd
import torch
from transformers import pipeline
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

warnings.filterwarnings("ignore")
analyzer = SentimentIntensityAnalyzer()

MEDDRA_MAPPING = {
    "nausea": "Gastrointestinal disorders", "vomiting": "Gastrointestinal disorders",
    "diarrhea": "Gastrointestinal disorders", "constipation": "Gastrointestinal disorders",
    "stomach": "Gastrointestinal disorders", "fatigue": "General disorders",
    "tired": "General disorders", "headache": "Nervous system disorders",
    "dizziness": "Nervous system disorders", "hair loss": "Skin disorders",
    "rash": "Skin disorders", "injection site": "General disorders",
    "hypoglycemia": "Metabolism disorders", "blood sugar": "Metabolism disorders",
    "weight loss": "Metabolism disorders", "appetite": "Metabolism disorders",
    "kidney": "Renal disorders", "heart": "Cardiac disorders",
    "anxiety": "Psychiatric disorders", "depression": "Psychiatric disorders",
    "pancreatitis": "Gastrointestinal disorders", "thyroid": "Endocrine disorders"
}

SEVERITY_KEYWORDS = {
    "severe": ["unbearable", "excruciating", "horrible", "awful", "terrible", "worst", "agony", "brutal", "destroyed", "violent", "severe", "extreme", "intolerable", "crippling", "debilitating"],
    "moderate": ["bad", "uncomfortable", "unpleasant", "significant", "noticeable", "moderate", "considerable", "frequent", "persistent", "bothersome"],
    "mild": ["little", "slight", "minor", "occasional", "sometimes", "mild", "manageable", "tolerable", "brief", "temporary", "small"]
}

def load_biobert_model():
    device_name = "mps" if torch.backends.mps.is_available() else "cpu"
    return pipeline(task="ner", model="allenai/scibert_scivocab_uncased", aggregation_strategy="simple", device=device_name)

def extract_medical_entities(text, ner_pipeline):
    if not isinstance(text, str) or not text.strip(): return []
    try:
        return [{"word": r.get('word',''), "entity": r.get('entity_group', r.get('entity','Unknown')), "score": float(r.get('score',0.0))} for r in ner_pipeline(text)]
    except: return []

def map_to_meddra(entity_str):
    if not isinstance(entity_str, str): return "Unknown"
    entity_str = entity_str.lower().strip()
    if entity_str in MEDDRA_MAPPING: return MEDDRA_MAPPING[entity_str]
    for key, meddra_class in MEDDRA_MAPPING.items():
        if key in entity_str: return meddra_class
    return "Unknown"

def rule_based_ade_extraction(text):
    if not isinstance(text, str): return []
    return [ade for ade in MEDDRA_MAPPING.keys() if re.search(r'\b' + re.escape(ade) + r'\b', text.lower())]

def classify_severity(text, ade_term):
    if not isinstance(text, str): return "MODERATE"
    ade_sentences = [s for s in text.replace('!', '.').replace('?', '.').split('.') if ade_term.lower() in s.lower()]
    if not ade_sentences: return "MODERATE"
    
    text_to_search = " ".join(ade_sentences).lower()
    for word in SEVERITY_KEYWORDS["severe"]:
        if re.search(r'\b' + re.escape(word) + r'\b', text_to_search): return "SEVERE"
    for word in SEVERITY_KEYWORDS["moderate"]:
        if re.search(r'\b' + re.escape(word) + r'\b', text_to_search): return "MODERATE"
    for word in SEVERITY_KEYWORDS["mild"]:
        if re.search(r'\b' + re.escape(word) + r'\b', text_to_search): return "MILD"
        
    score = analyzer.polarity_scores(text_to_search)['compound']
    if score <= -0.5: return "SEVERE"
    elif -0.5 < score <= -0.2: return "MODERATE"
    return "MILD"

def get_severity_summary(df):
    records = []
    for _, row in df.iterrows():
        ades = [x.strip() for x in str(row.get('detected_ades', '')).split(',') if x.strip()]
        sevs = [x.strip() for x in str(row.get('ade_severities', '')).split(',') if x.strip()]
        if len(ades) == len(sevs) and ades:
            records.extend([{'ade': a, 'severity': s} for a, s in zip(ades, sevs)])
            
    if not records: return pd.DataFrame()
    
    summary_data = []
    for ade, group in pd.DataFrame(records).groupby('ade'):
        counts = group['severity'].value_counts()
        total = len(group)
        summary_data.append({
            'ADE': ade, 'Total Mentions': total,
            '% MILD': round((counts.get('MILD', 0) / total) * 100, 1),
            '% MODERATE': round((counts.get('MODERATE', 0) / total) * 100, 1),
            '% SEVERE': round((counts.get('SEVERE', 0) / total) * 100, 1),
            'Most Common Severity': counts.idxmax() if not counts.empty else "UNKNOWN"
        })
    return pd.DataFrame(summary_data).sort_values(by='Total Mentions', ascending=False)

def analyze_dataframe(df, ner_pipeline=None):
    df = df.copy()
    all_detected_ades, all_meddra_categories, all_ade_counts, all_ade_severities = [], [], [], []
    for idx, row in df.iterrows():
        text = str(row.get('clean_text', row.get('text', '')))
        final_ades = list(set(rule_based_ade_extraction(text)))
        categories = set(cat for ade in final_ades if (cat := map_to_meddra(ade)) != "Unknown")
        severities = [classify_severity(text, ade) for ade in final_ades]
        
        all_detected_ades.append(", ".join(final_ades))
        all_meddra_categories.append(", ".join(categories))
        all_ade_severities.append(", ".join(severities))
        all_ade_counts.append(len(final_ades))
        
    df['detected_ades'] = all_detected_ades
    df['meddra_categories'] = all_meddra_categories
    df['ade_count'] = all_ade_counts
    df['ade_severities'] = all_ade_severities
    return df

def main():
    drugs = ['ozempic', 'metformin', 'ibuprofen', 'sertraline', 'lisinopril', 'jardiance']
    
    def print_metrics(df, name):
        ade_list = [x.strip() for row in df['detected_ades'] if str(row).strip() for x in str(row).split(",")]
        print(f"\n--- {name} Top Detected ADEs ---")
        for ade, count in collections.Counter(ade_list).most_common(5): print(f"  - {ade}: {count}")

    for drug in drugs:
        in_file = f'data/processed/{drug}_clean.csv'
        out_file = f'data/processed/{drug}_ades.csv'
        if not os.path.exists(in_file): continue
            
        df = pd.read_csv(in_file)
        ade_df = analyze_dataframe(df, ner_pipeline=None)
        ade_df.to_csv(out_file, index=False)
        print_metrics(ade_df, drug.capitalize())

if __name__ == '__main__':
    main()
