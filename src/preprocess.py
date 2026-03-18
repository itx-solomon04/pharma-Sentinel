"""
preprocess.py — Text Preprocessing Pipeline
Cleans strings natively bypassing HTML and standardizing tokens while preserving medical semantics.
"""
import html
import os
import re

import nltk
import pandas as pd
from nltk.corpus import stopwords

nltk.download('stopwords', quiet=True)

def clean_text(text):
    if not isinstance(text, str): return ""
    text = html.unescape(text).lower()
    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r"[^a-z0-9'-]", ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

def remove_stopwords(text):
    if not isinstance(text, str): return ""
    stop_words = set(stopwords.words('english')) - {"not", "no", "never", "without", "pain", "side", "effect"}
    return " ".join([w for w in text.split() if w not in stop_words])

def preprocess_dataframe(df):
    df = df.copy().drop_duplicates(subset=['text'])
    df['clean_text'] = df['text'].apply(clean_text).apply(remove_stopwords)
    df['word_count'] = df['clean_text'].apply(lambda x: len(x.split()))
    df = df[df['text'].str.len() >= 20]
    return df[df['clean_text'].str.strip() != ""]

def main():
    ozempic_in, metformin_in = 'data/raw/ozempic_posts.csv', 'data/raw/metformin_posts.csv'
    ozempic_out, metformin_out = 'data/processed/ozempic_clean.csv', 'data/processed/metformin_clean.csv'
    
    os.makedirs('data/processed', exist_ok=True)
    
    if os.path.exists(ozempic_in):
        oz_clean = preprocess_dataframe(pd.read_csv(ozempic_in))
        oz_clean.to_csv(ozempic_out, index=False)
        print(f"Saved {len(oz_clean)} cleaned Ozempic observations")
        
    if os.path.exists(metformin_in):
        met_clean = preprocess_dataframe(pd.read_csv(metformin_in))
        met_clean.to_csv(metformin_out, index=False)
        print(f"Saved {len(met_clean)} cleaned Metformin observations")

if __name__ == '__main__':
    main()
