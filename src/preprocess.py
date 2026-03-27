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
    drugs = ['ozempic', 'metformin', 'ibuprofen', 'sertraline', 'lisinopril', 'jardiance']
    os.makedirs('data/processed', exist_ok=True)
    
    for drug in drugs:
        in_path = f'data/raw/{drug}_posts.csv'
        out_path = f'data/processed/{drug}_clean.csv'
        if os.path.exists(in_path):
            df_clean = preprocess_dataframe(pd.read_csv(in_path))
            df_clean.to_csv(out_path, index=False)
            print(f"Saved {len(df_clean)} cleaned {drug.capitalize()} observations")

if __name__ == '__main__':
    main()
