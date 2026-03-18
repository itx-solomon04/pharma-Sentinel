"""
collect.py — Data Collection Pipeline
Handles loading of raw datasets, combinations, and synthetic data generation for testing.
"""
import pandas as pd
import random
import os
from datetime import datetime, timedelta

def generate_synthetic_ozempic_reviews(count=500):
    conditions = ["Type 2 Diabetes", "Obesity", "Weight Management"]
    positive_experiences = [
        "I've experienced significant weight loss.", "My appetite is so much reduced.", "I finally have much better blood sugar control.", "My A1C points dropped tremendously.", "I actually have more energy now.", "This drug changed my life completely.", "The best medication I have ever taken."
    ]
    negative_experiences = [
        "The nausea is overwhelming.", "I've been dealing with frequent vomiting.", "Horrible diarrhea nearly every day.", "Severe constipation that won't go away.", "I have constant fatigue.", "There is terrible injection site pain.", "I'm noticing significant hair loss.", "I have major pancreatitis concerns now.", "Dealing with symptoms of gastroparesis."
    ]
    mixed_experiences = [
        "It works really well, but it's incredibly expensive.", "The medication is effective, but side effects are barely manageable.", "I lose weight, but I'm always tired.", "Blood sugar is down, though the nausea restricts my life.", "It gets the job done, but I feel horrible on injection day."
    ]
    
    def build_review(rating):
        num_sentences = random.randint(1, 4)
        if rating >= 8: pool = positive_experiences * 4 + mixed_experiences
        elif rating <= 3: pool = negative_experiences * 4 + mixed_experiences
        else: pool = positive_experiences + negative_experiences + mixed_experiences * 3
            
        sentences = []
        for _ in range(num_sentences): sentences.append(random.choice(pool))
        unique_sentences = list(dict.fromkeys(sentences))
        if not unique_sentences: unique_sentences = [random.choice(mixed_experiences)]
            
        if len(unique_sentences) > 1 and random.random() > 0.5:
            unique_sentences.insert(0, random.choice(["Honestly,", "To be clear,", "In my experience,"]))
        return " ".join(unique_sentences).replace(" ,", ",")

    data = []
    start_date, end_date = datetime(2022, 1, 1), datetime(2024, 12, 31)
    days_between = (end_date - start_date).days
    
    for _ in range(count):
        dist = random.random()
        if dist < 0.45: rating = random.randint(8, 10)
        elif dist < 0.85: rating = random.randint(1, 3)
        else: rating = random.randint(4, 7)
            
        data.append({
            'drug': 'Ozempic (Synthetic)', 'condition': random.choice(conditions),
            'text': build_review(rating), 'rating': rating,
            'date': (start_date + timedelta(days=random.randrange(days_between))).strftime("%d-%b-%y"),
            'helpful_votes': random.randint(0, 150)
        })
    return pd.DataFrame(data)

def load_and_combine_data(train_path, test_path):
    dfs = []
    if os.path.exists(train_path): dfs.append(pd.read_csv(train_path))
    if os.path.exists(test_path): dfs.append(pd.read_csv(test_path))
    return pd.concat(dfs, ignore_index=True) if dfs else None

def process_and_save_data(df):
    df = df.rename(columns={'drugName': 'drug', 'review': 'text', 'usefulCount': 'helpful_votes'})
    if 'drug' not in df.columns: return None, None
    df = df.dropna(subset=['drug'])

    metformin_df = df[df['drug'].str.contains("metformin|glucophage|fortamet", case=False, na=False)].copy()
    ozempic_df = generate_synthetic_ozempic_reviews(count=500)

    os.makedirs('data/raw', exist_ok=True)
    ozempic_df.to_csv('data/raw/ozempic_posts.csv', index=False)
    metformin_df.to_csv('data/raw/metformin_posts.csv', index=False)
    return ozempic_df, metformin_df

def main():
    combined_df = load_and_combine_data("data/raw/drug_reviews_train.csv", "data/raw/drug_reviews_test.csv")
    if combined_df is not None and not combined_df.empty: process_and_save_data(combined_df)
    else:
        ozempic_df = generate_synthetic_ozempic_reviews(count=500)
        os.makedirs('data/raw', exist_ok=True)
        ozempic_df.to_csv('data/raw/ozempic_posts.csv', index=False)

if __name__ == "__main__":
    main()
