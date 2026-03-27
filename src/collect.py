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

def generate_synthetic_reviews(drug_name, condition, n=300):
    """Generate realistic synthetic patient reviews for drugs with limited dataset coverage"""
    DRUG_PROFILES = {
        "Ibuprofen": {
            "positive": [
                "Ibuprofen works fast for my pain relief, highly recommend",
                "Great for headaches and inflammation, very effective",
                "Works well for my arthritis pain, significant improvement",
                "Quick relief from fever and body aches",
                "Effective anti-inflammatory, helps with my joint pain"
            ],
            "negative": [
                "Severe stomach pain and nausea after taking ibuprofen",
                "Caused heartburn and acid reflux, very uncomfortable",
                "Got terrible headaches and dizziness from this drug",
                "Stomach ulcer developed after long term use, awful",
                "Kidney pain and water retention, very concerned"
            ],
            "ades": ["stomach pain", "nausea", "heartburn", "dizziness", "headache", "kidney pain"]
        },
        "Sertraline": {
            "positive": [
                "Sertraline has helped my depression significantly after 6 weeks",
                "Anxiety is much better controlled, finally feeling normal",
                "Life changing medication for my OCD and depression",
                "Mood improved dramatically, sleeping better too",
                "Finally found an antidepressant that works without bad side effects"
            ],
            "negative": [
                "Severe nausea and diarrhea for the first month",
                "Sexual dysfunction is a major side effect, very frustrating",
                "Insomnia got worse when I started sertraline",
                "Weight gain of 20 pounds in 3 months, very unhappy",
                "Headaches and sweating constantly, considering stopping"
            ],
            "ades": ["nausea", "insomnia", "weight gain", "headache", "sweating", "diarrhea"]
        },
        "Lisinopril": {
            "positive": [
                "Blood pressure perfectly controlled with lisinopril",
                "Great medication for hypertension, minimal side effects",
                "BP dropped from 160/100 to 120/80, very effective",
                "Protecting my kidneys while controlling blood pressure",
                "Easy once daily dosing, blood pressure well managed"
            ],
            "negative": [
                "Persistent dry cough that won't go away since starting lisinopril",
                "Dizziness and lightheadedness when standing up",
                "Severe cough is unbearable, need to switch medications",
                "Swelling in my lips and throat, very scary reaction",
                "Fatigue and weakness, hard to get through the day"
            ],
            "ades": ["dry cough", "dizziness", "fatigue", "swelling", "headache", "weakness"]
        },
        "Jardiance": {
            "positive": [
                "Jardiance lowered my A1C from 9.2 to 6.8 in 3 months",
                "Lost 15 pounds as a bonus while controlling blood sugar",
                "Heart benefits are well documented, feel much safer",
                "Blood sugar finally under control after years of struggling",
                "Energy improved significantly, A1C dropping steadily"
            ],
            "negative": [
                "Recurrent UTI infections since starting Jardiance",
                "Yeast infections constantly, very uncomfortable",
                "Genital itching and burning, embarrassing side effect",
                "Frequent urination disrupting my sleep every night",
                "Dehydration and dizziness, had to reduce dose"
            ],
            "ades": ["UTI", "yeast infection", "frequent urination", "dizziness", "dehydration", "fatigue"]
        }
    }

    profile = DRUG_PROFILES.get(drug_name, {})
    positive_reviews = profile.get("positive", [])
    negative_reviews = profile.get("negative", [])

    reviews = []
    for i in range(n):
        is_positive = random.random() > 0.45
        base = random.choice(positive_reviews if is_positive else negative_reviews)
        rating = random.randint(7, 10) if is_positive else random.randint(1, 4)
        date = (datetime(2022, 1, 1) + timedelta(days=random.randint(0, 900))).strftime("%Y-%m-%d")
        reviews.append({
            "drug": drug_name,
            "condition": condition,
            "text": base,
            "rating": rating,
            "date": date,
            "helpful_votes": random.randint(0, 50),
            "source": "synthetic"
        })
    return pd.DataFrame(reviews)

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
    if combined_df is not None and not combined_df.empty:
        process_and_save_data(combined_df)
    else:
        ozempic_df = generate_synthetic_ozempic_reviews(count=500)
        os.makedirs('data/raw', exist_ok=True)
        ozempic_df.to_csv('data/raw/ozempic_posts.csv', index=False)

    # Generate synthetic data for 4 new drugs
    new_drugs = [
        ("Ibuprofen", "Pain / Inflammation"),
        ("Sertraline", "Depression / Anxiety"),
        ("Lisinopril", "Hypertension"),
        ("Jardiance", "Type 2 Diabetes"),
    ]

    os.makedirs('data/raw', exist_ok=True)
    for drug_name, condition in new_drugs:
        df_new = generate_synthetic_reviews(drug_name, condition, n=300)
        out_path = f"data/raw/{drug_name.lower()}_posts.csv"
        df_new.to_csv(out_path, index=False)
        print(f"✅ {drug_name}: {len(df_new)} reviews → {out_path}")

if __name__ == "__main__":
    main()
