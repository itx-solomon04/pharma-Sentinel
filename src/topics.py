"""
topics.py — Dense Topic Modeling
Processes language vectors mapping to isolated topic arrays via robust BERTopic configurations.
"""
import os
import signal
import pandas as pd

def timeout_handler(signum, frame):
    raise Exception("BERTopic processing exceeded 3 minute timeout. Aborting computation.")

def run_topic_model(texts, drug_name):
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(180)
    try:
        from bertopic import BERTopic
        model = BERTopic(min_topic_size=5, nr_topics="auto")
        topics, probabilities = model.fit_transform(texts)
        topic_info = model.get_topic_info()
        signal.alarm(0)
        return model, topic_info
    except Exception as e:
        signal.alarm(0)
        return None, pd.DataFrame()

def get_topic_summary(topic_model):
    if not topic_model: return pd.DataFrame()
    records = []
    
    for idx, row in topic_model.get_topic_info().iterrows():
        topic_words = topic_model.get_topic(row['Topic'])
        keywords = ", ".join([word for word, score in topic_words][:5]) if topic_words else row['Name']
        records.append({'topic_id': row['Topic'], 'keywords': keywords, 'size': row['Count']})
        
    return pd.DataFrame(records)

def print_themes(df, name):
    print(f"\n--- {name} CORE THEMES ---")
    if df.empty: print("No topics generated.")
    else:
        for _, row in df[df['topic_id'] != -1].head(10).iterrows():
            print(f"Topic {row['topic_id']} (Size: {row['size']}): {row['keywords']}")

def main():
    drugs = ['ozempic', 'metformin', 'ibuprofen', 'sertraline', 'lisinopril', 'jardiance']
    for drug in drugs:
        in_file = f'data/processed/{drug}_sentiment.csv'
        out_file = f'data/processed/{drug}_topics.csv'
        if not os.path.exists(in_file): continue
            
        df = pd.read_csv(in_file).dropna(subset=['clean_text'])
        if df.empty: continue
            
        model, _ = run_topic_model(df['clean_text'].tolist(), drug.capitalize())
        summary = get_topic_summary(model)
        summary.to_csv(out_file, index=False)
        print_themes(summary, drug.upper())

if __name__ == '__main__':
    main()
