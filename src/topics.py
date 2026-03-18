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

def compare_drug_topics(oz_topics, met_topics):
    def print_themes(df, name):
        print(f"\n--- {name} CORE THEMES ---")
        if df.empty: print("No topics generated.")
        else:
            for _, row in df[df['topic_id'] != -1].head(10).iterrows():
                print(f"Topic {row['topic_id']} (Size: {row['size']}): {row['keywords']}")
                
    print_themes(oz_topics, "OZEMPIC")
    print_themes(met_topics, "METFORMIN")

def main():
    ozempic_file, metformin_file = 'data/processed/ozempic_sentiment.csv', 'data/processed/metformin_sentiment.csv'
    if not (os.path.exists(ozempic_file) and os.path.exists(metformin_file)): return
        
    oz_df = pd.read_csv(ozempic_file).dropna(subset=['clean_text'])
    met_df = pd.read_csv(metformin_file).dropna(subset=['clean_text'])
    
    oz_model, _ = run_topic_model(oz_df['clean_text'].tolist(), "Ozempic")
    oz_summary = get_topic_summary(oz_model)
    oz_summary.to_csv('data/processed/ozempic_topics.csv', index=False)
    
    met_model, _ = run_topic_model(met_df['clean_text'].tolist(), "Metformin")
    met_summary = get_topic_summary(met_model)
    met_summary.to_csv('data/processed/metformin_topics.csv', index=False)
    
    compare_drug_topics(oz_summary, met_summary)

if __name__ == '__main__':
    main()
