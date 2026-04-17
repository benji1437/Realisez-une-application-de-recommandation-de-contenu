import pandas as pd
import pickle
import os

def load_data():
    df_art = pd.read_csv('articles_metadata.csv')
    df_clicks = pd.read_csv('clicks_sample.csv')

    with open('articles_embeddings.pickle', 'rb') as f:
        embeddings = pickle.load(f)

    return df_art, df_clicks, embeddings


def load_clicks_folder(path='clicks/'):
    files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.csv')]
    data = [pd.read_csv(f) for f in files]
    return pd.concat(data, ignore_index=True)