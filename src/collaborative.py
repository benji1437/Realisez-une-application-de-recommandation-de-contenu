import numpy as np
import pandas as pd

from surprise import Reader, Dataset, SVDpp, accuracy
from surprise.model_selection import train_test_split


def scale_rating(df):

    df = df[['user_id', 'article_id', 'normalized_popularity']]

    return df

def train_svd(df):

    df = df[['user_id', 'article_id', 'normalized_popularity']].copy()

    df['user_id'] = df['user_id'].astype(int)
    df['article_id'] = df['article_id'].astype(int)

    reader = Reader(rating_scale=(0, 1))
    data = Dataset.load_from_df(df, reader)

    trainset, testset = train_test_split(data, test_size=0.25)

    algo = SVDpp()
    algo.fit(trainset)

    preds = algo.test(testset)
    accuracy.rmse(preds)

    return algo





def svd_predict(user, items, algo, n=10):

    preds = [(i, algo.predict(user, i).est) for i in items]

    return sorted(preds, key=lambda x: x[1], reverse=True)[:n]