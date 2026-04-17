# =====================================================
# IMPORTS
# =====================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from time import time
from collections import defaultdict

from scipy.spatial import distance
from scipy.sparse import csr_matrix

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split as skl_split
from sklearn.preprocessing import MinMaxScaler

from surprise import Reader, Dataset, SVDpp, accuracy
from surprise.model_selection import train_test_split

from implicit.als import AlternatingLeastSquares
from implicit.bpr import BayesianPersonalizedRanking
from implicit.lmf import LogisticMatrixFactorization


# =====================================================
# DATA LOADING
# =====================================================

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


# =====================================================
# PREPROCESSING
# =====================================================

def interaction(df, n_users=100, n_articles=100):
    top_users = df['user_id'].value_counts().head(n_users).index
    top_articles = df['click_article_id'].value_counts().head(n_articles).index

    filtered = df[
        df['user_id'].isin(top_users) &
        df['click_article_id'].isin(top_articles)
    ]

    return filtered.pivot_table(
        index='user_id',
        columns='click_article_id',
        aggfunc='size',
        fill_value=0
    )


def rating(df):
    user_total = df.groupby('user_id').size()

    df_rating = (
        df.groupby(['user_id', 'article_id'])
        .size()
        .rename('article_user_clicks')
        .reset_index()
    )

    df_rating['user_total_clicks'] = df_rating['user_id'].map(user_total)

    df_rating['normalized_popularity'] = (
        df_rating['article_user_clicks'] /
        df_rating['user_total_clicks']
    )

    return df_rating


# =====================================================
# CONTENT-BASED
# =====================================================

def popular(df, n=10):
    return df['article_id'].value_counts().index.tolist()[:n]


def user_embedding(user_articles, clicks, emb, mapping):
    vec = np.zeros(emb.shape[1])
    total = sum(clicks)

    for a, c in zip(user_articles, clicks):
        if a in mapping:
            vec += emb[mapping[a]] * (c / total)

    return vec


def articles_recomm(user_id, n, emb, df):

    mapping = {aid: i for i, aid in enumerate(df['article_id'].unique())}
    user_data = df[df['user_id'] == user_id]

    if user_data.empty:
        return [(a, None) for a in popular(df, n)]

    user_vec = user_embedding(
        user_data['article_id'],
        user_data['article_user_clicks'],
        emb,
        mapping
    )

    unread = list(set(mapping.keys()) - set(user_data['article_id']))
    idxs = [mapping[a] for a in unread if a in mapping]

    sim = 1 - distance.cdist([user_vec], emb[idxs], metric='cosine')[0]

    top = np.argsort(sim)[::-1][:n]
    return [(unread[i], sim[i]) for i in top]


# =====================================================
# PCA
# =====================================================

def make_pca(emb, n=100):
    pca = PCA(n_components=n)
    reduced = pca.fit_transform(emb)
    print("Variance:", pca.explained_variance_ratio_.sum())
    return reduced


# =====================================================
# SVD++
# =====================================================

def scale_rating(df):
    df = df[['user_id', 'article_id', 'normalized_popularity']]
    train, test = skl_split(df, test_size=0.25)

    scaler = MinMaxScaler()
    train['normalized_popularity'] = scaler.fit_transform(train[['normalized_popularity']])
    test['normalized_popularity'] = scaler.transform(test[['normalized_popularity']])

    return pd.concat([train, test])


def train_svd(df):
    reader = Reader(rating_scale=(0,1))
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


# =====================================================
# IMPLICIT MODELS
# =====================================================

def build_sparse(df):
    return csr_matrix(
        (df['normalized_popularity'],
         (df['user_id'], df['article_id']))
    )


def train_implicit(train_df, test_df):

    train = build_sparse(train_df)
    test = build_sparse(test_df)

    models = [
        AlternatingLeastSquares(),
        BayesianPersonalizedRanking(),
        LogisticMatrixFactorization()
    ]

    for m in models:
        print("Training:", m.__class__.__name__)
        m.fit(train)


# =====================================================
# HYBRID
# =====================================================

def hybrid(user, emb, df, algo, n=10, w=0.5):

    cbf = articles_recomm(user, n, emb, df)
    svd = svd_predict(user, df['article_id'].unique(), algo, n)

    scores = {}

    for a, s in cbf:
        if s:
            scores[a] = w * s

    for a, s in svd:
        scores[a] = scores.get(a, 0) + (1 - w) * s

    return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:n]


# =====================================================
# MAIN
# =====================================================

if __name__ == "__main__":

    print("\n===== RECOMMENDER SYSTEM =====")

    # Load
    df_art, df_clicks, emb = load_data()
    df_data = load_clicks_folder()

    df = df_data.merge(df_art, left_on='click_article_id', right_on='article_id')

    # Rating
    df_rating = rating(df)

    # Content-based
    print("\nContent-based:")
    print(articles_recomm(16280, 5, emb, df_rating))

    # PCA
    emb_reduced = make_pca(emb)

    # SVD
    X = scale_rating(df_rating)
    algo = train_svd(X)

    # Hybrid
    print("\nHybrid:")
    print(hybrid(16280, emb, df_rating, algo))

    print("\n✔ DONE")