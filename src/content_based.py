import numpy as np
from scipy.spatial import distance

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

def recommend_new_items(emb, existing_ids, n=10):
    """
    Recommande des nouveaux articles uniquement (cold start items)
    """
    all_ids = list(range(len(emb)))
    new_items = list(set(all_ids) - set(existing_ids))

    return new_items[:n]

def is_new_item(article_id, df):
    return article_id not in df['article_id'].values