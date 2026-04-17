import numpy as np

from .content_based import articles_recomm
from .collaborative import svd_predict
from .cold_start import is_new_user, popular_recommendation


def hybrid(user, emb, df, algo, n=10, w=0.5):

    # 🧊 COLD START USER
    if is_new_user(user, df):
        print("Cold start user → popularité")
        return [(a, None) for a in popular_recommendation(df, n)]

    # 🔥 Normal case
    cbf = articles_recomm(user, n, emb, df)
    svd = svd_predict(user, df['article_id'].unique(), algo, n)

    scores = {}

    for a, s in cbf:
        if s:
            scores[a] = w * s

    for a, s in svd:
        scores[a] = scores.get(a, 0) + (1 - w) * s

    return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:n]


