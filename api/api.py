from fastapi import FastAPI
import numpy as np

from src.data_loader import load_data, load_clicks_folder
from src.preprocessing import rating
from src.content_based import articles_recomm
from src.collaborative import train_svd, svd_predict
from src.hybrid import hybrid
from src.cold_start import is_new_user

app = FastAPI(title="Recommender API")

# =========================
# LOAD DATA (au démarrage)
# =========================

df_art, df_clicks, emb = load_data()
df_data = load_clicks_folder()
df = df_data.merge(df_art, left_on='click_article_id', right_on='article_id')
df_rating = rating(df)

algo = train_svd(df_rating)

# =========================
# ENDPOINTS
# =========================

@app.get("/")
def root():
    return {"message": "API Recommender OK"}

@app.get("/recommend")
def recommend(user_id: int, model: str = "hybrid", n: int = 5):

    if model == "content":
        recs = articles_recomm(user_id, n, emb, df_rating)

    elif model == "svd":
        recs = svd_predict(user_id, df_rating['article_id'].unique(), algo, n)

    elif model == "hybrid":
        recs = hybrid(user_id, emb, df_rating, algo, n)

    else:
        return {"error": "model inconnu"}

    return {
        "user_id": user_id,
        "model": model,
        "recommendations": recs
    }