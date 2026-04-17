from src.data_loader import load_data, load_clicks_folder
from src.preprocessing import rating
from src.content_based import articles_recomm
from src.collaborative import train_svd, svd_predict
from src.hybrid import hybrid
from src.utils import save
import numpy as np

def main():

    print("\n===== RECOMMENDER SYSTEM =====")

    # LOAD DATA
    df_art, df_clicks, emb = load_data()
    df_data = load_clicks_folder()

    df = df_data.merge(df_art, left_on='click_article_id', right_on='article_id')

    # PREPROCESSING
    df_rating = rating(df)

    # CONTENT BASED
    print("\n[CONTENT-BASED]")
    print(articles_recomm(16280, 5, emb, df_rating))

    # SVD
    print("\n[SVD++]")
    algo = train_svd(df_rating)
    save(algo, "models/svd.pkl")

    # HYBRID
    print("\n[HYBRID]")
    print(hybrid(16280, emb, df_rating, algo))

    print("\n✔ PIPELINE DONE")

    # 🧊 TEST COLD START USER
    print("\n[COLD START USER]")
    print(hybrid(999999, emb, df_rating, algo))  # utilisateur inconnu


    
    # =========================
    # 📰 TEST COLD START ITEM (VERSION PRO)
    # =========================

    print("\n[COLD START ITEM - PRO]")

    # Simuler un nouvel embedding (comme si nouvel article)
    new_embedding = np.random.rand(emb.shape[1])

    # Ajouter au tableau embeddings
    emb_extended = np.vstack([emb, new_embedding])

    new_article_id = len(emb)  # nouvel ID

    print(f"Nouvel article ajouté avec ID: {new_article_id}")

    # Recommandation avec embeddings mis à jour
    print(articles_recomm(16280, 5, emb_extended, df_rating))

if __name__ == "__main__":
    main()