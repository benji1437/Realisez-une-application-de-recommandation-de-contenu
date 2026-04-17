import pandas as pd
import pickle

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