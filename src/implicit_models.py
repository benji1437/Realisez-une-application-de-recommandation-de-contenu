import numpy as np
from scipy.sparse import csr_matrix
from time import time

from implicit.als import AlternatingLeastSquares
from implicit.bpr import BayesianPersonalizedRanking
from implicit.lmf import LogisticMatrixFactorization


def build_sparse(df):

    return csr_matrix(
        (df['normalized_popularity'],
         (df['user_id'], df['article_id']))
    )


def train_implicit(train_df, test_df):

    train = build_sparse(train_df)

    models = [
        AlternatingLeastSquares(),
        BayesianPersonalizedRanking(),
        LogisticMatrixFactorization()
    ]

    results = {}

    for m in models:
        start = time()
        m.fit(train)
        results[m.__class__.__name__] = time() - start

    return results