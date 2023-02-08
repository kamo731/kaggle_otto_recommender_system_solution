"""
word2vecを使ったaidの特徴ベクトルを作成し、svdで10次元に圧縮
https://www.kaggle.com/code/duuuscha/train-submit-word2vec-optimized-hparams
"""
import sys

sys.path.append("src")
import multiprocessing

import joblib
import pandas as pd
import polars as pl
from gensim.models import Word2Vec
from gensim.test.utils import common_texts
from sklearn import decomposition

from globals import FEATURES, TEST_ORIGINAL, TRAIN_ORIGINAL, TRAIN_STEP1, TRAIN_STEP2


def train_and_save_w2v_model(sentences, filename):
    w2vec = Word2Vec(
        sentences=sentences,
        vector_size=50,
        epochs=5,
        sg=1,
        window=3,
        sample=1e-3,
        ns_exponent=1,
        min_count=1,
        workers=12,
    )
    joblib.dump(w2vec, filename)


def main():
    # item2vecの学習
    ## train
    df_train_step1 = pl.read_parquet(TRAIN_STEP1)
    df_train_step2 = pl.read_parquet(TRAIN_STEP2)
    df_train = pl.concat([df_train_step1, df_train_step2])
    sentences_df_train = df_train.groupby("session").agg(pl.col("aid").alias("sentence"))
    sentences_train = sentences_df_train["sentence"].to_list()

    train_and_save_w2v_model(sentences_train, str(FEATURES / "item2vec_train_val.joblib"))
    print("1")

    ## test
    df_train_original = pl.read_parquet(TRAIN_ORIGINAL)
    df_test_original = pl.read_parquet(TEST_ORIGINAL)
    df_test = pl.concat([df_train_original, df_test_original])
    sentences_df_test = df_test.groupby("session").agg(pl.col("aid").alias("sentence"))
    sentences_test = sentences_df_test["sentence"].to_list()

    train_and_save_w2v_model(sentences_test, str(FEATURES / "item2vec_train_original_test.joblib"))
    print("2")

    # make row vector
    ## train
    item2vec_model_train_val = joblib.load(str(FEATURES / "item2vec_train_val.joblib"))

    w2v_dict = {}
    for aid in sorted(item2vec_model_train_val.wv.index_to_key):
        w2v_dict[aid] = item2vec_model_train_val.wv.get_vector(aid)

    item2vec_train_val_df = pd.DataFrame.from_dict(w2v_dict).T
    item2vec_train_val_df = item2vec_train_val_df.add_prefix("vec_").reset_index().rename(columns={"index": "aid"})
    item2vec_train_val_df.to_parquet(FEATURES / "item2vec_raw_train_val.parquet", index=False)

    print(item2vec_train_val_df.head())

    ## test
    item2vec_model_train_original_test = joblib.load(str(FEATURES / "item2vec_train_original_test.joblib"))
    w2v_dict = {}
    for aid in sorted(item2vec_model_train_original_test.wv.index_to_key):
        w2v_dict[aid] = item2vec_model_train_original_test.wv.get_vector(aid)

    item2vec_model_train_original_test_df = pd.DataFrame.from_dict(w2v_dict).T
    item2vec_model_train_original_test_df = (
        item2vec_model_train_original_test_df.add_prefix("vec_").reset_index().rename(columns={"index": "aid"})
    )
    item2vec_model_train_original_test_df.to_parquet(
        FEATURES / "item2vec_raw_train_original_test.parquet", index=False
    )
    print(item2vec_model_train_original_test_df.head())

    # SVDによる次元圧縮
    ## train
    svd = decomposition.TruncatedSVD(n_components=10, random_state=3655)
    item2vec_raw_train_val = pd.read_parquet(FEATURES / "item2vec_raw_train_val.parquet")
    item2vec_svd_train_val = pd.DataFrame(item2vec_raw_train_val["aid"])
    item2vec_svd_train_val = pd.concat(
        [
            item2vec_svd_train_val,
            pd.DataFrame(svd.fit_transform(item2vec_raw_train_val.filter(like="vec_"))).add_prefix("svd_"),
        ],
        axis=1,
    )
    item2vec_svd_train_val.to_parquet(FEATURES / "item2vec_svd_train_val.parquet", index=False)
    print(item2vec_svd_train_val.head())
    ## test
    svd = decomposition.TruncatedSVD(n_components=10, random_state=3655)
    item2vec_raw_train_original_test = pd.read_parquet(FEATURES / "item2vec_raw_train_original_test.parquet")
    item2vec_svd_train_original_test = pd.DataFrame(item2vec_raw_train_original_test["aid"])
    item2vec_svd_train_original_test = pd.concat(
        [
            item2vec_svd_train_original_test,
            pd.DataFrame(svd.fit_transform(item2vec_raw_train_original_test.filter(like="vec_"))).add_prefix("svd_"),
        ],
        axis=1,
    )
    item2vec_svd_train_original_test.to_parquet(FEATURES / "item2vec_svd_train_original_test.parquet", index=False)
    print(item2vec_svd_train_original_test.head())


if __name__ == "__main__":
    main()
