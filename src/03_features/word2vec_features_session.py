"""
word2vecを用いたsession粒度の特徴量作成
"""
import sys

sys.path.append("src/03_features")
import argparse
import datetime

import pandas as pd
from feature_utils import (
    load_item2vec_svd_train_step1_step2,
    load_item2vec_svd_train_test,
    load_test,
    load_train_step2,
    reduce_mem_usage,
)

from globals import FEATURES

parser = argparse.ArgumentParser()
parser.add_argument("phase", choices=["train", "test"])


def main():
    args = parser.parse_args()
    phase = args.phase

    if phase == "train":
        df_last1week = load_train_step2()
    else:  # phase == "test":
        df_last1week = load_test()

    print(datetime.datetime.fromtimestamp(df_last1week["ts"].min()))
    print(datetime.datetime.fromtimestamp(df_last1week["ts"].max()))

    # baseとなるdfの作成
    df_base = pd.DataFrame()
    df_base["session"] = df_last1week["session"].unique()
    df_base = df_base.sort_values("session")

    # ## read word2vec
    if phase == "train":
        df_w = load_item2vec_svd_train_step1_step2()
    else:  # phase == "test":
        df_w = load_item2vec_svd_train_test()

    # ## セッション内のaidに対して、word2vecを紐づけて特徴量化
    df_last1week = df_last1week.merge(df_w, how="left", on="aid")
    svd_cols = [x for x in df_last1week.columns if "svd" in x]

    # 最新
    df_last = df_last1week.groupby("session")[svd_cols].last().reset_index()
    df_last.columns = ["session"] + [f"{x}_last1" for x in svd_cols]
    df_base = df_base.merge(df_last, how="left", on="session")

    # df_last5
    df_last5 = df_last1week.groupby("session").tail(5)
    df_last5 = df_last5.groupby("session")[svd_cols].mean().reset_index()
    df_last5.columns = ["session"] + [f"{x}_last5" for x in svd_cols]
    df_base = df_base.merge(df_last5, how="left", on="session")

    # df_all
    df_all = df_last1week.groupby("session")[svd_cols].mean().reset_index()
    df_all.columns = ["session"] + [f"{x}_all" for x in svd_cols]
    df_base = df_base.merge(df_all, how="left", on="session")
    # -

    df_base = reduce_mem_usage(df_base)

    assert df_base.isna().sum().sum() == 0, "nullの特徴量が存在します。"
    df_base.to_parquet(FEATURES / f"word2vec_features_session_{phase}.parquet")


if __name__ == "__main__":
    main()
