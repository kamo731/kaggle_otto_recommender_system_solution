"""
word2vecを用いたaid*session粒度の特徴量作成
session内の最後のデータを利用
"""
import sys

sys.path.append("src/03_features")
import argparse

import pandas as pd
from feature_utils import load_candidates, reduce_mem_usage

from globals import FEATURES

parser = argparse.ArgumentParser()
parser.add_argument("phase", choices=["train", "test"])
parser.add_argument("type_id", choices=["0", "1", "2"])


def main():
    args = parser.parse_args()
    phase = args.phase
    type_id = int(args.type_id)

    ## read candidates
    candidates = load_candidates(phase, type_id)

    ## read features
    df_aid = pd.read_parquet(FEATURES / f"word2vec_features_aid_{phase}.parquet")
    df_session = pd.read_parquet(FEATURES / f"word2vec_features_session_{phase}.parquet")

    ## merge features
    candidates = candidates.merge(df_aid, how="left", on="aid")

    ## last1 diff
    svd_cols_candidates = [f"svd_{x}_candidates" for x in range(10)]

    # merge
    svd_cols_last1 = [f"svd_{x}_last1" for x in range(10)]
    use_cols = ["session"] + svd_cols_last1
    candidates = candidates.merge(df_session[use_cols], how="left", on="session")

    # calc diff
    diff_cols_candidates_last1 = [f"svd_{x}_diff_candidates_last1" for x in range(10)]
    candidates[diff_cols_candidates_last1] = candidates[svd_cols_candidates].values - candidates[svd_cols_last1].values
    candidates[f"svd_total_diff_candidates_last1"] = candidates[diff_cols_candidates_last1].abs().sum(axis=1)

    # drop cols
    candidates = candidates.drop(svd_cols_candidates + svd_cols_last1, axis=1)

    candidates = reduce_mem_usage(candidates)

    assert candidates.isna().sum().sum() == 0, "nullの特徴量が存在します。"
    candidates.to_parquet(FEATURES / f"word2vec_features_session_and_aid_last1_{phase}_type_{type_id}.parquet")


if __name__ == "__main__":
    main()
