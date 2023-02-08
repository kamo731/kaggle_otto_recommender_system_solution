"""
word2vecを用いたaid粒度の特徴量作成
"""
import sys

sys.path.append("src/03_features")
import argparse

from feature_utils import load_item2vec_svd_train_step1_step2, load_item2vec_svd_train_test, reduce_mem_usage

from globals import FEATURES

parser = argparse.ArgumentParser()
parser.add_argument("phase", choices=["train", "test"])


def main():
    args = parser.parse_args()
    phase = args.phase

    # ## read word2vec
    if phase == "train":
        df_w = load_item2vec_svd_train_step1_step2()
    else:  # phase == "test":
        df_w = load_item2vec_svd_train_test()

    # 列名の変更(candidatesのaidに紐づけるため)
    df_w.columns = [f"{x}_candidates" if x not in ["aid"] else x for x in df_w.columns]

    df_w = reduce_mem_usage(df_w)

    df_w.to_parquet(FEATURES / f"word2vec_features_aid_{phase}.parquet")


if __name__ == "__main__":
    main()
