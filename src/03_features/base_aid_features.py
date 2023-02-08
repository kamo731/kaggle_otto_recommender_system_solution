"""
aid粒度の基本的な特徴量を作成
"""
import sys

sys.path.append("../")
import argparse
import datetime

import numpy as np
import pandas as pd
from feature_utils import load_test, load_train_step1_step2, load_train_step2, load_train_test, reduce_mem_usage

from globals import FEATURES

parser = argparse.ArgumentParser()
parser.add_argument("phase", choices=["train", "test"])


def make_type_rate_features(df, df_base):
    """
    type別のカウント、割合
    どのtypeの行動が多いのか（click、carts、orders）
    """
    type_list = [0, 1, 2]
    type_mapper = {0: "clicks", 1: "carts", 2: "orders"}
    for t in type_list:
        df_part = df[df["type"] == t]
        df_part = df_part.groupby("aid")["type"].count().reset_index()
        df_part.columns = ["aid", f"{type_mapper[t]}_count"]
        df_base = df_base.merge(df_part, how="left", on="aid")
    df_base = df_base.fillna(0)
    df_base["total_action_count"] = df_base["clicks_count"] + df_base["carts_count"] + df_base["orders_count"]

    # 全体に占める割合
    df_base["clicks_rate_per_action"] = (df_base["carts_count"] / df_base["total_action_count"]).fillna(0)
    df_base["carts_rate_per_action"] = (df_base["carts_count"] / df_base["total_action_count"]).fillna(0)
    df_base["orders_rate_per_action"] = (df_base["orders_count"] / df_base["total_action_count"]).fillna(0)

    # それぞれの行動同士の割合
    df_base["carts_rate_per_clicks"] = (
        (df_base["carts_count"] / df_base["clicks_count"]).fillna(0).replace(np.inf, 100_000_000)
    )
    df_base["orders_rate_per_clicks"] = (
        (df_base["orders_count"] / df_base["clicks_count"]).fillna(0).replace(np.inf, 100_000_000)
    )
    df_base["orders_rate_per_carts"] = (
        (df_base["orders_count"] / df_base["carts_count"]).fillna(0).replace(np.inf, 100_000_000)
    )

    # 人気度
    df_base["popularity_all_actions"] = df_base["total_action_count"].rank() / len(df_base)
    df_base["popularity_clicks"] = df_base["clicks_count"].rank() / len(df_base)
    df_base["popularity_carts"] = df_base["carts_count"].rank() / len(df_base)
    df_base["popularity_orders"] = df_base["orders_count"].rank() / len(df_base)

    df_base = df_base.drop(["clicks_count", "carts_count", "orders_count", "total_action_count"], axis=1)

    return df_base


def main():
    args = parser.parse_args()
    phase = args.phase
    FEATURES.mkdir(parents=True, exist_ok=True)

    if phase == "train":
        df_all = load_train_step1_step2()
    else:  # phase == "test":
        df_all = load_train_test()

    print(datetime.datetime.fromtimestamp(df_all["ts"].min()))
    print(datetime.datetime.fromtimestamp(df_all["ts"].max()))

    # baseとなるdfの作成
    df_base = pd.DataFrame()
    df_base["aid"] = df_all["aid"].unique()
    df_base = df_base.sort_values("aid").reset_index(drop=True)
    df_base = make_type_rate_features(df_all, df_base)

    # ## week4のみでaid_featuresを計算
    # - 直近の情報のみを使うことで、現在のトレンドを把握する
    if phase == "train":
        df_last1week = load_train_step2()
    else:  # phase == "test":
        df_last1week = load_test()

    # baseとなるdfの作成
    df_base_val = pd.DataFrame()
    df_base_val["aid"] = df_all["aid"].unique()  # 全商品をとってくる
    df_base_val = df_base_val.sort_values("aid").reset_index(drop=True)
    df_base_val = make_type_rate_features(df_last1week, df_base_val)

    # 列名を変更
    df_base_val.columns = [f"{x}_last1week" if x != "aid" else x for x in df_base_val.columns]

    # df_baseにweek4をマージ
    df_base = df_base.merge(df_base_val, how="left", on="aid")

    # ## 変化に関する特徴量を作成
    # 全体から直近1週間への人気度の変化
    df_base["popularity_all_diff_from_all_to_last1week"] = (
        df_base["popularity_all_actions_last1week"] - df_base["popularity_all_actions"]
    )
    df_base["popularity_clicks_diff_from_all_to_last1week"] = (
        df_base["popularity_clicks_last1week"] - df_base["popularity_clicks"]
    )
    df_base["popularity_carts_diff_from_all_to_last1week"] = (
        df_base["popularity_carts_last1week"] - df_base["popularity_carts"]
    )
    df_base["popularity_orders_diff_from_all_to_last1week"] = (
        df_base["popularity_orders_last1week"] - df_base["popularity_orders"]
    )

    # 全体から直近1週間への行動割合の変化
    df_base["clicks_rate_per_action_diff_from_all_to_last1week"] = (
        df_base["clicks_rate_per_action"] - df_base["clicks_rate_per_action_last1week"]
    )
    df_base["carts_rate_per_action_diff_from_all_to_last1week"] = (
        df_base["carts_rate_per_action"] - df_base["carts_rate_per_action_last1week"]
    )
    df_base["orders_rate_per_action_diff_from_all_to_last1week"] = (
        df_base["orders_rate_per_action"] - df_base["orders_rate_per_action_last1week"]
    )
    df_base["carts_rate_per_clicks_diff_from_all_to_last1week"] = (
        df_base["carts_rate_per_clicks"] - df_base["carts_rate_per_clicks_last1week"]
    )
    df_base["orders_rate_per_clicks_diff_from_all_to_last1week"] = (
        df_base["orders_rate_per_clicks"] - df_base["orders_rate_per_clicks_last1week"]
    )
    df_base["orders_rate_per_carts_diff_from_all_to_last1week"] = (
        df_base["orders_rate_per_carts"] - df_base["orders_rate_per_carts_last1week"]
    )

    # null チェック
    print(df_base.isna().sum())
    assert df_base.isna().sum().sum() == 0, "nullの特徴量が存在します。"

    # ## メモリ削減&出力
    df_base = reduce_mem_usage(df_base)
    df_base.to_parquet(FEATURES / f"base_aid_features_{phase}.parquet")


if __name__ == "__main__":
    main()
