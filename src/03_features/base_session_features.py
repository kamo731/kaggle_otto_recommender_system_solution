"""
session粒度の基本的な特徴量を作成
"""

import argparse
import datetime

import numpy as np
import pandas as pd
from feature_utils import load_test, load_train_step1_step2, load_train_step2, load_train_test, reduce_mem_usage

from globals import FEATURES

parser = argparse.ArgumentParser()
parser.add_argument("phase", choices=["train", "test"])


def make_base_session_features(df, df_base):
    """
    ## ベース特徴量
    - セッション内のクリック数、カート数、オーダー数、合計
    - セッション内での各行動の割合（クリック、カート、オーダー）
    """
    type_list = [0, 1, 2]
    type_mapper = {0: "clicks", 1: "carts", 2: "orders"}
    for t in type_list:
        df_part = df[df["type"] == t]
        df_part = df_part.groupby("session")["type"].count().reset_index()
        df_part.columns = ["session", f"{type_mapper[t]}_count_of_session"]
        df_base = df_base.merge(df_part, how="left", on="session")

    df_base = df_base.fillna(0)
    df_base["total_action_count_of_session"] = (
        df_base["clicks_count_of_session"] + df_base["carts_count_of_session"] + df_base["orders_count_of_session"]
    )

    # 全体に占める割合
    df_base["clicks_rate_per_action_of_session"] = (
        df_base["clicks_count_of_session"] / df_base["total_action_count_of_session"]
    ).fillna(0)
    df_base["carts_rate_per_action_of_session"] = (
        df_base["carts_count_of_session"] / df_base["total_action_count_of_session"]
    ).fillna(0)
    df_base["orders_rate_per_action_of_session"] = (
        df_base["orders_count_of_session"] / df_base["total_action_count_of_session"]
    ).fillna(0)

    # それぞれの行動同士の割合
    df_base["carts_rate_per_clicks_of_session"] = (
        (df_base["carts_count_of_session"] / df_base["clicks_count_of_session"]).fillna(0).replace(np.inf, 100_000_000)
    )
    df_base["orders_rate_per_clicks_of_session"] = (
        (df_base["orders_count_of_session"] / df_base["clicks_count_of_session"])
        .fillna(0)
        .replace(np.inf, 100_000_000)
    )
    df_base["orders_rate_per_carts_of_session"] = (
        (df_base["orders_count_of_session"] / df_base["carts_count_of_session"]).fillna(0).replace(np.inf, 100_000_000)
    )

    # int型への変換
    int_cols = [
        "clicks_count_of_session",
        "carts_count_of_session",
        "orders_count_of_session",
        "total_action_count_of_session",
    ]
    df_base[int_cols] = df_base[int_cols].astype(int)

    return df_base


# ## 最後の行動type
# 最新の行動を取得
def make_last_features(df, df_base):
    df_last = df.groupby("session")["type"].last(1).reset_index()
    df_last.columns = ["session", "last_type_of_session"]
    df_base = df_base.merge(df_last, how="left", on="session")
    return df_base


# ## 直近3回、直近5回のカウント
def make_features_count_action_latest(df, df_base):
    """
    セッション内の直近n回の行動typeをカウントする
    """
    type_list = [0, 1, 2]
    type_mapper = {0: "clicks", 1: "carts", 2: "orders"}
    n_records = [3, 10, 30]  # 直近何個のレコードを使用するか
    for n in n_records:
        df_tail = df.groupby("session").tail(n).reset_index()
        for t in type_list:
            df_part = df_tail[df_tail["type"] == t]
            df_part = df_part.groupby("session")["type"].count().reset_index()
            df_part.columns = ["session", f"{type_mapper[t]}_count_of_session_{n}"]
            df_base = df_base.merge(df_part, how="left", on="session")

        df_base = df_base.fillna(0)
        df_base[f"total_action_count_of_session_{n}"] = (
            df_base[f"clicks_count_of_session_{n}"]
            + df_base[f"carts_count_of_session_{n}"]
            + df_base[f"orders_count_of_session_{n}"]
        )

        # 全体に占める割合
        df_base[f"clicks_rate_per_action_of_session_{n}"] = (
            df_base[f"clicks_count_of_session_{n}"] / df_base[f"total_action_count_of_session_{n}"]
        ).fillna(0)
        df_base[f"carts_rate_per_action_of_session_{n}"] = (
            df_base[f"carts_count_of_session_{n}"] / df_base[f"total_action_count_of_session_{n}"]
        ).fillna(0)
        df_base[f"orders_rate_per_action_of_session_{n}"] = (
            df_base[f"orders_count_of_session_{n}"] / df_base[f"total_action_count_of_session_{n}"]
        ).fillna(0)

        # それぞれの行動同士の割合
        df_base[f"carts_rate_per_clicks_of_session_{n}"] = (
            (df_base[f"carts_count_of_session_{n}"] / df_base[f"clicks_count_of_session_{n}"])
            .fillna(0)
            .replace(np.inf, 100_000_000)
        )
        df_base[f"orders_rate_per_clicks_of_session_{n}"] = (
            (df_base[f"orders_count_of_session_{n}"] / df_base[f"clicks_count_of_session_{n}"])
            .fillna(0)
            .replace(np.inf, 100_000_000)
        )
        df_base[f"orders_rate_per_carts_of_session_{n}"] = (
            (df_base[f"orders_count_of_session_{n}"] / df_base[f"carts_count_of_session_{n}"])
            .fillna(0)
            .replace(np.inf, 100_000_000)
        )

        int_cols = [
            f"clicks_count_of_session_{n}",
            f"carts_count_of_session_{n}",
            f"orders_count_of_session_{n}",
            f"total_action_count_of_session_{n}",
        ]
        df_base[int_cols] = df_base[int_cols].astype(int)

    return df_base


def main():
    args = parser.parse_args()
    phase = args.phase
    FEATURES.mkdir(parents=True, exist_ok=True)

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

    df_base = make_base_session_features(df_last1week, df_base)
    df_base = make_last_features(df_last1week, df_base)
    df_base = make_features_count_action_latest(df_last1week, df_base)

    # ## 変化に関する特徴量

    # 直近10→3への変化
    df_base["clicks_rate_per_action_of_session_diff_3_10"] = (
        df_base["clicks_rate_per_action_of_session_3"] - df_base["clicks_rate_per_action_of_session_10"]
    )
    df_base["carts_rate_per_action_of_session_diff_3_10"] = (
        df_base["carts_rate_per_action_of_session_3"] - df_base["carts_rate_per_action_of_session_10"]
    )
    df_base["orders_rate_per_action_of_session_diff_3_10"] = (
        df_base["orders_rate_per_action_of_session_3"] - df_base["orders_rate_per_action_of_session_10"]
    )
    df_base["carts_rate_per_clicks_of_session_diff_3_10"] = (
        df_base["carts_rate_per_clicks_of_session_3"] - df_base["carts_rate_per_clicks_of_session_10"]
    )
    df_base["orders_rate_per_clicks_of_session_diff_3_10"] = (
        df_base["orders_rate_per_clicks_of_session_3"] - df_base["orders_rate_per_clicks_of_session_10"]
    )
    df_base["orders_rate_per_carts_of_session_diff_3_10"] = (
        df_base["orders_rate_per_carts_of_session_3"] - df_base["orders_rate_per_carts_of_session_10"]
    )

    # 直近30→3への変化
    df_base["clicks_rate_per_action_of_session_diff_3_30"] = (
        df_base["clicks_rate_per_action_of_session_3"] - df_base["clicks_rate_per_action_of_session_30"]
    )
    df_base["carts_rate_per_action_of_session_diff_3_30"] = (
        df_base["carts_rate_per_action_of_session_3"] - df_base["carts_rate_per_action_of_session_30"]
    )
    df_base["orders_rate_per_action_of_session_diff_3_30"] = (
        df_base["orders_rate_per_action_of_session_3"] - df_base["orders_rate_per_action_of_session_30"]
    )
    df_base["carts_rate_per_clicks_of_session_diff_3_30"] = (
        df_base["carts_rate_per_clicks_of_session_3"] - df_base["carts_rate_per_clicks_of_session_30"]
    )
    df_base["orders_rate_per_clicks_of_session_diff_3_30"] = (
        df_base["orders_rate_per_clicks_of_session_3"] - df_base["orders_rate_per_clicks_of_session_30"]
    )
    df_base["orders_rate_per_carts_of_session_diff_3_30"] = (
        df_base["orders_rate_per_carts_of_session_3"] - df_base["orders_rate_per_carts_of_session_30"]
    )

    assert df_base.isna().sum().sum() == 0, "nullの特徴量が存在します。"
    assert df_base.replace([np.inf, -np.inf], np.nan).isna().sum().sum() == 0, "infの特徴量が存在します。"

    df_base = reduce_mem_usage(df_base)

    df_base.to_parquet(FEATURES / f"base_session_features_{phase}.parquet")


if __name__ == "__main__":
    main()
