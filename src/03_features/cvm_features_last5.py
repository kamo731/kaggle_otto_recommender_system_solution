"""
co-visitation matrixを用いた特徴量の作成
session内の最後5つのデータを利用
"""
import gc
import sys

sys.path.append("src/03_features")
import argparse

import polars as pl
from feature_utils import (
    load_buy2buy_test,
    load_buy2buy_train,
    load_carts_orders_test,
    load_carts_orders_train,
    load_clicks_test,
    load_clicks_train,
    load_test,
    load_train_step2,
    reduce_mem_usage,
)

from globals import FEATURES

parser = argparse.ArgumentParser()
parser.add_argument("phase", choices=["train", "test"])
parser.add_argument("type_id", choices=["0", "1", "2"])


def calc_weight_of_session(candidates, df, cvm, n_record, cvm_name):
    """
    sessionのaidにcvmをマージして、weightを計算する。
    計算したweightをcandidatesに紐づける。
    """
    cvm = pl.DataFrame(cvm)
    cvm = cvm.with_column(cvm["aid_x"].cast(pl.Int32))
    cvm = cvm.with_column(cvm["aid_y"].cast(pl.Int32))
    # 列名を変更
    cvm = cvm.rename({"aid_x": "aid"})
    # slice&join
    df_part = df.groupby("session").tail(n_record)
    df_part = df.join(cvm, on="aid")

    # calc wgt
    df_part = df_part.groupby(["session", "aid_y"]).agg(
        [
            pl.mean("wgt").suffix(f"_{cvm_name}_mean_{n_record}"),
            pl.min("wgt").suffix(f"_{cvm_name}_min_{n_record}"),
            pl.max("wgt").suffix(f"_{cvm_name}_max_{n_record}"),
            pl.median("wgt").suffix(f"_{cvm_name}_median_{n_record}"),
            pl.sum("wgt").suffix(f"_{cvm_name}_sum_last{n_record}"),
        ]
    )

    # join
    df_part = df_part.rename({"aid_y": "aid"})
    candidates = candidates.join(df_part, how="left", on=["session", "aid"])
    candidates = candidates.fill_null(0)

    return candidates


def main():
    args = parser.parse_args()
    phase = args.phase
    type_id = int(args.type_id)

    if phase == "train":
        df_last1week = pl.DataFrame(load_train_step2())
    else:  # phase == "test":
        df_last1week = pl.DataFrame(load_test())

    # dtypeの変更(mergeのため)
    df_last1week = df_last1week.with_column(df_last1week["session"].cast(pl.Int32))
    df_last1week = df_last1week.with_column(df_last1week["aid"].cast(pl.Int32))

    candidates = pl.read_parquet(FEATURES / f"base_preprocess_candidates_{phase}_type_{type_id}.parquet")
    # dtypeの変更(mergeのため)
    candidates = candidates.with_column(candidates["session"].cast(pl.Int32))
    candidates = candidates.with_column(candidates["aid"].cast(pl.Int32))

    # carts_orders
    if phase == "train":
        df_carts_orders = load_carts_orders_train()
    else:  # phase == "test":
        df_carts_orders = load_carts_orders_test()

    candidates = calc_weight_of_session(candidates, df_last1week, df_carts_orders, n_record=5, cvm_name="buys")

    del df_carts_orders
    gc.collect()

    # buy2buy
    if phase == "train":
        df_buy2buy = load_buy2buy_train()
    else:  # phase == "test":
        df_buy2buy = load_buy2buy_test()

    candidates = calc_weight_of_session(candidates, df_last1week, df_buy2buy, n_record=5, cvm_name="buy2buy")

    del df_buy2buy
    gc.collect()

    # clicks
    if phase == "train":
        df_clicks = load_clicks_train()
    else:  # phase == "test":
        df_clicks = load_clicks_test()

    candidates = calc_weight_of_session(candidates, df_last1week, df_clicks, n_record=5, cvm_name="clicks")

    del df_clicks
    gc.collect()

    candidates = candidates.drop(["type", "target", "__index_level_0__"])
    candidates = candidates.to_pandas()
    candidates = reduce_mem_usage(candidates)

    if phase == "test":
        # trainは3.5週間、test予測は4.5週間と作成期間が違うので調整
        features_col = [col for col in candidates.columns if col not in ["session", "aid"]]
        candidates[features_col] = candidates[features_col] * (7 / 9)

    assert candidates.isna().sum().sum() == 0, "nullの特徴量が存在します。"
    candidates.to_parquet(FEATURES / f"cvm_features_last5_{phase}_type_{type_id}.parquet")


if __name__ == "__main__":
    main()
