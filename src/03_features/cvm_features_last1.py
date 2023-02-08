"""
co-visitation matrixを用いた特徴量の作成
session内の最後のデータを利用
"""
import gc
import sys

sys.path.append("src/03_features")
import argparse

from feature_utils import (
    load_buy2buy_test,
    load_buy2buy_train,
    load_candidates,
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


def main():
    args = parser.parse_args()
    phase = args.phase
    type_id = int(args.type_id)

    if phase == "train":
        df_last1week = load_train_step2()
    else:  # phase == "test":
        df_last1week = load_test()
    candidates = load_candidates(phase, type_id)

    # candidatesのdfに、session内の最後のaidを紐づける
    df_last = df_last1week.groupby("session")["aid"].last().reset_index()
    df_last.columns = ["session", "aid_last"]
    candidates = candidates.merge(df_last, how="left", on="session")

    # carts_orders
    if phase == "train":
        df_carts_orders = load_carts_orders_train()
    else:  # phase == "test":
        df_carts_orders = load_carts_orders_test()

    # cvmの前処理
    df_carts_orders.columns = ["aid_last", "aid", "cvm_weight_buys_last1"]
    # wgtのmerge(polarsで実行)
    candidates = candidates.merge(df_carts_orders, how="left", on=["aid_last", "aid"])
    # candidates = pl.DataFrame(candidates)
    # candidates = candidates.join(cvm, on=['aid_last', 'aid'])
    # candidates = candidates.to_pandas()
    candidates["cvm_weight_buys_last1"] = candidates["cvm_weight_buys_last1"].fillna(0)
    candidates["cvm_weight_buys_last1"] = candidates["cvm_weight_buys_last1"].astype(int)

    del df_carts_orders
    gc.collect()

    # buy2buy
    if phase == "train":
        df_buy2buy = load_buy2buy_train()
    else:  # phase == "test":
        df_buy2buy = load_buy2buy_test()

    # cvmの前処理
    df_buy2buy.columns = ["aid_last", "aid", "cvm_weight_buy2buy_last1"]

    # wgtのmerge
    candidates = candidates.merge(df_buy2buy, how="left", on=["aid_last", "aid"])
    candidates["cvm_weight_buy2buy_last1"] = candidates["cvm_weight_buy2buy_last1"].fillna(0)
    candidates["cvm_weight_buy2buy_last1"] = candidates["cvm_weight_buy2buy_last1"].astype(int)

    del df_buy2buy
    gc.collect()

    # clicks
    if phase == "train":
        df_clicks = load_clicks_train()
    else:  # phase == "test":
        df_clicks = load_clicks_test()

    # cvmの前処理
    df_clicks.columns = ["aid_last", "aid", "cvm_weight_clicks_last1"]

    # wgtのmerge
    candidates = candidates.merge(df_clicks, how="left", on=["aid_last", "aid"])
    candidates["cvm_weight_clicks_last1"] = candidates["cvm_weight_clicks_last1"].fillna(0)
    candidates["cvm_weight_clicks_last1"] = candidates["cvm_weight_clicks_last1"].astype(int)

    del df_clicks
    gc.collect()

    candidates = candidates.drop("aid_last", axis=1)

    candidates = reduce_mem_usage(candidates)

    if phase == "test":
        # trainは3.5週間、test予測は4.5週間と作成期間が違うので調整
        features_col = [col for col in candidates.columns if col not in ["session", "aid"]]
        candidates[features_col] = candidates[features_col] * (7 / 9)

    assert candidates.isna().sum().sum() == 0, "nullの特徴量が存在します。"
    candidates.to_parquet(FEATURES / f"cvm_features_last1_{phase}_type_{type_id}.parquet")


if __name__ == "__main__":
    main()
