"""
session*aid粒度の基本的な特徴量作成
"""
import argparse
import datetime

from feature_utils import load_candidates, load_test, load_train_step2, reduce_mem_usage

from globals import FEATURES

parser = argparse.ArgumentParser()
parser.add_argument("phase", choices=["train", "test"])
parser.add_argument("type_id", choices=["0", "1", "2"])


# ## そのaidがsession内で登場する回数
def make_features_count_aid_of_session(candidates, df):
    """
    session内で候補となるaidがどれくらい登場したかを計算する
    """
    df_count = df.groupby(["session", "aid", "type"])["ts"].count().reset_index()
    df_count.columns = ["session", "aid", "type", "count"]

    type_labels = {0: "clicks", 1: "carts", 2: "orders"}
    for i in range(3):
        print("type:", i)
        df_count_part = df_count[df_count["type"] == i]
        df_count_part = df_count_part.drop("type", axis=1)
        candidates = candidates.merge(df_count_part, how="left", on=["session", "aid"])
        candidates[f"{type_labels[i]}_count_all"] = candidates["count"].fillna(0).astype("int16")
        candidates = candidates.drop("count", axis=1)

    # countのdiff。どの行動がどれくらい多いか
    candidates["carts_count_diff_from_clicks_all"] = candidates["carts_count_all"] - candidates["clicks_count_all"]
    candidates["orders_count_diff_from_clicks_all"] = candidates["orders_count_all"] - candidates["clicks_count_all"]
    candidates["orders_count_diff_from_carts_all"] = candidates["orders_count_all"] - candidates["carts_count_all"]

    return candidates


def make_features_count_aid_of_session_latest(candidates, df):
    """
    session内で候補となるaidがどれくらい登場したかを計算する。(直近n行ごとに計算)
    """
    type_labels = {0: "clicks", 1: "carts", 2: "orders"}
    n_records = [3, 10, 30]  # 直近何個のレコードを使用するか

    # 直近n個でループ
    for n in n_records:
        print("n_records:", n, "=" * 30)
        df_tail = df.groupby("session").tail(n).reset_index(drop=True)

        df_count = df_tail.groupby(["session", "aid", "type"])["ts"].count().reset_index()
        df_count.columns = ["session", "aid", "type", "count"]

        # typeごとにループ
        for i in range(3):
            print("type:", i)
            df_count_part = df_count[df_count["type"] == i]
            df_count_part = df_count_part.drop("type", axis=1)
            candidates = candidates.merge(df_count_part, how="left", on=["session", "aid"])
            candidates[f"{type_labels[i]}_count_last{n}rows"] = candidates["count"].fillna(0).astype("int16")
            candidates = candidates.drop("count", axis=1)

        # countのdiff。どの行動がどれくらい多いか
        candidates[f"carts_count_diff_from_clicks_last{n}rows"] = (
            candidates[f"carts_count_last{n}rows"] - candidates[f"clicks_count_last{n}rows"]
        )
        candidates[f"orders_count_diff_from_clicks_last{n}rows"] = (
            candidates[f"orders_count_last{n}rows"] - candidates[f"clicks_count_last{n}rows"]
        )
        candidates[f"orders_count_diff_from_carts_last{n}rows"] = (
            candidates[f"orders_count_last{n}rows"] - candidates[f"carts_count_last{n}rows"]
        )

    return candidates


def main():
    args = parser.parse_args()
    phase = args.phase
    type_id = int(args.type_id)
    FEATURES.mkdir(parents=True, exist_ok=True)

    if phase == "train":
        df_last1week = load_train_step2()
    else:  # phase == "test":
        df_last1week = load_test()

    print(datetime.datetime.fromtimestamp(df_last1week["ts"].min()))
    print(datetime.datetime.fromtimestamp(df_last1week["ts"].max()))

    candidates = load_candidates(phase, type_id)

    # ## candidatesの順位
    candidates["candidates_rank"] = candidates.groupby("session").cumcount() + 1

    candidates = make_features_count_aid_of_session(candidates, df_last1week)
    candidates = make_features_count_aid_of_session_latest(candidates, df_last1week)

    # ## 変化に関する特徴量
    # - 直近30行→10行でどう変化したか など
    # clicks
    candidates["clicks_count_diff_last3rows_last10rows"] = (
        candidates["clicks_count_last3rows"] - candidates["clicks_count_last10rows"]
    )
    candidates["clicks_count_diff_last10rows_last30rows"] = (
        candidates["clicks_count_last10rows"] - candidates["clicks_count_last30rows"]
    )
    candidates["clicks_count_diff_last30rows_all"] = (
        candidates["clicks_count_last30rows"] - candidates["clicks_count_all"]
    )

    # carts
    candidates["carts_count_diff_last3rows_last10rows"] = (
        candidates["carts_count_last3rows"] - candidates["carts_count_last10rows"]
    )
    candidates["carts_count_diff_last10rows_last30rows"] = (
        candidates["carts_count_last10rows"] - candidates["carts_count_last30rows"]
    )
    candidates["carts_count_diff_last30rows_all"] = (
        candidates["carts_count_last30rows"] - candidates["carts_count_all"]
    )

    # orders
    candidates["orders_count_diff_last3rows_last10rows"] = (
        candidates["orders_count_last3rows"] - candidates["orders_count_last10rows"]
    )
    candidates["orders_count_diff_last10rows_last30rows"] = (
        candidates["orders_count_last10rows"] - candidates["orders_count_last30rows"]
    )
    candidates["orders_count_diff_last30rows_all"] = (
        candidates["orders_count_last30rows"] - candidates["orders_count_all"]
    )

    assert candidates.isna().sum().sum() == 0, "nullの特徴量が存在します。"
    # ## メモリ削減＆出力
    candidates = reduce_mem_usage(candidates)

    candidates.to_parquet(FEATURES / f"base_session_and_aid_features_{phase}_type_{type_id}.parquet")


if __name__ == "__main__":
    main()
