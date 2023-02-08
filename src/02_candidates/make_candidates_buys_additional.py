"""
session外の商品をcarts/ordersの候補としてco-visitation matrixを用いて取得
https://www.kaggle.com/code/cdeotte/candidate-rerank-model-lb-0-575
"""
from __future__ import annotations

import gc
import os
import sys

sys.path.append("src")

import cudf as cd
from loguru import logger
from tqdm import tqdm

from globals import CANDIDATES, CO_VISITATION_MATRIX, INPUT, LOGS, TEST_ORIGINAL, TRAIN_STEP2
from utils import read_carts_orders


def separate_dataframe_based_on_aid_n_unique(df, n_candidates) -> tuple[cd.DataFrame, cd.DataFrame]:
    """n_candidatesを超えるsessionとそうでないsessionを分離"""
    df = df.assign(n_unique_aid=df.groupby("session").aid.transform("nunique"))
    df_over_n_candidates = df.query(f"n_unique_aid >= {n_candidates}").drop("n_unique_aid", axis=1)
    df_under_n_candidates = df.query(f"n_unique_aid < {n_candidates}").drop("n_unique_aid", axis=1)

    logger.info(f"over_n_candidates: {len(df_over_n_candidates)}")
    logger.info(f"under_n_candidates: {len(df_under_n_candidates)}")
    return df_over_n_candidates, df_under_n_candidates


def add_candidates(
    df_all: cd.DataFrame,
    df_under_n_candidates: cd.DataFrame,
    co_visitation_matrix_carts_orders: cd.DataFrame,
    co_visitation_matrix_buy2buy: cd.DataFrame,
    n_candidates: int,
):
    """
    n_candidatesに満たない側
    carts_ordersとbuy2buyのco-visitation matrixから候補を取得し、出現回数の大きい順にn個候補を追加する

    """
    # cvmからの候補と重複する可能性があるため、多めに1.5倍取得しておく
    top_orders = df_all.query("type==2")["aid"].value_counts().index.values[: n_candidates * 1.5]
    df_top_orders = cd.DataFrame({"aid": top_orders.tolist(), "rank": [*range(1, len(top_orders) + 1, 1)]})
    df_result = cd.DataFrame()
    nparts = 10
    for df_part in tqdm(df_under_n_candidates.partition_by_hash(columns=["session"], nparts=nparts)):
        df_part_session_aid_unique = df_part[["session", "aid"]].drop_duplicates().reset_index(drop=True)
        df_part_session_aid_unique_buys = (
            df_part.query("type in [1,2]")[["session", "aid"]].drop_duplicates().reset_index(drop=True)
        )

        # session内に現れたaidからcarts_orders co-visitation matrixに紐づくものをjoin
        df_under_n_candidates_carts_orders = df_part_session_aid_unique.merge(
            co_visitation_matrix_carts_orders, how="left", left_on="aid", right_on="aid_x"
        ).reset_index(drop=True)
        # df_under_n_candidates_carts_orders["carts_orders"] = 1
        # df_under_n_candidates_carts_orders["wgt"] = 1

        # session内のcarts, ordersが発生したaidからbuy2buy co-visitation matrixに紐づくものをjoin
        df_under_n_candidates_buy2buy = df_part_session_aid_unique_buys.merge(
            co_visitation_matrix_buy2buy, how="left", left_on="aid", right_on="aid_x"
        ).reset_index(drop=True)
        # df_under_n_candidates_buy2buy["carts_orders"] = 0
        # df_under_n_candidates_buy2buy["wgt"] = 1

        df_under_n_candidates_added_ = (
            cd.concat([df_under_n_candidates_carts_orders, df_under_n_candidates_buy2buy])
            .drop(["aid", "aid_x"], axis="columns")
            .groupby(["session", "aid_y"])
            .size()
            .rename("wgt")
            .reset_index()
            .sort_values(["session", "wgt"], ascending=[True, False])
            .reset_index(drop=True)
        )

        df_under_n_candidates_added = (
            # 30個を新たに追加する
            df_under_n_candidates_added_.assign(n=df_under_n_candidates_added_.groupby("session").cumcount())
            .query(f"n<{n_candidates}")
            .drop("n", axis=1)
        )
        df_additional_candidates = df_under_n_candidates_added[["session", "aid_y", "wgt"]].rename(
            columns={"aid_y": "aid"}
        )

        # top_ordersを使ってn_candidatesに満たないsessionに商品を追加
        # co-visitation matrixと結合せず欠落してしまったsessionが存在する
        df_unique_session = df_part[["session"]].drop_duplicates().reset_index(drop=True)
        df_session_missing_aid_ = (
            n_candidates - df_additional_candidates.groupby("session").aid.count().rename("n_missing_aid")
        ).reset_index()
        df_session_missing_aid = (
            df_unique_session.merge(df_session_missing_aid_, how="left", on="session")
            .fillna(n_candidates)
            .query("n_missing_aid > 0")
        )
        df_further_candidates_ = (
            df_session_missing_aid.assign(key=1)
            .merge(df_top_orders.assign(key=1), how="outer")
            .drop("key", axis="columns")
            .merge(df_additional_candidates, how="left", on=["session", "aid"])
            .merge(df_part_session_aid_unique.assign(exist_session=1), how="left", on=["session", "aid"])
            .fillna({"exist_session": 0})
        )
        # すでにsession, df_additional_candidatesに含まれるものは除外する
        df_further_candidates_1 = (
            df_further_candidates_[df_further_candidates_.wgt.isna()]
            .query("exist_session == 0")
            .drop("wgt", axis="columns")
            .sort_values(["session", "rank"])
            .reset_index(drop=True)
        )
        df_further_candidates = (
            df_further_candidates_1.assign(
                n=df_further_candidates_1.groupby("session").cumcount()
            )  # rankが欠落したので改めてふり直す
            .query("n < n_missing_aid")
            .reset_index(drop=True)[["session", "aid", "n"]]
        )
        # n(rank)の符号を反転してwgtとして残すことで、wgtの並び替え時にadditionalの続きがソートされて得られる。
        df_further_candidates = df_further_candidates.assign(wgt=df_further_candidates["n"] * (-1)).drop(
            "n", axis="columns"
        )

        df_additional_candidates = cd.concat([df_additional_candidates, df_further_candidates], axis=0)

        df_result = cd.concat([df_result, df_additional_candidates], axis=0)

    return df_result


def save_with_log(df_result, output_dir, output_filename):
    df_result.to_pandas().to_parquet(output_dir / output_filename)
    logger.info(f"head: \n{df_result.head()}")
    logger.info(f"shape: {df_result.shape}")
    logger.info(f"{output_filename} saved")


def main():
    use_chris_cvm = False
    output_dir = CANDIDATES
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"{use_chris_cvm=}")

    if use_chris_cvm:
        co_visitation_matrix_carts_orders: cd.DataFrame = cd.concat(
            [
                cd.read_parquet(INPUT / "chris_san_score/carts_orders/top_15_carts_orders_0.parquet"),
                cd.read_parquet(INPUT / "chris_san_score/carts_orders/top_15_carts_orders_1.parquet"),
            ],
            ignore_index=True,
        )
        co_visitation_matrix_buy2buy: cd.DataFrame = cd.read_parquet(
            INPUT / "chris_san_score/buy2buy/top_15_buy2buy_0.parquet"
        )
    else:
        co_visitation_matrix_carts_orders: cd.DataFrame = read_carts_orders(phase="train")
        co_visitation_matrix_buy2buy: cd.DataFrame = cd.read_parquet(CO_VISITATION_MATRIX / "buy2buy_train.pqt")

    n_candidates = 120  # session内のaid unique数がこの数に満たない場合候補をco-visitation matrixから追加する
    logger.info(f"{n_candidates=}")

    logger.info("train_step2用のcandidates追加")
    df_train_step2: cd.DataFrame = cd.read_parquet(TRAIN_STEP2)
    df_train_step2_over_n_candidates, df_train_step2_under_n_candidates = separate_dataframe_based_on_aid_n_unique(
        df_train_step2, n_candidates
    )
    df_train_step2_added_candidates = add_candidates(
        df_train_step2,
        df_train_step2_under_n_candidates,
        co_visitation_matrix_carts_orders,
        co_visitation_matrix_buy2buy,
        n_candidates,
    )
    save_with_log(df_train_step2_added_candidates, output_dir, "train_step2_candidates_buys_additional.parquet")

    del (
        df_train_step2,
        df_train_step2_under_n_candidates,
        co_visitation_matrix_carts_orders,
        co_visitation_matrix_buy2buy,
    )
    gc.collect()

    # --------------
    logger.info("test用のcandidates追加")
    co_visitation_matrix_carts_orders_test: cd.DataFrame = read_carts_orders(phase="test")
    co_visitation_matrix_buy2buy_test: cd.DataFrame = cd.read_parquet(CO_VISITATION_MATRIX / "buy2buy_test.pqt")
    df_test = cd.read_parquet(TEST_ORIGINAL)
    df_test_over_n_candidates, df_test_under_n_candidates = separate_dataframe_based_on_aid_n_unique(
        df_test, n_candidates
    )
    df_test_added_candidates = add_candidates(
        df_test,
        df_test_under_n_candidates,
        co_visitation_matrix_carts_orders_test,
        co_visitation_matrix_buy2buy_test,
        n_candidates,
    )
    save_with_log(df_test_added_candidates, output_dir, "test_candidates_buys_additional.parquet")


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    filename = os.path.basename(__file__).replace(".py", "")
    logger.add(LOGS / f"{filename}.log", rotation="1 MB", level="TRACE")
    main()
