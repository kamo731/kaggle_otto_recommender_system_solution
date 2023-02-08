"""
session内に含まれる商品をcarts/ordersの候補として取得
https://www.kaggle.com/code/cdeotte/candidate-rerank-model-lb-0-575
"""
import itertools
import os
import sys

sys.path.append("src")

import cudf as cd
import numpy as np
from loguru import logger

from globals import (CANDIDATES, CO_VISITATION_MATRIX, INPUT, LOGS,
                     TEST_ORIGINAL, TRAIN_STEP2)
from utils import save_with_log, timer


def add_chirs_session_score(df_raw_session_: cd.DataFrame, df_co_visitation_matrix_buy2buy: cd.DataFrame) -> cd.Series:
    type_weight_multipliers = {0: 1, 1: 6, 2: 3}

    session_length = df_raw_session_.groupby("session").size().sort_index().values

    with timer("session_length"):
        log_time_wgt = [
            (np.logspace(0.5, 1, length, base=2, endpoint=True) - 1).tolist() for length in session_length.tolist()
        ]
    # flatten
    log_time_wgt = list(itertools.chain.from_iterable(log_time_wgt))
    df_raw_session = df_raw_session_.assign(
        type_wgt=df_raw_session_.type.map(type_weight_multipliers),
        log_time_wgt=log_time_wgt,
    )
    df_buys = df_raw_session.query("type in [1, 2]")[["session", "aid"]].drop_duplicates().reset_index(drop=True)

    df_candidates_scored = (
        df_raw_session.assign(score=df_raw_session.type_wgt * df_raw_session.log_time_wgt)
        .groupby(["session", "aid"])
        .score.sum()
        .reset_index()
    )
    # session内でcarts/ordersが発生した商品からbuy2buyで紐づけ、aid_yの商品のscoreに0.1加算する
    df_ = (
        df_buys.merge(df_co_visitation_matrix_buy2buy, how="left", left_on="aid", right_on="aid_x")
        .assign(additional_score=0.1)
        .groupby(["session", "aid_y"])["additional_score"]
        .sum()
        .reset_index()
        .rename(columns={"aid_y": "aid"})[["session", "aid", "additional_score"]]
    )

    df_candidates_scored = df_candidates_scored.merge(df_, how="left", on=["session", "aid"]).fillna(
        {"additional_score": 0}
    )[["session", "aid", "score", "additional_score"]]
    df_candidates_scored["session_score"] = df_candidates_scored["score"] + df_candidates_scored["additional_score"]

    assert (
        df_raw_session[["session", "aid"]].drop_duplicates().shape[0] == df_candidates_scored.shape[0]
    ), "assign先のテーブルとのサイズが異なります"

    return df_candidates_scored.sort_values(["session", "aid"]).reset_index(drop=True)["session_score"]


def make_candidates(df_raw_session: cd.DataFrame, df_co_visitation_matrix_buy2buy: cd.DataFrame):
    df_base = (
        df_raw_session[["session", "aid"]].drop_duplicates().sort_values(["session", "aid"]).reset_index(drop=True)
    )
    # 特徴量を追加
    df_candidates = df_base.assign(
        session_score=add_chirs_session_score(df_raw_session, df_co_visitation_matrix_buy2buy),
    )

    return df_candidates.sort_values(["session", "aid"]).reset_index(drop=True)


def main():
    use_chris_cvm = False
    logger.info(f"{use_chris_cvm=}")
    CANDIDATES.mkdir(parents=True, exist_ok=True)
    df_train_step2: cd.DataFrame = cd.read_parquet(TRAIN_STEP2)

    if use_chris_cvm:
        co_visitation_matrix_buy2buy: cd.DataFrame = cd.read_parquet(
            INPUT / "chris_san_score/buy2buy/top_15_buy2buy_0.parquet"
        )
    else:
        co_visitation_matrix_buy2buy: cd.DataFrame = cd.read_parquet(CO_VISITATION_MATRIX / "buy2buy_train.pqt")

    df_train_step2_candidates = make_candidates(df_train_step2, co_visitation_matrix_buy2buy)

    save_with_log(df_train_step2_candidates, CANDIDATES, "train_step2_candidates_buys_from_session.parquet", logger)

    df_test: cd.DataFrame = cd.read_parquet(TEST_ORIGINAL)
    co_visitation_matrix_buy2buy_test: cd.DataFrame = cd.read_parquet(CO_VISITATION_MATRIX / "buy2buy_test.pqt")
    df_test_candidates = make_candidates(df_test, co_visitation_matrix_buy2buy_test)

    save_with_log(df_test_candidates, CANDIDATES, "test_candidates_buys_from_session.parquet", logger)


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    filename = os.path.basename(__file__).replace(".py", "")
    LOGS.mkdir(parents=True, exist_ok=True)
    logger.add(LOGS / f"{filename}.log", rotation="1 MB", level="TRACE")
    main()
