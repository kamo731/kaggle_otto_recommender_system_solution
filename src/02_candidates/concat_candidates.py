"""
session内とsession外の候補を結合してまとめる
"""
import gc
import os
import sys

sys.path.append("src")
import cudf as cd
from loguru import logger

from globals import CANDIDATES, LOGS
from utils import concat_session_and_additional_candidates, save_with_log


def concat_session_additional(
    filename_session_candidates: str, filename_additional_candidates: str, n_candidates: int
) -> cd.DataFrame:
    df_additional_candidates: cd.DataFrame = cd.read_parquet(
        CANDIDATES / filename_additional_candidates, columns=["session", "aid", "wgt"]
    )
    df_session_candidates: cd.DataFrame = cd.read_parquet(
        CANDIDATES / filename_session_candidates, columns=["session", "aid", "session_score"]
    )
    df_candidates = concat_session_and_additional_candidates(
        df_session_candidates, df_additional_candidates, n_candidates=n_candidates
    )
    return df_candidates


def main():
    n_candidates = 100

    # carts/ordersの候補 train
    df_candidates_buys_train = concat_session_additional(
        "train_step2_candidates_buys_from_session.parquet",
        "train_step2_candidates_buys_additional.parquet",
        n_candidates,
    )
    save_with_log(df_candidates_buys_train, CANDIDATES, f"candidates_buys_train_n_{n_candidates}.parquet", logger)

    del df_candidates_buys_train
    gc.collect()

    # carts/ordersの候補 test
    df_candidates_buys_test = concat_session_additional(
        "test_candidates_buys_from_session.parquet",
        "test_candidates_buys_additional.parquet",
        n_candidates,
    )
    save_with_log(df_candidates_buys_test, CANDIDATES, f"candidates_buys_test_n_{n_candidates}.parquet", logger)

    del df_candidates_buys_test
    gc.collect()

    # FIXME: 候補数が足りない
    # clicksの候補 train
    df_candidates_clicks_train = concat_session_additional(
        "train_step2_candidates_clicks_from_session.parquet",
        "train_step2_candidates_clicks_additional.parquet",
        n_candidates,
    )
    save_with_log(df_candidates_clicks_train, CANDIDATES, f"candidates_clicks_train_n_{n_candidates}.parquet", logger)

    del df_candidates_clicks_train
    gc.collect()

    # FIXME: 候補数が足りない
    # clicksの候補 test
    df_candidates_clicks_test = concat_session_additional(
        "test_candidates_clicks_from_session.parquet",
        "test_candidates_clicks_additional.parquet",
        n_candidates,
    )
    save_with_log(df_candidates_clicks_test, CANDIDATES, f"candidates_clicks_test_n_{n_candidates}.parquet", logger)

    del df_candidates_clicks_test
    gc.collect()


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    filename = os.path.basename(__file__).replace(".py", "")
    logger.add(LOGS / f"{filename}.log", rotation="1 MB", level="TRACE")
    main()
