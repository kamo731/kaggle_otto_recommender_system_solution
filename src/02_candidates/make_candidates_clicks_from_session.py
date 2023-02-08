"""
session内に含まれる商品をclicksの候補として取得
https://www.kaggle.com/code/cdeotte/candidate-rerank-model-lb-0-575
"""
import itertools
import os
import sys

sys.path.append("src")

import cudf as cd
import numpy as np
from loguru import logger

from globals import CANDIDATES, LOGS, TEST_ORIGINAL, TRAIN_STEP2
from utils import save_with_log, timer


def add_chirs_session_score(df_raw_session_: cd.DataFrame) -> cd.Series:
    type_weight_multipliers = {0: 1, 1: 6, 2: 3}
    session_length = df_raw_session_.groupby("session").size().sort_index().values
    with timer("session_length"):
        log_time_wgt = [
            (np.logspace(0.1, 1, length, base=2, endpoint=True) - 1).tolist() for length in session_length.tolist()
        ]
    # flatten
    log_time_wgt = list(itertools.chain.from_iterable(log_time_wgt))
    df_raw_session = df_raw_session_.assign(
        type_wgt=df_raw_session_.type.map(type_weight_multipliers), log_time_wgt=log_time_wgt
    )

    df_candidates_scored = (
        df_raw_session.assign(session_score=df_raw_session.type_wgt * df_raw_session.log_time_wgt)
        .groupby(["session", "aid"])
        .session_score.sum()
        .reset_index()
    )

    return df_candidates_scored.sort_values(["session", "aid"]).reset_index(drop=True)["session_score"]


def add_last_event_ts(df_raw_session: cd.DataFrame) -> cd.Series:
    df_last_ts = df_raw_session.groupby(["session", "aid"])["ts"].max().rename("last_event_ts").reset_index()

    assert df_last_ts.shape[0] == df_raw_session[["session", "aid"]].drop_duplicates().shape[0]
    return df_last_ts.sort_values(["session", "aid"]).reset_index(drop=True)["last_event_ts"]


def add_session_length(df_raw_session: cd.DataFrame, df_base: cd.DataFrame) -> cd.Series:
    df_raw_session["session_length"] = df_raw_session.groupby("session")["ts"].transform("count")
    df_session_length = df_raw_session.groupby("session").size().rename("session_length").reset_index()

    return (
        df_base.merge(df_session_length, how="left", on="session")
        .sort_values(["session", "aid"])
        .reset_index(drop=True)["session_length"]
    )


def add_exists_type_event(df_raw_session: cd.DataFrame, df_base: cd.DataFrame, type_id: int) -> cd.Series:
    """session, aidについて、type_idのeventが発生している場合1となるフラグ"""
    df_session_aid = (
        df_raw_session.query(f"type=={type_id}")[["session", "aid"]].drop_duplicates().reset_index(drop=True)
    )
    df_session_aid = df_session_aid.assign(exists_type_id=1)
    df_result = df_base.merge(df_session_aid, how="left", on=["session", "aid"]).fillna({"exists_type_id": 0})
    return df_result.sort_values(["session", "aid"]).reset_index(drop=True)["exists_type_id"]


def make_clicks_feature(df_raw_session: cd.DataFrame) -> cd.DataFrame:
    df_base = (
        df_raw_session[["session", "aid"]].drop_duplicates().sort_values(["session", "aid"]).reset_index(drop=True)
    )

    # sessionベースの特徴量を追加
    df_candidates = df_base.assign(
        session_score=add_chirs_session_score(df_raw_session),
        session_length=add_session_length(df_raw_session, df_base),
        last_event_ts=add_last_event_ts(df_raw_session),
        exists_clicks=add_exists_type_event(df_raw_session, df_base, 0),
        exists_carts=add_exists_type_event(df_raw_session, df_base, 1),
        exists_orders=add_exists_type_event(df_raw_session, df_base, 2),
    )
    # session*aid毎に最後のeventの時刻(last_event_ts)を考え、新しい順に50個で切る
    max_n_candidates = 50
    df_candidates = (
        df_candidates.sort_values(["session", "last_event_ts"], ascending=[True, False])
        .assign(n=df_candidates.groupby("session").cumcount())
        .query(f"n<{max_n_candidates}")
        .drop("n", axis=1)
    )
    return df_candidates


def main():
    CANDIDATES.mkdir(parents=True, exist_ok=True)
    df_train_step2: cd.DataFrame = cd.read_parquet(TRAIN_STEP2)

    df_train_step2_candidates = make_clicks_feature(df_train_step2)

    save_with_log(df_train_step2_candidates, CANDIDATES, "train_step2_candidates_clicks_from_session.parquet", logger)

    df_test: cd.DataFrame = cd.read_parquet(TEST_ORIGINAL)
    df_test_candidates = make_clicks_feature(df_test)

    save_with_log(df_test_candidates, CANDIDATES, "test_candidates_clicks_from_session.parquet", logger)


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    filename = os.path.basename(__file__).replace(".py", "")
    LOGS.mkdir(parents=True, exist_ok=True)
    logger.add(LOGS / f"{filename}.log", rotation="1 MB", level="TRACE")
    main()
