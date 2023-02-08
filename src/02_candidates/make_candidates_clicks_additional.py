"""
session外の商品をclicksの候補としてco-visitation matrixを用いて取得
https://www.kaggle.com/code/cdeotte/candidate-rerank-model-lb-0-575
"""
import os
import sys

sys.path.append("src")

import cudf as cd
from loguru import logger

from globals import CANDIDATES, CO_VISITATION_MATRIX, INPUT, LOGS, TEST_ORIGINAL, TRAIN_STEP2


def separate_dataframe_based_on_aid_n_unique(df, n_candidates):
    """n_candidatesを超えるsessionとそうでないsessionを分離"""
    df = df.assign(n_unique_aid=df.groupby("session").aid.transform("nunique"))
    df_over_n_candidates = df.query(f"n_unique_aid >= {n_candidates}").drop("n_unique_aid", axis=1)
    df_under_n_candidates = df.query(f"n_unique_aid < {n_candidates}").drop("n_unique_aid", axis=1)

    logger.info(f"over_n_candidates: {len(df_over_n_candidates)}")
    logger.info(f"under_n_candidates: {len(df_under_n_candidates)}")

    return df_over_n_candidates, df_under_n_candidates


def add_candidates(df_all, df_under_n_candidates, co_visitation_matrix_clicks, n_candidates):
    top_clicks = df_all.query("type==0")["aid"].value_counts().index.values[:n_candidates]
    df_top_clicks = cd.DataFrame({"aid": top_clicks.tolist(), "rank": [*range(1, len(top_clicks) + 1, 1)]})

    df_result = cd.DataFrame()
    nparts = 10
    for df_part in df_under_n_candidates.partition_by_hash(columns=["session"], nparts=nparts):
        df_part_session_aid_unique = df_part[["session", "aid"]].drop_duplicates().reset_index(drop=True)
        # sessionに存在したaidをco-visitation matrixと紐づけ、wgtを合計し(異なるaidから同じaid_yが紐づく場合がある)、大きい順に並べ替える
        df_under_n_candidates_added_ = (
            df_part_session_aid_unique.merge(co_visitation_matrix_clicks, how="left", left_on="aid", right_on="aid_x")
            .drop(["aid", "aid_x"], axis="columns")
            .groupby(["session", "aid_y"])
            .size()
            .rename("wgt")
            .reset_index()
            .sort_values(["session", "wgt"], ascending=[True, False])
            .reset_index(drop=True)
            # .drop("wgt", axis="columns")
        )

        df_under_n_candidates_added = (
            # n_candidates個を新たに追加する
            df_under_n_candidates_added_.assign(n=df_under_n_candidates_added_.groupby("session").cumcount())
            .query(f"n<{n_candidates}")
            .drop("n", axis=1)
        )
        df_additional_candidates = df_under_n_candidates_added[["session", "aid_y", "wgt"]].rename(
            columns={"aid_y": "aid"}
        )

        # top_clicksを使ってn_candidatesに満たないsessionに商品を追加
        # co-visitation matrixと結合せず欠落してしまったsessionが存在する
        df_unique_session = df_part[["session"]].drop_duplicates().reset_index(drop=True)
        df_session_missing_aid_ = (
            n_candidates - df_additional_candidates.groupby("session").aid.count().rename("n_missing_aid")
        ).reset_index()
        # 各sessionのn_candidatesに対する不足数をjoinし、1よりも大きいsessionを残す
        df_session_missing_aid = (
            df_unique_session.merge(df_session_missing_aid_, how="left", on="session")
            .fillna(n_candidates)
            .query("n_missing_aid > 0")
        )
        # n_candidatesに満たない分だけ追加
        df_further_candidates = (
            df_session_missing_aid.assign(key=1)
            .merge(df_top_clicks.assign(key=1), how="outer")
            .drop("key", axis="columns")
            .sort_values(["session", "rank"])
            .query("rank <= n_missing_aid")
            .reset_index(drop=True)
            .drop("n_missing_aid", axis="columns")
        )
        # rankの符号を反転してwgtとして残すことで、wgtの並び替え時にadditionalの続きがソートされて得られる。
        df_further_candidates = df_further_candidates.assign(wgt=df_further_candidates["rank"] * (-1)).drop(
            "rank", axis="columns"
        )
        df_result = cd.concat([df_result, df_additional_candidates, df_further_candidates], axis=0)

    return df_result


def save_with_log(df_result, output_dir, output_filename):
    df_result.to_pandas().to_parquet(output_dir / output_filename)
    logger.info(f"head: \n{df_result.head()}")
    logger.info(f"shape: {df_result.shape}")
    logger.info(f"{output_filename} saved")


def main():
    use_chris_cvm = False
    logger.info(f"{use_chris_cvm=}")

    output_dir = CANDIDATES
    output_dir.mkdir(parents=True, exist_ok=True)

    if use_chris_cvm:
        co_visitation_matrix_clicks = cd.concat(
            [
                cd.read_parquet(INPUT / "chris_san_score/clicks_time_weight/top_20_clicks_0.parquet"),
                cd.read_parquet(INPUT / "chris_san_score/clicks_time_weight/top_20_clicks_1.parquet"),
            ],
            ignore_index=True,
        )
    else:
        co_visitation_matrix_clicks = cd.read_parquet(CO_VISITATION_MATRIX / "clicks_train.pqt")

    n_candidates = 120  # session内のaid unique数がこの数に満たない場合候補をco-visitation matrixから追加する
    logger.info(f"{n_candidates=}")

    logger.info("train_step2用のcandidates追加")
    df_train_step2 = cd.read_parquet(TRAIN_STEP2)
    df_train_step2_over_n_candidates, df_train_step2_under_n_candidates = separate_dataframe_based_on_aid_n_unique(
        df_train_step2, n_candidates
    )
    df_train_step2_added_candidates = add_candidates(
        df_train_step2, df_train_step2_under_n_candidates, co_visitation_matrix_clicks, n_candidates
    )
    save_with_log(df_train_step2_added_candidates, output_dir, "train_step2_candidates_clicks_additional.parquet")

    logger.info("test用のcandidates追加")
    df_test = cd.read_parquet(TEST_ORIGINAL)
    df_test_over_n_candidates, df_test_under_n_candidates = separate_dataframe_based_on_aid_n_unique(
        df_test, n_candidates
    )
    co_visitation_matrix_clicks_test = cd.read_parquet(CO_VISITATION_MATRIX / "clicks_test.pqt")
    df_test_added_candidates = add_candidates(
        df_test, df_test_under_n_candidates, co_visitation_matrix_clicks_test, n_candidates
    )
    save_with_log(df_test_added_candidates, output_dir, "test_candidates_clicks_additional.parquet")


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    filename = os.path.basename(__file__).replace(".py", "")
    logger.add(LOGS / f"{filename}.log", rotation="1 MB", level="TRACE")
    main()
