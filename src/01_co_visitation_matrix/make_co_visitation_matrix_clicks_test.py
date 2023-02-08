"""
推論用co-visitation matrixのclicks (Time Weighted)を作成
https://www.kaggle.com/code/cdeotte/candidate-rerank-model-lb-0-575
"""
import gc
import os
import sys

sys.path.append("src")
import cudf
from loguru import logger
from tqdm import tqdm

from globals import CO_VISITATION_MATRIX, LOGS, TEST_ORIGINAL, TRAIN_ORIGINAL
from utils import timer


def main():
    aids_per_aidx = 40
    logger.info(f"{aids_per_aidx=}")

    output_dir = CO_VISITATION_MATRIX
    output_dir.mkdir(parents=True, exist_ok=True)

    df_train_original: cudf.DataFrame = cudf.read_parquet(TRAIN_ORIGINAL)
    df_test: cudf.DataFrame = cudf.read_parquet(TEST_ORIGINAL)

    df_train = cudf.concat([df_train_original, df_test], ignore_index=True)
    del df_train_original, df_test
    gc.collect()

    df_clicks = cudf.DataFrame(
        columns=[
            "aid_x",
            "aid_y",
            "wgt",
        ]
    )
    nparts = 200
    for df_part in tqdm(df_train.partition_by_hash(columns=["session"], nparts=nparts)):

        # sessionは昇順、tsは降順へ
        df_part = df_part.sort_values(["session", "ts"], ascending=[True, False]).reset_index(drop=True)
        # 直近のものを取り出す
        df_part = (
            # 直近30のlogへ絞る
            df_part.assign(n=df_part.groupby("session").cumcount())
            .query("n<30")
            .drop("n", axis=1)
        )

        # 同一session(ユーザ)のペアを作成
        df_part_merged = df_part.merge(df_part, on="session")

        # 1日以内 & 異なる商品ペア
        df_part_merged = df_part_merged.loc[
            ((df_part_merged.ts_x - df_part_merged.ts_y).abs() < 24 * 60 * 60)
            & (df_part_merged.aid_x != df_part_merged.aid_y)
        ]

        # session内でペアとそのタイプをuniqueにする
        df_part_merged = df_part_merged[["session", "aid_x", "aid_y", "ts_x"]].drop_duplicates(
            ["session", "aid_x", "aid_y", "ts_x"]
        )

        # 各ペアのaid_yのtypeごとに重みを付けてscoreを算出
        df_pair_score = (
            df_part_merged.assign(wgt=1 + 3 * (df_part_merged.ts_x - 1659304800) / (1662328791 - 1659304800))
            # .assign(wgt=lambda row: row.wgt.astype("float32"))
            .groupby(["aid_x", "aid_y"])["wgt"]
            .sum()
            .reset_index()
        )

        del df_part_merged
        gc.collect()

        # 最終的に出力するテーブルへscoreをを加える
        df_clicks = (
            cudf.concat([df_clicks, df_pair_score], axis="index").groupby(["aid_x", "aid_y"], as_index=False).sum()
        )

    df_clicks = df_clicks.sort_values(["aid_x", "wgt", "aid_y"], ascending=[True, False, True]).reset_index(drop=True)
    df_clicks = (
        # aid_x毎に、よりwgtが大きい組み合わせのtop 20を取得
        df_clicks.assign(n=df_clicks.groupby("aid_x").aid_y.cumcount())
        .query(f"n<{aids_per_aidx}")
        .drop("n", axis=1)
    )
    df_clicks.to_pandas().to_parquet(output_dir / "clicks_test.pqt")
    logger.info(f"file head:\n{df_clicks.head()}")
    logger.info(f"shape: {df_clicks.shape}")


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    filename = os.path.basename(__file__).replace(".py", "")
    logger.add(LOGS / f"{filename}.log", rotation="1 MB", level="TRACE")
    with timer("all process", logger):
        main()
