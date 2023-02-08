"""
学習用co-visitation matrixのbuy2buyを作成
https://www.kaggle.com/code/cdeotte/candidate-rerank-model-lb-0-575
"""
import gc
import os
import sys

sys.path.append("src")

import cudf
from loguru import logger
from tqdm import tqdm

from globals import CO_VISITATION_MATRIX, LOGS, TRAIN_STEP1, TRAIN_STEP2
from utils import timer


def main():
    aids_per_aidx = 40
    logger.info(f"{aids_per_aidx=}")
    output_dir = CO_VISITATION_MATRIX
    output_dir.mkdir(parents=True, exist_ok=True)

    df_train_step1: cudf.DataFrame = cudf.read_parquet(TRAIN_STEP1)
    df_train_step2: cudf.DataFrame = cudf.read_parquet(TRAIN_STEP2)

    df_train = cudf.concat([df_train_step1, df_train_step2], ignore_index=True)
    df_train_orders_carts = df_train.query("type in [1, 2]")
    del df_train, df_train_step1, df_train_step2
    gc.collect()

    # sessionは昇順、tsは降順へ
    df_train_orders_carts = df_train_orders_carts.sort_values(["session", "ts"], ascending=[True, False]).reset_index(
        drop=True
    )
    df_train_orders_carts = (
        # 直近30のlogへ絞る
        df_train_orders_carts.assign(
            n=df_train_orders_carts.groupby("session").cumcount())
        .query("n<30")
        .drop("n", axis=1)
    )

    df_buy2buy = cudf.DataFrame(
        columns=[
            "aid_x",
            "aid_y",
            "wgt",
        ]
    )
    nparts = 20
    for df_part in tqdm(df_train_orders_carts.partition_by_hash(columns=["session"], nparts=nparts)):
        # 同一session(ユーザ)のペアを作成
        df_part_merged = df_part.merge(df_part, on="session")
        # 14日以内 & 異なる商品ペア
        df_part_merged = df_part_merged.loc[
            ((df_part_merged.ts_x - df_part_merged.ts_y).abs() < 14 * 24 * 60 * 60)
            & (df_part_merged.aid_x != df_part_merged.aid_y)
        ]
        df_part_merged = df_part_merged[["session", "aid_x", "aid_y", "type_y"]].drop_duplicates(
            ["session", "aid_x", "aid_y", "type_y"]
        )

        # 各ペアが何回現れるのかを算出し、wgtという名前で格納
        df_pair_counts = df_part_merged.groupby(
            ["aid_x", "aid_y"]).size().reset_index().rename(columns={0: "wgt"})

        # 最終的に出力するテーブルへカウント数を加える
        df_buy2buy = (
            cudf.concat([df_buy2buy, df_pair_counts], axis="index").groupby(
                ["aid_x", "aid_y"], as_index=False).sum()
        )

    df_buy2buy = df_buy2buy.sort_values(["aid_x", "wgt", "aid_y"], ascending=[True, False, True]).reset_index(
        drop=True
    )
    df_buy2buy = (
        # aid_x毎に、よりwgtが大きい組み合わせのtop 15を取得
        df_buy2buy.assign(n=df_buy2buy.groupby("aid_x").aid_y.cumcount())
        .query(f"n<{aids_per_aidx}")
        .drop("n", axis=1)
    )
    df_buy2buy.to_pandas().to_parquet(output_dir / f"buy2buy_train.pqt")
    logger.info(f"file head:\n{df_buy2buy.head()}")
    logger.info(f"shape: {df_buy2buy.shape}")


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    filename = os.path.basename(__file__).replace(".py", "")
    logger.add(LOGS / f"{filename}.log", rotation="1 MB", level="TRACE")
    with timer("all process", logger):
        main()
