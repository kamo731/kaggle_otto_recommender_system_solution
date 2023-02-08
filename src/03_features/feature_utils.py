import sys

sys.path.append("src")
import glob

import numpy as np
import pandas as pd

from globals import (
    CANDIDATES,
    CO_VISITATION_MATRIX,
    FEATURES,
    TEST_ORIGINAL,
    TRAIN_ORIGINAL,
    TRAIN_STEP1,
    TRAIN_STEP2,
    TRAIN_STEP2_LABEL,
)


def reduce_mem_usage(df):
    """iterate through all the columns of a dataframe and modify the data type
    to reduce memory usage.
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print("Memory usage of dataframe is {:.2f} MB".format(start_mem))

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == "int":
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    # df[col] = df[col].astype(np.float16)
                    df[col] = df[col].astype(np.float32)  # parquetはfloat16をサポートしていない
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype("category")

    end_mem = df.memory_usage().sum() / 1024**2
    print("Memory usage after optimization is: {:.2f} MB".format(end_mem))
    print("Decreased by {:.1f}%".format(100 * (start_mem - end_mem) / start_mem))

    return df


def load_carts_orders_train():
    filenames = glob.glob(str(CO_VISITATION_MATRIX / "carts_orders_train_*.pqt"))
    cvm = pd.concat(
        [pd.read_parquet(filename) for filename in filenames],
        ignore_index=True,
    )
    return cvm


def load_carts_orders_test():
    filenames = glob.glob(str(CO_VISITATION_MATRIX / "carts_orders_test_*.pqt"))
    cvm = pd.concat(
        [pd.read_parquet(filename) for filename in filenames],
        ignore_index=True,
    )
    return cvm


def load_buy2buy_train():
    return pd.read_parquet(CO_VISITATION_MATRIX / "buy2buy_train.pqt")


def load_buy2buy_test():
    return pd.read_parquet(CO_VISITATION_MATRIX / "buy2buy_test.pqt")


def load_clicks_train():
    return pd.read_parquet(CO_VISITATION_MATRIX / "clicks_train.pqt")


def load_clicks_test():
    return pd.read_parquet(CO_VISITATION_MATRIX / "clicks_test.pqt")


def load_train_step1_step2():
    """
    train_step1 + train_step2
    """
    return pd.concat([pd.read_parquet(TRAIN_STEP1), pd.read_parquet(TRAIN_STEP2)], ignore_index=True)


def load_train_step2():
    """
    train step2
    """
    return pd.read_parquet(TRAIN_STEP2)


def load_train_test():
    """
    train_original + test_original
    """
    return pd.concat([pd.read_parquet(TRAIN_ORIGINAL), pd.read_parquet(TEST_ORIGINAL)], ignore_index=True)


def load_test():
    """
    test_original
    """
    return pd.read_parquet(TEST_ORIGINAL)


def load_train_step2_label():
    return pd.read_parquet(TRAIN_STEP2_LABEL)


def load_item2vec_svd_train_step1_step2():
    """item2vec_svd_train_val.parquet"""
    df_w = pd.read_parquet(FEATURES / "item2vec_svd_train_val.parquet")
    return df_w


def load_item2vec_svd_train_test():
    """item2vec_svd_train_original_test.parquet"""
    df_w = pd.read_parquet(FEATURES / "item2vec_svd_train_original_test.parquet")
    return df_w


def load_candidates_buys_train(n_candidates=100):
    """carts/ordersの候補(train)"""
    return pd.read_parquet(CANDIDATES / f"candidates_buys_train_n_{n_candidates}.parquet", columns=["session", "aid"])


def load_candidates_buys_test(n_candidates=100):
    """carts/ordersの候補(test)"""
    return pd.read_parquet(CANDIDATES / f"candidates_buys_test_n_{n_candidates}.parquet", columns=["session", "aid"])


def load_candidates_clicks_train(n_candidates=100):
    """clicksの候補(train)"""
    return pd.read_parquet(
        CANDIDATES / f"candidates_clicks_train_n_{n_candidates}.parquet", columns=["session", "aid"]
    )


def load_candidates_clicks_test(n_candidates=100):
    """clicksの候補(test)"""
    return pd.read_parquet(CANDIDATES / f"candidates_clicks_test_n_{n_candidates}.parquet", columns=["session", "aid"])


def load_candidates(phase, type_id):
    assert phase == "train" or phase == "test"
    assert type_id in [0, 1, 2]

    if phase == "train":
        if type_id == 1 or type_id == 2:
            candidates = load_candidates_buys_train()
        else:  # type_id==0
            candidates = load_candidates_clicks_train()
    else:  # "phase=="test"
        if type_id == 1 or type_id == 2:
            candidates = load_candidates_buys_test()
        else:  # type_id==0
            candidates = load_candidates_clicks_test()

    return candidates
