from __future__ import annotations

import gc
import glob
import os
import time
from contextlib import contextmanager
from pathlib import Path

import cudf as cd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from globals import CO_VISITATION_MATRIX, ID2TYPE, TRAIN_STEP2_LABEL


@contextmanager
def timer(title, logger=None):
    t0 = time.time()
    yield
    if logger is None:
        print("{} - done in {:.0f}s".format(title, time.time() - t0))
    else:
        logger.info(f"{title} - done in {time.time() - t0:.0f}s")


def read_buy2buy(phase: str):
    if phase == "train":
        return cd.read_parquet(CO_VISITATION_MATRIX / "buy2buy_train.pqt")
    elif phase == "test":
        return cd.read_parquet(CO_VISITATION_MATRIX / "buy2buy_test.pqt")


def read_carts_orders(phase: str):
    if phase == "train":
        filenames = glob.glob(str(CO_VISITATION_MATRIX / "carts_orders_train_*.pqt"))
        cvm = cd.concat(
            [cd.read_parquet(filename) for filename in filenames],
            ignore_index=True,
        )
        return cvm
    elif phase == "test":
        filenames = glob.glob(str(CO_VISITATION_MATRIX / "carts_orders_test_*.pqt"))
        cvm = cd.concat(
            [cd.read_parquet(filename) for filename in filenames],
            ignore_index=True,
        )
        return cvm


def save_with_log(
    df_result: cd.DataFrame | pd.DataFrame, output_dir: Path, output_filename: str, logger, save_format="parquet"
):
    if save_format == "parquet":
        df_result.to_parquet(output_dir / output_filename)
    elif save_format == "csv":
        # submission file用
        df_result.to_csv(output_dir / output_filename, index=False)

    logger.info(f"head: \n{df_result.head()}")
    logger.info(f"shape: {df_result.shape}")
    logger.info(f"{output_filename} saved")


def concat_session_and_additional_candidates(
    df_session: cd.DataFrame, df_additional: cd.DataFrame, n_candidates=100, logger=None
) -> cd.DataFrame:

    df_additional_ = df_additional.merge(df_session, how="left", on=["session", "aid"])

    # sessionに既に存在している候補は重複するため不要
    df_additional_ = df_additional_[df_additional_.session_score.isna()]

    df_all_candidates = cd.concat([df_session, df_additional_], ignore_index=True)

    # session毎に候補数を同じにする
    df_all_candidates = df_all_candidates.sort_values(
        ["session", "session_score", "wgt"], ascending=[True, False, False]
    )
    df_all_candidates = (
        df_all_candidates.assign(chris_rank=df_all_candidates.groupby("session").cumcount()).query(
            f"chris_rank<{n_candidates}"
        )
        # .drop("chris_rank", axis=1)
    )
    if logger is not None:
        df_n_aid_describe = df_all_candidates.groupby("session").aid.nunique().describe().round(2)
        logger.info(f"候補数:\n{df_n_aid_describe}")
    return df_all_candidates


def make_sub(df: cd.DataFrame, model_type_id: int) -> pd.DataFrame:
    """
    input
                   session      aid  gt     score
        0         11098528    11830   1  3.338930
        1         11098528  1157882   0 -0.623595
        2         11098528   571762   0 -0.369184
        3         11098528   588923   0 -0.337537
        4         11098528   231487   0 -0.851709
    """
    type_to_make_sub = ID2TYPE[model_type_id]

    df_preds = df.sort_values(["session", "score"], ascending=[True, False])
    df_preds = df_preds.assign(n=df_preds.groupby("session").cumcount()).query("n<40").drop("n", axis=1)

    df_preds = df_preds.to_pandas().groupby("session")["aid"].apply(list).reset_index()

    session_types = []
    labels = []

    for session, preds in zip(df_preds["session"].to_numpy(), df_preds["aid"].to_numpy()):
        l = " ".join(str(p) for p in preds)
        # for session_type in types_to_make_sub:
        labels.append(l)
        session_types.append(f"{session}_{type_to_make_sub}")

    df_submission = pd.DataFrame({"session_type": session_types, "labels": labels})

    return df_submission


def calc_type_recall(df_sub_format: pd.DataFrame, train_labels_: pd.DataFrame, event_type: str, logger):
    """event_typeのrecallを計算"""
    train_labels = train_labels_.copy()
    df_sub = df_sub_format.loc[df_sub_format.session_type.str.contains(event_type)].copy()
    df_sub["session"] = df_sub.session_type.apply(lambda x: int(x.split("_")[0]))
    df_sub.labels = df_sub.labels.apply(lambda x: [int(i) for i in x.split(" ")[:20]])

    train_labels_type = train_labels.loc[train_labels["type"] == event_type]
    train_labels_type = train_labels_type.merge(df_sub, how="left", on=["session"])
    train_labels_type["hits"] = train_labels_type.apply(
        lambda df: len(set(df.ground_truth).intersection(set(df.labels))), axis=1
    )
    train_labels_type["gt_count"] = train_labels_type.ground_truth.str.len().clip(0, 20)

    recall = train_labels_type["hits"].sum() / train_labels_type["gt_count"].sum()
    logger.info(
        f"{event_type} recall = {recall:.5}",
    )
    return recall


def calc_local_cv(df_sub_format: pd.DataFrame, logger):

    score = 0
    weights = {"clicks": 0.10, "carts": 0.30, "orders": 0.60}
    train_labels = pd.read_parquet(TRAIN_STEP2_LABEL)
    for event_type in ["clicks", "carts", "orders"]:
        recall = calc_type_recall(df_sub_format, train_labels, event_type, logger)
        score += weights[event_type] * recall

    logger.info(
        f"Score = {score:.5}",
    )
    return score


def save_feature_importance(feature_importance_df, model_output_dir, plt_feature_num=20):
    save_feature_importance_dir = model_output_dir / "feature_importance/"
    save_feature_importance_dir.mkdir(parents=True, exist_ok=True)
    feature_importance_df.to_csv(save_feature_importance_dir / "feature_importance.csv")

    folds_mean_importance = (
        feature_importance_df.groupby("feature", as_index=False)
        .importance.mean()
        .sort_values(by="importance", ascending=False)
    )
    plt.figure(figsize=(16, 10))
    plt.subplot(1, 2, 1)
    sns.barplot(x="importance", y="feature", data=folds_mean_importance.head(plt_feature_num))
    plt.title(f"Best Importance Features")
    plt.subplot(1, 2, 2)
    sns.barplot(x="importance", y="feature", data=folds_mean_importance.tail(plt_feature_num))
    plt.title(f"Worst Importance Features")
    plt.tight_layout()
    plt.savefig(save_feature_importance_dir / "folds_mean_feature_importance.png")


# -----------------
# features
# -----------------
def merge_features(x_train, x_val, f_name, m_keys, trn_ind, val_ind):
    """
    特徴量をマージする（df_featureによるメモリ圧迫を避けるため関数化）
    """
    df_feature = cd.read_parquet(f_name)
    # concatでよいときは、concatを使う
    if m_keys == ["session", "aid"]:
        # print(f"concat...")
        use_cols = [x for x in df_feature.columns if x not in m_keys]
        # slice
        df_feature_train = df_feature.iloc[trn_ind, :]
        df_feature_val = df_feature.iloc[val_ind, :]
        # sessionが一致しているかチェック（concatミスのチェック）
        assert (x_train["session"].values == df_feature_train["session"].values).sum() == len(x_train)
        assert (x_val["session"].values == df_feature_val["session"].values).sum() == len(x_val)
        # memory management
        del df_feature
        gc.collect()
        # concat
        x_train = cd.concat([x_train, df_feature_train[use_cols]], axis=1)
        x_val = cd.concat([x_val, df_feature_val[use_cols]], axis=1)
    # それ以外はmerge
    else:
        # print(f"merge...")
        x_train = x_train.merge(df_feature, how="left", on=m_keys)
        x_val = x_val.merge(df_feature, how="left", on=m_keys)
    return x_train, x_val


def merge_features_valid_only(x_val, f_name, m_keys, val_ind):
    """
    特徴量をマージする（df_featureによるメモリ圧迫を避けるため関数化）
    """
    df_feature = cd.read_parquet(f_name)
    # concatでよいときは、concatを使う
    if m_keys == ["session", "aid"]:
        use_cols = [x for x in df_feature.columns if x not in m_keys]
        # slice
        df_feature_val = df_feature.iloc[val_ind, :]
        # sessionが一致しているかチェック（concatミスのチェック）
        assert (x_val["session"].values == df_feature_val["session"].values).sum() == len(x_val)
        # memory management
        del df_feature
        gc.collect()
        x_val = cd.concat([x_val, df_feature_val[use_cols]], axis=1)
    # それ以外はmerge
    else:
        # print(f"merge...")
        x_val = x_val.merge(df_feature, how="left", on=m_keys)
    return x_val


def downsample_train(train, trn_ind, negative_rate=20):
    """
    trainデータをダウンサンプリングする
    """
    train_part = train.iloc[trn_ind]
    positives = train_part.loc[train_part["target"] == 1]
    n_negatives = len(positives) * negative_rate
    negatives = train_part.loc[train_part["target"] == 0].sample(n=n_negatives, random_state=42)
    train_part = cd.concat([positives, negatives], axis=0, ignore_index=False)  # indexは後で使うので残しておく
    print("after rows:", round(len(train_part) / len(trn_ind), 3) * 100, "%")  # どれくらい減ったか
    return train_part.index


def shuffle_keep_session(df: cd.DataFrame) -> cd.DataFrame:
    """https://www.kaggle.com/competitions/otto-recommender-system/discussion/377094"""
    df["_noise"] = np.random.randn(len(df))
    df = df.sort_values(["session", "_noise"])
    df = df.drop("_noise", axis=1)
    return df.reset_index(drop=True)
