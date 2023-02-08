"""
xgboostランキング
orders用モデル学習
"""
import gc
import os
import sys

sys.path.append("src")
import cudf as cd
import numpy as np
import pandas as pd
import xgboost as xgb
from loguru import logger
from sklearn.model_selection import GroupKFold

from globals import FEATURES, LOGS, OUTPUT
from utils import merge_features, save_feature_importance, timer

np.random.seed(42)


# data_path
common_settings = {
    "version": "orders_100",
    # pathとmergeのkey
    "features_info": {
        # TODO: concatのmergeを先にしないとindexがおかしくなる
        FEATURES / "base_session_and_aid_features_train_type_2.parquet": ["session", "aid"],
        FEATURES / "word2vec_features_session_and_aid_last1_train_type_2.parquet": ["session", "aid"],
        FEATURES / "word2vec_features_session_and_aid_last5_train_type_2.parquet": ["session", "aid"],
        FEATURES / "word2vec_features_session_and_aid_all_train_type_2.parquet": ["session", "aid"],
        FEATURES / "cvm_features_all_train_type_2.parquet": ["session", "aid"],
        FEATURES / "cvm_features_last1_train_type_2.parquet": ["session", "aid"],
        FEATURES / "cvm_features_last5_train_type_2.parquet": ["session", "aid"],
        # base
        FEATURES / "base_aid_features_train.parquet": ["aid"],
        FEATURES / "base_session_features_train.parquet": ["session"],
        # word2vec
        FEATURES / "word2vec_features_aid_train.parquet": ["aid"],
        FEATURES / "word2vec_features_session_train.parquet": ["session"],
        # INPUT / "aids-concat-feature/train.parquet": ["session"],
    },
    "xgb_parms": {
        # "objective": "rank:ndcg",
        "objective": "rank:pairwise",
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.2,
        "tree_method": "gpu_hist",
        "min_child_weight": 20,
        "random_state": 3655,
    },
    "num_boost_round": 1000,
    "verbose_eval": 50,
    "n_fold": 5,
    "early_stopping_rounds": 50,
    "target_col": "target",
    "model_type_id": 2,  # orders
}


def shuffle_keep_session(df: cd.DataFrame) -> cd.DataFrame:
    """https://www.kaggle.com/competitions/otto-recommender-system/discussion/377094"""
    df["_noise"] = np.random.randn(len(df))
    df = df.sort_values(["session", "_noise"])
    df = df.drop("_noise", axis=1)
    return df.reset_index(drop=True)


def drop_negative_only_session(df_: cd.DataFrame) -> cd.DataFrame.index:
    """
    target=1が存在するsessionだけ残す
    全体の7%くらい
    """
    df = df_.copy().reset_index().rename(columns={"index": "original_index"})
    df_use_sessions = cd.DataFrame(df.query("target == 1").session.unique(), columns=["session"])
    df = df.merge(df_use_sessions, how="inner", on="session")
    # use_cols = [x for x in x_val.columns if x not in ["target"]]
    df = df.set_index("original_index")
    logger.info(f"session w/ positive target: {(len(df)/len(df_)) * 100: .3} %")
    return df.index  # x_val[use_cols], x_val["target"]


def get_session_lengths(df):
    return df.groupby("session").size().sort_index().values


def main():
    MODEL = OUTPUT / "model" / common_settings["version"]
    model_type_id = common_settings["model_type_id"]
    logger.info(f"start xgb training version={common_settings['version']}")

    if MODEL.exists():
        input_str = input('既にこのversionのmodelディレクトリは存在します。続ける場合は"y"を入力してください。: ')
        if input_str != "y":
            sys.exit()

    MODEL.mkdir(parents=True, exist_ok=True)
    logger.info(f"{common_settings=}")

    target_col = "target"

    df_train = cd.read_parquet(FEATURES / f"base_preprocess_candidates_train_type_{model_type_id}.parquet")

    # train parameters
    logger.info(f"{df_train.shape=}")

    feature_importance_df = pd.DataFrame()
    pdf_train = df_train.to_pandas()
    group_k_fold = GroupKFold(n_splits=common_settings["n_fold"])
    for fold, (train_idx, valid_idx) in enumerate(
        group_k_fold.split(pdf_train, pdf_train[target_col], groups=pdf_train["session"])
    ):
        logger.info("-" * 100)
        logger.info(f"Training fold {fold}...")

        train_idx_ds = drop_negative_only_session(df_train.iloc[train_idx])
        valid_idx_ds = drop_negative_only_session(df_train.iloc[valid_idx])

        key_cols = ["session", "aid"]
        df_train_sampled = df_train.iloc[train_idx_ds][[*key_cols, target_col]]
        df_valid_sampled = df_train.iloc[valid_idx_ds][[*key_cols, target_col]]

        for f_name, m_keys in common_settings["features_info"].items():
            df_train_sampled, df_valid_sampled = merge_features(
                df_train_sampled, df_valid_sampled, f_name, m_keys, train_idx_ds, valid_idx_ds
            )

        df_train_sampled = shuffle_keep_session(df_train_sampled)
        X_train = df_train_sampled
        y_train = df_train_sampled[target_col].to_pandas()

        df_valid_sampled = df_valid_sampled.sort_values("session").reset_index(drop=True)
        X_valid = df_valid_sampled
        y_valid = df_valid_sampled[target_col].to_pandas()

        session_lengths_train = get_session_lengths(X_train)
        X_train = X_train.drop(["session", "aid", target_col], axis=1).to_pandas()

        session_lengths_valid = get_session_lengths(X_valid)
        X_valid = X_valid.drop(["session", "aid", target_col], axis=1).to_pandas()

        if fold == 0:
            logger.info(f"n features {X_train.shape[1]}")
            logger.info(f"n features {X_train.columns}")

        dtrain = xgb.DMatrix(
            X_train,
            y_train,
            group=session_lengths_train,
        )
        dvalid = xgb.DMatrix(
            X_valid,
            y_valid,
            group=session_lengths_valid,
        )
        model = xgb.train(
            common_settings["xgb_parms"],
            dtrain=dtrain,
            evals=[(dtrain, "train"), (dvalid, "valid")],
            num_boost_round=common_settings["num_boost_round"],
            early_stopping_rounds=common_settings["early_stopping_rounds"],
            verbose_eval=common_settings["verbose_eval"],
        )

        fold_importance_df = pd.DataFrame(
            {
                "feature": model.get_score(importance_type="gain").keys(),
                "importance": model.get_score(importance_type="gain").values(),
                "fold": fold,
            }
        )
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

        train_score = model.eval(dtrain, iteration=model.best_iteration, name="train")
        valid_score = model.eval(dvalid, iteration=model.best_iteration, name="valid")
        logger.info(f"{train_score[:27]}")
        logger.info(f"{valid_score[:27]}")

        model.save_model(MODEL / f"XGB_fold{fold}.xgb")

    save_feature_importance(feature_importance_df, MODEL)

    logger.info("complete xgb training", logger)


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    filename = os.path.basename(__file__).replace(".py", "")
    logger.add(LOGS / f"{filename}.log", rotation="1 MB", level="TRACE")
    with timer("all process", logger):
        main()
