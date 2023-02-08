"""
xgboost分類
carts用モデル学習
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
from utils import downsample_train, merge_features, save_feature_importance, timer

np.random.seed(108)


common_settings = {
    "version": "carts_000",
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
    "xgb_params": {
        "objective": "binary:logistic",
        "learning_rate": 0.01,
        "max_depth": 10,
        "subsample": 0.8,
        "colsample_bytree": 0.2,
        "tree_method": "gpu_hist",
        "min_child_weight": 20,
        "random_state": 3655,
    },
    "num_boost_round": 100_000_000,
    "verbose_eval": 100,
    "n_fold": 5,
    "target_col": "target",
    "early_stopping_rounds": 100,
    "model_type_id": 1,  # carts
}


def main():

    MODEL = OUTPUT / "model" / common_settings["version"]
    logger.info(f"start xgb training version={common_settings['version']}")

    if MODEL.exists():
        input_str = input('既にこのversionのmodelディレクトリは存在します。続ける場合は"y"を入力してください。: ')
        if input_str != "y":
            sys.exit()

    MODEL.mkdir(parents=True, exist_ok=True)

    logger.info(f"{common_settings['n_fold']=}")
    logger.info(f"{common_settings['num_boost_round']=}")
    logger.info(f"{common_settings['xgb_params']=}")
    target_col = "target"

    df_train = cd.read_parquet(
        FEATURES / f"base_preprocess_candidates_train_type_{common_settings['model_type_id']}.parquet"
    )

    logger.info(f"{df_train.shape=}")

    feature_importance_df = pd.DataFrame()
    pdf_train = df_train.to_pandas()
    group_k_fold = GroupKFold(n_splits=common_settings["n_fold"])
    for fold, (train_idx, valid_idx) in enumerate(
        group_k_fold.split(pdf_train, pdf_train[target_col], groups=pdf_train["session"])
    ):
        logger.info("-" * 100)
        logger.info(f"Training fold {fold}...")

        train_idx_ds = downsample_train(df_train, train_idx, negative_rate=20)
        valid_idx_ds = downsample_train(df_train, valid_idx, negative_rate=20)

        key_cols = ["session", "aid"]
        df_train_sampled = df_train.iloc[train_idx_ds][[*key_cols, target_col]]
        df_valid_sampled = df_train.iloc[valid_idx_ds][[*key_cols, target_col]]

        for f_name, m_keys in common_settings["features_info"].items():
            df_train_sampled, df_valid_sampled = merge_features(
                df_train_sampled, df_valid_sampled, f_name, m_keys, train_idx_ds, valid_idx_ds
            )

        X_train = df_train_sampled.drop([*key_cols, target_col], axis=1).to_pandas()
        y_train = df_train_sampled[target_col].to_pandas()

        X_valid = df_valid_sampled.drop([*key_cols, target_col], axis=1).to_pandas()
        y_valid = df_valid_sampled[target_col].to_pandas()

        if fold == 0:
            logger.info(f"n features {X_train.shape[1]}")
            logger.info(f"n features {X_train.columns}")

        dtrain = xgb.DMatrix(
            X_train,
            y_train,
        )
        dvalid = xgb.DMatrix(
            X_valid,
            y_valid,
        )
        model = xgb.train(
            common_settings["xgb_params"],
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
