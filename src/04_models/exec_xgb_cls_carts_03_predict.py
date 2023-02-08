"""
xgboost分類
carts用モデルの推論
"""
import datetime as dt
import glob
import math
import os
import sys

sys.path.append("src")

from datetime import timedelta, timezone

import cudf as cd
import numpy as np
import pandas as pd
import xgboost as xgb
from exec_xgb_cls_carts_01_train import common_settings
from loguru import logger

from globals import FEATURES, ID2TYPE, LOGS, OUTPUT, SUBMISSION
from utils import make_sub, merge_features_valid_only, timer

np.random.seed(42)
feature_info_test = {
    # TODO: concatのmergeを先にしないとindexがおかしくなる
    FEATURES / "base_session_and_aid_features_test_type_1.parquet": ["session", "aid"],
    FEATURES / "word2vec_features_session_and_aid_last1_test_type_1.parquet": ["session", "aid"],
    FEATURES / "word2vec_features_session_and_aid_last5_test_type_1.parquet": ["session", "aid"],
    FEATURES / "word2vec_features_session_and_aid_all_test_type_1.parquet": ["session", "aid"],
    FEATURES / "cvm_features_all_test_type_1.parquet": ["session", "aid"],
    FEATURES / "cvm_features_last1_test_type_1.parquet": ["session", "aid"],
    FEATURES / "cvm_features_last5_test_type_1.parquet": ["session", "aid"],
    # base
    FEATURES / "base_aid_features_test.parquet": ["aid"],
    FEATURES / "base_session_features_test.parquet": ["session"],
    # word2vec
    FEATURES / "word2vec_features_aid_test.parquet": ["aid"],
    FEATURES / "word2vec_features_session_test.parquet": ["session"],
}


def main():
    version = common_settings["version"]

    model_type_id = common_settings["model_type_id"]
    model_type = ID2TYPE[model_type_id]  # "orders"
    MODEL = OUTPUT / "model" / version

    # predict
    logger.info(f"start xgb prediction version={version}")
    with timer("predict"):

        df_test = cd.read_parquet(
            FEATURES / f"base_preprocess_candidates_test_type_{model_type_id}.parquet",
            columns=["session", "aid"],
        ).reset_index(drop=True)

        model_filenames = sorted(glob.glob(str(MODEL / "*.xgb")))
        n_fold = len(model_filenames)

        df_result = pd.DataFrame()
        for fold, model_filename in enumerate(model_filenames):
            logger.info(f"{fold=}")
            model = xgb.Booster()
            model.load_model(model_filename)
            model.set_param({"predictor": "gpu_predictor"})

            n_split = 10
            SIZE = math.ceil(len(df_test) / n_split)
            df_preds_per_model = pd.DataFrame()  # modelごとの予測値を格納
            for i in range(n_split):
                logger.info(f"split_{i}...")
                start, end = SIZE * i, SIZE * (i + 1)
                df_test_part = df_test.iloc[start:end, :]
                for f_name, m_keys in feature_info_test.items():
                    df_test_part = merge_features_valid_only(df_test_part, f_name, m_keys, df_test_part.index)

                df_pred_part = df_test_part[["session", "aid"]].copy().to_pandas()
                df_test_part = df_test_part.drop(["session", "aid"], axis=1).to_pandas()

                dtest = xgb.DMatrix(data=df_test_part)
                df_pred_part["score"] = model.predict(dtest) / n_fold
                df_preds_per_model = pd.concat([df_preds_per_model, df_pred_part], ignore_index=True, axis="index")

            df_result = pd.concat([df_result, df_preds_per_model], ignore_index=True, axis="index")

        df_result = df_result.groupby(["session", "aid"]).score.sum().reset_index()

    with timer("create submission", logger):
        submission = make_sub(cd.DataFrame.from_pandas(df_result), model_type_id)

        # current_time = dt.datetime.now(tz=timezone(timedelta(hours=+9))).strftime("%y%m%d_%H%M%S")
        sub_dir = SUBMISSION / model_type
        sub_dir.mkdir(parents=True, exist_ok=True)
        submission.to_csv(
            # sub_dir / f"{current_time}_ver_{version}_submission.csv",
            sub_dir / f"ver_{version}_submission.csv",
            index=False,
        )
    logger.info(f"submission file head:\n{submission.head()}")
    logger.info(f"shape: {submission.shape}")
    logger.info("complete xgb prediction")


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    filename = os.path.basename(__file__).replace(".py", "")
    logger.add(LOGS / f"{filename}.log", rotation="1 MB", level="TRACE")
    with timer("all process", logger):
        main()
