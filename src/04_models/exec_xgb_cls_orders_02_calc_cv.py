"""
xgboost分類
orders用モデルのcv(recall)を計算
"""
import math
import os
import sys

sys.path.append("src")

import cudf as cd
import pandas as pd
import xgboost as xgb
from exec_xgb_cls_orders_01_train import common_settings
from loguru import logger
from sklearn.model_selection import GroupKFold

from globals import FEATURES, ID2TYPE, LOGS, OOF, OUTPUT, TRAIN_STEP2_LABEL
from utils import calc_type_recall, make_sub, merge_features_valid_only, timer


def main():
    OOF.mkdir(parents=True, exist_ok=True)
    MODEL = OUTPUT / "model" / common_settings["version"]
    model_type = ID2TYPE[common_settings["model_type_id"]]

    logger.info(f"start xgb cv calc version={common_settings['version']}")

    df_train = cd.read_parquet(
        FEATURES / f"base_preprocess_candidates_train_type_{common_settings['model_type_id']}.parquet"
    )

    pdf_train = df_train.to_pandas()
    group_k_fold = GroupKFold(n_splits=common_settings["n_fold"])
    # oof_pred = np.zeros(len(df_train))
    df_oof = pd.DataFrame()
    for fold, (_, valid_idx) in enumerate(
        group_k_fold.split(pdf_train, pdf_train[common_settings["target_col"]], groups=pdf_train["session"])
    ):
        logger.info(f"fold {fold}...")
        model = xgb.Booster()
        model.load_model(MODEL / f"XGB_fold{fold}.xgb")
        model.set_param({"predictor": "gpu_predictor"})

        X_valid = df_train.iloc[valid_idx][["session", "aid"]]

        n_split = 3
        SIZE = math.ceil(len(X_valid) / n_split)
        df_preds_per_model = pd.DataFrame()  # modelごとの予測値を格納
        for i in range(n_split):
            print(f"split_{i}...")
            start, end = SIZE * i, SIZE * (i + 1)
            x_val_part = X_valid.iloc[start:end, :]

            for f_name, m_keys in common_settings["features_info"].items():
                x_val_part = merge_features_valid_only(x_val_part, f_name, m_keys, x_val_part.index)

            df_pred_part = x_val_part[["session", "aid"]].copy().to_pandas()
            x_val_part = x_val_part.drop(["session", "aid"], axis=1).to_pandas()

            dvalid = xgb.DMatrix(data=x_val_part)
            df_pred_part["score"] = model.predict(dvalid)
            df_preds_per_model = pd.concat([df_preds_per_model, df_pred_part], ignore_index=True, axis="index")
            # preds_per_model.append(val_pred)

        df_oof = pd.concat([df_oof, df_preds_per_model], ignore_index=True, axis="index")
        # preds_per_model = np.concatenate(preds_per_model)  # 結合
        # oof_pred[valid_idx] = preds_per_model

    # df_oof = df_oof.sort_values(["session", "aid"])
    # df_oof = df_train[["session", "aid", common_settings["target_col"]]].assign(score=oof_pred)
    df_sub_format_oof = make_sub(cd.DataFrame.from_pandas(df_oof), common_settings["model_type_id"])

    train_labels = pd.read_parquet(TRAIN_STEP2_LABEL)
    recall = calc_type_recall(df_sub_format_oof, train_labels, model_type, logger)
    df_sub_format_oof.to_csv(OOF / f"{common_settings['version']}_cv{round(recall,5)}.csv", index=False)


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    filename = os.path.basename(__file__).replace(".py", "")
    logger.add(LOGS / f"{filename}.log", rotation="1 MB", level="TRACE")
    with timer("all process", logger):
        main()
