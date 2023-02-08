"""
candidatesに対して学習用にラベルを付与する。
trainではこのデータをベースとして特徴量をjoinしていく。
"""
import argparse

from feature_utils import load_candidates, load_train_step2_label

from globals import FEATURES

parser = argparse.ArgumentParser()
parser.add_argument("phase", choices=["train", "test"])
parser.add_argument("type_id", choices=["0", "1", "2"])


def preprocess_labels(labels):
    labels = labels.explode("ground_truth")
    type_labels = {"clicks": 0, "carts": 1, "orders": 2}
    labels["type"] = labels["type"].map(type_labels).astype("int8")

    labels.columns = ["session", "type", "aid"]
    labels["target"] = 1
    return labels


def main():
    args = parser.parse_args()
    phase = args.phase
    type_id = int(args.type_id)
    FEATURES.mkdir(parents=True, exist_ok=True)

    ## read candidates
    candidates = load_candidates(phase, type_id)
    candidates["type"] = type_id

    ## read labels
    labels = load_train_step2_label()
    labels = preprocess_labels(labels)

    # ## merge labels
    candidates = candidates.merge(labels, how="left", on=["session", "aid", "type"])
    candidates["target"] = candidates["target"].fillna(0)
    candidates["target"] = candidates["target"].astype("int8")

    candidates.to_parquet(FEATURES / f"base_preprocess_candidates_{phase}_type_{type_id}.parquet")


if __name__ == "__main__":
    main()
