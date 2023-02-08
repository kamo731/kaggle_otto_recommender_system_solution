import os
import sys

sys.path.append("src")
import cudf as cd
from loguru import logger

import globals


def main():
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    filename = os.path.basename(__file__).replace(".py", "")
    logger.add(globals.LOGS / f"{filename}.log", rotation="1 MB", level="TRACE")

    output_dir = globals.SUBMISSION / "concatenated"
    output_dir.mkdir(parents=True, exist_ok=True)

    # ここを書き換える
    output_filename = "submission.csv"
    clicks_file_name = globals.SUBMISSION / "clicks/ver_clicks_000_submission.csv"
    carts_file_name = globals.SUBMISSION / "carts/ver_carts_000_submission.csv"
    orders_file_name = globals.SUBMISSION / "orders/ver_orders_000_submission.csv"
    logger.info(f"{output_filename=}")
    logger.info("concatenated files are")
    logger.info(f"{clicks_file_name=}")
    logger.info(f"{carts_file_name=}")
    logger.info(f"{orders_file_name=}")

    df_sub_clicks = cd.read_csv(clicks_file_name)
    df_sub_clicks = df_sub_clicks[df_sub_clicks.session_type.str.contains("clicks")]

    df_sub_carts = cd.read_csv(carts_file_name)
    df_sub_carts = df_sub_carts[df_sub_carts.session_type.str.contains("carts")]
    df_sub_carts["labels"] = df_sub_carts.labels.str.slice(stop=8 * 40)  # size exceeded の回避 aid 7桁 + 空白1 が40候補くらいで打ち切り

    df_sub_orders = cd.read_csv(orders_file_name)
    df_sub_orders = df_sub_orders[df_sub_orders.session_type.str.contains("orders")]
    df_sub_orders["labels"] = df_sub_orders.labels.str.slice(
        stop=8 * 40
    )  # size exceeded の回避 aid 7桁 + 空白1 が40候補くらいで打ち切り

    df_concat = cd.concat([df_sub_clicks, df_sub_carts, df_sub_orders]).reset_index(drop=True)
    df_concat = df_concat.sort_values("session_type").reset_index(drop=True)
    df_concat.to_csv(output_dir / output_filename, index=False)
    logger.info(f"file head:\n{df_concat.head()}")
    logger.info(f"shape: {df_concat.shape}")


if __name__ == "__main__":
    main()
