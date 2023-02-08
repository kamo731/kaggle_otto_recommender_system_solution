# include .env
IMAGE=gcp-a100-cudf/otto:latest
BASH_CONTAINER=otto-gpu-bash

.PHONY: build-env
build-env:
	cd environment && \
	sh install_gpu_driver.sh && \
	curl https://get.docker.com | sh && \
	sh install_nvidia_docker.sh

.PHONY: build-image
build-image:
	cd environment && \
	docker build -t $(IMAGE) .


# データセットのinstall
.PHONY: install-dataset
install-dataset:
	mkdir input
	apt install unzip
	kaggle datasets download -d radek1/otto-full-optimized-memory-footprint -p ./input
	unzip input/otto-full-optimized-memory-footprint.zip -d input/otto-full-optimized-memory-footprint
	kaggle datasets download -d radek1/otto-train-and-test-data-for-local-validation -p ./input
	unzip input/otto-train-and-test-data-for-local-validation.zip -d input/otto-train-and-test-data-for-local-validation


# スクリプトの実行

# src/01_co_visitation_matrix: co-visitation matrixの作成/更新
.PYTHON: update-cvm-all
update-cvm-all: update-clicks update-carts-orders update-buy2buy

.PYTHON: update-clicks
update-clicks:
	docker run -it --rm --gpus all --volume $(PWD):$(PWD) --workdir $(PWD) --name otto-gpu-bash gcp-a100-cudf/otto:latest python3 src/01_co_visitation_matrix/make_co_visitation_matrix_clicks_train.py
	docker run -it --rm --gpus all --volume $(PWD):$(PWD) --workdir $(PWD) --name otto-gpu-bash gcp-a100-cudf/otto:latest python3 src/01_co_visitation_matrix/make_co_visitation_matrix_clicks_test.py

.PYTHON: update-carts-orders
update-carts-orders:
	docker run -it --rm --gpus all --volume $(PWD):$(PWD) --workdir $(PWD) --name otto-gpu-bash gcp-a100-cudf/otto:latest python3 src/01_co_visitation_matrix/make_co_visitation_matrix_carts_orders_train.py
	docker run -it --rm --gpus all --volume $(PWD):$(PWD) --workdir $(PWD) --name otto-gpu-bash gcp-a100-cudf/otto:latest python3 src/01_co_visitation_matrix/make_co_visitation_matrix_carts_orders_test.py

.PYTHON: update-buy2buy
update-buy2buy:
	docker run -it --rm --gpus all --volume $(PWD):$(PWD) --workdir $(PWD) --name otto-gpu-bash gcp-a100-cudf/otto:latest python3 src/01_co_visitation_matrix/make_co_visitation_matrix_buy2buy_train.py
	docker run -it --rm --gpus all --volume $(PWD):$(PWD) --workdir $(PWD) --name otto-gpu-bash gcp-a100-cudf/otto:latest python3 src/01_co_visitation_matrix/make_co_visitation_matrix_buy2buy_test.py


# src/02_candidates: 候補の作成/更新
.PYTHON: update-candidates-all
update-candidates-all: update-candidates-session update-candidates-additional update-candidates-concat

.PYTHON: update-candidates-session
update-candidates-session:
	docker run -it --rm --gpus all --volume $(PWD):$(PWD) --workdir $(PWD) --name otto-gpu-bash gcp-a100-cudf/otto:latest python3 src/02_candidates/make_candidates_buys_from_session.py
	docker run -it --rm --gpus all --volume $(PWD):$(PWD) --workdir $(PWD) --name otto-gpu-bash gcp-a100-cudf/otto:latest python3 src/02_candidates/make_candidates_clicks_from_session.py

.PYTHON: update-candidates-additional
update-candidates-additional:
	docker run -it --rm --gpus all --volume $(PWD):$(PWD) --workdir $(PWD) --name otto-gpu-bash gcp-a100-cudf/otto:latest python3 src/02_candidates/make_candidates_clicks_additional.py
	docker run -it --rm --gpus all --volume $(PWD):$(PWD) --workdir $(PWD) --name otto-gpu-bash gcp-a100-cudf/otto:latest python3 src/02_candidates/make_candidates_buys_additional.py

.PYTHON: update-candidates-concat
update-candidates-concat:
	docker run -it --rm --gpus all --volume $(PWD):$(PWD) --workdir $(PWD) --name otto-gpu-bash gcp-a100-cudf/otto:latest python3 src/02_candidates/concat_candidates.py


# src/03_features: 特徴量の作成 (このプロセスはcudfを用いていないため、docker環境ではない通常のpython環境で実行)
## clicks/carts/ordersすべてに共通するtrain/testの特徴量

.PYTHON: update-features-all
update-features-all: make-common-features make-features-train-clicks make-features-test-clicks make-features-train-carts make-features-test-carts make-features-train-orders make-features-test-orders

.PYTHON: make-common-features
make-common-features
	python src/03_features/base_aid_features.py train
	python src/03_features/base_session_features.py train
	python src/03_features/base_aid_features.py test
	python src/03_features/base_session_features.py test
	python src/03_features/prepare_word2vec_feature.py
	python src/03_features/word2vec_features_aid.py train
	python src/03_features/word2vec_features_session.py train
	python src/03_features/word2vec_features_aid.py test
	python src/03_features/word2vec_features_session.py test

## clicksのtrain用特徴量
.PYTHON: make-features-train-clicks
make-features-train-clicks:
	python src/03_features/base_preprocess_candidates.py train 0
	python src/03_features/base_session_and_aid_features.py train 0
	python src/03_features/word2vec_features_session_and_aid_all.py train 0
	python src/03_features/word2vec_features_session_and_aid_last1.py train 0
	python src/03_features/word2vec_features_session_and_aid_last5.py train 0
	python src/03_features/cvm_features_all.py train 0
	python src/03_features/cvm_features_last1.py train 0
	python src/03_features/cvm_features_last5.py train 0

## clicksのtest用特徴量
.PYTHON: make-features-test-clicks
make-features-test-clicks:
	python src/03_features/base_preprocess_candidates.py test 0
	python src/03_features/base_session_and_aid_features.py test 0
	python src/03_features/word2vec_features_session_and_aid_all.py test 0
	python src/03_features/word2vec_features_session_and_aid_last1.py test 0
	python src/03_features/word2vec_features_session_and_aid_last5.py test 0
	python src/03_features/cvm_features_all.py test 0
	python src/03_features/cvm_features_last1.py test 0
	python src/03_features/cvm_features_last5.py test 0

## cartsのtrain用特徴量
.PYTHON: make-features-train-carts
make-features-train-carts:
	python src/03_features/base_preprocess_candidates.py train 1
	python src/03_features/base_session_and_aid_features.py train 1
	python src/03_features/word2vec_features_session_and_aid_all.py train 1
	python src/03_features/word2vec_features_session_and_aid_last1.py train 1
	python src/03_features/word2vec_features_session_and_aid_last5.py train 1
	python src/03_features/cvm_features_all.py train 1
	python src/03_features/cvm_features_last1.py train 1
	python src/03_features/cvm_features_last5.py train 1

## cartsのtest用特徴量
.PYTHON: make-features-test-carts
make-features-test-carts:
	python src/03_features/base_preprocess_candidates.py test 1
	python src/03_features/base_session_and_aid_features.py test 1
	python src/03_features/word2vec_features_session_and_aid_all.py test 1
	python src/03_features/word2vec_features_session_and_aid_last1.py test 1
	python src/03_features/word2vec_features_session_and_aid_last5.py test 1
	python src/03_features/cvm_features_all.py test 1
	python src/03_features/cvm_features_last1.py test 1
	python src/03_features/cvm_features_last5.py test 1


## ordersのtrain用特徴量
.PYTHON: make-features-train-orders
make-features-train-orders:
	python src/03_features/base_preprocess_candidates.py train 2
	python src/03_features/base_session_and_aid_features.py train 2
	python src/03_features/word2vec_features_session_and_aid_all.py train 2
	python src/03_features/word2vec_features_session_and_aid_last1.py train 2
	python src/03_features/word2vec_features_session_and_aid_last5.py train 2
	python src/03_features/cvm_features_all.py train 2
	python src/03_features/cvm_features_last1.py train 2
	python src/03_features/cvm_features_last5.py train 2

## ordersのtest用特徴量
.PYTHON: make-features-test-orders
make-features-test-orders:
	python src/03_features/base_preprocess_candidates.py test 2
	python src/03_features/base_session_and_aid_features.py test 2
	python src/03_features/word2vec_features_session_and_aid_all.py test 2
	python src/03_features/word2vec_features_session_and_aid_last1.py test 2
	python src/03_features/word2vec_features_session_and_aid_last5.py test 2
	python src/03_features/cvm_features_all.py test 2
	python src/03_features/cvm_features_last1.py test 2
	python src/03_features/cvm_features_last5.py test 2



# 04_model: モデルの学習と推論
.PYTHON: exec-xgb-cls-clicks
exec-xgb-cls-clicks:
	docker run -it --rm --gpus all --volume $(PWD):$(PWD) --workdir $(PWD) --name otto-gpu-bash gcp-a100-cudf/otto:latest python3 src/04_models/exec_xgb_cls_clicks_01_train.py
	docker run -it --rm --gpus all --volume $(PWD):$(PWD) --workdir $(PWD) --name otto-gpu-bash gcp-a100-cudf/otto:latest python3 src/04_models/exec_xgb_cls_clicks_02_calc_cv.py
	docker run -it --rm --gpus all --volume $(PWD):$(PWD) --workdir $(PWD) --name otto-gpu-bash gcp-a100-cudf/otto:latest python3 src/04_models/exec_xgb_cls_clicks_03_predict.py

.PYTHON: exec-xgb-cls-carts
exec-xgb-cls-carts:
	docker run -it --rm --gpus all --volume $(PWD):$(PWD) --workdir $(PWD) --name otto-gpu-bash gcp-a100-cudf/otto:latest python3 src/04_models/exec_xgb_cls_carts_01_train.py
	docker run -it --rm --gpus all --volume $(PWD):$(PWD) --workdir $(PWD) --name otto-gpu-bash gcp-a100-cudf/otto:latest python3 src/04_models/exec_xgb_cls_carts_02_calc_cv.py
	docker run -it --rm --gpus all --volume $(PWD):$(PWD) --workdir $(PWD) --name otto-gpu-bash gcp-a100-cudf/otto:latest python3 src/04_models/exec_xgb_cls_carts_03_predict.py

.PYTHON: exec-xgb-cls-orders
exec-xgb-cls-orders:
	docker run -it --rm --gpus all --volume $(PWD):$(PWD) --workdir $(PWD) --name otto-gpu-bash gcp-a100-cudf/otto:latest python3 src/04_models/exec_xgb_cls_orders_01_train.py
	docker run -it --rm --gpus all --volume $(PWD):$(PWD) --workdir $(PWD) --name otto-gpu-bash gcp-a100-cudf/otto:latest python3 src/04_models/exec_xgb_cls_orders_02_calc_cv.py
	docker run -it --rm --gpus all --volume $(PWD):$(PWD) --workdir $(PWD) --name otto-gpu-bash gcp-a100-cudf/otto:latest python3 src/04_models/exec_xgb_cls_orders_03_predict.py

# ordersのみランキングモデルの例
.PYTHON: exec-xgb-rank-orders
exec-xgb-rank-orders:
	docker run -it --rm --gpus all --volume $(PWD):$(PWD) --workdir $(PWD) --name otto-gpu-bash gcp-a100-cudf/otto:latest python3 src/04_models/exec_xgb_rank_orders_01_train.py
	docker run -it --rm --gpus all --volume $(PWD):$(PWD) --workdir $(PWD) --name otto-gpu-bash gcp-a100-cudf/otto:latest python3 src/04_models/exec_xgb_rank_orders_02_calc_cv.py
	docker run -it --rm --gpus all --volume $(PWD):$(PWD) --workdir $(PWD) --name otto-gpu-bash gcp-a100-cudf/otto:latest python3 src/04_models/exec_xgb_rank_orders_03_predict.py

# 05_make_submission
# predictの推論結果のファイル名をスクリプト内に記載してください
.PYTHON: concat-sub
concat-sub:
	docker run -it --rm --gpus all --volume $(PWD):$(PWD) --workdir $(PWD) --name otto-gpu-bash gcp-a100-cudf/otto:latest python3 src/05_make_submission/concat_sub.py


# example 1
.PYTHON: exec-xgb-example
exec-xgb-example: update-cvm-all update-candidates-all update-features-all exec-xgb-cls-clicks exec-xgb-cls-carts exec-xgb-cls-orders concat-sub

# example 2
.PYTHON: exec-xgb-ranking-example
exec-xgb-example: update-cvm-all update-candidates-all update-features-all exec-xgb-rank-orders
