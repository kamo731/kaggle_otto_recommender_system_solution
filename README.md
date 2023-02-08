# kaggle_otto_recommender_system_solution
## 概要
- Kaggle [OTTO – Multi-Objective Recommender System](https://www.kaggle.com/competitions/otto-recommender-system/overview) のチーム FOMKA (LB 40位 top2% Silver Medal)のソリューションの一部です。
- xgboostでの単体モデルを作成し提出するファイルを作成します。

## 環境構築
### 動作環境
- Google Compute Engine
  - NVIDIA A100 80GB
  - Ubuntu 20.04 LTS (x86/64)
  - Disk size 128GB以上推奨

### 実行手順
```
$ sudo apt install make
$ make build-env
# docker daemonの再起動。
$ sudo service docker restart

# ユーザをdockerグループに追加。
$ sudo usermod -aG docker <USER_NAME>

# docker:x:998:<USER_NAME> という出力が出る。
$ cat /etc/group | grep docker

# permission deniedが出る場合、一度セッション切断が必要がある。
$ make build-image
```

注意:
- 特徴量作成のスクリプト(src/03_features)はdocker上での実行検証をしていません。
- リポジトリ直下のrequirements.txtでvenvなどの通常のpython環境を利用になるか、docker上で実行する場合は適宜モジュール追加をお願いします。
- 複数名でのコードをつなぎ合わせた関係で実行環境が異なっています。ご了承ください。

## 入力データの取得
```
$ make install-dataset
```

## example
### 例1 xgbを用いた全パイプラインの実行
```
# output/sub/concatenated/submission.csvが提出ファイルとして最後に出力される
$ make exec-xgb-example
```

### 例2 ランキングモデルの例 (ordersのみ)
```
# 例1「xgbを用いた全パイプラインの実行」の手順を既に実行している場合
$ make exec-xgb-rank-orders

# 例1「xgbを用いた全パイプラインの実行」の手順を既に実行していない場合
$ make exec-xgb-ranking-example
```

## 独自用語
- train original: コンペで配布された[オリジナルデータ](https://www.kaggle.com/competitions/otto-recommender-system/data)のtrain (week1~week4)
- test original: コンペで配布された[オリジナルデータ](https://www.kaggle.com/competitions/otto-recommender-system/data)のtest (week4~)
- train step1: train originalのうち、前3週分のデータ
- train step2: train originalのうち、最後1週分のデータ(labelを除く)
  - 詳細は[local validation tracks public LB perfecty -- here is the setup](https://www.kaggle.com/competitions/otto-recommender-system/discussion/364991)
- cvm: co-visitation matrixの省略


## 参考
- [候補作成までの土台](https://www.kaggle.com/code/cdeotte/candidate-rerank-model-lb-0-575)
- [word2vec特徴量](https://www.kaggle.com/code/duuuscha/train-submit-word2vec-optimized-hparams)
- [validation](https://www.kaggle.com/competitions/otto-recommender-system/discussion/364991)
- [学習パイプライン作成方法](https://www.kaggle.com/competitions/otto-recommender-system/discussion/370210)


## チームFOMKAメンバー
- kaggle アカウント
  - [moka](https://www.kaggle.com/kmoka731)
  - [Shaorooon](https://www.kaggle.com/syaorn13)
  - [amataro](https://www.kaggle.com/amataro224)
  - [minamoto](https://www.kaggle.com/usermina)
  - [Ranchantan](https://www.kaggle.com/ranchantan)
- github アカウント
  - [moka](https://github.com/kamo731)
  - [shaoroon](https://github.com/shaoroon)
  - [amataro224](https://github.com/amataro224)
  - [minamoto](https://github.com/mina-moto)
  - [yomura-yomura](https://github.com/yomura-yomura)