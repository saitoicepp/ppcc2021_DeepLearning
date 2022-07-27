# 粒子物理コンピューティングサマースクール 「Deep Learning」 講習資料
粒子物理コンピューティングサマースクールの講習 「Deep Learning 基礎編」、「Deep Learning 発展編」 のための講習資料です。

## 構成
- basic: 「Deep Learning 基礎編」のためのJupyter notebookです。
  - メインの講習資料: [DeepLearning.ipynb](basic/DeepLearning.ipynb)
  - 実習課題例:
    - 消失勾配問題 [ActivationFunction.ipynb](basic/ActivationFunction.ipynb)
    - 最適化手法 [Optimization.ipynb](basic/Optimization.ipynb)
- advanced: 「Deep Learning 発展編」のためのJupyter notebookです。
  - 多層パーセプトロン [MLP.ipynb](advanced/MLP.ipynb)
  - 過学習を抑えるテクニック
    - 早期終了　[EarlyStopping.ipynb](advanced/EarlyStopping.ipynb)
    - L1/L2 正則化 [Regularization.ipynb](advanced/Regularization.ipynb)
    - ドロップアウト [Dropout.ipynb](advanced/Dropout.ipynb)
  - 畳み込みニューラルネットワーク [CNN.ipynb](advanced/CNN.ipynb)
  - 実習課題例:
    - ヒッグスチャレンジ [HiggsChallenge.ipynb](advanced/HiggsChallenge.ipynb)
- pytorch: 上記のjupyter notebookでは深層学習ライブラリとしてKeras/Tensorflowが使われています。PyTorchを使った講習資料がこのディレクトリにあります。notebook名と内容はKeras/Tensorflow用のものと同一です。
