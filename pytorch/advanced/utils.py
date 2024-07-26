import japanize_matplotlib
import numpy as np
import torch.nn as nn
from numpy.random import default_rng
from torch import from_numpy


# 二次元ガウス分布と一様分布
def dataset_for_mlp():
    rng = default_rng(seed=0)  # 今回はデータセットの乱数を固定させます。

    num_pos = 100  # 生成するシグナル(positive)イベントの数
    num_neg = 1000  # 生成するバックグラウンド(negative)イベントの数

    # === データ点の生成 === #
    # 平均(x1,x2) = (1.0, 0.0)、分散=1の２次元ガウス分布
    x_pos = rng.multivariate_normal(mean=[1.0, 0], cov=np.eye(2, 2), size=num_pos)
    t_pos = np.ones((num_pos, 1))  # Signalは1にラベリング

    # (-5, +5)の一様分布
    x_neg = rng.uniform(low=-5, high=5, size=(num_neg, 2))
    t_neg = np.zeros((num_neg, 1))  # Backgroundは0にラベリング

    # === 2つのラベルを持つ学習データを1つにまとめる === #
    x = np.concatenate([x_pos, x_neg])
    t = np.concatenate([t_pos, t_neg])

    # === データをランダムに並び替える === #
    p = rng.permutation(len(x))
    x, t = x[p], t[p]

    return x, t


def dataset_overtraining(n=30, noise_scale=0.3):
    rng = default_rng(seed=0)  # 今回はデータセットの乱数を固定させます。

    # === データ点の生成 === #
    x = np.linspace(-np.pi, np.pi, n)
    t = np.sin(x) + rng.normal(loc=0.0, scale=noise_scale, size=n)

    # === データをランダムに並び替える === #
    p = rng.permutation(len(x))
    x, t = x[p], t[p]

    return x, t


# ラベル t={0,1}を持つデータ点のプロット
def plot_datapoint(x, t, ax):

    # シグナル/バックグラウンドの抽出
    x_pos = x[t[:, 0] == 1]  # シグナルのラベルだけを抽出
    x_neg = x[t[:, 0] == 0]  # バックグラウンドのラベルだけを抽出

    # プロット
    ax.scatter(x_pos[:, 0], x_pos[:, 1], label="Signal", c="red", s=10)
    ax.scatter(x_neg[:, 0], x_neg[:, 1], label="Background", c="blue", s=10)
    ax.set_xlabel("x1")  # x軸ラベルの設定
    ax.set_ylabel("x2")  # y軸ラベルの設定
    ax.legend()  # labelの表示


# predictionメソッドの等高線プロット(fill)を作成する
def plot_prediction(prediction, *args, ax, ngrid=100, **kwargs):
    # 等高線を描くためのメッシュの生成
    # x1 = (-5, 5), x2 = (-5, 5) の範囲で100点x100点のメッシュを作成
    x1 = np.linspace(-5, 5, ngrid)
    x2 = np.linspace(-5, 5, ngrid)
    x1v, x2v = np.meshgrid(x1, x2)
    x1v = x1v.flatten()  # 二次元配列を一次元配列に変換 ( shape=(100, 100) => shape=(10000, ))
    x2v = x2v.flatten()  # 二次元配列を一次元配列に変換 ( shape=(100, 100) => shape=(10000, ))
    x = np.array([x1v, x2v]).T

    # 関数predictionを使って入力xから出力yを計算し、等高線プロットを作成
    if isinstance(prediction, nn.Module):
        x_tensor = from_numpy(x).float()
        y = prediction(x_tensor, *args, **kwargs)
        y = y.detach().numpy()
    else:
        y = prediction(x, *args, **kwargs)

    _ = ax.tricontourf(x[:, 0], x[:, 1], y.flatten(), levels=10)
