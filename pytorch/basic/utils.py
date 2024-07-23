import gc

import japanize_matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
from matplotlib.widgets import Slider
from numpy.random import default_rng
from torch import from_numpy

import tensorflow as tf


def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))


def perceptron(x, w, b):
    a = np.dot(x, w) + b
    return sigmoid(a)


def multilayer_perceptron(x, w1, b1, w2, b2):
    z1 = perceptron(x, w1, b1)
    y = perceptron(z1, w2, b2)
    return y


# 中心値の異なる2つの二次元ガウス分布
def dataset_for_perceptron():
    rng = default_rng(seed=0)  # 今回はデータセットの乱数を固定させます。

    num_pos = 100  # 生成するシグナル(positive)イベントの数
    num_neg = 100  # 生成するバックグラウンド(negative)イベントの数

    # === データ点の生成 ===#
    # 平均(x1,x2) = (1.0, 0.0)、分散=1の２次元ガウス分布
    x_pos = rng.multivariate_normal(mean=[1.0, 0.0], cov=np.eye(2, 2), size=num_pos)
    t_pos = np.ones((num_pos, 1))  # Signalは1にラベリング

    # 平均(x1,x2) = (-1.0, 0.0)、分散=1の２次元ガウス分布
    x_neg = rng.multivariate_normal(mean=[-1.0, 0.0], cov=np.eye(2, 2), size=num_neg)
    t_neg = np.zeros((num_neg, 1))  # Backgroundは0にラベリング

    # === 2つのラベルを持つ学習データを1つにまとめる ===#
    x = np.concatenate([x_pos, x_neg])
    t = np.concatenate([t_pos, t_neg])

    # === データをランダムに並び替える ===#
    p = rng.permutation(len(x))
    x, t = x[p], t[p]

    return x, t


# 二次元ガウス分布と一様分布
def dataset_for_mlp():
    rng = default_rng(seed=0)  # 今回はデータセットの乱数を固定させます。

    num_pos = 100  # 生成するシグナル(positive)イベントの数
    num_neg = 1000  # 生成するバックグラウンド(negative)イベントの数

    # === データ点の生成 ===#
    # 平均(x1,x2) = (1.0, 0.0)、分散=1の２次元ガウス分布
    x_pos = rng.multivariate_normal(mean=[1.0, 0], cov=np.eye(2, 2), size=num_pos)
    t_pos = np.ones((num_pos, 1))  # Signalは1にラベリング

    # (-5, +5)の一様分布
    x_neg = rng.uniform(low=-5, high=5, size=(num_neg, 2))
    t_neg = np.zeros((num_neg, 1))  # Backgroundは0にラベリング

    # === 2つのラベルを持つ学習データを1つにまとめる ===#
    x = np.concatenate([x_pos, x_neg])
    t = np.concatenate([t_pos, t_neg])

    # === データをランダムに並び替える ===#
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
def plot_prediction(prediction, *args, ax, ngrid=100):
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
        y = prediction(x_tensor, *args)
        y = y.detach().numpy()
    else:
        y = prediction(x, *args)
    _ = ax.tricontourf(x[:, 0], x[:, 1], y.flatten(), levels=10)


# predictionメソッドの等高線プロット(line)を作成する
def plot_prediction_line(prediction, *args, ax, ngrid=100):
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
        y = prediction(x_tensor, *args)
        y = y.detach().numpy()
    else:
        y = prediction(x, *args)
    _ = ax.tricontour(x[:, 0], x[:, 1], y.flatten(), levels=10)


def plot_prediction_regression(x, t, prediction, w1, b1, w2, b2, ax):
    #  データ点のプロット
    ax.scatter(x, t, s=10, c="black")

    #  関数predictionの出力をプロット
    y = prediction(x, w1, b1, w2, b2)
    ax.plot(x, y, c="red")

    # 中間層の各ノードの出力をプロット
    num_nodes = len(w2)
    for i in range(num_nodes):
        y = w2[i] * perceptron(x, w1[:, i], b1[i])
        ax.plot(x, y, linestyle="dashed")  # (中間層のノードの出力 * 重み)をプロット
    ax.plot(x, np.full_like(x, b2[0]), linestyle="dashed")  # 最後の層のバイアスタームのプロット


def sigmoid_with_slider():
    fig, ax = plt.subplots(figsize=(7, 5))
    fig.subplots_adjust(left=0.15, bottom=0.20)

    x = np.linspace(-10, 10, 100)
    y1 = np.where(x > 0, 1, 0)
    y2 = sigmoid(x)
    y3 = sigmoid(1.0 * x)
    ax.plot(x, y1, label="Step function", c="black")
    ax.plot(x, y2, label="Sigmoid function", c="red")
    (line,) = ax.plot(x, y3, label="Sigmoid function (x' = x / 1)", c="orange", linestyle="dashed")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend()

    ax_scale = fig.add_axes([0.2, 0.0, 0.65, 0.10])
    scale_slider = Slider(ax=ax_scale, label="x scale", valmin=1, valmax=15, valinit=1.0)

    def update(val):
        line.set_ydata(sigmoid(scale_slider.val * x))
        line.set_label(f"Sigmoid function (x' = x /{scale_slider.val: 1.0f})")
        ax.legend()
        fig.canvas.draw_idle()

    scale_slider.on_changed(update)
    plt.show()


def simple_perceptron_with_slider(w1=1.0, w2=1.0, b=-1.5):

    def calculate_x2(x1, w1, w2, b):
        return -(w1 * x1 + b) / w2

    def update_fill(fill, x, y1, y2):
        if isinstance(y1, (int, float)):
            y1 = np.full_like(x, y1)
        if isinstance(y2, (int, float)):
            y2 = np.full_like(x, y2)

        for path in fill.get_paths():
            vertices = path.vertices
            vertices[0] = [x[0], y2[0]]
            vertices[1 : len(x) + 1] = np.stack([x, y1]).T
            vertices[len(x) + 1] = [x[-1], y2[-1]]
            vertices[-len(x) - 1 : -1] = np.stack([x, y2]).T[::-1]
            vertices[-1] = vertices[0]

    def update_ax(val):
        w1 = w1_slider.val
        w2 = w2_slider.val
        b = b_slider.val

        x2 = calculate_x2(x1, w1, w2, b)

        line.set_ydata(x2)
        line.set_label(f"{w1:1.1f} * x1 + {w2:1.1f} * x2 + {b:1.1f} = 0")

        update_fill(fill_b, x1, ymin, x2)
        update_fill(fill_t, x1, x2, ymax)
        fill_t.set_facecolor("red" if w2 > 0 else "blue")
        fill_b.set_facecolor("blue" if w2 > 0 else "red")

        ax.legend()
        fig.canvas.draw_idle()

    xmin, xmax = -0.5, +1.5
    ymin, ymax = -0.5, +1.5

    x1 = np.linspace(xmin, xmax, 2)
    x2 = calculate_x2(x1, w1, w2, b)

    fig, ax = plt.subplots(figsize=(5, 6))
    fig.subplots_adjust(left=0.15, bottom=0.25)

    (line,) = ax.plot(x1, x2, c="black")
    line.set_label(f"{w1:1.1f} * x1 + {w2:1.1f} * x2 + {b:1.1f} = 0")
    fill_b = ax.fill_between(x1, ymin, x2, alpha=0.3)
    fill_t = ax.fill_between(x1, x2, ymax, alpha=0.3)
    fill_t.set_facecolor("red" if w2 > 0 else "blue")
    fill_b.set_facecolor("blue" if w2 > 0 else "red")

    ax.scatter([0, 0, 1], [0, 1, 0], label="False", c="blue", s=100)
    ax.scatter([1], [1], label="True", c="red", s=100)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_xlim([xmin, xmax])
    ax.set_ylim([xmin, xmax])
    ax.legend()

    ax_w1 = fig.add_axes([0.125, 0.13, 0.775, 0.035])
    ax_w2 = fig.add_axes([0.125, 0.09, 0.775, 0.035])
    ax_b = fig.add_axes([0.125, 0.05, 0.775, 0.035])

    w1_slider = Slider(ax=ax_w1, label="w1", valmin=-3, valmax=3, valinit=w1)
    w2_slider = Slider(ax=ax_w2, label="w2", valmin=-3, valmax=3, valinit=w2)
    b_slider = Slider(ax=ax_b, label="b", valmin=-3, valmax=3, valinit=b)

    w1_slider.on_changed(update_ax)
    w2_slider.on_changed(update_ax)
    b_slider.on_changed(update_ax)

    plt.show()


def perceptron_with_slider(w=[1.0, 1.0], b=-1.5):
    def update_ax(val):
        ax.clear()

        w[0] = w1_slider.val
        w[1] = w2_slider.val
        b = b_slider.val

        plot_prediction(perceptron, w, b, ax=ax, ngrid=30)
        plot_datapoint(*data, ax=ax)

        fig.canvas.draw_idle()

        gc.collect()

    w = np.array(w)
    b = np.array(b)

    fig, ax = plt.subplots(figsize=(5, 6))
    fig.subplots_adjust(left=0.15, bottom=0.25)

    plot_prediction(perceptron, w, b, ax=ax, ngrid=30)

    data = dataset_for_perceptron()
    plot_datapoint(*data, ax=ax)

    ax_w1 = fig.add_axes([0.125, 0.13, 0.775, 0.035])
    ax_w2 = fig.add_axes([0.125, 0.09, 0.775, 0.035])
    ax_b = fig.add_axes([0.125, 0.05, 0.775, 0.035])

    w1_slider = Slider(ax=ax_w1, label="w1", valmin=-3, valmax=3, valinit=w[0])
    w2_slider = Slider(ax=ax_w2, label="w2", valmin=-3, valmax=3, valinit=w[1])
    b_slider = Slider(ax=ax_b, label="b", valmin=-3, valmax=3, valinit=b)

    w1_slider.on_changed(update_ax)
    w2_slider.on_changed(update_ax)
    b_slider.on_changed(update_ax)

    plt.show()


def mlp_with_slider():
    def update_ax(val):
        ax.clear()

        for j, i in w_slider.keys():
            if i < 3:
                w1[j][i] = w_slider[(j, i)].val
            else:
                w2[j][0] = w_slider[(j, 3)].val

        plot_prediction(multilayer_perceptron, w1[:-1], w1[-1], w2[:-1], w2[-1], ax=ax)
        plot_datapoint(*data, ax=ax)

        fig.canvas.draw_idle()

        gc.collect()

    w1 = np.array([[-1.0, 0.0, +1.0], [+0.0, +1.0, +1.0], [+3.0, +2.0, +2.0]])
    w2 = np.array([[+1.0], [+1.0], [+1.0], [+1.0]])

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.subplots_adjust(left=0.125, bottom=0.10, right=0.6)

    plot_prediction(multilayer_perceptron, w1[:-1], w1[-1], w2[:-1], w2[-1], ax=ax)

    data = dataset_for_mlp()
    plot_datapoint(*data, ax=ax)

    w_slider = {}
    for i in range(4):
        if i < 3:
            text = f"1層目{i + 1}つ目のパーセプトロン"
        else:
            text = "2層目のパーセプトロン"
        fig.text(
            1.3,
            0.95 - 0.19 * i,
            text,
            ha="center",
            va="center",
            fontsize=13,
            transform=ax.transAxes,
        )

        for j in range(4):
            if i < 3 and j == 3:
                continue

            position = [0.68, 0.80 - 0.15 * i - 0.03 * j, 0.15, 0.02]
            if i < 3:
                label = f"w1[{j}][{i}]" if j < 2 else f"b1[{i}]"
                valint = w1[j][i]
            else:
                label = f"w2[{j}][0]" if j < 3 else "b2[0]"
                valint = w2[j][0]

            w_slider[(j, i)] = Slider(
                ax=fig.add_axes(position),
                label=label,
                valmin=-3,
                valmax=3,
                valinit=valint,
            )

    for key in w_slider.keys():
        w_slider[key].on_changed(update_ax)

    plt.show()


def perceptron_gd_with_slider(parameter_history):

    x, t = dataset_for_perceptron()

    fig, ax = plt.subplots(figsize=(6, 6))
    fig.subplots_adjust(left=0.15, bottom=0.20)

    w, b = parameter_history[0]
    plot_prediction(perceptron, w, b, ax=ax)
    plot_datapoint(x, t, ax=ax)

    ax_step = fig.add_axes([0.2, 0.0, 0.65, 0.10])
    step_slider = Slider(
        ax=ax_step,
        label="steps",
        valmin=0,
        valmax=len(parameter_history) - 1,
        valinit=0,
        valstep=len(parameter_history) // 50,
    )

    def update_ax(val):
        ax.clear()

        i = step_slider.val
        w, b = parameter_history[i]

        plot_prediction(perceptron, w, b, ax=ax)
        plot_datapoint(x, t, ax=ax)

        fig.canvas.draw_idle()

        gc.collect()

    step_slider.on_changed(update_ax)

    plt.show()


def mlp_gradient_descent_slider(parameter_history):

    x, t = dataset_for_mlp()

    fig, ax = plt.subplots(figsize=(6, 6))
    fig.subplots_adjust(left=0.15, bottom=0.20)

    w1, b1, w2, b2 = parameter_history[0]
    plot_prediction(multilayer_perceptron, w1, b1, w2, b2, ax=ax)
    plot_datapoint(x, t, ax=ax)

    ax_step = fig.add_axes([0.2, 0.0, 0.65, 0.10])
    step_slider = Slider(
        ax=ax_step,
        label="steps",
        valmin=0,
        valmax=len(parameter_history) - 1,
        valinit=0,
        valstep=len(parameter_history) // 50,
    )

    def update_ax(val):
        ax.clear()

        i = step_slider.val
        w1, b1, w2, b2 = parameter_history[i]

        plot_prediction(multilayer_perceptron, w1, b1, w2, b2, ax=ax)
        plot_datapoint(x, t, ax=ax)

        fig.canvas.draw_idle()

        gc.collect()

    step_slider.on_changed(update_ax)

    plt.show()


# ウェイト(wij)の初期値をプロット
def plot_model_weights(model, ax):
    iLayers = [0, 3, 6, 10]
    labels = [
        " 0th layer",
        " 3th layer",
        " 6th layer",
        "Last layer",
    ]

    if isinstance(model, tf.keras.Model):
        # get only kernel (not bias)
        values = [model.weights[i * 2].numpy().flatten() for i in iLayers]
    elif isinstance(model, nn.Module):
        values = [model.linears[i].weight.flatten().detach().numpy() for i in iLayers]
    else:
        raise NotImplementedError

    ax.hist(values, bins=50, stacked=False, density=True, label=labels, histtype="step")
    ax.set_xlabel("weight")
    ax.set_ylabel("Probability density")
    ax.legend(loc="upper left", fontsize="x-small")


# 各ノードの出力(sigma(ai))をプロット
def plot_model_hidden_nodes(model, x, ax):

    # i 番目のレイヤーの出力を取得する
    def _get_activation(model, i, x):
        if isinstance(model, tf.keras.Model):
            activation = tf.keras.Model(model.layers[0].input, model.layers[i].output)(x)
            return activation.numpy()
        elif isinstance(model, nn.Module):
            return model(x, i).flatten().detach().numpy()
        else:
            raise NotImplementedError

    iLayers = [0, 3, 6, 10]
    labels = [
        " 0th layer",
        " 3th layer",
        " 6th layer",
        "Last layer",
    ]

    values = [_get_activation(model, i, x).flatten() for i in iLayers]
    ax.hist(values, bins=50, stacked=False, density=True, label=labels, histtype="step")
    ax.set_xlabel("activation")
    ax.set_ylabel("Probability density")
    ax.legend(loc="upper center", fontsize="x-small")


# ウェイト(wij)の微分(dE/dwij)をプロット
def plot_model_weight_gradients(model, x, t, ax):

    # i 番目のレイヤーにつながる重み(wij)の勾配を取得する
    def _get_gradients(model, i, x, t):
        if isinstance(model, tf.keras.Model):
            # from tensorflow.python.eager import backprop

            weights = model.layers[i].weights[0]  # get only kernel (not bias)
            with tf.GradientTape() as tape:
                pred = model(x)
                loss = model.compiled_loss(tf.constant(t), pred)

            gradients = tape.gradient(loss, weights)
            return gradients.numpy()
        elif isinstance(model, nn.Module):
            return np.abs(model.linears[i].weight.grad.flatten().detach().numpy())
        else:
            raise NotImplementedError

    iLayers = [0, 3, 6, 10]
    labels = [
        " 0th layer",
        " 3th layer",
        " 6th layer",
        "Last layer",
    ]

    grads = [np.abs(_get_gradients(model, i, x, t).flatten()) for i in iLayers]
    grads = [np.log10(x[x > 0]) for x in grads]
    ax.hist(grads, bins=50, stacked=False, density=True, label=labels, histtype="step")
    ax.set_xlabel("log10(|gradient of weights|)")
    ax.set_ylabel("Probability density")
