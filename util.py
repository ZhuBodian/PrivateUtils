import json
import torch
import pandas as pd
from pathlib import Path
from itertools import repeat
from collections import OrderedDict
from PrivateUtils import global_var
import time
import pickle
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import random
from sklearn.decomposition import PCA,FactorAnalysis
from factor_analyzer import FactorAnalyzer
import winsound
from sklearn.decomposition import PCA
from yellowbrick.cluster import KElbowVisualizer


mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

class ProcessBar:
    def __init__(self, n, delta_ratio=None):
        self.n = n
        if delta_ratio is None:
            self.delta = int(self.n * 0.25) + 1
        else:
            assert 0 < delta_ratio < 1, 'delta_ratio应该在0与1之间'
            self.delta = int(self.n * delta_ratio) + 1

        idx = self.delta
        self.diplay_points = []
        for i in range(int(n/self.delta) + 1):
            self.diplay_points.append(idx)
            idx += self.delta

        self.displat_idx = 0

    def display(self, i, **kwargs):
        if i == self.diplay_points[self.displat_idx]:
            block_length = 50

            progress = (i / self.n) * 100
            temp = int(i / self.n * block_length)
            finish = "▓" * temp
            need_do = "-" * (block_length - temp)
            print(f'进度：{progress:^3.0f}%[{finish}->{need_do}] {i}/{self.n}；', end='')
            for key, value in kwargs.items():
                print(f'{key}：{value:.5f}，', end='')

            print('\n', end='')

            self.displat_idx += 1


def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)


def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)


def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)


def inf_loop(data_loader):
    ''' wrapper function for endless data loader. '''
    for loader in repeat(data_loader):
        yield from loader


def prepare_device(n_gpu_use):
    """
    setup GPU device if available. get gpu device indices which are used for DataParallel
    """
    n_gpu = torch.cuda.device_count()
    if n_gpu_use > 0 and n_gpu == 0:
        print("Warning: There\'s no GPU available on this machine,"
              "training will be performed on CPU.")
        global_var.get_value('email_log').add_log(
            "Warning: There\'s no GPU available on this machine,training will be performed on CPU.")

        n_gpu_use = 0
    if n_gpu_use > n_gpu:
        print(f"Warning: The number of GPU\'s configured to use is {n_gpu_use}, but only {n_gpu} are "
              "available on this machine.")
        global_var.get_value('email_log').add_log(
            f"Warning: The number of GPU\'s configured to use is {n_gpu_use}, but only {n_gpu} are available on this machine.")

        n_gpu_use = n_gpu
    device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
    list_ids = list(range(n_gpu_use))
    return device, list_ids


def save_as_pickle(path, name, data):
    os.makedirs(path, exist_ok=True)
    with open(path + name + '.pickle', 'wb') as f:
        pickle.dump(data, f)


def load_from_pickle(path, name):
    full_path = path + name + '.pickle'
    assert os.path.exists(full_path), print(f'文件{full_path}不存在')
    with open(full_path, "rb") as fh:
        data = pickle.load(fh)
    return data



class MetricTracker:
    """用于跟踪tensorboard scalar中的数据的类"""

    def __init__(self, *keys, writer=None):
        self.writer = writer
        self._data = pd.DataFrame(index=keys, columns=['total', 'counts', 'average'])
        self.reset()

    def reset(self):
        # 全置0
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        if self.writer is not None:
            self.writer.add_scalar(key, value)  # 如果没有这个属性，由__getattr__函数，其会创建一个
        self._data.total[key] += value * n
        self._data.counts[key] += n
        self._data.average[key] = self._data.total[key] / self._data.counts[key]

    def avg(self, key):
        return self._data.average[key]

    def result(self):
        # 返回计算跟踪的metric的平均值的字典形式
        return dict(self._data.average)


class MyTimer:
    def __init__(self, timer_reason):
        self._start_timer = time.time()
        self._stop_timer = time.time()
        self._timer_reason = timer_reason

    def stop(self):
        self._stop_timer = time.time()
        print(f'"{self._timer_reason}" cost time: {self._stop_timer - self._start_timer}s')


def save_fig(path, fig_id, tight_layout=True, fig_extension="png", resolution=300):
    os.makedirs(path, exist_ok=True)
    path = path + fig_id + "." + fig_extension
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)


def plot_cm(cm, text_label=None, process_cm=False):
    if process_cm:
        # 获得分类错误率，而不是错误的绝对值
        row_sums = cm.sum(axis=1, keepdims=True)
        norm_conf_mx = cm / row_sums

        # 用0填充对角线，仅保留错误，重新画图
        np.fill_diagonal(norm_conf_mx, 0)
        # 行代表实际类，列代表预测类，第8列看起来很亮，说明许多图片被错误分类为8
        # PS：第8行不那么差，说明数字8被正确分类了，且注意到错误不完全对称
        # 说明精力可以花在改进8的错误上（如进一步搜集8的数据，或者添加特征写个算法统计闭环数量）
        cm = norm_conf_mx

    if text_label is not None:
        cm = pd.DataFrame(cm, columns=text_label, index=text_label)

    plt.figure()
    sns.heatmap(cm, annot=True, linewidths=0.5, fmt=".2f", cmap="YlGnBu")
    plt.xlabel('True class')
    plt.ylabel('Predict class')


def plot_confusion_image(X_train, y_train, y_train_pred, cl_a, cl_b):
    # 分析单个错误也可以帮助获得洞见
    # 通过画图分析35正确分类，与错误分类的图像，获得洞见
    X_aa = X_train[(y_train == cl_a) & (y_train_pred == cl_a)]
    X_ab = X_train[(y_train == cl_a) & (y_train_pred == cl_b)]
    X_ba = X_train[(y_train == cl_b) & (y_train_pred == cl_a)]
    X_bb = X_train[(y_train == cl_b) & (y_train_pred == cl_b)]

    plt.figure(figsize=(8, 8))

    plt.subplot(221)
    plot_digits(X_aa[:25], images_per_row=5)
    plt.title(f'True {cl_a}, Predict{cl_a}')

    plt.subplot(222)
    plot_digits(X_ab[:25], images_per_row=5)
    plt.title(f'True {cl_a}, Predict{cl_b}')

    plt.subplot(223)
    plot_digits(X_ba[:25], images_per_row=5)
    plt.title(f'True {cl_b}, Predict{cl_a}')

    plt.subplot(224)
    plot_digits(X_bb[:25], images_per_row=5)
    plt.title(f'True {cl_b}, Predict{cl_b}')


def plot_digits(instances, images_per_row=10, **options):
    size = 28
    images_per_row = min(len(instances), images_per_row)
    # This is equivalent to n_rows = ceil(len(instances) / images_per_row):
    n_rows = (len(instances) - 1) // images_per_row + 1

    # Append empty images to fill the end of the grid, if needed:
    n_empty = n_rows * images_per_row - len(instances)
    padded_instances = np.concatenate([instances, np.zeros((n_empty, size * size))], axis=0)

    # Reshape the array so it's organized as a grid containing 28×28 images:
    image_grid = padded_instances.reshape((n_rows, images_per_row, size, size))

    # Combine axes 0 and 2 (vertical image grid axis, and vertical image axis),
    # and axes 1 and 3 (horizontal axes). We first need to move the axes that we
    # want to combine next to each other, using transpose(), and only then we
    # can reshape:
    big_image = image_grid.transpose(0, 2, 1, 3).reshape(n_rows * size,
                                                         images_per_row * size)
    # Now that we have a big image, we just need to show it:
    plt.imshow(big_image, cmap = mpl.cm.binary, **options)
    plt.axis("off")


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def plot_pca_var(df, additional_plot=None):
    """
    additional_plot是将整体的可解释方差图只看局部的图，是一个[left, right]的列表
    """
    col_nums = df.shape[1]

    pca = PCA(n_components=col_nums)
    df_transformed = pca.fit_transform(df)
    var = np.cumsum(pca.explained_variance_ratio_)
    var = np.append(0, var)

    # 可视化
    # plot横轴是指标个数，纵轴是ev值
    # scatter横轴是指标个数，纵轴是ev值

    if additional_plot is None:
        plt.figure(figsize=(8, 8))
        plt.plot(range(0, df.shape[1] + 1), var, markersize=3, marker='o')
        plt.plot(range(0, df.shape[1] + 1), var)
        plt.xlabel('Factors')
        plt.ylabel('Var cum sum')
        plt.grid()
    else:
        left, right = additional_plot[0], additional_plot[1]

        plt.figure(figsize=(16, 8))
        plt.subplot(1, 2, 1)
        plt.plot(range(0, df.shape[1] + 1), var, markersize=3, marker='o')
        plt.plot(range(0, df.shape[1] + 1), var)
        plt.xlabel('Factors')
        plt.ylabel('Var cum sum')
        plt.grid()

        plt.subplot(1, 2, 2)
        plt.plot(range(left, right + 1), var[left:(right+1)], markersize=3, marker='o')
        plt.plot(range(left, right + 1), var[left:(right+1)])
        plt.xlabel('Partial Factors')
        plt.ylabel('Var cum sum')
        plt.grid()


def program_done_sound(duration = 2000, frequency = 500):
    # duration为毫秒， frequency为频率
    winsound.Beep(frequency, duration)


def set_output_width(width = 300):
    pd.set_option('display.width', width)
    np.set_printoptions(linewidth=width)


def plot_confusion_image(X_train, y_train, y_train_pred, cl_a, cl_b):
    # 分析单个错误也可以帮助获得洞见
    # 通过画图分析35正确分类，与错误分类的图像，获得洞见
    X_aa = X_train[(y_train == cl_a) & (y_train_pred == cl_a)]
    X_ab = X_train[(y_train == cl_a) & (y_train_pred == cl_b)]
    X_ba = X_train[(y_train == cl_b) & (y_train_pred == cl_a)]
    X_bb = X_train[(y_train == cl_b) & (y_train_pred == cl_b)]

    plt.figure(figsize=(8, 8))

    plt.subplot(221)
    plot_digits(X_aa[:25], images_per_row=5)
    plt.title(f'True {cl_a}, Predict{cl_a}')

    plt.subplot(222)
    plot_digits(X_ab[:25], images_per_row=5)
    plt.title(f'True {cl_a}, Predict{cl_b}')

    plt.subplot(223)
    plot_digits(X_ba[:25], images_per_row=5)
    plt.title(f'True {cl_b}, Predict{cl_a}')

    plt.subplot(224)
    plot_digits(X_bb[:25], images_per_row=5)
    plt.title(f'True {cl_b}, Predict{cl_b}')


def plot_cm(cm, text_label=None, process_cm=False):
    if process_cm:
        # 获得分类错误率，而不是错误的绝对值
        row_sums = cm.sum(axis=1, keepdims=True)
        norm_conf_mx = cm / row_sums

        # 用0填充对角线，仅保留错误，重新画图
        np.fill_diagonal(norm_conf_mx, 0)
        # 行代表实际类，列代表预测类，第8列看起来很亮，说明许多图片被错误分类为8
        # PS：第8行不那么差，说明数字8被正确分类了，且注意到错误不完全对称
        # 说明精力可以花在改进8的错误上（如进一步搜集8的数据，或者添加特征写个算法统计闭环数量）
        cm = norm_conf_mx

    if text_label is not None:
        cm = pd.DataFrame(cm, columns=text_label, index=text_label)

    plt.figure()
    sns.heatmap(cm, annot=True, linewidths=0.5, fmt=".2f", cmap="YlGnBu")
    plt.xlabel('True class')
    plt.ylabel('Predict class')


def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--') # dashed diagonal
    plt.axis([0, 1, 0, 1])                                    # Not shown in the book
    plt.xlabel('False Positive Rate (Fall-Out)', fontsize=16) # Not shown
    plt.ylabel('True Positive Rate (Recall)', fontsize=16)    # Not shown
    plt.grid(True)


def find_proper_k(model, X, k_tuple, image_path=None, name=None):
    """
    找到k均值聚类合适的K
    """
    print(f'K均值寻找范围：{k_tuple}')
    visualizer = KElbowVisualizer(model, k=k_tuple)
    visualizer.fit(X)  # Fit the data to the visualizer

    if image_path is not None:
        # save_fig(image_path, name)
        visualizer.show(outpath=image_path+ '\\' + name, clear_figure=True)
    else:
        visualizer.show(clear_figure=True)

