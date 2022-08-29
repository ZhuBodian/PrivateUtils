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
from math import sqrt, exp


mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

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


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def program_done_sound(duration = 2000, frequency = 500):
    # duration为毫秒， frequency为频率
    winsound.Beep(frequency, duration)


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


def euc_dist(x1, x2):
    # 方便广播机制
    # if x1.ndim == 1:
    #     x1 = x1[:, np.newaxis]
    # if x2.ndim == 1:
    #     x2 = x2[:, np.newaxis]
    # if x1.shape[1] != x2.shape[1]:
    #     x2 = x2.T

    return sqrt(np.sum((x1 - x2) * (x1 - x2), axis=1))


# 多分类，positive_label对应阳，其它均为阴。其实sklearn.metrics有accuracy_score, classification_report
def classifier_effect(real_label, cal_label, positive_label):
    TP = sum((real_label == positive_label) * (cal_label == positive_label))
    FN = sum((real_label == positive_label) * (cal_label != positive_label))
    FP = sum((real_label != positive_label) * (cal_label == positive_label))
    TN = sum((real_label != positive_label) * (cal_label != positive_label))
    Sn = TP / (TP + FN)
    Sp = TN / (TN + FP)
    Acc = (TP + TN) / (TP + TN + FP + FN)
    Rec = TP / (TP + FN)
    Pre = TP / (TP + FP)
    F = 2 * Rec * Pre / (Rec + Pre)
    return Sn, Sp, Acc, Rec, Pre, F


# 防止下溢的sigmoid函数，（处理多维数组）
def sigmoid(data):
    # lambda a: 1 / (1 + exp(-a)) if a>=0 else lambda a: exp(a) / (1 + exp(a))
    idxs = np.where(data >= 0)
    data[idxs] = 1 / (1 + exp(-data[idxs]))
    idxs = np.where(data < 0)
    data[idxs] = exp(data[idxs]) / (1 + exp(data[idxs]))  # 如果x<0，还是exp(-x)，当x绝对值很大时，如10000，那么exp（10000）溢出
    return data


# 线性核直接返回两个向量的相乘
def linear_kernal(x1, x2):
    return x1.dot(x2)


# rbf核
def rbf_kernal(x1, x2, sigma_2=256 * 0.3):
    # 为了方便使用广播机制
    if x1.ndim == 1:
        x1 = x1[:, np.newaxis]
    if x2.ndim == 1:
        x2 = x2[:, np.newaxis]
    if x1.shape[1] != x2.shape[1]:
        x2 = x2.T

    return exp(euc_dist(x1, x2) ** 2 / sigma_2)[:, np.newaxis]


# 根据指定概率生成数字
def number_of_certain_probability(sequence, probability):
    x = np.random.uniform(0, 1)
    cumulative_probability = 0.0
    for item, item_probability in zip(sequence, probability):
        cumulative_probability += item_probability
        if x < cumulative_probability:
            break
    return item

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

class MyTimer:
    def __init__(self, timer_reason):
        self._start_timer = time.time()
        self._stop_timer = time.time()
        self._timer_reason = timer_reason

    def stop(self):
        self._stop_timer = time.time()
        print(f'"{self._timer_reason}" cost time: {self._stop_timer - self._start_timer}s')