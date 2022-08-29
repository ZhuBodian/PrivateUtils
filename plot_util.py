import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.decomposition import PCA
import os



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


def set_output_width(width = 300):
    pd.set_option('display.width', width)
    np.set_printoptions(linewidth=width)


def plot_bound(data, ratio=1.1):
    left, right = min(data[:, 0]), max(data[:, 0])
    bottom, top = min(data[:, 1]), max(data[:, 1])
    left, right, bottom, top = ratio * left, ratio * right, ratio * bottom, ratio * top

    return left, right, bottom, top


# 给定画布边界，计算决策面（暂用于2维2分类，且有些决策面不是函数，如一个圆，那时候可以考虑画散点图）
# predict_fun为预测函数，给定输入，输出标签
def cal_decision_curve(left, right, bottom, top, predict_fun, step=200):
    grid = np.meshgrid(np.linspace(left, right, step),
                       np.linspace(bottom, top, step))  # list of array, array为(step,step)
    grid = np.hstack((grid[0].reshape((step * step, 1)),
                      grid[1].reshape((step * step, 1))))  # 转为(step*step, 2)
    temp = predict_fun(grid).reshape(step, step)  # 如果不reshape，直接差分，图像最右边会有一道线
    gap = temp.max() - temp.min()
    idxs = np.vstack((np.diff(temp, axis=0), np.zeros((1, step), dtype=np.int8)))
    idxs = idxs.reshape(step * step)
    return grid[np.where((idxs == gap) + (idxs == -gap))]


# predict_fun为预测函数，给定输入，输出标签
def cal_decision_area(left, right, bottom, top, predict_fun, step=300):
    grid = np.meshgrid(np.linspace(left, right, step),
                       np.linspace(bottom, top, step))  # list of array, array为(step,step)
    grid = np.hstack((grid[0].reshape((step * step, 1)),
                      grid[1].reshape((step * step, 1))))  # 转为(step*step, 2)
    return grid, predict_fun(grid)


def save_fig(path, fig_id, tight_layout=True, fig_extension="png", resolution=300):
    os.makedirs(path, exist_ok=True)
    path = path + fig_id + "." + fig_extension
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)