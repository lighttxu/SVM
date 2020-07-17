# @Author  : lightXu
# @File    : svm_classify.py
import numpy as np
import matplotlib.pyplot as plt
import random


def round_float(value):
    return round(value, 8)


def load_data(file_name):
    data_x = []
    data_y = []

    with open(file_name, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split('\t')
            xi = line[:-1]
            data_x.append(xi)
            data_y.append(line[-1])

    data_x = np.array(data_x, dtype=np.float)
    label = np.array(data_y, dtype=np.float)

    return data_x, label


def select_j_random(i, m):
    j = i
    while j == i:
        j = int(random.uniform(0, m))

    return j


def clip_alpha(alpha, L, H):
    if alpha > H:
        alpha = H
    if alpha < L:
        alpha = L

    return alpha


def smo_simple(data_x, label, C, toler, max_iter):
    """
    :param data_x:数据矩阵
    :param label:数据标签
    :param C:松弛变量
    :param toler:容错率  1 - epsilon
    :param max_iter:最大迭代次数
    :return:
    """

    # data_x = np.array(data_x, dtype=np.float)
    # label = np.array(label, dtype=np.float)
    row, col = data_x.shape[0], data_x.shape[1]
    alpha = np.zeros(row)

    b = 0

    iter_num = 0

    while iter_num < max_iter:
        alpha_pairs_changed = 0
        for i in range(row):
            # i = 1
            fxi = np.dot((alpha * label).T, np.dot(data_x, data_x[i, :])) + b

            Ei = fxi - label[i]
            # 优化alpha，更设定一定的容错率。
            cond1 = (label[i] * Ei < -toler) and (alpha[i] < C)
            cond2 = (label[i] * Ei > toler) and (alpha[i] > 0)
            # if cond1 or cond2:

            # 统计学习方法  P147
            cond3 = label[i] * fxi < 1 and alpha[i] == 0
            cond4 = label[i] * fxi != 1 and 0 < alpha[i] < C
            cond5 = label[i] * fxi > 1 and alpha[i] == C
            if cond3 or cond4 or cond5:
                j = select_j_random(i, row)
                # j = 51
                # 1 计算误差Ej
                fxj = np.dot((alpha * label).T, np.dot(data_x, data_x[j, :])) + b
                Ej = fxj - label[j]
                alpha_i_old = alpha[i].copy()
                alpha_j_old = alpha[j].copy()
                # 2：计算上下界L和H
                if label[i] != label[j]:
                    L = max(0, alpha[j] - alpha[i])
                    H = min(C, C + alpha[j] - alpha[i])
                else:
                    L = max(0, alpha[j] + alpha[i] - C)
                    H = min(C, alpha[j] + alpha[i])
                if L == H:
                    # print("L==H")
                    continue

                # 3: 计算 eta = k11+k22-2k12
                eta = (np.dot(data_x[i, :], data_x[i, :])
                       + np.dot(data_x[j, :], data_x[j, :])
                       - 2 * np.dot(data_x[i, :], data_x[j, :]))
                eta = round_float(eta)
                if eta <= 0:
                    # print("eta<=0")
                    continue
                # 4：更新alpha_j
                alpha_j_tmp = round_float(label[j] * (Ei - Ej) / eta)
                alpha[j] = alpha_j_old + alpha_j_tmp
                # 5：修剪alpha_j
                alpha[j] = clip_alpha(alpha[j], L, H)

                if abs(alpha[j] - alpha_j_old) < 0.00001:
                    # print("alpha_j变化太小")
                    continue

                # 6：更新alpha_i
                alpha_i_tmp = round_float(label[i] * label[j] * (alpha_j_old - alpha[j]))

                alpha_tmp = alpha_i_old + alpha_i_tmp
                # if 0 < alpha_tmp < 0.1e-10:
                #     alpha_tmp = 0.0

                alpha[i] = alpha_tmp
                # 7:更新b
                bi = (b - Ei
                      - label[i] * (alpha[i] - alpha_i_old) * np.dot(data_x[i, :], data_x[i, :])
                      - label[j] * (alpha[j] - alpha_j_old) * np.dot(data_x[i, :], data_x[j, :]))
                bj = (b - Ej
                      - label[i] * (alpha[i] - alpha_i_old) * np.dot(data_x[i, :], data_x[j, :])
                      - label[j] * (alpha[j] - alpha_j_old) * np.dot(data_x[j, :], data_x[j, :]))

                if 0 < alpha[i] < C:
                    b = bi
                elif 0 < alpha[j] < C:
                    b = bj
                else:
                    b = (bi + bj) / 2.0
                # 统计优化次数
                print("i, j, b: ", i, j, alpha[i], alpha[j], b)
                alpha_pairs_changed += 1
                # 打印统计信息
                print("第%d次迭代 样本:%d, alpha优化次数:%d" % (iter_num, i, alpha_pairs_changed))
        if alpha_pairs_changed == 0:
            iter_num += 1
        else:
            iter_num = 0
        print("迭代次数: %d" % iter_num)
    return b, alpha


def cal_w(data_x, label, alpha):
    w = np.dot((alpha*label).T, data_x)
    return w


def show_classifier(data_x, label, w, b, alpha, seed):
    positive_index = np.where(label == 1)[0]
    negative_index = np.where(label == -1)[0]
    data_x_positive = data_x[positive_index]
    data_x_negative = data_x[negative_index]

    plt.scatter(data_x_positive[:, 0], data_x_positive[:, 1],
                s=30, alpha=0.7, c='green')  # 正样本散点图
    plt.scatter(data_x_negative[:, 0], data_x_negative[:, 1],
                s=30, alpha=0.7, c='pink')  # 负样本散点图

    x_max = np.max(data_x, axis=0)[0]
    x_min = np.min(data_x, axis=0)[0]
    a1, a2 = w
    b = float(b)
    y1, y2 = (-b - a1 * x_max) / a2, (-b - a1 * x_min) / a2
    plt.plot([x_max, x_min], [y1, y2])

    # 找出支持向量点
    for i, alp in enumerate(alpha):
        if abs(alp) > 0:
            x_max, x_min = data_x[i]
            plt.scatter([x_max], [x_min], s=150, c='none', alpha=0.7, linewidth=1.5, edgecolor='red')

    # plt.savefig("seed_{}.png".format(seed))
    plt.show()
    plt.close()


def show_classifier1(data_x, label):
    positive_index = np.where(label == 1)[0]
    negative_index = np.where(label == -1)[0]
    data_x_positive = data_x[positive_index]
    data_x_negative = data_x[negative_index]

    plt.scatter(data_x_positive[:, 0], data_x_positive[:, 1],
                s=30, alpha=0.7, c='green')  # 正样本散点图
    plt.scatter(data_x_negative[:, 0], data_x_negative[:, 1],
                s=30, alpha=0.7, c='pink')  # 负样本散点图

    plt.show()


if __name__ == '__main__':
    seed = random.randint(1, 2000)
    random.seed(seed)
    dataMat, labelMat = load_data('testSet.txt')
    b, alphas = smo_simple(dataMat, labelMat, 0.6, 0.001, 100)
    w = cal_w(dataMat, labelMat, alphas)
    print(w, b)
    show_classifier(dataMat, labelMat, w, b, alphas, seed)


