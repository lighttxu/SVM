# @Author  : lightXu
# @File    : svm_platt_array.py

import numpy as np
import matplotlib.pyplot as plt
import random


class OptStruct:
    """
    数据结构，维护所有需要操作的值
    Parameters：
        dataMatIn - 数据矩阵
        classLabels - 数据标签
        C - 松弛变量
        toler - 容错率
    """

    def __init__(self, data_x, label, C, toler):
        self.X = data_x
        self.label = label
        self.C = C
        self.toler = toler
        self.row = data_x.shape[0]
        self.alpha = np.zeros(self.row)
        self.b = 0
        self.e_cache = np.zeros(self.row)
        # self.e_cache = label * (-1)


def cal_Ek(ost, k):
    """
    计算误差
    Parameters：
        ost - 数据结构
        k - 标号为k的数据
    Returns:
        Ek - 标号为k的数据误差
    """
    fxk = np.dot((ost.alpha * ost.label).T, np.dot(ost.X, ost.X[k, :])) + ost.b
    Ek = fxk - ost.label[k]
    return round_float(Ek), round_float(fxk)


def round_float(value):
    return round(value, 15)


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


def select_j(i, ost, Ei):
    """
    内循环启发方式2
    Parameters：
        i - 标号为i的数据的索引值
        oS - 数据结构
        Ei - 标号为i的数据误差
    Returns:
        j, maxK - 标号为j或maxK的数据的索引值
        Ej - 标号为j的数据误差
    """
    maxK = 0
    maxDeltaE = 0
    Ej = 0
    # ost.e_cache[i] = Ei

    # valid_ecache_list = np.nonzero(ost.e_cache)[0]
    valid_ecache_list = np.nonzero(ost.alpha)[0]
    if (len(valid_ecache_list)) > 1:
        for k in valid_ecache_list:
            if k == i:
                continue
            Ek, _ = cal_Ek(ost, k)
            deltaE = abs(Ei - Ek)
            if deltaE > maxDeltaE:
                maxK = k
                maxDeltaE = deltaE
                Ej = Ek

        return maxK, Ej
    else:
        j = select_j_random(i, ost.row)
        Ej, _ = cal_Ek(ost, j)
        return j, Ej


def updateEk(ost, k):
    """
    计算Ek,并更新误差缓存
    Parameters：
        oS - 数据结构
        k - 标号为k的数据的索引值
    Returns:
    """
    Ek, _ = cal_Ek(ost, k)
    ost.e_cache[k] = Ek


def clip_alpha(alpha, L, H):
    if alpha > H:
        alpha = H
    if alpha < L:
        alpha = L

    return alpha


def innerL(i, ost, Ei):
    """
    优化的SMO算法
    Parameters：
        i - 标号为i的数据的索引值
        oS - 数据结构
    Returns:
        1 - 有任意一对alpha值发生变化
        0 - 没有任意一对alpha值发生变化或变化太小
    """
    # 1: 计算误差Ei


    # cond3 = ost.label[i] * fxi < 1 and ost.alpha[i] == 0
    # cond4 = ost.label[i] * fxi != 1 and 0 < ost.alpha[i] < ost.C
    # cond5 = ost.label[i] * fxi > 1 and ost.alpha[i] == ost.C
    # if cond3 or cond4 or cond5:
    # 使用内循环启发方式2选择alpha_j,并计算Ej
    j, Ej = select_j(i, ost, Ei)
    print('i, j, Ej', i, j, Ej)
    alpha_i_old = ost.alpha[i].copy()
    alpha_j_old = ost.alpha[j].copy()

    # 2：计算上下界L和H
    if ost.label[i] != ost.label[j]:
        L = max(0, ost.alpha[j] - ost.alpha[i])
        H = min(ost.C, ost.C + ost.alpha[j] - ost.alpha[i])
    else:
        L = max(0, ost.alpha[j] + ost.alpha[i] - ost.C)
        H = min(ost.C, ost.alpha[j] + ost.alpha[i])
    if L == H:
        # print("L==H")
        return 0

    # 3: 计算 eta
    eta = (np.dot(ost.X[i, :], ost.X[i, :])
           + np.dot(ost.X[j, :], ost.X[j, :])
           - 2 * np.dot(ost.X[i, :], ost.X[j, :]))
    eta = round_float(eta)
    if eta <= 0:
        # print("eta<=0")
        return 0

    # 4：更新alpha j
    alpha_j_tmp = round_float(ost.label[j] * (Ei - Ej) / eta)
    ost.alpha[j] = alpha_j_old + alpha_j_tmp
    # 5: 修剪alpha_j
    ost.alpha[j] = clip_alpha(ost.alpha[j], L, H)

    if abs(ost.alpha[j] - alpha_j_old) < 0.1e-6:
        print("alpha_j变化太小")
        return 0

    # 6: 更新 alpha i
    alpha_i_tmp = round_float(ost.label[i] * ost.label[j] * (alpha_j_old - ost.alpha[j]))
    ost.alpha[i] = alpha_i_old + alpha_i_tmp

    # 7: 更新bi， bj
    bi = (ost.b - Ei
          - ost.label[i] * (ost.alpha[i] - alpha_i_old) * np.dot(ost.X[i, :], ost.X[i, :])
          - ost.label[j] * (ost.alpha[j] - alpha_j_old) * np.dot(ost.X[i, :], ost.X[j, :]))
    bj = (ost.b - Ej
          - ost.label[i] * (ost.alpha[i] - alpha_i_old) * np.dot(ost.X[i, :], ost.X[j, :])
          - ost.label[j] * (ost.alpha[j] - alpha_j_old) * np.dot(ost.X[j, :], ost.X[j, :]))

    if 0 < ost.alpha[i] < ost.C:
        ost.b = bi
    elif 0 < ost.alpha[j] < ost.C:
        ost.b = bj
    else:
        ost.b = (bi + bj) / 2.0

    ost.b = round_float(float(ost.b))
    # 更新Ej至误差缓存
    updateEk(ost, j)
    updateEk(ost, i)

    return 1
    # else:
    #     return 0


def smo_platt(dataMatIn, classLabels, C, toler, maxIter):
    ost = OptStruct(dataMatIn, classLabels, C, toler)
    iter_num = 0
    entireSet = True
    alphaPairsChanged = 0

    while (iter_num < maxIter) and alphaPairsChanged > 0 or entireSet:
        alphaPairsChanged = 0
        if entireSet:
            for i in range(ost.row):
                """
                The outer loop first iterates over the entire training set, 
                determining whether each example violates the KKT conditions (12). 
                """
                Ei, fxi = cal_Ek(ost, i)
                cond1 = ((ost.label[i] * Ei < -ost.toler) and (ost.alpha[i] < ost.C))
                cond2 = ((ost.label[i] * Ei > ost.toler) and (ost.alpha[i] > 0))
                if cond1 or cond2:
                    alphaPairsChanged = alphaPairsChanged + innerL(i, ost, Ei)
                else:
                    pass
                print("全样本遍历:第%d次迭代 样本:%d, alpha优化次数:%d" % (iter_num, i, alphaPairsChanged))
            iter_num += 1
        else:
            """
            After one pass through the entire training set, the outer loop iterates over all examples whose
            Lagrange multipliers are neither 0 nor C (the non-bound examples). Again, each example is
            checked against the KKT conditions and violating examples are eligible for optimization. 
            """
            # nonBoundIs = np.nonzero((ost.alpha > 0) * (ost.alpha < C))[0]
            nonBoundIs = np.where((0 < ost.alpha) & (ost.alpha < C))[0]
            for i in nonBoundIs:
                Ei, fxi = cal_Ek(ost, i)
                cond1 = ((ost.label[i] * Ei < -ost.toler) and (ost.alpha[i] < ost.C))
                cond2 = ((ost.label[i] * Ei > ost.toler) and (ost.alpha[i] > 0))
                if cond1 or cond2:
                    alphaPairsChanged = alphaPairsChanged + innerL(i, ost, Ei)
                else:
                    pass
                print("非边界遍历:第%d次迭代 样本:%d, alpha优化次数:%d" % (iter_num, i, alphaPairsChanged))
            iter_num += 1

        if entireSet:
            entireSet = False
        elif alphaPairsChanged == 0:
            entireSet = True
        print("迭代次数: %d" % iter_num)
    return ost.b, ost.alpha


def cal_w(data_x, label, alpha):
    w = np.dot((alpha * label).T, data_x)
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

    plt.savefig("./fig/array_seed_{}.png".format(seed))
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

    seed = 10
    random.seed(seed)
    dataMat, labelMat = load_data('testSet.txt')
    b, alphas = smo_platt(dataMat, labelMat, 0.6, 0.001, 100)
    w = cal_w(dataMat, labelMat, alphas)
    print(w, b)
    show_classifier(dataMat, labelMat, w, b, alphas, seed)
