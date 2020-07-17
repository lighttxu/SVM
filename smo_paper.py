# @Author  : lightXu
# @File    : smo_paper.py

import numpy as np
import matplotlib.pyplot as plt
import random
import copy


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
    # 避免浮点精度对影像支持向量
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


def select_j(eligible_list, i, ost, Ei):
    eligible_list = list(eligible_list)
    if i in eligible_list:
        eligible_list.remove(i)

    E_list = [cal_Ek(ost, k)[0] for k in eligible_list]
    if Ei < 0:
        value = max(E_list)
    elif Ei > 0:
        value = min(E_list)
    else:
        value = max([abs(cal_Ek(ost, k)[0]) for k in eligible_list])
    max_k = eligible_list[E_list.index(value)]

    # E_list1 = [(Ei - cal_Ek(ost, k)[0]) for k in eligible_list]
    # value1 = max(E_list1)
    # max_k1 = eligible_list[E_list1.index(value1)]
    # if max_k != max_k1:
    #     print('!=', i)

    Ej, _ = cal_Ek(ost, max_k)
    return max_k, Ej


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


def cal_w(data_x, label, alpha):
    w = np.dot((alpha * label).T, data_x)
    return w


def objective_func(ost, i1, i2, alpha1, alpha2, L, H):
    s = ost.label[i1] * ost.label[i2]
    k11 = np.dot(ost.X[i1, :], ost.X[i1, :])
    k12 = np.dot(ost.X[i1, :], ost.X[i2, :])
    k22 = np.dot(ost.X[i2, :], ost.X[i2, :])
    f1 = (ost.label[i1] * (cal_Ek(ost, i1) + ost.b) - alpha1 * k11 - s * alpha2 * k12)
    f2 = (ost.label[i2] * (cal_Ek(ost, i2) + ost.b) - s * alpha1 * k12 - alpha2 * k12)

    L1 = alpha1 + s * (alpha2 - L)
    H1 = alpha1 + s * (alpha2 - H)

    obj_L = L1 * f1 + L * f2 + 0.5 * L1 * L1 * k11 + 0.5 * L * L * k22 + s * L * L1 * k12
    obj_H = H1 * f1 + H * f2 + 0.5 * H1 * H1 * k11 + 0.5 * H * H * k22 + s * H * H1 * k12

    return obj_L, obj_H


def take_step(ost, i1, i2, E2):
    if i1 == i2:
        return 0
    alpha1 = ost.alpha[i1].copy()
    y1 = ost.label[i1]
    alpha2 = ost.alpha[i2].copy()
    y2 = ost.label[i2]

    E1, _ = cal_Ek(ost, i1)
    s = y1 * y2

    if ost.label[i1] != ost.label[i2]:
        L = max(0, ost.alpha[i2] - ost.alpha[i1])
        H = min(ost.C, ost.C + ost.alpha[i2] - ost.alpha[i1])
    else:
        L = max(0, ost.alpha[i2] + ost.alpha[i1] - ost.C)
        H = min(ost.C, ost.alpha[i2] + ost.alpha[i1])
    if L == H:
        # print("L==H")
        return 0

    eta = (np.dot(ost.X[i1, :], ost.X[i1, :])
           + np.dot(ost.X[i2, :], ost.X[i2, :])
           - 2 * np.dot(ost.X[i1, :], ost.X[i2, :]))
    eta = round_float(eta)
    if eta > 0:
        a2 = alpha2 + y2 * (E1 - E2) / eta
        a2 = a2
        if a2 < L:
            a2 = L
        if a2 > H:
            a2 = H

    else:
        Lobj, _ = objective_func(ost, i1, i2, alpha1, L, L, H)
        _, Hobj = objective_func(ost, i1, i2, alpha1, H, L, H)
        if Lobj < Hobj - 0.0001:
            a2 = L
        elif Lobj > Hobj + 0.001:
            a2 = H
        else:
            a2 = alpha2

    if abs(a2 - alpha2) < 0.0001:
        return 0

    a1 = alpha1 + s * (alpha2 - a2)

    # 7: 更新bi， bj
    b1 = (ost.b - E1
          - ost.label[i1] * (ost.alpha[i1] - alpha1) * np.dot(ost.X[i1, :], ost.X[i1, :])
          - ost.label[i2] * (ost.alpha[i2] - alpha2) * np.dot(ost.X[i1, :], ost.X[i2, :]))
    b2 = (ost.b - E2
          - ost.label[i1] * (ost.alpha[i1] - alpha1) * np.dot(ost.X[i1, :], ost.X[i2, :])
          - ost.label[i2] * (ost.alpha[i2] - alpha2) * np.dot(ost.X[i2, :], ost.X[i2, :]))

    if 0 < ost.alpha[i1] < ost.C:
        ost.b = b1
    elif 0 < ost.alpha[i2] < ost.C:
        ost.b = b2
    else:
        ost.b = (b1 + b2) / 2.0

    updateEk(ost, i1)
    updateEk(ost, i2)

    ost.alpha[i1] = round_float(a1)
    ost.alpha[i2] = round_float(a2)

    return 1


def violate_kkt(ost, alpha2, E2, fx2, y2):
    r2 = E2 * y2
    violate_cond1 = r2 < -ost.toler and alpha2 < ost.C
    violate_cond2 = r2 > ost.toler and alpha2 > 0

    violate12 = violate_cond1 or violate_cond2

    # 原始kkt
    # y2 * fx2 - 1 = y2*(fx2-y2) = y2*E2
    violate_cond3 = (not y2 * fx2 - 1 >= 0) and alpha2 == 0
    violate_cond4 = (not y2 * fx2 - 1 != 0) and 0 < alpha2 < ost.C
    violate_cond5 = (not y2 * fx2 - 1 <= 0) and alpha2 == ost.C

    # Notice that the KKT conditions are checked to be within ε of fulfillment.
    # 论文中引入了一个误差eps, 此时
    violate_cond3_ = (not y2 * fx2 - 1 >= -ost.toler) and alpha2 == 0
    violate_cond4_ = (not abs(y2 * fx2 - 1) <= ost.toler) and 0 < alpha2 < ost.C
    violate_cond5_ = (not y2 * fx2 - 1 <= ost.toler) and alpha2 == ost.C

    violate345 = violate_cond3_ or violate_cond4_ or violate_cond5_

    return violate345


def examine_example(ost, i2):
    y2 = ost.label[i2]
    alpha2 = ost.alpha[i2]
    E2, fx2 = cal_Ek(ost, i2)

    # 是非违反kkt条件
    cond = violate_kkt(ost, alpha2, E2, fx2, y2)
    if cond:
        non_0_non_C_alpha_list = np.where((ost.alpha != 0) & (ost.alpha != ost.C))[0]
        if (len(non_0_non_C_alpha_list)) > 1:
            i1, _ = select_j(non_0_non_C_alpha_list, i2, ost, E2)
            if take_step(ost, i1, i2, E2):
                return 1

        non_tmp = non_0_non_C_alpha_list.copy().tolist()
        while len(non_tmp) > 0:
            i1 = random.choice(non_tmp)
            if take_step(ost, i1, i2, E2):
                return 1
            else:
                non_tmp.remove(i1)

        tmp_list = list(range(0, ost.row))
        while len(tmp_list) > 0:
            i1 = random.choice(tmp_list)
            if take_step(ost, i1, i2, E2):
                return 1
            else:
                tmp_list.remove(i1)

    return 0


def main(dataMatIn, classLabels, C, toler, maxIter):
    ost = OptStruct(dataMatIn, classLabels, C, toler)
    iter_num = 0
    num_changed = 0
    examine_all = 1

    while (iter_num < maxIter) and num_changed > 0 or examine_all:
        num_changed = 0
        if examine_all:
            for i in range(ost.row):
                """
                The outer loop first iterates over the entire training set, 
                determining whether each example violates the KKT conditions (12). 
                """
                num_changed = num_changed + examine_example(ost, i)
                print("全样本遍历:第%d次迭代 样本:%d, alpha优化次数:%d" % (iter_num, i, num_changed))
            iter_num += 1

        else:
            """
            After one pass through the entire training set, the outer loop iterates over all examples whose
            Lagrange multipliers are neither 0 nor C (the non-bound examples). Again, each example is
            checked against the KKT conditions and violating examples are eligible for optimization. 
            """
            non_bound_index = np.where((0 < ost.alpha) & (ost.alpha < C))[0]
            for i in non_bound_index:
                num_changed = num_changed + examine_example(ost, i)
                print("非边界:第%d次迭代 样本:%d, alpha优化次数:%d" % (iter_num, i, num_changed))
            iter_num += 1
        if examine_all:
            examine_all = 0
        elif num_changed == 0:
            examine_all = 1

    return ost.b, ost.alpha


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
            print(i)
            x_max, x_min = data_x[i]
            plt.scatter([x_max], [x_min], s=150, c='none', alpha=0.7, linewidth=1.5, edgecolor='red')

    # plt.savefig("./fig/seed_{}.png".format(seed))
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
    b, alphas = main(dataMat, labelMat, 0.6, 0.0001, 100)
    w = cal_w(dataMat, labelMat, alphas)
    print(w, b)
    show_classifier(dataMat, labelMat, w, b, alphas, seed)
