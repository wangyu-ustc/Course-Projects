import os
import numpy as np
import pandas as pd
import argparse
import matplotlib.pyplot as plt



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="../data/", help="data directory")
    return parser.parse_args()


def load_data(data_dir):
    x_train = np.load(os.path.join(data_dir, "train_data.npy"))
    y_train = np.load(os.path.join(data_dir, "train_target.npy"))
    x_test = np.load(os.path.join(data_dir, "test_data.npy"))
    y_test = np.load(os.path.join(data_dir, "test_target.npy"))
    return x_train, y_train, x_test, y_test


# Tree结点类型：回归树
def regLeaf(dataSet):  # 生成叶结点，在回归树中是目标变量特征的均值
    return np.mean(dataSet[:, -1])

# 误差计算函数：回归误差
def regErr(dataSet):  # 计算目标的平方误差（均方误差*总样本数）
    return np.var(dataSet[:, -1]) * np.shape(dataSet)[0]

def sigmoid(x):
    return 1/(1+np.exp(-x))

def Loss_function(pred, y, loss_type='mse'):
    return np.mean((y-pred) ** 2) / 2 if loss_type == 'mse' else - np.mean(y * np.log(pred + 1e-5) + (1-y) * np.log(1-pred + 1e-5))

def calculate_g(pred, y):
    return (pred - y)

def calculate_h(pred, y, loss_type):
    return np.array([1] * len(y)) if loss_type == 'mse' else (pred * (1 - pred))

class Xgboost():
    def __init__(self, leaf_lambda=0.2, epsilon=0.01, threshold=0.01,
                 gamma=1, max_depth=10, max_iter=10, loss_type='log'):
        self.leaf_lambda = leaf_lambda
        self.threshold = threshold
        self.max_depth = max_depth
        self.loss_type = loss_type
        self.max_iter = max_iter
        self.epsilon = epsilon
        self.gamma = gamma
        self.trees = []

    def fit(self, x_train, y_train):
        self.trees.append(0)
        predictions = self.predict(x_train)
        self.g = calculate_g(predictions, y_train)
        self.h = calculate_h(predictions, y_train, self.loss_type)
        loss = Loss_function(predictions, y_train)
        count = 1
        while loss > self.epsilon:
            if count > self.max_iter:
                break
            new_tree, depth = self.createTree(x_train, y_train - predictions, self.g, self.h, 0)
            self.trees.append(new_tree)
            predictions = self.predict(x_train)
            self.g = calculate_g(predictions, y_train)
            self.h = calculate_h(predictions, y_train, self.loss_type)
            loss = Loss_function(predictions, y_train, self.loss_type)
            print("generate a new tree with depth %d, loss down to %.2f" % (depth, loss))
            count += 1
        # self.tree = self.createTree(np.concatenate([x_train, y_train.reshape(-1, 1)], axis=1))

    def predict(self, x_test):
        prediction = []
        for x in x_test:
            prediction.append(self.predict_single(x))
        return np.array(prediction)

    def predict_single(self, x):
        prediction = 0
        for tree in self.trees:
            while isinstance(tree, dict):
                if x[tree['spInd']] < tree['spVal']:
                    tree = tree['left']
                else:
                    tree = tree['right']
            # Now 'tree' is a float
            prediction += tree
        if self.loss_type != 'mse':
            prediction = sigmoid(prediction)
        return prediction

    # 切分数据集为两个子集
    def binSplitDataSet(self, dataSet, feature, value):  # 数据集 待切分特征 特征值
        indexes0 = np.where(dataSet[:, feature] <= value)[0]
        indexes1 = np.where(dataSet[:, feature] > value)[0]
        # 下面原书代码报错 index 0 is out of bounds,使用上面两行代码
        # mat0 = dataSet[nonzero(dataSet[:, feature] > value)[0], :][0]
        # mat1 = dataSet[nonzero(dataSet[:, feature] <= value)[0], :][0]
        return indexes0, indexes1

    # 二元切分
    def chooseBestSplit(self, x_train, y_train, g, h, depth):

        m, n = x_train.shape
        bestGain = 0
        bestIndex = 0
        bestValue = 0
        G, H = np.sum(g), np.sum(h)

        # 若所有特征值都相同，停止切分
        if len(np.unique(y_train)) == 1:  # 倒数第一列转化成list 不重复
            return None, - G/(H + self.leaf_lambda)  # 如果剩余特征数为1，停止切分1

        for featIndex in range(n):
            for i, splitVal in enumerate(np.sort(np.unique(x_train[:, featIndex]))):
                indexes0, indexes1 = self.binSplitDataSet(x_train, featIndex, splitVal)
                G_L = np.sum(g[indexes0])
                G_R = np.sum(g[indexes1])
                H_L = np.sum(h[indexes0])
                H_R = np.sum(h[indexes1])
                Gain = (G_L**2/(H_L + self.leaf_lambda) + G_R**2/(H_R + self.leaf_lambda)
                    - G**2 / (H + self.leaf_lambda))/2 - self.gamma

                if Gain > bestGain:
                    bestGain = Gain
                    bestIndex = featIndex
                    bestValue = splitVal

        # 如果切分后误差效果下降不大，则取消切分，直接创建叶结点
        if bestGain < self.threshold:
            return None, - G/(H + self.leaf_lambda)  # 停止切分2
        return bestIndex, bestValue  # 返回特征编号和用于切分的特征值

    # 构建tree
    def createTree(self, x_train, y_train, g, h, depth):
        if depth >= self.max_depth:
            return np.mean(y_train), depth

        feat, val = self.chooseBestSplit(x_train, y_train, g, h, depth)
        if feat == None: return val, depth  # 满足停止条件时返回叶结点值
        # 切分后赋值
        retTree = {}
        retTree['spInd'] = feat
        retTree['spVal'] = val
        retTree['depth'] = depth
        # 切分后的左右子树
        lSet, rSet = self.binSplitDataSet(x_train, feat, val)
        retTree['left'], left_max_depth = self.createTree(x_train[lSet], y_train[lSet], g[lSet], h[lSet], depth + 1)
        retTree['right'], right_max_depth = self.createTree(x_train[rSet], y_train[rSet], g[rSet] ,h[rSet], depth + 1)
        return retTree, max(left_max_depth, right_max_depth)

if __name__ == "__main__":
    args = parse_args()
    x_train, y_train, x_test, y_test = load_data(args.data_dir)

    ### Using mse loss
    xgboost = Xgboost(leaf_lambda=0.2, epsilon=0.01, threshold=0.01,
                 gamma=1, max_depth=3, max_iter=3, loss_type='mse')
    xgboost.fit(x_train, y_train)

    predictions = xgboost.predict(x_train)
    print("MSE Train: prediction accuracy: %.4f" % (np.sum((predictions > 0.5) == y_train) / float(len(y_train))))

    predictions = xgboost.predict(x_test)
    print("MSE Test: prediction accuracy: %.4f" % (np.sum((predictions > 0.5) == y_test) / float(len(y_test))))