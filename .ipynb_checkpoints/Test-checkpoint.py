import numpy as np
from collections import defaultdict

class NaiveBayesClassifier:
    def __init__(self):
        """初始化分类器参数"""
        self.prior = {}                # 存储每个类别的先验概率
        self.conditional = defaultdict(dict)  # 存储每个特征在每个类别下的条件概率
        self.classes = None            # 存储所有类别标签

    def fit(self, X, y):
        """训练朴素贝叶斯分类器"""
        n_samples, n_features = X.shape
        self.classes = np.unique(y)

        # 计算每个类别的先验概率 P(y)
        for c in self.classes:
            self.prior[c] = np.sum(y == c) / n_samples

        # 计算每个特征在每个类别下的条件概率 P(x_i|y)
        for c in self.classes:
            X_c = X[y == c]  # 属于类别c的所有样本
            for i in range(n_features):
                # 计算特征i在类别c下的每个可能取值的出现频率
                values, counts = np.unique(X_c[:, i], return_counts=True)
                for v, cnt in zip(values, counts):
                    self.conditional[c][(i, v)] = cnt / len(X_c)

        return self

    def predict(self, X):
        """对输入样本进行分类预测"""
        return [self._predict_one(x) for x in X]

    def _predict_one(self, x):
        """预测单个样本的类别"""
        # 计算每个类别下的后验概率的对数（避免数值下溢）
        posteriors = {c: np.log(self.prior[c]) for c in self.classes}

        # 累加每个特征的条件概率对数
        for c in self.classes:
            for i, xi in enumerate(x):
                # 如果特征值在训练集中出现过，则使用其条件概率
                # 否则假设其概率为一个很小的值（拉普拉斯平滑）
                if (i, xi) in self.conditional[c]:
                    posteriors[c] += np.log(self.conditional[c][(i, xi)])
                else:
                    posteriors[c] += np.log(1e-10)

        # 返回后验概率最大的类别
        return max(posteriors, key=posteriors.get)


# 简单测试示例
if __name__ == "__main__":
    # 创建一个简单的数据集（天气、温度、是否打球）
    X = np.array([
        ['晴', '热', '是'],
        ['晴', '热', '是'],
        ['阴', '热', '是'],
        ['雨', '适中', '是'],
        ['雨', '冷', '否'],
        ['阴', '冷', '否'],
        ['晴', '适中', '是'],
        ['晴', '冷', '否'],
        ['雨', '适中', '否'],
        ['晴', '适中', '是'],
        ['阴', '适中', '是'],
        ['阴', '热', '是'],
        ['雨', '适中', '否']
    ])

    # 特征列和目标列
    X_train = X[:, :-1]  # 前两列是特征（天气、温度）
    y_train = X[:, -1]   # 最后一列是目标（是否打球）

    # 训练模型
    clf = NaiveBayesClassifier()
    clf.fit(X_train, y_train)

    # 预测新样本
    X_test = np.array([
        ['晴', '适中'],
        ['雨', '冷']
    ])

    predictions = clf.predict(X_test)
    print("预测结果:", predictions)  # 输出应为 ['是', '否']