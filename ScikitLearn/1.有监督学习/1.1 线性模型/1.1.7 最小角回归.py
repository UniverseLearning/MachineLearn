import numpy as np
from sklearn.datasets import make_regression

def lars(X, y, max_iter=100):
    n_samples, n_features = X.shape
    coef = np.zeros(n_features)  # 初始化系数
    active_set = []  # 活跃特征集
    residuals = y.copy()  # 初始化残差

    for _ in range(min(max_iter, n_features)):
        # 计算当前残差与各特征的相关性
        correlations = np.dot(X.T, residuals)

        # 找出未在活跃集中且相关性最大的特征
        inactive_indices = [i for i in range(n_features) if i not in active_set]
        if not inactive_indices:
            break

        j = inactive_indices[np.argmax(np.abs(correlations[inactive_indices]))]
        active_set.append(j)

        # 计算等角方向（角平分线）
        X_active = X[:, active_set]
        G_active = np.dot(X_active.T, X_active)
        G_inv = np.linalg.inv(G_active)
        signs = np.sign(correlations[active_set])
        equiangular = np.dot(X_active, np.dot(G_inv, signs))

        # 计算步长（沿着等角方向前进多远）
        step = np.inf
        for i in range(n_features):
            if i in active_set:
                continue

            a = np.dot(X[:, i], equiangular)
            c = correlations[i]

            # 计算可能的步长
            step1 = (correlations[j] - c) / (a - correlations[j])
            step2 = (correlations[j] + c) / (a + correlations[j])

            # 选择有效的正步长
            valid_steps = [s for s in [step1, step2] if s > 0]
            if valid_steps:
                step = min(step, min(valid_steps))

        # 更新系数和残差
        coef[active_set] += step * signs
        residuals = y - np.dot(X, coef)

    return coef

if __name__ == '__main__':
    # 示例数据
    # 生成多任务数据（前10个特征对所有任务有用，其余特征随机）
    n_samples,n_features,n_tasks = 1000, 20, 1

    X, Y = make_regression(
        n_samples= n_samples,
        n_features=n_features,
        n_informative=10,
        n_targets=n_tasks,
        noise=20,
        random_state=49
    )
    # 运行LARS算法
    coef = lars(X, Y)
    print("Estimated coefficients:", coef)