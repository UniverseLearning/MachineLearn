import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # 或者 'QtAgg'
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

# 设置中文字体
plt.rcParams["font.family"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

class LinearRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        """初始化线性回归模型"""
        self.learning_rate = learning_rate  # 学习率
        self.n_iterations = n_iterations  # 迭代次数
        self.weights = None  # 权重参数
        self.bias = None  # 偏置参数
        self.history = {"loss": [], "weights": [], "bias": []}  # 训练历史记录

    def fit(self, X, y):
        """训练线性回归模型"""
        n_samples, n_features = X.shape

        # 初始化参数
        self.weights = np.zeros((n_features, 1))
        self.bias = 0

        # 梯度下降迭代
        for iteration in range(self.n_iterations):
            # 计算预测值
            y_pred = np.dot(X, self.weights) + self.bias

            # 计算损失 (均方误差)
            loss = np.mean((y_pred - y) ** 2) / 2

            # 记录训练历史
            self.history["loss"].append(loss)
            self.history["weights"].append(self.weights.copy())
            self.history["bias"].append(self.bias)

            # 计算梯度
            dw = np.dot(X.T, (y_pred - y)) / n_samples
            db = np.sum(y_pred - y) / n_samples

            # 更新参数
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            # 每100次迭代打印一次损失
            if iteration % 100 == 0:
                print(f"迭代 {iteration}, 损失: {loss:.4f}")

        return self

    def predict(self, X):
        """使用模型进行预测"""
        return np.dot(X, self.weights) + self.bias

    def plot_loss_curve(self):
        """绘制损失曲线"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.history["loss"], 'b-', linewidth=2)
        plt.title('训练损失随迭代次数变化', fontsize=14)
        plt.xlabel('迭代次数', fontsize=12)
        plt.ylabel('均方误差损失', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.show()

    def visualize_fit(self, X, y, feature_indices=[0, 1], animate=False):
        """可视化模型拟合结果
        feature_indices: 选择要可视化的两个特征的索引
        """
        if X.shape[1] < 2:
            print("需要至少两个特征进行可视化")
            return

        # 提取指定的两个特征
        X_vis = X[:, feature_indices]

        if animate:
            # 创建3D动画
            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111, projection='3d')

            # 绘制数据点
            scatter = ax.scatter(X_vis[:, 0], X_vis[:, 1], y, c='blue', alpha=0.7, label='数据点')

            # 初始化平面
            x_min, x_max = X_vis[:, 0].min(), X_vis[:, 0].max()
            y_min, y_max = X_vis[:, 1].min(), X_vis[:, 1].max()
            xx, yy = np.meshgrid(np.linspace(x_min, x_max, 20),
                                 np.linspace(y_min, y_max, 20))
            zz = np.zeros_like(xx)
            plane = ax.plot_surface(xx, yy, zz, alpha=0.5, color='r')

            # 设置坐标轴标签
            feature_names = ["房屋面积", "卧室数量", "房龄", "卫生间数量"]
            ax.set_title('多元线性回归模型拟合过程', fontsize=14)
            ax.set_xlabel(feature_names[feature_indices[0]], fontsize=12)
            ax.set_ylabel(feature_names[feature_indices[1]], fontsize=12)
            ax.set_zlabel('房价 (万元)', fontsize=12)
            ax.legend()

            def init():
                plane.remove()
                plane = ax.plot_surface(xx, yy, zz, alpha=0.5, color='r')
                return plane,

            def update(frame):
                w1 = self.history["weights"][frame][feature_indices[0]][0]
                w2 = self.history["weights"][frame][feature_indices[1]][0]
                b = self.history["bias"][frame]

                # 计算平面 z = w1*x + w2*y + b
                plane.remove()
                zz = w1 * xx + w2 * yy + b
                plane = ax.plot_surface(xx, yy, zz, alpha=0.5, color='r')
                ax.set_title(f'多元线性回归模型拟合过程 (迭代 {frame})')
                return plane,

            ani = FuncAnimation(fig, update, frames=len(self.history["weights"]),
                                init_func=init, interval=100, blit=False)
            plt.show()
            return ani
        else:
            # 静态3D可视化
            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111, projection='3d')

            # 绘制数据点
            ax.scatter(X_vis[:, 0], X_vis[:, 1], y, c='blue', alpha=0.7, label='数据点')

            # 绘制拟合平面
            x_min, x_max = X_vis[:, 0].min(), X_vis[:, 0].max()
            y_min, y_max = X_vis[:, 1].min(), X_vis[:, 1].max()
            xx, yy = np.meshgrid(np.linspace(x_min, x_max, 20),
                                 np.linspace(y_min, y_max, 20))

            w1 = self.weights[feature_indices[0]][0]
            w2 = self.weights[feature_indices[1]][0]
            b = self.bias
            zz = w1 * xx + w2 * yy + b

            ax.plot_surface(xx, yy, zz, alpha=0.5, color='r')

            # 设置坐标轴标签
            feature_names = ["房屋面积", "卧室数量", "房龄", "卫生间数量"]
            ax.set_title('多元线性回归模型拟合结果', fontsize=14)
            ax.set_xlabel(feature_names[feature_indices[0]], fontsize=12)
            ax.set_ylabel(feature_names[feature_indices[1]], fontsize=12)
            ax.set_zlabel('房价 (万元)', fontsize=12)
            ax.legend()
            plt.show()

# 生成多个自变量的模拟数据
def generate_multifeature_data(n_samples=100, noise=10, random_state=42):
    """生成具有多个自变量的房价数据
    特征包括：房屋面积、卧室数量、房龄
    """
    np.random.seed(random_state)

    # 房屋面积 (50-200平方米)
    area = np.random.rand(n_samples, 1) * 150 + 50

    # 卧室数量 (1-5间)
    bedrooms = np.random.randint(1, 6, size=(n_samples, 1))

    # 房龄 (1-30年)
    age = np.random.rand(n_samples, 1) * 29 + 1

    # 卫生间数量 (1-4间)
    bathrooms = np.random.randint(1, 5, size=(n_samples, 1))

    # 合并所有特征
    X = np.hstack([area, bedrooms, age, bathrooms])

    # 真实权重和偏置
    true_weights = np.array([[3.0],    # 房屋面积每增加1平方米，房价增加3万元
                             [15.0],   # 每增加1间卧室，房价增加15万元
                             [-1.0],   # 房龄每增加1年，房价减少1万元
                             [8.0]])   # 每增加1间卫生间，房价增加8万元
    true_bias = 30  # 基础价格30万元

    # 生成房价，添加随机噪声
    y = np.dot(X, true_weights) + true_bias + np.random.randn(n_samples, 1) * noise

    return X, y

# 主函数
def main():
    # 生成多特征数据
    X, y = generate_multifeature_data()

    # 创建并训练模型
    model = LinearRegression(learning_rate=0.0001, n_iterations=1000)
    model.fit(X, y)

    # 打印最终参数
    print(f"\n训练完成!")
    print(f"真实参数:")
    print(f"  房屋面积系数: 3.0 万元/平方米")
    print(f"  卧室数量系数: 15.0 万元/间")
    print(f"  房龄系数: -1.0 万元/年")
    print(f"  卫生间数量系数: 8.0 万元/间")
    print(f"  基础价格: 30 万元")

    print(f"\n模型参数:")
    print(f"  房屋面积系数: {model.weights[0][0]:.4f} 万元/平方米")
    print(f"  卧室数量系数: {model.weights[1][0]:.4f} 万元/间")
    print(f"  房龄系数: {model.weights[2][0]:.4f} 万元/年")
    print(f"  卫生间数量系数: {model.weights[3][0]:.4f} 万元/间")
    print(f"  基础价格: {model.bias:.4f} 万元")

    # 可视化损失曲线
    model.plot_loss_curve()

    # 可视化拟合结果（选择房屋面积和卧室数量这两个特征）
    model.visualize_fit(X, y, feature_indices=[0, 1], animate=True)  # 可视化房屋面积和卧室数量对房价的影响

if __name__ == "__main__":
    main()