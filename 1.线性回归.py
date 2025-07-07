import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # 或者 'QtAgg'
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

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

    def visualize_fit(self, X, y, animate=False):
        """可视化模型拟合过程"""
        if X.shape[1] != 1:
            print("只能可视化单特征数据")
            return

        if animate:
            # 创建动画
            fig, ax = plt.subplots(figsize=(10, 6))
            scatter = ax.scatter(X, y, c='blue', alpha=0.7, label='数据点')
            line, = ax.plot([], [], 'r-', linewidth=2, label='拟合直线')
            ax.set_title('线性回归模型拟合过程', fontsize=14)
            ax.set_xlabel('房屋面积 (平方米)', fontsize=12)
            ax.set_ylabel('房价 (万元)', fontsize=12)
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.legend()

            def init():
                line.set_data([], [])
                return line,

            def update(frame):
                w = self.history["weights"][frame][0][0]
                b = self.history["bias"][frame]
                x_vals = np.linspace(X.min(), X.max(), 100)
                y_vals = w * x_vals + b
                line.set_data(x_vals, y_vals)
                ax.set_title(f'线性回归模型拟合过程 (迭代 {frame})')
                return line,

            ani = FuncAnimation(fig, update, frames=len(self.history["weights"]),
                                init_func=init, interval=50, blit=True)
            plt.show()
            return ani
        else:
            # 静态可视化
            plt.figure(figsize=(10, 6))
            plt.scatter(X, y, c='blue', alpha=0.7, label='数据点')

            # 绘制最终拟合直线
            x_vals = np.linspace(X.min(), X.max(), 100)
            y_vals = self.weights[0][0] * x_vals + self.bias
            plt.plot(x_vals, y_vals, 'r-', linewidth=2, label='拟合直线')

            plt.title('线性回归模型拟合结果', fontsize=14)
            plt.xlabel('房屋面积 (平方米)', fontsize=12)
            plt.ylabel('房价 (万元)', fontsize=12)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend()
            plt.show()

# 生成模拟数据
def generate_data(n_samples=100, noise=10, random_state=42):
    """生成房屋面积与价格的模拟数据"""
    np.random.seed(random_state)
    X = np.random.rand(n_samples, 1) * 100  # 房屋面积 (0-100平方米)
    true_weights = np.array([[3]])  # 每平方米3万元
    true_bias = 50  # 基础价格50万元
    y = np.dot(X, true_weights) + true_bias + np.random.randn(n_samples, 1) * noise  # 添加随机噪声
    return X, y

# 主函数
def main():
    # 生成数据
    X, y = generate_data()

    # 创建并训练模型
    model = LinearRegression(learning_rate=0.0001, n_iterations=500)
    model.fit(X, y)

    # 打印最终参数
    print(f"\n训练完成!")
    print(f"真实参数: 斜率=3, 截距=50")
    print(f"模型参数: 斜率={model.weights[0][0]:.4f}, 截距={model.bias:.4f}")

    # 可视化
    model.plot_loss_curve()
    model.visualize_fit(X, y, animate=True)  # 设置animate=True可查看拟合过程动画

if __name__ == "__main__":
    main()