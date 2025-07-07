import numpy as np
import matplotlib.pyplot as plt

# 生成数据
x_left = np.linspace(-5, 0, 100)
x_right = np.linspace(0.1, 10, 500)  # x>0时避开x=0

# 计算函数值
y_left = x_left + 1
y_right = np.sin(x_right) / x_right

# 绘制图像
plt.figure(figsize=(10, 6))
plt.plot(x_left, y_left, label='x + 1 (x ≤ 0)', color='blue')
plt.plot(x_right, y_right, label='sinx/x (x > 0)', color='red')
plt.scatter(0, 1, color='blue', s=50, zorder=2, label='f(0) = 1')  # 实心点
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('分段函数图像')
plt.ylim(-2, 2)
plt.legend()
plt.grid(True)
plt.show()