import numpy as np
import matplotlib.pyplot as plt

x = np.array([
-0.02,
0,
0.02
])

y = np.array([
0.069453,
0.109787,
0.151105
])

# 使用numpy进行线性拟合 (y = ax + b)
coefficients = np.polyfit(x, y, 1)  # 1表示一次多项式（直线）
a, b = coefficients  # a是斜率，b是截距

print(f"拟合直线方程: y = {a:.6f} * x + {b:.6f}")
print(f"斜率 (a): {a:.6f}")
print(f"截距 (b): {b:.6f}")

# 计算拟合质量（R²）
y_pred = a * x + b
ss_res = np.sum((y - y_pred) ** 2)
ss_tot = np.sum((y - np.mean(y)) ** 2)
r_squared = 1 - (ss_res / ss_tot)
print(f"R²: {r_squared:.6f}")

# 可视化
plt.figure(figsize=(10, 6))
plt.scatter(x, y, color='red', s=100, label='data points', zorder=3)
x_line = np.linspace(x.min() - 0.01, x.max() + 0.01, 100)
y_line = a * x_line + b
plt.plot(x_line, y_line, 'b-', label=f'linear fitting: y = {a:.6f}x + {b:.6f}', linewidth=2)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('linear fitting result')
plt.legend()
plt.grid(True, alpha=0.3)
# plt.tight_layout()
plt.show()
