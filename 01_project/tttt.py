import matplotlib.pyplot as plt

# 起点和方向
x_start, y_start = 0, 0
dx, dy = 2, 1  # 向量方向

plt.figure()
plt.arrow(x_start, y_start, dx, dy, head_width=0.1, head_length=0.1, fc='green', ec='blue')
plt.xlim(-1, 3)
plt.ylim(-1, 2)
plt.gca().set_aspect('equal')
plt.grid(True)
plt.show()
