import matplotlib.pyplot as plt

x = [1, 2, 4, 8, 16, 32, 64, 128]

original1 = [0.165, 0.175, 0.1625, 0.1775, 0.15, 0.17, 0.17, 0.17]
new1 = [0.165, 0.165, 0.165, 0.165, 0.165, 0.165, 0.165, 0.165]

original2 = [0.1675, 0.2275, 0.2475, 0.315, 0.3525, 0.4225, 0.4575, 0.4975]
new2 = [0.1561875, 0.20873682, 0.25976871, 0.30762292, 0.35460739, 0.40033108, 0.44483431, 0.48965766]

original3 = [0.16, 0.25, 0.33, 0.4275, 0.515, 0.605, 0.63, 0.7175]
new3 = [0.145875, 0.21950358, 0.30459949, 0.39345965, 0.48020121, 0.5613553, 0.63205111, 0.69218137]

fig, axs = plt.subplots(1, 3, figsize=(18, 5))

# 第一张图
axs[0].plot(x, original1, marker='o', label='original')
axs[0].plot(x, new1, marker='s', label='new')
axs[0].set_xlabel('n')
axs[0].set_ylabel('pass rate')
axs[0].set_title('Figure 1')
axs[0].legend()
axs[0].grid(True)
axs[0].set_ylim(0.1, 0.8)  # 设置纵轴范围

# 第二张图
axs[1].plot(x, original2, marker='o', label='original')
axs[1].plot(x, new2, marker='s', label='new')
axs[1].set_xlabel('n')
axs[1].set_ylabel('pass rate')
axs[1].set_title('Figure 2')
axs[1].legend()
axs[1].grid(True)
axs[1].set_ylim(0.1, 0.8)  # 设置纵轴范围

# 第三张图
axs[2].plot(x, original3, marker='o', label='original')
axs[2].plot(x, new3, marker='s', label='new')
axs[2].set_xlabel('n')
axs[2].set_ylabel('pass rate')
axs[2].set_title('Figure 3')
axs[2].legend()
axs[2].grid(True)
axs[2].set_ylim(0.1, 0.8)  # 设置纵轴范围

plt.tight_layout()
plt.savefig("pass_rate_comparison.png")
plt.show()