import matplotlib.pyplot as plt

# 横纵坐标数据
# x = [0.331, 0.468, 0.457, 0.636, 0.548, 0.539]
# y = [0.44, 0.742, 0.493, 1.232, 0.906, 0.917]

mse = [0.524, 0.633, 0.711, 1.422, 1.647, 0.989, 1.097]
crps = [0.252, 0.365, 0.562, 0.774, 0.694, 0.477, 0.504]
qice = [1.24, 0.78, 2.35, 14.82, 5.03, 3.04, 5.12]
# time = [0.11, 0.27, 11.54]
# sizes = [0.11, 0.42, 11.54, 30.45, 88.70, 83, 24.82]
sizes = [1.16, 2.34, 11.20, 117.65, 34.92, 26.74, 10.97]
sizes = list(i*100 for i in sizes)

labels = ['PPM', 'NsDiff', 'TMDM', 'TimeDiff', 'TimeGrad', 'DiffusionTS', 'D3VAE']
colors = ['#B4512D', "#206060", "#44C39B", "#D8B365", "#185395", "#C32340", "#85542D"]  

y = mse
x = crps

time = [0.1]

# 计算边界范围
x_margin = (max(x) - min(x)) * 0.1
y_margin = (max(y) - min(y)) * 0.1

x_min, x_max = min(x) - x_margin, max(x) + x_margin + 0.1
y_min, y_max = min(y) - y_margin, max(y) + y_margin + 0.2

# 绘图
plt.figure(figsize=(6, 4.5), dpi=230)
ax = plt.gca()
# 设置图像背景颜色
plt.gca().set_facecolor('none')
plt.grid(True, linestyle='--')


for i in range(len(x)):
    plt.scatter(x[i], y[i], color=colors[i], s=sizes[i], alpha=0.9, edgecolors='none')

legend_handles = [plt.scatter([], [], color=colors[i], s=60, label=labels[i], alpha=0.9, edgecolors='none', linewidths=0)
                  for i in range(len(labels))]

time_list = [5, 20, 45, 80]
bubble_sizes = [s*100 for s in time_list]  # 显示用的大小，调小一点更美观
 
# 图例起始位置
legend_center_x = 0.83
legend_center_y = 0.65

for i, (time, size) in enumerate(zip(reversed(time_list), reversed(bubble_sizes))):
    # 每个圆的位置：中心不变，大小不同
    ax.scatter(legend_center_x, legend_center_y - i*0.058, s=size,
               color='grey', alpha=0.6, zorder=10+i,
               edgecolors='white', linewidths=1.2)

for i, (time, size) in enumerate(zip(reversed(time_list), reversed(bubble_sizes))):
    label_y = legend_center_y - i*0.094 + 0.17 # 调整文本位置
    ax.text(legend_center_x, label_y, f"{int(time)}", ha='center', va='center',
            fontsize=8, color='black', zorder=20+i)
    
plt.text(0.745, 0.92, 'Inference time(s)')
# plt.text(0.732, 0.905, 'Inference time(s)',)

# 设置边界范围
plt.xlim(x_min, x_max)
plt.ylim(0.35, y_max)

# 标注
# plt.xlabel('CRPS', fontsize=12)
# plt.ylabel('MSE', fontsize=12)
plt.xlabel('CRPS', fontsize=12, )
plt.ylabel('MSE', fontsize=12,)
# plt.title('Scatter Plot from Rows 2 and 3', fontsize=14)


#刻度
# plt.xticks([0.3, 0.5, 0.7, 0.9, 1.1], ['0.3', '0.5', '0.7', '0.9', '1.1'])
# plt.yticks([0.4, 0.6, 0.8, 1.0, 1.2], ['0.4', '0.6', '0.8', '1.0', '1.2'])
plt.tick_params(axis='x', length=3, width=1, colors='black', grid_color='grey', grid_alpha=0.3)
plt.tick_params(axis='y', length=3, width=1, colors='black', grid_color='grey', grid_alpha=0.3)

ax = plt.gca()  # 获取当前轴
for spine in ax.spines.values():
    spine.set_edgecolor('black')  # 设置边框颜色
    spine.set_linewidth(1)        # 设置边框宽度

# plt.legend(loc='upper left', frameon=True)
legend = ax.legend(loc='upper left', frameon=True,)
# plt.setp(legend.get_title(), fontweight='bold')
# legend = ax.legend(loc='upper left', frameon=True)
legend.get_frame().set_edgecolor('black') 
legend.get_frame().set_facecolor('white')
plt.tight_layout()
plt.show()