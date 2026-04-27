import matplotlib.pyplot as plt
from pathlib import Path

# NeurIPS-style publication settings
plt.rcParams.update({
    'figure.dpi': 600,
    'savefig.dpi': 600,
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 11,
    'axes.linewidth': 1.0,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})

# 横纵坐标数据
# mse = [0.300, 0.316, 0.413, 0.469, 0.663, 1.298]
# crps = [0.246, 0.257, 0.327, 0.383, 0.487, 0.689]

mse = [0.444, 0.629, 0.721, 1.465, 0.932]
crps = [0.229, 0.378, 0.557, 0.671, 0.657]
# qice = [1.24, 0.78, 2.35, 14.82, 5.03, 3.04, 5.12]
# time = [0.11, 0.27, 11.54]
# sizes = [0.11, 0.42, 11.54, 30.45, 88.70, 83, 24.82]
sizes = [2.72, 2.34, 11.20, 117.65, 34.92]
bubble_scale = 100
sizes = [i * bubble_scale for i in sizes]

labels = ['LS-Flow', 'NsDiff', 'TMDM', 'TimeDiff', 'TimeGrad']
colors = ["#C32340", "#206060", "#44C39B", "#D8B365", "#185395", ]

y = mse
x = crps

time = [0.1]

# 计算边界范围
x_margin = (max(x) - min(x)) * 0.1
y_margin = (max(y) - min(y)) * 0.1

x_min, x_max = min(x) - x_margin - 0.015, max(x) + x_margin + 0.085
y_min, y_max = max(0.0, min(y) - y_margin - 0.04), max(y) + y_margin + 0.23

# 绘图
plt.figure(figsize=(6.6, 5.0), dpi=600)
ax = plt.gca()
# 设置图像背景颜色
plt.gca().set_facecolor('white')
plt.grid(True, linestyle='--', linewidth=0.6, alpha=0.28, color='#9E9E9E')


# 先画大气泡再画小气泡，避免左下角小气泡被完全遮住
plot_order = sorted(range(len(x)), key=lambda i: sizes[i], reverse=True)
for i in plot_order:
    plt.scatter(
        x[i], y[i], color=colors[i], s=sizes[i], alpha=0.82,
        edgecolors='white', linewidths=0.75, zorder=3
    )

legend_handles = [plt.scatter([], [], color=colors[i], s=60, label=labels[i], alpha=0.9, edgecolors='white', linewidths=0.4)
                  for i in range(len(labels))]

time_list = [5, 20, 45, 80]
bubble_sizes = [s * bubble_scale for s in time_list]  # 与主图气泡保持一致比例
 
# 图例起始位置
legend_center_x = 0.85
legend_center_y = 0.16
bubble_step = 0.036

for i, (time, size) in enumerate(zip(reversed(time_list), reversed(bubble_sizes))):
    # 每个圆的位置：中心不变，大小不同
    ax.scatter(
        legend_center_x, legend_center_y - i * bubble_step, s=size,
        color='grey', alpha=0.6, zorder=10 + i,
        edgecolors='white', linewidths=1.2,
        transform=ax.transAxes
    )

label_start_y = legend_center_y + 0.102
label_step = 0.050
for i, (time, size) in enumerate(zip(reversed(time_list), reversed(bubble_sizes))):
    label_y = label_start_y - i * label_step  # 标签单独排布，避免数字挤在一起
    ax.text(legend_center_x, label_y, f"{int(time)}", ha='center', va='center',
        fontsize=10, color='black', zorder=20 + i,
        transform=ax.transAxes)
    
ax.text(0.85, 0.31, 'Inference time(s)', fontsize=11,
    ha='center', va='bottom', transform=ax.transAxes,
    bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, pad=1.2))
# plt.text(0.732, 0.905, 'Inference time(s)',)

# 设置边界范围
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)

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
legend = ax.legend(loc='upper left', frameon=True)
# plt.setp(legend.get_title(), fontweight='bold')
# legend = ax.legend(loc='upper left', frameon=True)
legend.get_frame().set_edgecolor('black') 
legend.get_frame().set_facecolor('white')
legend.get_frame().set_alpha(0.96)
plt.tight_layout()

output_dir = Path('results/analysis')
output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / 'inference_time.pdf'
plt.savefig(output_path, format='pdf', bbox_inches='tight', dpi=600)

plt.show()