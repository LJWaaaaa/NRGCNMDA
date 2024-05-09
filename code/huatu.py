import numpy as np
import matplotlib.pyplot as plt
plt.axis('off')  # 隐藏横纵坐标
heatmap = np.random.rand(4, 4)
plt.imshow(heatmap, cmap='Oranges', alpha=0.5)
plt.savefig('heatmap.png', dpi=300, bbox_inches='tight')  # 保存为图片文件

