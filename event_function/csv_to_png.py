import pandas as pd
import numpy as np
import cv2

# 1. 读取 CSV
df = pd.read_csv('/home/qiangubuntu/research/data_collection/src/data/0/event_csv/1000.csv')

# 2. 固定图像尺寸
width, height = 346, 260

# 3. 准备空白 RGB 图像（背景默认为白色）
canvas = np.ones((height, width, 3), dtype=np.uint8) * 255

# 4. 着色
#    polarity == 1: 红色 [255,0,0]
#    polarity == 0: 绿色 [0,255,0]
pos = df[df['polarity'] == 1]
neg = df[df['polarity'] == 0]

# 注意 NumPy 索引是 [row=y, col=x]，并且要确保坐标在画布范围内
# 正极性点（1）
ys = pos['y'].astype(int).clip(0, height-1).values
xs = pos['x'].astype(int).clip(0, width-1).values
canvas[ys, xs] = [(0.0, 255.0, 0.0)]

# 负极性点（0）
ys = neg['y'].astype(int).clip(0, height-1).values
xs = neg['x'].astype(int).clip(0, width-1).values
canvas[ys, xs] = [255.0, 0.0, 0.0]
canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)

# show canvas
cv2.imshow('canvas', canvas)
cv2.waitKey(0)

# 5. 保存成 PNG
# cv2.imwrite('/home/qiangubuntu/research/data_collection/src/data/0/output.png', canvas)
# print(f"已生成 output.png，尺寸：{width}×{height}")
