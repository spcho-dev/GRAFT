import numpy as np

# 加载npz文件
data = np.load("PP.adj.npz")

print(data)

# 关闭文件
data.close()