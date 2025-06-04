# from scipy.io import loadmat
#
# data = loadmat('Cdataset.mat')  # 替换为你的文件名
# print(data.keys())  # 查看有哪些变量
# data_keys=data.keys()
# for key in data:
#     if not key.startswith('__'):
#         print(f"{key}: type={type(data[key])}, shape={getattr(data[key], 'shape', 'N/A')}")
import os
import numpy as np
import pandas as pd
from scipy.io import loadmat

# 你的 .mat 文件路径
mat_path = 'Cdataset.mat'  # 修改为实际路径
output_dir = 'excel_outputs'
os.makedirs(output_dir, exist_ok=True)

# 加载 .mat 文件
data = loadmat(mat_path)

# 排除系统字段
mat_keys = [k for k in data.keys() if not k.startswith('__')]

for key in mat_keys:
    value = data[key]
    print(f"处理变量: {key}, 类型: {type(value)}, 形状: {value.shape if isinstance(value, np.ndarray) else '未知'}")

    try:
        # 特殊处理 1D 或 2D 矩阵（如相似性、关系矩阵）
        if isinstance(value, np.ndarray):
            if value.ndim == 2:
                df = pd.DataFrame(value)
            elif value.ndim == 1:
                df = pd.DataFrame(value.reshape(-1, 1))
            else:
                raise ValueError(f"暂不支持维度为 {value.ndim} 的数组")

            # 字符串矩阵处理
            if value.dtype == object:
                try:
                    df = pd.DataFrame([str(x[0]) if isinstance(x[0], str) else x[0][0] for x in value])
                except Exception:
                    df = pd.DataFrame(value)

            # 保存为 Excel 文件
            excel_path = os.path.join(output_dir, f"{key}.xlsx")
            df.to_excel(excel_path, index=False)
            print(f"✅ 已保存为: {excel_path}")

        else:
            print(f"⚠️ 变量 {key} 不是 ndarray，跳过")

    except Exception as e:
        print(f"❌ 处理变量 {key} 出错: {e}")

