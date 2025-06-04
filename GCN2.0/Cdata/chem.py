import pandas as pd
import pubchempy as pcp

# 读取 Excel 文件
df = pd.read_excel("excel_outputs\Wrname.xlsx")  # 替换为你的 Excel 文件名
mesh_ids = df['MeSH_ID'].astype(str).tolist()
print("ok")
# 创建一个新列存储分子式
formulas = []

for mesh_id in mesh_ids:
    try:
        compounds = pcp.get_compounds(mesh_id, namespace='name')
        if compounds:
            formulas.append(compounds[0].molecular_formula)
            print(compounds[0].molecular_formula)
        else:
            formulas.append("Not Found")
            print("not found")
    except Exception as e:
        formulas.append("Error")
        print("error")

# 加入新列并保存
df['molecular_formula'] = formulas
df.to_excel("output_with_formula.xlsx", index=False)
