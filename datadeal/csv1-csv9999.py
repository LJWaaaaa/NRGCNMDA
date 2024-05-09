import pandas as pd

# 读取第一个CSV文件
df1 = pd.read_csv('../MIC-DRUG/interaction.csv',header=None)

# 读取第二个CSV文件
df2 = pd.read_csv('../code/average_predict_y_proba.csv',header=None)

# 将第一个CSV文件中数值为1的位置对应的第二个CSV文件的值设置为9999
df2[df1 == 1] = 9999

# 将结果保存到新的CSV文件
df2.to_csv('output.csv', index=False)
