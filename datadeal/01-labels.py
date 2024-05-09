import pandas as pd

# 从 CSV 文件中读取行和列标签
row_labels = pd.read_csv('../data-mic-drug/drug_number_627.csv')['row_labels'].tolist()
column_labels = pd.read_csv('../data-mic-drug/microbe_number_142_T.csv')['column_labels'].tolist()

# 从两个 CSV 文件中读取 0-1 关系矩阵
relationship_matrix = pd.read_csv('../data-mic-drug/interaction.csv', header=None).values

# 定义标签映射
label_mapping = {0: 'Label_0', 1: 'Label_1'}

# 将关系矩阵转换为标签矩阵
label_matrix = []
for i in range(len(relationship_matrix)):
    row = []
    for j in range(len(relationship_matrix[i])):
        label = label_mapping[relationship_matrix[i][j]]
        row.append(label)
    label_matrix.append(row)

# 打印标签矩阵
for row_label, row in zip(row_labels, label_matrix):
    print(row_label, row)
