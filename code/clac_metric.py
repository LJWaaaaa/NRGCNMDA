import numpy as np


def get_metrics(real_score, predict_score):
    sorted_predict_score = np.array(
        sorted(list(set(np.array(predict_score).flatten()))))
    sorted_predict_score_num = len(sorted_predict_score)
    thresholds = sorted_predict_score[np.int32(
        sorted_predict_score_num*np.arange(1, 1000)/1000)]
    thresholds = np.mat(thresholds)
    thresholds_num = thresholds.shape[1]

    predict_score_matrix = np.tile(predict_score, (thresholds_num, 1))
    negative_index = np.where(predict_score_matrix < thresholds.T)
    positive_index = np.where(predict_score_matrix >= thresholds.T)
    predict_score_matrix[negative_index] = 0
    predict_score_matrix[positive_index] = 1
    TP = predict_score_matrix.dot(real_score.T)
    FP = predict_score_matrix.sum(axis=1)-TP
    FN = real_score.sum()-TP
    TN = len(real_score.T)-TP-FP-FN

    fpr = FP/(FP+TN)
    tpr = TP/(TP+FN)
    ROC_dot_matrix = np.mat(sorted(np.column_stack((fpr, tpr)).tolist())).T
    ROC_dot_matrix.T[0] = [0, 0]
    ROC_dot_matrix = np.c_[ROC_dot_matrix, [1, 1]]
    x_ROC = ROC_dot_matrix[0].T
    y_ROC = ROC_dot_matrix[1].T
    auc = 0.5*(x_ROC[1:]-x_ROC[:-1]).T*(y_ROC[:-1]+y_ROC[1:])

    recall_list = tpr
    precision_list = TP/(TP+FP)
    PR_dot_matrix = np.mat(sorted(np.column_stack(
        (recall_list, precision_list)).tolist())).T
    PR_dot_matrix.T[0] = [0, 1]
    PR_dot_matrix = np.c_[PR_dot_matrix, [1, 0]]
    x_PR = PR_dot_matrix[0].T
    y_PR = PR_dot_matrix[1].T
    aupr = 0.5*(x_PR[1:]-x_PR[:-1]).T*(y_PR[:-1]+y_PR[1:])

    f1_score_list = 2*TP/(len(real_score.T)+TP-TN)
    accuracy_list = (TP+TN)/len(real_score.T)
    specificity_list = TN/(TN+FP)

    max_index = np.argmax(f1_score_list)
    f1_score = f1_score_list[max_index]
    accuracy = accuracy_list[max_index]
    specificity = specificity_list[max_index]
    recall = recall_list[max_index]
    precision = precision_list[max_index]
    return [aupr[0, 0], auc[0, 0], f1_score, accuracy, recall, specificity, precision]
     # ,fpr,tpr,recall,precision

# def get_metrics(real_score, predict_score):
#     sorted_predict_score = np.array(
#         sorted(list(set(np.array(predict_score).flatten()))))
#     sorted_predict_score_num = len(sorted_predict_score)
#
#     thresholds = sorted_predict_score[np.int32(
#         sorted_predict_score_num * np.arange(1, 1000) / 1000)]
#     thresholds = np.mat(thresholds)
#     thresholds_num = thresholds.shape[1]
#
#     predict_score_matrix = np.tile(predict_score, (thresholds_num, 1))
#     negative_index = np.where(predict_score_matrix < thresholds.T)
#     positive_index = np.where(predict_score_matrix >= thresholds.T)
#     predict_score_matrix[negative_index] = 0
#     predict_score_matrix[positive_index] = 1
#     TP = predict_score_matrix.dot(real_score.T)
#     FP = predict_score_matrix.sum(axis=1) - TP
#     FN = real_score.sum() - TP
#     TN = len(real_score.T) - TP - FP - FN
#
#     fpr = FP / (FP + TN)
#     tpr = TP / (TP + FN)
#
#     f1_score_list = 2 * TP / (len(real_score.T) + TP - TN)
#     accuracy_list = (TP + TN) / len(real_score.T)
#     specificity_list = TN / (TN + FP)
#
#     max_f1_index = np.argmax(f1_score_list)
#     max_accuracy_index = np.argmax(accuracy_list)
#     max_recall_index = np.argmax(tpr)
#
#     # 使用F1 score最大的索引作为自适应阈值的索引
#     index = max_f1_index
#
#     f1_score = f1_score_list[index]
#     accuracy = accuracy_list[index]
#     specificity = specificity_list[index]
#     recall = tpr[index]
#     precision = TP[index] / (TP[index] + FP[index])
#     auc = 0.5 * (fpr[index:] - fpr[index:-1]) * (tpr[index:-1] + tpr[index + 1:])
#     aupr = 0.5 * (recall[:-1] - recall[1:]) * (precision[:-1] + precision[1:])
#
#     return [aupr[0, 0], auc.sum().item(), f1_score.item(), accuracy.item(), recall.item(), specificity.item(),
#             precision.item()]

def cv_model_evaluate(interaction_matrix, predict_matrix, train_matrix):
    test_index = np.where(train_matrix == 0)    #train_matrix 中的值为0的位置，将这些位置的索引保存在 test_index 中
    # test_index = np.unique(np.where(train_matrix == 0)[0])
    real_score = interaction_matrix[test_index] #从 interaction_matrix 中提取与测试集对应位置的真实关联情况，这些值构成了真实的标签（真实得分）
    predict_score = predict_matrix[test_index]  #从predict_matrix 中提取与测试集对应位置的模型预测的得分
    return get_metrics(real_score, predict_score)

# 这段Python代码定义了两个函数：get_metrics 和 cv_model_evaluate，用于评估二分类问题中预测模型的性能。以下是每个函数及其用途的详细说明：
#
# get_metrics(real_score, predict_score):
#
# 这个函数计算了针对二分类模型预测的各种评价指标。
#
# real_score：包含样本的真实二元标签（实际情况）的1D numpy数组。
# predict_score：包含样本的预测分数（概率或置信度分数）的1D numpy数组。
# 函数执行步骤如下：
#
# 它按升序对唯一的预测分数进行排序，并创建一组用于分类的阈值。
# 它基于这些阈值构建了一个二进制矩阵，其中的元素被设置为1，如果预测分数高于阈值，则设置为0。
# 使用混淆矩阵方法计算了真正例（TP）、假正例（FP）、假反例（FN）和真反例（TN）的数量。
# 基于计算的值，它计算了真正例率（TPR）、假正例率（FPR）和ROC曲线下的面积（AUC）。
# 基于计算的值，它计算了召回率-精确率对和召回率-精确率曲线下的面积（AUPR）。
# 它计算了不同阈值水平的F1分数、准确率、特异度、召回率和精确率，并选择最大的F1分数作为最优指标。
# 函数返回一个指标列表：[AUPR, AUC, F1分数, 准确率, 召回率, 特异度, 精确率]。
#
# cv_model_evaluate(interaction_matrix, predict_matrix, train_matrix):
#
# 这个函数通过比较预测分数与未包含在训练数据集中的样本的真实二元标签，执行模型评估。
#
# interaction_matrix：表示样本与特征之间交互（或标签）的矩阵。
# predict_matrix：包含交互的预测分数的矩阵。
# train_matrix：表示用于训练的样本的矩阵（使用为1，未使用为0）。
# 函数执行以下操作：
#
# 它识别测试样本的索引（其中 train_matrix 为0）。
# 提取测试样本的真实二元标签（real_score）和预测分数（predict_score）。
# 调用 get_metrics 函数，基于测试样本的真实和预测分数，计算并返回评价指标。
# 总之，这些函数旨在通过将预测分数与真实二元标签进行比较，评估二分类模型的性能，并计算与ROC曲线、召回率-精确率曲线和其他分类性能指标相关的各种指标。