# 阶段2: 嵌套交叉验证报告

## 模型配置

- **Pipeline**: StandardScaler → SelectKBest → ElasticNet Logistic Regression
- **外层CV**: 5-Fold × 3 重复 = 15次训练
- **内层CV**: 5-Fold GridSearchCV
- **超参数空间**:
  - selector__k: [300, 500, 700, all]
  - classifier__l1_ratio: [0.2, 0.5, 0.8]
  - classifier__C: [0.001, 0.01, 0.1, 1, 10, 100]
  - 总组合数: 72

## 外层交叉验证性能

### 主要指标 (Mean ± Std)

| 指标 | 均值 | 标准差 |
|------|------|--------|
| **AUROC** | 0.6570 | 0.0937 |
| **AUPRC** | 0.4223 | 0.0893 |
| **F1-Score** | 0.4101 | 0.1285 |
| **Recall (Sensitivity)** | 0.5007 | 0.1715 |
| **Specificity** | 0.7256 | 0.0843 |
| **Accuracy** | 0.6724 | 0.0593 |

### 解读

- **AUROC > 0.7**: 模型具有良好的区分能力
- **AUPRC**: 考虑类别不平衡，PR曲线下面积
- **Recall**: 高风险患者的召回率（灵敏度）
- **Specificity**: 低风险患者的正确识别率

## 测试集性能

| 指标 | 值 |
|------|------|
| **AUROC** | 0.7017 |
| **AUPRC** | 0.3734 |
| **F1-Score** | 0.4889 |
| **Recall** | 0.6471 |
| **Precision** | 0.3929 |
| **Specificity** | 0.6964 |
| **Accuracy** | 0.6849 |

### 混淆矩阵

|  | 预测: 低风险 | 预测: 高风险 |
|---|---|---|
| **实际: 低风险** | 39 (TN) | 17 (FP) |
| **实际: 高风险** | 6 (FN) | 11 (TP) |

## 最优超参数

最常见的最佳参数组合:
- **selector__k**: 300
- **classifier__l1_ratio**: 0.2
- **classifier__C**: 0.01

## 特征选择统计

- **平均选择特征数**: 528 ± 264
- **平均非零系数数**: 164 ± 284
- **稀疏化比例**: 81.9%

## 输出文件

- `outer_cv_results.csv`: 15次外层CV的详细结果
- `feature_selection_history.json`: 每次选择的特征列表
- `coefficient_history.json`: 每次的模型系数
- `best_params_history.csv`: 每次的最佳超参数
- `test_set_results.json`: 测试集评估结果
- `test_predictions.csv`: 测试集预测值
- `final_model.pkl`: 最终训练的模型
- `summary.json`: 汇总统计
- `nested_cv_results.png`: 可视化结果

---
生成时间: 2025-11-13 18:59:15
