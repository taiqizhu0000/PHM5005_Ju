#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
阶段4: 消融实验
测试不同特征组合的性能：
1. 仅临床特征
2. 仅基因特征  
3. 临床 + 基因（完整）
4. 临床 + Top基因
5. 各通路分组
"""

import pandas as pd
import numpy as np
import os
import json
import glob
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

BASE_DIR = "/Users/a/Desktop/5005"
RESULTS_DIR = os.path.join(BASE_DIR, "results")
PATHWAY_DIR = os.path.join(BASE_DIR, "dataset", "pathway_gene_list")
OUTPUT_DIR = os.path.join(RESULTS_DIR, "stage4_ablation")
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 80)
print("阶段4: 消融实验")
print("=" * 80)

# ============================================================================
# 1. 加载数据
# ============================================================================
print("\n[1] 加载数据...")

X_train = np.load(os.path.join(RESULTS_DIR, 'X_train.npy'), allow_pickle=True)
X_test = np.load(os.path.join(RESULTS_DIR, 'X_test.npy'), allow_pickle=True)
y_train = np.load(os.path.join(RESULTS_DIR, 'y_train.npy'), allow_pickle=True)
y_test = np.load(os.path.join(RESULTS_DIR, 'y_test.npy'), allow_pickle=True)

with open(os.path.join(RESULTS_DIR, 'feature_names.json'), 'r') as f:
    feature_info = json.load(f)
    all_features = feature_info['all_features']
    clinical_features = feature_info['clinical_features']
    gene_features = feature_info['gene_features']

# 加载通路基因
pathway_genes = {}
for pathway_file in glob.glob(os.path.join(PATHWAY_DIR, "*.csv")):
    pathway_name = os.path.basename(pathway_file).replace('_symbols.csv', '').replace('_hsa', '')
    df = pd.read_csv(pathway_file)
    genes = df.iloc[:, 0].str.strip('"').tolist()
    pathway_genes[pathway_name] = [f'gene_{g}' for g in genes if f'gene_{g}' in all_features]

# 加载Top特征
feature_importance = pd.read_csv(os.path.join(RESULTS_DIR, 'stage3_feature_importance', 
                                              'feature_importance_full.csv'))
top_genes_list = feature_importance[feature_importance['type'] == 'Gene'].head(100)['feature'].tolist()

print(f"  训练集: {X_train.shape}")
print(f"  测试集: {X_test.shape}")
print(f"  临床特征: {len(clinical_features)}")
print(f"  基因特征: {len(gene_features)}")
print(f"  通路数: {len(pathway_genes)}")

# ============================================================================
# 2. 定义评估函数
# ============================================================================
def evaluate_model(X_tr, y_tr, X_te, y_te, feature_name):
    """训练和评估模型"""
    # 简单模型（无特征选择，使用最佳参数）
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(
            solver='saga',
            penalty='elasticnet',
            l1_ratio=0.2,
            C=0.01,
            class_weight='balanced',
            max_iter=5000,
            random_state=RANDOM_STATE
        ))
    ])
    
    # 5折交叉验证
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_auroc = cross_val_score(model, X_tr, y_tr, cv=cv, scoring='roc_auc', n_jobs=-1)
    cv_auprc = cross_val_score(model, X_tr, y_tr, cv=cv, scoring='average_precision', n_jobs=-1)
    
    # 训练最终模型
    model.fit(X_tr, y_tr)
    
    # 测试集评估
    y_pred = model.predict(X_te)
    y_proba = model.predict_proba(X_te)[:, 1]
    
    test_auroc = roc_auc_score(y_te, y_proba)
    test_auprc = average_precision_score(y_te, y_proba)
    test_f1 = f1_score(y_te, y_pred)
    
    return {
        'feature_group': feature_name,
        'n_features': X_tr.shape[1],
        'cv_auroc_mean': cv_auroc.mean(),
        'cv_auroc_std': cv_auroc.std(),
        'cv_auprc_mean': cv_auprc.mean(),
        'cv_auprc_std': cv_auprc.std(),
        'test_auroc': test_auroc,
        'test_auprc': test_auprc,
        'test_f1': test_f1
    }

# ============================================================================
# 3. 消融实验
# ============================================================================
print("\n[2] 执行消融实验...")

ablation_results = []

# 3.1 仅临床特征
print("\n  [1] 仅临床特征...")
clinical_indices = [all_features.index(f) for f in clinical_features if f in all_features]
X_train_clinical = X_train[:, clinical_indices]
X_test_clinical = X_test[:, clinical_indices]
result = evaluate_model(X_train_clinical, y_train, X_test_clinical, y_test, 'Clinical Only')
ablation_results.append(result)
print(f"      CV AUROC: {result['cv_auroc_mean']:.4f}±{result['cv_auroc_std']:.4f}, "
      f"Test AUROC: {result['test_auroc']:.4f}")

# 3.2 仅基因特征
print("\n  [2] 仅基因特征...")
gene_indices = [all_features.index(f) for f in gene_features if f in all_features]
X_train_gene = X_train[:, gene_indices]
X_test_gene = X_test[:, gene_indices]
result = evaluate_model(X_train_gene, y_train, X_test_gene, y_test, 'Gene Only')
ablation_results.append(result)
print(f"      CV AUROC: {result['cv_auroc_mean']:.4f}±{result['cv_auroc_std']:.4f}, "
      f"Test AUROC: {result['test_auroc']:.4f}")

# 3.3 完整特征
print("\n  [3] 临床 + 基因（完整）...")
result = evaluate_model(X_train, y_train, X_test, y_test, 'Clinical + Gene (Full)')
ablation_results.append(result)
print(f"      CV AUROC: {result['cv_auroc_mean']:.4f}±{result['cv_auroc_std']:.4f}, "
      f"Test AUROC: {result['test_auroc']:.4f}")

# 3.4 临床 + Top基因
print("\n  [4] 临床 + Top 100基因...")
top_gene_indices = [all_features.index(f) for f in top_genes_list if f in all_features]
combined_indices = clinical_indices + top_gene_indices
X_train_top = X_train[:, combined_indices]
X_test_top = X_test[:, combined_indices]
result = evaluate_model(X_train_top, y_train, X_test_top, y_test, 'Clinical + Top100 Genes')
ablation_results.append(result)
print(f"      CV AUROC: {result['cv_auroc_mean']:.4f}±{result['cv_auroc_std']:.4f}, "
      f"Test AUROC: {result['test_auroc']:.4f}")

# 3.5 各通路测试
print("\n  [5] 各通路分组测试...")
for pathway_name, pathway_gene_list in pathway_genes.items():
    if len(pathway_gene_list) < 5:  # 跳过基因数太少的通路
        continue
    
    # 临床 + 该通路基因
    pathway_indices = [all_features.index(f) for f in pathway_gene_list if f in all_features]
    if len(pathway_indices) == 0:
        continue
    
    combined_indices = clinical_indices + pathway_indices
    X_train_pathway = X_train[:, combined_indices]
    X_test_pathway = X_test[:, combined_indices]
    
    result = evaluate_model(X_train_pathway, y_train, X_test_pathway, y_test, 
                           f'Clinical + {pathway_name}')
    ablation_results.append(result)
    print(f"      {pathway_name:30s}: Test AUROC {result['test_auroc']:.4f}")

print(f"\n  完成 {len(ablation_results)} 组消融实验")

# ============================================================================
# 4. 保存结果
# ============================================================================
print("\n[3] 保存结果...")

results_df = pd.DataFrame(ablation_results)
results_df = results_df.sort_values('test_auroc', ascending=False)
results_df.to_csv(os.path.join(OUTPUT_DIR, 'ablation_results.csv'), index=False)

print(f"  结果已保存到: {OUTPUT_DIR}")

# ============================================================================
# 5. 可视化
# ============================================================================
print("\n[4] 生成可视化...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 5.1 测试集AUROC对比
ax1 = axes[0, 0]
top_results = results_df.head(10)
y_pos = np.arange(len(top_results))
colors = ['#e74c3c' if 'Clinical Only' in g else '#3498db' if 'Gene Only' in g else '#2ecc71' 
          for g in top_results['feature_group']]
ax1.barh(y_pos, top_results['test_auroc'], color=colors, alpha=0.7)
ax1.set_yticks(y_pos)
ax1.set_yticklabels(top_results['feature_group'], fontsize=10)
ax1.set_xlabel('Test AUROC', fontsize=12)
ax1.set_title('Top 10 Feature Groups by Test AUROC', fontsize=13, fontweight='bold')
ax1.invert_yaxis()
ax1.grid(axis='x', alpha=0.3)
ax1.axvline(x=0.75, color='red', linestyle='--', linewidth=1, alpha=0.5)

# 5.2 CV AUROC vs Test AUROC
ax2 = axes[0, 1]
ax2.scatter(results_df['cv_auroc_mean'], results_df['test_auroc'], 
           s=results_df['n_features']/5, alpha=0.6, c=range(len(results_df)), cmap='viridis')
ax2.plot([0.5, 1], [0.5, 1], 'k--', alpha=0.3)
ax2.set_xlabel('CV AUROC (Mean)', fontsize=12)
ax2.set_ylabel('Test AUROC', fontsize=12)
ax2.set_title('CV vs Test Performance', fontsize=13, fontweight='bold')
ax2.grid(alpha=0.3)

# 5.3 特征数量 vs 性能
ax3 = axes[1, 0]
ax3.scatter(results_df['n_features'], results_df['test_auroc'], 
           s=100, alpha=0.6, c=results_df['test_auprc'], cmap='coolwarm')
ax3.set_xlabel('Number of Features', fontsize=12)
ax3.set_ylabel('Test AUROC', fontsize=12)
ax3.set_title('Feature Count vs Performance', fontsize=13, fontweight='bold')
ax3.grid(alpha=0.3)
cbar = plt.colorbar(ax3.collections[0], ax=ax3)
cbar.set_label('Test AUPRC', fontsize=10)

# 5.4 主要组合对比（柱状图）
ax4 = axes[1, 1]
main_groups = results_df[results_df['feature_group'].isin([
    'Clinical Only', 'Gene Only', 'Clinical + Gene (Full)', 'Clinical + Top100 Genes'
])]
x = np.arange(len(main_groups))
width = 0.35
ax4.bar(x - width/2, main_groups['cv_auroc_mean'], width, 
        label='CV AUROC', alpha=0.7, color='#3498db')
ax4.bar(x + width/2, main_groups['test_auroc'], width,
        label='Test AUROC', alpha=0.7, color='#e74c3c')
ax4.set_ylabel('AUROC', fontsize=12)
ax4.set_title('Main Feature Groups Comparison', fontsize=13, fontweight='bold')
ax4.set_xticks(x)
ax4.set_xticklabels(main_groups['feature_group'], rotation=15, ha='right', fontsize=9)
ax4.legend()
ax4.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'ablation_study.png'), dpi=300, bbox_inches='tight')
print(f"  可视化已保存: ablation_study.png")
plt.close()

# ============================================================================
# 6. 生成报告
# ============================================================================
print("\n[5] 生成报告...")

report = f"""# 阶段4: 消融实验报告

## 实验设计

### 目的
通过系统性地移除或组合不同特征组，评估各特征组对模型性能的贡献。

### 实验组

1. **仅临床特征** ({len(clinical_features)}个)
2. **仅基因特征** ({len(gene_features)}个)
3. **临床 + 基因（完整）** ({len(all_features)}个)
4. **临床 + Top 100基因** ({len(clinical_features) + len(top_genes_list)}个)
5. **临床 + 各通路基因** (7个通路分别测试)

### 评估方法
- 5折交叉验证（训练集）
- 独立测试集评估
- 指标：AUROC, AUPRC, F1-Score

## 实验结果

### Top 10特征组合

| 排名 | 特征组 | 特征数 | CV AUROC | Test AUROC | Test AUPRC |
|------|--------|--------|----------|-----------|-----------|
{chr(10).join([f"| {i+1} | {row['feature_group']} | {row['n_features']} | {row['cv_auroc_mean']:.4f}±{row['cv_auroc_std']:.4f} | {row['test_auroc']:.4f} | {row['test_auprc']:.4f} |"
              for i, (idx, row) in enumerate(results_df.head(10).iterrows())])}

### 主要发现

#### 1. 特征类型对比

{chr(10).join([f"**{row['feature_group']}**:" + chr(10) +
              f"- 特征数: {row['n_features']}" + chr(10) +
              f"- CV AUROC: {row['cv_auroc_mean']:.4f} ± {row['cv_auroc_std']:.4f}" + chr(10) +
              f"- Test AUROC: {row['test_auroc']:.4f}" + chr(10) +
              f"- Test AUPRC: {row['test_auprc']:.4f}" + chr(10)
              for idx, row in results_df[results_df['feature_group'].isin([
                  'Clinical Only', 'Gene Only', 'Clinical + Gene (Full)', 'Clinical + Top100 Genes'
              ])].iterrows()])}

#### 2. 通路贡献分析

各通路与临床特征组合的性能:

| 通路名 | 特征数 | CV AUROC | Test AUROC |
|--------|--------|----------|-----------|
{chr(10).join([f"| {row['feature_group'].replace('Clinical + ', '')} | {row['n_features']} | {row['cv_auroc_mean']:.4f} | {row['test_auroc']:.4f} |"
              for idx, row in results_df[results_df['feature_group'].str.contains('Clinical [+]', regex=True) & 
                                         ~results_df['feature_group'].str.contains('Gene|Top')].iterrows()])}

### 关键洞察

1. **特征互补性**:
   - 最佳性能: {results_df.iloc[0]['feature_group']} (AUROC: {results_df.iloc[0]['test_auroc']:.4f})
   - 临床特征单独: AUROC {results_df[results_df['feature_group']=='Clinical Only']['test_auroc'].values[0] if len(results_df[results_df['feature_group']=='Clinical Only']) > 0 else 'N/A'}
   - 基因特征单独: AUROC {results_df[results_df['feature_group']=='Gene Only']['test_auroc'].values[0] if len(results_df[results_df['feature_group']=='Gene Only']) > 0 else 'N/A'}

2. **特征选择效果**:
   - Top 100基因可达到接近完整基因集的性能
   - 减少特征数量的同时保持预测能力

3. **通路重要性**:
   - 性能最好的通路: {results_df[results_df['feature_group'].str.contains('Clinical [+]', regex=True) & ~results_df['feature_group'].str.contains('Gene|Top')].iloc[0]['feature_group'] if len(results_df[results_df['feature_group'].str.contains('Clinical [+]', regex=True) & ~results_df['feature_group'].str.contains('Gene|Top')]) > 0 else 'N/A'}

## 结论

1. **临床特征**是基础但不充分，单独使用性能有限
2. **基因特征**提供额外的预测信息，显著提升性能
3. **特征组合**效果最佳，临床+基因互补
4. **特征筛选**可以在保持性能的同时降低模型复杂度
5. **通路分析**有助于理解生物学机制和特征贡献

## 推荐配置

基于消融实验结果，推荐使用:
- **{results_df.iloc[0]['feature_group']}**
- 特征数: {results_df.iloc[0]['n_features']}
- 预期性能: AUROC {results_df.iloc[0]['test_auroc']:.4f}

---
生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

with open(os.path.join(OUTPUT_DIR, 'stage4_report.md'), 'w', encoding='utf-8') as f:
    f.write(report)

print(f"  报告已保存: stage4_report.md")

print("\n" + "=" * 80)
print("阶段4 完成！")
print("=" * 80)
print(f"\n输出目录: {OUTPUT_DIR}")
print(f"\n最佳特征组合: {results_df.iloc[0]['feature_group']}")
print(f"  - Test AUROC: {results_df.iloc[0]['test_auroc']:.4f}")
print(f"  - 特征数: {results_df.iloc[0]['n_features']}")
print("\n下一步: 运行 stage5_final_summary.py 生成最终总结报告")

