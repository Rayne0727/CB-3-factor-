import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import roc_auc_score, classification_report, roc_curve
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings('ignore')

# 中文字体设置
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

BASE_PATH = r"C:\Users\Rayne\Desktop"

print("="*60)
print("阶段三：XGBoost 强赎概率预测模型训练")
print("="*60)

# --- 1. 加载数据 ---
file_path = os.path.join(BASE_PATH, "ML_02_强赎训练集_完整特征.csv")
df = pd.read_csv(file_path)

# 按时间严格排序
df['trigger_date'] = pd.to_datetime(df['trigger_date'])
df = df.sort_values('trigger_date').reset_index(drop=True)

# 定义特征列
features = [
    'cb_close', 'cb_prem', 'cb_vol_60d', 'cb_amt_20d',
    'pe_ttm', 'pb', 'stk_vol_60d', 'stk_mom_20d',
    'remain_years', 'remain_size'
]

X = df[features]
y = df['Y']

# --- 2. OOT (Out-of-Time) 时间序列划分 ---
# 前 80% 的时间作为训练集，后 20% 作为测试集
split_idx = int(len(df) * 0.8)

X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
dates_test = df['trigger_date'].iloc[split_idx:]

print(f"-> 训练集大小: {len(X_train)} 条 (时间跨度: {df['trigger_date'].iloc[0].date()} 至 {df['trigger_date'].iloc[split_idx-1].date()})")
print(f"-> 测试集大小: {len(X_test)} 条 (时间跨度: {dates_test.iloc[0].date()} 至 {dates_test.iloc[-1].date()})")

# --- 3. 模型构建与训练 ---
# 计算正负样本比例，用于 scale_pos_weight
pos_weight = (len(y_train) - sum(y_train)) / sum(y_train)

print(f"\n-> 正在启动 XGBoost 引擎 (正负样本权重均衡比: {pos_weight:.2f})...")
model = xgb.XGBClassifier(
    n_estimators=200,          # 树的棵树
    max_depth=4,               # 树的深度（设小一点防止过拟合）
    learning_rate=0.05,        # 学习率
    subsample=0.8,             # 随机采样比例
    colsample_bytree=0.8,      # 特征采样比例
    scale_pos_weight=pos_weight, # 解决样本不平衡
    eval_metric='auc',
    random_state=42
)

# 训练模型
model.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_test, y_test)],
    verbose=False
)

# --- 4. 模型评估 ---
# 预测概率
y_pred_prob = model.predict_proba(X_test)[:, 1]
# 预测标签 (默认0.5阈值)
y_pred = model.predict(X_test)

auc_score = roc_auc_score(y_test, y_pred_prob)

print(f"测试集 AUC = {auc_score:.4f}")

print("\n分类报告 (Classification Report):")
print(classification_report(y_test, y_pred, target_names=['不强赎 (0)', '强赎 (1)']))

# --- 5. 保存模型  ---
model_path = os.path.join(BASE_PATH, "xgb_call_model.json")
model.save_model(model_path)
print(f"-> 模型已固化保存至: {model_path}")

# --- 6. 绘图：ROC曲线与特征重要性 ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# 图1：ROC 曲线
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
ax1.plot(fpr, tpr, color='#D32F2F', lw=2, label=f'XGBoost (AUC = {auc_score:.3f})')
ax1.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
ax1.set_xlabel('False Positive Rate')
ax1.set_ylabel('True Positive Rate')
ax1.set_title('强赎预测 ROC 曲线 (测试集)')
ax1.legend(loc="lower right")
ax1.grid(alpha=0.3)

# 图2：特征重要性
importances = model.feature_importances_
indices = np.argsort(importances)
features_names = [features[i] for i in indices]

ax2.barh(range(len(indices)), importances[indices], color='#1976D2', align='center')
ax2.set_yticks(range(len(indices)))
ax2.set_yticklabels(features_names)
ax2.set_title('因子特征重要性 (Feature Importance)')
ax2.set_xlabel('相对重要度')

plt.tight_layout()
save_img = os.path.join(BASE_PATH, "XGB_Model_Eval.png")
plt.savefig(save_img, dpi=300)
plt.show()

print(f"\n评估图表已保存至: {save_img}")