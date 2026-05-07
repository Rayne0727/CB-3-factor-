import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

BASE_PATH = r"C:\Users\Rayne\Desktop"

print("="*60)
print("阶段二：构建机器学习特征矩阵 (X)")
print("="*60)

# --- 1. 读取基础数据与标签 ---
print("正在加载数据...")
def parse_dt(series):
    s = series.astype(str).str.replace('-', '', regex=False).str.replace('/', '', regex=False).str.slice(0, 8)
    return pd.to_datetime(s, format='%Y%m%d', errors='coerce')

df_y = pd.read_csv(os.path.join(BASE_PATH, "ML_01_强赎标签样本.csv"))
df_y['trigger_date'] = pd.to_datetime(df_y['trigger_date'])

df_basic = pd.read_excel(os.path.join(BASE_PATH, "1_cb_basic_全字段.xlsx"))
df_cb = pd.read_excel(os.path.join(BASE_PATH, "2_cb_daily_日频.xlsx"))
df_stk = pd.read_csv(os.path.join(BASE_PATH, "7_stk_daily_正股.csv"))

df_cb['dt'] = parse_dt(df_cb['trade_date'])
df_stk['dt'] = parse_dt(df_stk['trade_date'])
df_basic['maturity_date'] = parse_dt(df_basic['maturity_date'])

# --- 2. 预计算转债因子  ---
print("正在计算转债特征...")
df_cb = df_cb.sort_values(['ts_code', 'dt']).reset_index(drop=True)

# 平价溢价率 = (转债价格 / 转股价值) - 1
df_cb['cb_prem'] = df_cb['close'] / df_cb['cb_value'] - 1

# 计算对数收益率
df_cb['cb_ret'] = np.log(df_cb['close'] / df_cb.groupby('ts_code')['close'].shift(1))

# 60日年化波动率 & 20日均成交额
df_cb['cb_vol_60d'] = df_cb.groupby('ts_code')['cb_ret'].transform(lambda x: x.rolling(60, min_periods=20).std() * np.sqrt(242))
df_cb['cb_amt_20d'] = df_cb.groupby('ts_code')['amount'].transform(lambda x: x.rolling(20, min_periods=5).mean())

# 提取转债核心列
cb_features = df_cb[['ts_code', 'dt', 'close', 'cb_prem', 'cb_vol_60d', 'cb_amt_20d']].rename(columns={'close': 'cb_close'})

# --- 3. 预计算正股因子 ---
print("正在计算正股特征...")
df_stk = df_stk.sort_values(['ts_code', 'dt']).reset_index(drop=True)

df_stk['stk_ret'] = np.log(df_stk['close_qfq'] / df_stk.groupby('ts_code')['close_qfq'].shift(1))

# 60日年化波动率
df_stk['stk_vol_60d'] = df_stk.groupby('ts_code')['stk_ret'].transform(lambda x: x.rolling(60, min_periods=20).std() * np.sqrt(242))
# 20日动量
df_stk['stk_mom_20d'] = df_stk['close_qfq'] / df_stk.groupby('ts_code')['close_qfq'].shift(20) - 1

stk_features = df_stk[['ts_code', 'dt', 'pe_ttm', 'pb', 'stk_vol_60d', 'stk_mom_20d']].rename(columns={'ts_code': 'stk_code'})

# --- 4. 特征矩阵 X ---

# 4.1 合并基础信息 (先不要 remain_size，避免数据污染)
df_final = pd.merge(df_y, df_basic[['ts_code', 'stk_code', 'maturity_date']], on='ts_code', how='left')

# 计算触发时的剩余期限 (年)
df_final['remain_years'] = (df_final['maturity_date'] - df_final['trigger_date']).dt.days / 365

# ================= 4.1.5 动态匹配历史真实剩余规模 =================
print("正在匹配历史真实动态余额 (Point-in-Time)...")
df_share = pd.read_excel(os.path.join(BASE_PATH, "6_cb_share_份额.xlsx"))
df_share['publish_date'] = parse_dt(df_share['publish_date'])

# 除以 1亿，统一成亿元
df_share['remain_size_dynamic'] = df_share['remain_size'] / 100000000
df_share = df_share.dropna(subset=['publish_date', 'remain_size_dynamic']).sort_values('publish_date')

# merge_asof 必须要求左表也是排序好的
df_final = df_final.sort_values('trigger_date')

# 按照时间点 (Point-in-Time) 向后寻找最近一次公布的余额
df_final = pd.merge_asof(
    df_final,
    df_share[['ts_code', 'publish_date', 'remain_size_dynamic']],
    by='ts_code',
    left_on='trigger_date',
    right_on='publish_date',
    direction='backward'
)
df_final = pd.merge(df_final, df_basic[['ts_code', 'remain_size', 'issue_size']], on='ts_code', how='left')

df_final['remain_size'] = df_final['remain_size_dynamic'].fillna(df_final['issue_size']).fillna(df_final['remain_size'])

df_final = df_final.drop(columns=['publish_date', 'remain_size_dynamic', 'issue_size'])
# =======================================================================
# 4.2 合并转债特征
df_final = pd.merge(df_final, cb_features, left_on=['ts_code', 'trigger_date'], right_on=['ts_code', 'dt'], how='left')
df_final = df_final.drop(columns=['dt'])

# 4.3 合并正股特征 (基于 stk_code 和 trigger_date)
df_final = pd.merge(df_final, stk_features, left_on=['stk_code', 'trigger_date'], right_on=['stk_code', 'dt'], how='left')
df_final = df_final.drop(columns=['dt'])

# --- 5. 清洗与保存 ---
print("数据清洗")
feature_cols = [
    'cb_close', 'cb_prem', 'cb_vol_60d', 'cb_amt_20d',
    'pe_ttm', 'pb', 'stk_vol_60d', 'stk_mom_20d',
    'remain_years', 'remain_size'
]

# 去除因停牌或上市时间太短导致特征缺失的极少数样本
df_ml = df_final[['ts_code', 'trigger_date', 'Y'] + feature_cols].dropna()

save_path = os.path.join(BASE_PATH, "ML_02_强赎训练集_完整特征.csv")
df_ml.to_csv(save_path, index=False, encoding='utf-8-sig')

print("\n" + "="*60)
print(f"特征矩阵已保存至: {save_path}")
print(f"原始事件数: {len(df_y)} -> 过滤缺失值后有效训练集: {len(df_ml)}")
print("="*60)