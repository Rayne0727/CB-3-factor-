import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

# ===================== 配置路径 =====================
BASE_PATH = r"C:\Users\Rayne\Desktop"

print("="*60)
print("阶段一：强赎事件捕捉与标签(Y)构建")
print("="*60)

# --- 1. 读取数据 ---
def parse_dt(series):
    s = series.astype(str).str.replace('-', '', regex=False).str.replace('/', '', regex=False).str.slice(0, 8)
    return pd.to_datetime(s, format='%Y%m%d', errors='coerce')

df_daily = pd.read_excel(os.path.join(BASE_PATH, "2_cb_daily_日频.xlsx"))
df_call = pd.read_excel(os.path.join(BASE_PATH, "4_cb_call_赎回.xlsx"))

df_daily['dt'] = parse_dt(df_daily['trade_date'])
df_call['ann_dt'] = parse_dt(df_call['ann_date'])

# 剔除空值
df_daily = df_daily.dropna(subset=['dt', 'cb_value']).sort_values(['ts_code', 'dt'])

# --- 2. 触发日 ---
trigger_events = []

for code, group in df_daily.groupby('ts_code'):
    group = group.reset_index(drop=True)
    
    # 判断单日是否大于等于 130 (即正股 >= 130%转股价)
    group['is_high'] = (group['cb_value'] >= 130).astype(int)
    
    # 滚动 30 个交易日求和
    group['high_days'] = group['is_high'].rolling(window=30, min_periods=1).sum()
    
    # 设置一个冷却期（如果触发后不强赎，通常承诺至少 3 个月不强赎）
    cooldown_until = pd.to_datetime('2000-01-01')
    
    for i, row in group.iterrows():
        if row['dt'] <= cooldown_until:
            continue
            
        # 触发条件：30天内有15天达标，且当天是达标日
        if row['high_days'] >= 15 and row['is_high'] == 1:
            trigger_events.append({
                'ts_code': code,
                'trigger_date': row['dt'],
                'cb_value': row['cb_value']
            })
            # 一旦记录为触发日，进入 90 天冷却期，避免同一个波段连续记录
            cooldown_until = row['dt'] + pd.Timedelta(days=90)

df_triggers = pd.DataFrame(trigger_events)
print(f" -> 共捕捉到 {len(df_triggers)} 次有效强赎触发事件！")

# --- 3. 匹配公告打标签 (Y) 终极修复版 ---
print("匹配赎回公告，生成可读标签 Y")

Y_labels = []
for _, row in df_triggers.iterrows():
    code = row['ts_code']
    t_date = row['trigger_date']
    
    # 窗口期：考虑到可能有提前预告或周末延迟，取触发日 [-5天, +20天] 的公告
    mask = (df_call['ts_code'] == code) & \
           (df_call['ann_dt'] >= t_date - pd.Timedelta(days=5)) & \
           (df_call['ann_dt'] <= t_date + pd.Timedelta(days=20))
    
    recent_calls = df_call[mask]
    Y = 0 
    ann_date = pd.NaT
    
    if not recent_calls.empty:
        # 遍历这个窗口期内的所有公告
        for idx, call_row in recent_calls.iterrows():
            is_call_val = str(call_row.get('is_call', '')).strip()
            
            # 核心判断逻辑：宣告了实施强赎，就打上 1 的标签
            if '公告实施强赎' in is_call_val:
                Y = 1
                break  
                
        # 记录最先发布的公告时间
        ann_date = recent_calls['ann_dt'].min()
    
    Y_labels.append({
        'ts_code': code,
        'trigger_date': t_date,
        'ann_date': ann_date,
        'Y': Y
    })

df_Y = pd.DataFrame(Y_labels)

# 将 Y 标签合并回原始触发事件表中
df_final_Y = pd.merge(df_triggers, df_Y, on=['ts_code', 'trigger_date'])

# ===================== 保存结果 =====================
save_path = os.path.join(BASE_PATH, "ML_01_强赎标签样本.csv")
df_final_Y.to_csv(save_path, index=False, encoding='utf-8-sig')

print("\n" + "="*60)
print(f"阶段一完成！样本已保存至: {save_path}")
print("整体正负样本分布：")
print(df_final_Y['Y'].value_counts().rename(index={0: "不强赎 (0)", 1: "强赎 (1)"}))
print("="*60)